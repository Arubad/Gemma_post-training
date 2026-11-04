"""Build, validate and zip a competition submission from a profile (profiles/*.yaml).

    python scripts/build_submission.py --profile profiles/e4b_lora.yaml \
        --adapter main_lora=/kaggle/working/adapters/main_lora --out /kaggle/working/submission.zip

Validation mirrors the scorer's rules (HARNESS_README sections 2-3) so a bad zip is caught
locally instead of after a ~20 h queue: one root config, allowed extensions only, no symlinks /
traversal, < 3 GiB, a single allowed base model across every agent, only harness tools or
agent_tools, adapters present + valid (+ keys checked against the base checkpoint if given),
generation-config bounds, eval_config keys, and no unresolvable ADK `{state}` placeholders.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from pathlib import Path

import yaml

from .adapter_utils import adapter_size_bytes, validate_adapter
from .constants import (
    ADAPTER_NAME_RE,
    ALLOWED_FILE_EXTENSIONS,
    ALLOWED_MODEL_NAMES,
    HARNESS_TOOLS,
    MAX_FILE_COUNT,
    MAX_LORA_RANK,
    MAX_OUTPUT_TOKENS_CAP,
    MAX_SUBMISSION_BYTES,
    MODELS,
    REPO_ROOT,
    ROOT_CONFIG_NAMES,
    TOOL_ORDER,
)

TEMPLATE_DIR = REPO_ROOT / "submission_template"
EVAL_KEYS = {"timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns"}
FORBIDDEN_GEN_KEYS = {"tools", "system_instruction", "http_options", "safety_settings", "response_schema"}
STATE_VARS_ALWAYS = {"problem_description"}
_STATE_RE = re.compile(r"{+[^{}]*}+")


class SubmissionError(ValueError):
    pass


# ------------------------------------------------------------------------------- yaml + include
class _IncludeLoader(yaml.SafeLoader):
    pass


def _make_include_loader(root: Path):
    class L(_IncludeLoader):
        pass

    def _include(loader: yaml.Loader, node):
        rel = loader.construct_scalar(node)
        base = Path(getattr(loader, "_gswe_dir", root))
        if rel.startswith("/") or "\x00" in rel or ".." in Path(rel).parts:
            raise SubmissionError(f"illegal !include path: {rel}")
        p = (base / rel).resolve()
        if root.resolve() not in p.parents and p != root.resolve():
            raise SubmissionError(f"!include escapes submission root: {rel}")
        if not p.exists():
            raise SubmissionError(f"!include target missing: {rel}")
        if p.suffix in (".md", ".txt"):
            return p.read_text(encoding="utf-8")
        return load_yaml(p, root)

    L.add_constructor("!include", _include)
    return L


def load_yaml(path: Path, root: Path):
    L = _make_include_loader(root)
    with open(path, encoding="utf-8") as f:
        loader = L(f)
        loader._gswe_dir = path.parent  # type: ignore[attr-defined]
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()


# ------------------------------------------------------------------------------- profile
def _expand(v):
    if isinstance(v, str):
        return os.path.expandvars(v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def load_profile(path: str | Path) -> dict:
    p = yaml.safe_load(Path(path).read_text())
    p = _expand(p)
    p["model_name"] = MODELS.get(p["model"], p["model"])
    return p


def _gen_config(s: dict) -> dict:
    g: dict = {}
    for k in ("temperature", "top_p", "top_k", "max_output_tokens", "seed", "presence_penalty", "frequency_penalty", "stop_sequences"):
        if s.get(k) is not None:
            g[k] = s[k]
    tb = s.get("thinking_budget")
    if tb is not None:
        g["thinking_config"] = {"thinking_budget": int(tb), "include_thoughts": bool(int(tb) > 0 and s.get("include_thoughts", True))}
    return g


def _ordered_tools(tools: list[str]) -> list[str]:
    unknown = [t for t in tools if t not in HARNESS_TOOLS]
    if unknown:
        raise SubmissionError(f"unknown tools in profile: {unknown}")
    return [t for t in TOOL_ORDER if t in tools]


def _agent_yaml(agent: dict, model: str, instruction_rel: str, sampling_rel: str, extra_tools: list[str] | None = None) -> str:
    lines = [f"name: {agent['name']}", f"model: {model}"]
    if agent.get("adapter"):
        lines.append(f"adapter: {agent['adapter']}")
    if agent.get("description"):
        lines.append(f"description: {json.dumps(agent['description'])}")
    lines.append(f"instruction: !include {instruction_rel}")
    lines.append("tools:")
    for t in _ordered_tools(agent["tools"]):
        lines.append(f"  - {t}")
    for t in extra_tools or []:
        lines.append(t)
    lines.append(f"generate_content_config: !include {sampling_rel}")
    return "\n".join(lines) + "\n"


def build_dir(profile: dict, out_dir: str | Path, adapters: dict[str, str] | None = None) -> Path:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "prompts").mkdir(parents=True)
    (out / "configs").mkdir()
    model = profile["model_name"]
    root = profile["root"]
    variant = profile.get("variant", "single")

    sys_text = (TEMPLATE_DIR / root["instruction"]).read_text(encoding="utf-8")
    extra_tools = []
    if variant == "localizer":
        loc = profile["localizer"]
        sys_text = sys_text.rstrip() + "\n" + (TEMPLATE_DIR / "prompts/localizer_addendum.md").read_text(encoding="utf-8")
        (out / "sub_agents").mkdir()
        # !include resolves relative to the including file and '..' is blocked by the scorer,
        # so the sub-agent's prompt + sampling live next to its YAML.
        shutil.copy(TEMPLATE_DIR / loc["instruction"], out / "sub_agents/localizer.md")
        (out / "sub_agents/localizer_sampling.yaml").write_text(
            yaml.safe_dump(_gen_config(loc.get("sampling", profile["sampling"])), sort_keys=False))
        (out / "sub_agents/localizer.yaml").write_text(
            _agent_yaml(loc, model, "localizer.md", "localizer_sampling.yaml")
        )
        extra_tools = ["  - agent_tool:", "      config_path: sub_agents/localizer.yaml", "      skip_summarization: true"]
    elif variant != "single":
        raise SubmissionError(f"unknown variant {variant}")
    (out / "prompts/system.md").write_text(sys_text, encoding="utf-8")
    (out / "configs/sampling.yaml").write_text(yaml.safe_dump(_gen_config(profile["sampling"]), sort_keys=False))
    (out / "agent.yaml").write_text(_agent_yaml(root, model, "prompts/system.md", "configs/sampling.yaml", extra_tools))
    (out / "eval_config.yaml").write_text(yaml.safe_dump({"evaluation": profile["evaluation"]}, sort_keys=False))

    wanted = {a for a in (root.get("adapter"), (profile.get("localizer") or {}).get("adapter")) if a}
    srcs = dict(profile.get("adapters") or {})
    srcs.update(adapters or {})
    for name in sorted(wanted):
        src = srcs.get(name)
        if not src or "${" in str(src) or not Path(src).exists():
            raise SubmissionError(f"adapter '{name}' referenced but no valid source dir given (got {src!r}); "
                                  f"pass --adapter {name}=PATH or set the env var")
        dst = out / "adapters" / name
        dst.mkdir(parents=True)
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy2(Path(src) / f, dst / f)
    return out


# ------------------------------------------------------------------------------- validation
def _iter_agents(cfg: dict, cfg_path: Path, root: Path, seen=None):
    seen = seen or set()
    yield cfg_path, cfg
    refs = []
    for sa in cfg.get("sub_agents") or []:
        if isinstance(sa, dict) and "config_path" in sa:
            refs.append(sa["config_path"])
    for t in cfg.get("tools") or []:
        if isinstance(t, dict) and "agent_tool" in t:
            refs.append(t["agent_tool"]["config_path"])
    for r in refs:
        p = (cfg_path.parent / r).resolve()
        if not p.exists():
            # ADK resolves config_path relative to the referencing file; tolerate root-relative too
            p = (root / r).resolve()
        if p in seen:
            continue
        seen.add(p)
        yield from _iter_agents(load_yaml(p, root), p, root, seen)


def validate_dir(sub_dir: str | Path, base_ckpt_dir: str | Path | None = None) -> dict:
    root = Path(sub_dir)
    rep = {"ok": True, "errors": [], "warnings": [], "models": [], "adapters": {}, "size_bytes": 0, "files": 0}

    def err(m):
        rep["ok"] = False
        rep["errors"].append(m)

    roots = [n for n in ROOT_CONFIG_NAMES if (root / n).exists()]
    if len(roots) != 1:
        err(f"need exactly one root config, found {roots}")
        return rep
    total, count = 0, 0
    for p in root.rglob("*"):
        if p.is_symlink():
            err(f"symlink not allowed: {p.relative_to(root)}")
            continue
        if p.is_file():
            count += 1
            total += p.stat().st_size
            if p.suffix not in ALLOWED_FILE_EXTENSIONS:
                err(f"disallowed extension: {p.relative_to(root)}")
    rep["size_bytes"], rep["files"] = total, count
    if total >= MAX_SUBMISSION_BYTES:
        err(f"submission {total / 2**30:.2f} GiB >= 3 GiB")
    if count > MAX_FILE_COUNT:
        err(f"{count} files > {MAX_FILE_COUNT}")

    try:
        agents = list(_iter_agents(load_yaml(root / roots[0], root), root / roots[0], root))
    except (SubmissionError, yaml.YAMLError) as e:
        err(f"yaml: {e}")
        return rep
    models, used_adapters = set(), set()
    for path, a in agents:
        rel = path.relative_to(root.resolve()) if path.is_absolute() else path
        cls = a.get("agent_class", "LlmAgent")
        if cls != "LlmAgent":
            continue
        m = str(a.get("model", "")).split("/")[-1]
        models.add(m)
        if a.get("adapter"):
            used_adapters.add(a["adapter"])
        for t in a.get("tools") or []:
            if isinstance(t, str) and t not in HARNESS_TOOLS:
                err(f"{rel}: tool '{t}' is not a harness tool (undeclared tools end the task)")
        instr = a.get("instruction") or ""
        for mm in _STATE_RE.finditer(instr):
            var = mm.group().lstrip("{").rstrip("}").strip()
            opt = var.endswith("?")
            var = var.rstrip("?")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", var) or re.fullmatch(r"(app|user|temp):[A-Za-z_]\w*", var):
                if var not in STATE_VARS_ALWAYS and not opt:
                    err(f"{rel}: instruction contains ADK placeholder {{{var}}} not guaranteed in session state")
        g = a.get("generate_content_config") or {}
        bad = FORBIDDEN_GEN_KEYS & set(g)
        if bad:
            err(f"{rel}: forbidden generate_content_config keys {bad}")
        mot = g.get("max_output_tokens")
        if mot is not None and not (1 <= int(mot) <= MAX_OUTPUT_TOKENS_CAP):
            err(f"{rel}: max_output_tokens {mot} out of range")
        tc = g.get("thinking_config") or {}
        if "thinking_budget" in tc and not (0 <= int(tc["thinking_budget"]) <= 32768):
            err(f"{rel}: thinking_budget out of range")
        if "thinking_level" in tc:
            rep["warnings"].append(f"{rel}: thinking_level is ignored/unsafe with vLLM+LoRA; use thinking_budget")
    rep["models"] = sorted(models)
    if len(models) != 1:
        err(f"exactly one base model required across all agents, found {sorted(models)}")
    elif next(iter(models)) not in ALLOWED_MODEL_NAMES:
        err(f"model {next(iter(models))} not in allowed list")

    adir = root / "adapters"
    present = {p.name for p in adir.iterdir() if p.is_dir()} if adir.exists() else set()
    for a in used_adapters - present:
        err(f"agent references adapter '{a}' but adapters/{a}/ is missing")
    for a in present - used_adapters:
        rep["warnings"].append(f"adapters/{a} is unused but still loaded by vLLM (costs KV cache)")
    for a in sorted(present):
        if not ADAPTER_NAME_RE.match(a):
            err(f"bad adapter name {a}")
        r = validate_adapter(adir / a, base_ckpt_dir)
        r["size_mb"] = round(adapter_size_bytes(adir / a) / 2**20, 1)
        rep["adapters"][a] = r
        for e in r["errors"]:
            err(f"adapter {a}: {e}")
        rep["warnings"] += [f"adapter {a}: {w}" for w in r["warnings"][:5]]
        if r.get("rank") and r["rank"] > MAX_LORA_RANK:
            err(f"adapter {a}: rank {r['rank']} > {MAX_LORA_RANK}")

    ev = root / "eval_config.yaml"
    if ev.exists():
        e = (yaml.safe_load(ev.read_text()) or {}).get("evaluation", {})
        unk = set(e) - EVAL_KEYS
        if unk:
            rep["warnings"].append(f"eval_config keys ignored by scorer: {sorted(unk)}")
        mins = float(e.get("max_time_minutes", 60))
        if mins * 120 / 60 > 11:
            rep["warnings"].append(f"max_time_minutes={mins}: 120 sequential tasks may exceed the 12 h cap (whole run errors)")
    else:
        rep["warnings"].append("no eval_config.yaml: default 60 min/task WILL exceed the 12 h cap")

    rep["official_check"] = _official_check(root)
    return rep


def _official_check(root: Path) -> str:
    """Use the real adk_submission validator when it is installed (Kaggle wheelhouse)."""
    try:
        import adk_submission  # type: ignore
    except Exception:
        return "adk_submission not installed (skipped)"
    for name in ("validate_directory", "validate_submission"):
        fn = getattr(adk_submission, name, None)
        if fn is None:
            try:
                from adk_submission import validation  # type: ignore

                fn = getattr(validation, name, None)
            except Exception:
                fn = None
        if fn is not None:
            try:
                fn(root)
                return f"adk_submission.{name}: OK"
            except Exception as e:  # report, do not crash
                return f"adk_submission.{name}: FAILED: {type(e).__name__}: {e}"
    return "adk_submission installed but no known validate function found"


def zip_dir(sub_dir: str | Path, zip_path: str | Path) -> Path:
    sub_dir, zip_path = Path(sub_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as z:
        for p in sorted(sub_dir.rglob("*")):
            if p.is_file():
                comp = zipfile.ZIP_STORED if p.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
                z.write(p, p.relative_to(sub_dir).as_posix(), compress_type=comp)
    return zip_path
