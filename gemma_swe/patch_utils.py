"""Unified-diff parsing and conversion of gold patches into edit_file calls."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    section: str = ""
    lines: list[tuple[str, str]] = field(default_factory=list)  # (' '|'-'|'+', text)

    @property
    def old_lines(self) -> list[str]:
        return [t for tag, t in self.lines if tag in " -"]

    @property
    def new_lines(self) -> list[str]:
        return [t for tag, t in self.lines if tag in " +"]

    def changed_old_linenos(self) -> list[int]:
        """1-indexed line numbers in the old file that are removed, or the insertion anchor."""
        out, ln = [], self.old_start
        for tag, _ in self.lines:
            if tag == " ":
                ln += 1
            elif tag == "-":
                out.append(ln)
                ln += 1
            elif tag == "+":
                out.append(max(ln - 1, 1) if ln > 1 else 1)
        return sorted(set(out))


@dataclass
class FilePatch:
    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path if self.new_path != "/dev/null" else self.old_path

    @property
    def is_new(self) -> bool:
        if self.old_path == "/dev/null":
            return True
        return len(self.hunks) == 1 and self.hunks[0].old_start == 0 and self.hunks[0].old_len == 0

    @property
    def is_deleted(self) -> bool:
        return self.new_path == "/dev/null"

    def n_changed(self) -> int:
        return sum(1 for h in self.hunks for tag, _ in h.lines if tag in "+-")


def _strip_prefix(p: str) -> str:
    p = p.split("\t")[0].strip()
    if p in ("/dev/null", "dev/null"):
        return "/dev/null"
    if p.startswith(("a/", "b/")):
        return p[2:]
    return p


def parse_unified_diff(text: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    cur: FilePatch | None = None
    hunk: Hunk | None = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            cur = FilePatch(_strip_prefix(ln[4:]), _strip_prefix(lines[i + 1][4:]))
            files.append(cur)
            hunk = None
            i += 2
            continue
        m = _HUNK_RE.match(ln)
        if m and cur is not None:
            hunk = Hunk(
                int(m.group(1)),
                int(m.group(2)) if m.group(2) is not None else 1,
                int(m.group(3)),
                int(m.group(4)) if m.group(4) is not None else 1,
                m.group(5).strip(),
            )
            cur.hunks.append(hunk)
            i += 1
            continue
        if hunk is not None and ln and ln[0] in " +-":
            hunk.lines.append((ln[0], ln[1:]))
        elif hunk is not None and ln == "":
            # some generators drop the single space on blank context lines
            hunk.lines.append((" ", ""))
        # '\ No newline at end of file', 'diff --git', 'index ...' are ignored
        i += 1
    return files


def new_file_content(fp: FilePatch) -> str:
    return "".join(t + "\n" for h in fp.hunks for t in h.new_lines)


def apply_hunks(original: str, hunks: list[Hunk]) -> str:
    """Apply hunks by position (gold patches apply cleanly on their base commit)."""
    src = original.splitlines()
    out: list[str] = []
    pos = 0  # 0-indexed into src
    for h in hunks:
        start = max(h.old_start - 1, 0) if h.old_len > 0 else h.old_start
        out.extend(src[pos:start])
        pos = start + h.old_len
        out.extend(h.new_lines)
    out.extend(src[pos:])
    trailing = "\n" if original.endswith("\n") or not original else ""
    return "\n".join(out) + trailing


def _merge_hunks(hunks: list[Hunk], gap: int) -> list[list[Hunk]]:
    groups: list[list[Hunk]] = []
    for h in sorted(hunks, key=lambda x: x.old_start):
        if groups:
            last = groups[-1][-1]
            if h.old_start - (last.old_start + last.old_len) <= gap:
                groups[-1].append(h)
                continue
        groups.append([h])
    return groups


def hunks_to_edits(original: str, hunks: list[Hunk], merge_gap: int = 3, trim_context: int = 2,
                   max_context: int = 12) -> list[tuple[str, str]]:
    """Turn hunks into (old_string, new_string) pairs whose old_string is unique in `original`.

    Context is trimmed to `trim_context` lines on each side, then grown until unique.
    Edits are returned in file order and must be applied sequentially.
    """
    src = original.splitlines()
    patched = apply_hunks(original, hunks).splitlines()
    edits: list[tuple[str, str]] = []
    # Work on merged groups so nearby hunks become a single edit.
    delta = 0  # line offset between original and patched coordinates
    for group in _merge_hunks(hunks, merge_gap):
        first, last = group[0], group[-1]
        o_start = (first.old_start - 1) if first.old_len > 0 else first.old_start
        o_end = (last.old_start - 1 + last.old_len) if last.old_len > 0 else last.old_start
        # trim pure-context lines at both ends of the group
        lead = 0
        for tag, _ in first.lines:
            if tag == " ":
                lead += 1
            else:
                break
        tail = 0
        for tag, _ in reversed(last.lines):
            if tag == " ":
                tail += 1
            else:
                break
        core_s = o_start + lead
        core_e = o_end - tail
        n_new = sum(h.new_len - h.old_len for h in group)
        ctx = trim_context
        while True:
            s = max(0, core_s - ctx)
            e = min(len(src), max(core_e, core_s) + ctx)
            old_block = src[s:e]
            old_str = "\n".join(old_block)
            ps = s + delta
            pe = e + delta + n_new
            new_str = "\n".join(patched[ps:pe])
            if old_str.strip() and original.count(old_str) == 1:
                break
            if ctx >= max_context or (s == 0 and e == len(src)):
                break
            ctx += 2
        edits.append((old_str, new_str))
        delta += n_new
    return edits


def apply_edits(text: str, edits: list[tuple[str, str]]) -> str:
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError(f"edit old_string not unique/absent ({text.count(old)} matches)")
        text = text.replace(old, new, 1)
    return text


def files_in_patch(text: str) -> list[str]:
    return [fp.path for fp in parse_unified_diff(text)]
