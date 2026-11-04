"""Code-graph utilities for the released NetworkX graphs + 256-d node embeddings.

Graph JSON schema (About_Dataset): {directed, multigraph, graph, nodes:[{id,name,text}],
edges:[{source,target,type,key}]} (older networkx dumps use "links" instead of "edges").
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Edge:
    src: str
    dst: str
    type: str


class CodeGraph:
    def __init__(self, nodes: dict[str, dict], edges: list[Edge]):
        self.nodes = nodes  # id -> {"name":..., "text":...}
        self.edges = edges
        self.out: dict[str, list[Edge]] = defaultdict(list)
        self.inc: dict[str, list[Edge]] = defaultdict(list)
        for e in edges:
            self.out[e.src].append(e)
            self.inc[e.dst].append(e)
        self._ids = list(nodes)
        self._by_last: dict[str, list[str]] = defaultdict(list)
        for nid in self._ids:
            self._by_last[_last(nid)].append(nid)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> "CodeGraph":
        d = json.loads(Path(path).read_text())
        nodes = {}
        for n in d.get("nodes", []):
            nid = n.get("id")
            if nid is None:
                continue
            nodes[nid] = {"name": n.get("name", nid), "text": n.get("text", "") or ""}
        raw_edges = d.get("edges", d.get("links", []))
        edges = [Edge(e["source"], e["target"], str(e.get("type", "calls"))) for e in raw_edges]
        return cls(nodes, edges)

    # ------------------------------------------------------------------ harness emulation
    def resolve(self, name: str) -> str | None:
        """4-tier resolve_node_name: exact -> ./;// suffix -> case-insensitive -> substring."""
        name = name.strip()
        if not name:
            return None
        if name in self.nodes:
            return name
        suffix_hits = [n for n in self._ids if any(n.endswith(sep + name) for sep in (".", ";", "/"))]
        if suffix_hits:
            return min(suffix_hits, key=len)
        low = name.lower()
        ci = [n for n in self._ids if n.lower() == low]
        if ci:
            return ci[0]
        sub = [n for n in self._ids if low in n.lower()]
        if sub:
            return min(sub, key=len)
        return None

    def neighbors(self, node: str, edge_type: str | None = None, max_neighbors: int = 50) -> list[str]:
        nid = self.resolve(node)
        if nid is None:
            return []
        et = edge_type.lower() if edge_type else None
        seen, out = set(), []
        for e in self.inc.get(nid, []) + self.out.get(nid, []):
            if et and e.type.lower() != et:
                continue
            other = e.src if e.dst == nid else e.dst
            if other not in seen:
                seen.add(other)
                out.append(other)
            if len(out) >= max_neighbors:
                break
        return out

    def callers(self, nid: str) -> list[str]:
        return sorted({e.src for e in self.inc.get(nid, [])})

    def callees(self, nid: str) -> list[str]:
        return sorted({e.dst for e in self.out.get(nid, [])})

    def subgraph(self, names: list[str]) -> tuple[list[str], list[tuple[str, str, str]]]:
        ids = [x for x in (self.resolve(n) for n in names) if x]
        s = set(ids)
        edges = [(e.src, e.dst, e.type) for nid in ids for e in self.out.get(nid, []) if e.dst in s]
        return ids, edges

    def undirected_adj(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = defaultdict(set)
        for e in self.edges:
            if e.src == e.dst:
                continue
            adj[e.src].add(e.dst)
            adj[e.dst].add(e.src)
        return adj

    def by_last_name(self, short: str) -> list[str]:
        return self._by_last.get(short, [])


def _last(nid: str) -> str:
    return re.split(r"[.;/]", nid)[-1]


# ------------------------------------------------------------------------------- embeddings
class Embeddings:
    def __init__(self, ids: list[str], mat: np.ndarray):
        self.ids = ids
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.mat = (mat / norms).astype(np.float32)
        self.index = {n: i for i, n in enumerate(ids)}

    @classmethod
    def load(cls, path: str | Path) -> "Embeddings":
        z = np.load(path, allow_pickle=False)
        ids = [k[:-4] if k.endswith(".npy") else k for k in z.files]
        mat = np.stack([z[k].astype(np.float32).reshape(-1) for k in z.files]) if ids else np.zeros((0, 256), np.float32)
        return cls(ids, mat)

    def search(self, graph: CodeGraph | None, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Emulate search_similar_code: resolve query to a node, cosine vs all others;
        lexical token-overlap fallback for multi-word queries."""
        nid = None
        if graph is not None:
            nid = graph.resolve(query)
        elif query in self.index:
            nid = query
        if nid is not None and nid in self.index:
            v = self.mat[self.index[nid]]
            sims = self.mat @ v
            order = np.argsort(-sims)
            # "computes cosine similarity across all *other* nodes"
            out = [(self.ids[i], float(sims[i])) for i in order[: k + 1] if self.ids[i] != nid]
            return out[:k]
        toks = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", query)}
        if not toks:
            return []
        scored = []
        for n in self.ids:
            nt = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", n.replace(".", " "))}
            ov = len(toks & nt)
            if ov:
                scored.append((n, ov / len(toks)))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ------------------------------------------------------------------------------- source <-> graph
@dataclass
class Def:
    qualname: str
    kind: str  # "function" | "class"
    start: int
    end: int


def collect_defs(source: str) -> list[Def]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[Def] = []

    def visit(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}.{ch.name}" if prefix else ch.name
                start = min([ch.lineno] + [d.lineno for d in getattr(ch, "decorator_list", [])])
                out.append(Def(q, "class" if isinstance(ch, ast.ClassDef) else "function", start, ch.end_lineno or ch.lineno))
                visit(ch, q)
            elif isinstance(ch, (ast.If, ast.Try, ast.With)):
                visit(ch, prefix)

    visit(tree, "")
    return out


def module_candidates(path: str) -> list[str]:
    p = path[:-3] if path.endswith(".py") else path
    parts = [x for x in p.split("/") if x]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    cands = [".".join(parts)]
    for strip in ("src", "lib"):
        if parts and parts[0] == strip:
            cands.append(".".join(parts[1:]))
    return [c for c in cands if c]


def graph_id_for(graph: CodeGraph, path: str, qualname: str, source_text: str | None = None) -> str | None:
    for mod in module_candidates(path):
        cand = f"{mod}.{qualname}"
        if cand in graph.nodes:
            return cand
    # fall back: any node ending with qualname whose module tail matches the file stem
    stem = Path(path).stem
    hits = [n for n in graph.by_last_name(qualname.split(".")[-1]) if n.endswith(qualname)]
    if len(hits) == 1:
        return hits[0]
    hits2 = [n for n in hits if f".{stem}." in f".{n}."]
    if hits2:
        return min(hits2, key=len)
    if source_text and hits:
        first_line = next((ln.strip() for ln in source_text.splitlines() if ln.strip().startswith(("def ", "async def ", "class "))), None)
        for n in hits:
            if first_line and first_line in graph.nodes[n]["text"]:
                return n
    return None


def gold_nodes_for_file(graph: CodeGraph, path: str, original: str, changed_lines: list[int]) -> list[str]:
    """Innermost def/class containing each changed line, mapped to graph ids."""
    defs = collect_defs(original)
    golds: list[str] = []
    for ln in changed_lines:
        enclosing = [d for d in defs if d.start <= ln <= d.end]
        if not enclosing:
            continue
        inner = min(enclosing, key=lambda d: d.end - d.start)
        src = "\n".join(original.splitlines()[inner.start - 1 : inner.end])
        gid = graph_id_for(graph, path, inner.qualname, src)
        if gid and gid not in golds:
            golds.append(gid)
    return golds


# ------------------------------------------------------------------------------- issue mentions
_STOP = {
    "self", "None", "True", "False", "this", "that", "with", "from", "import", "return", "class",
    "def", "type", "value", "values", "string", "list", "dict", "test", "tests", "issue", "error",
    "file", "files", "data", "name", "names", "default", "object", "function", "method", "param",
    "params", "example", "should", "would", "could", "using", "when", "then", "there", "where",
    "which", "what", "fix", "fixes", "added", "add", "support", "update", "docs", "make", "used",
}


def extract_mentions(text: str) -> list[str]:
    """Identifier-like mentions in an issue / PR text, most specific first."""
    cands: list[str] = []
    cands += re.findall(r"`([^`\n]{2,80})`", text)
    cands += re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b", text)
    cands += re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\(", text)
    cands += re.findall(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b", text)
    cands += re.findall(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b", text)
    out, seen = [], set()
    for c in cands:
        c = c.strip().strip("()").strip()
        c = re.sub(r"\(.*$", "", c)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", c or ""):
            continue
        if c in seen or c in _STOP or len(c) < 4:
            continue
        seen.add(c)
        out.append(c)
    return out


def find_anchors(graph: CodeGraph, mentions: list[str], max_anchors: int = 8, max_ambiguity: int = 6) -> list[str]:
    anchors: list[str] = []
    for m in mentions:
        hits: list[str] = []
        if m in graph.nodes:
            hits = [m]
        elif "." in m:
            hits = [n for n in graph.nodes if n.endswith("." + m)]
            if not hits:
                hits = graph.by_last_name(m.split(".")[-1])
        else:
            hits = graph.by_last_name(m)
        if 0 < len(hits) <= max_ambiguity:
            for h in sorted(hits, key=len):
                if h not in anchors:
                    anchors.append(h)
        if len(anchors) >= max_anchors:
            break
    return anchors[:max_anchors]


def shortest_path(adj: dict[str, set[str]], sources: list[str], targets: set[str], max_depth: int = 6) -> list[str] | None:
    if not sources or not targets:
        return None
    prev: dict[str, str | None] = {s: None for s in sources}
    q = deque((s, 0) for s in sources)
    while q:
        u, d = q.popleft()
        if u in targets:
            path = [u]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return list(reversed(path))
        if d >= max_depth:
            continue
        for v in adj.get(u, ()):
            if v not in prev:
                prev[v] = u
                q.append((v, d + 1))
    return None


def hop_bucket(hops: int | None) -> str:
    if hops is None:
        return "unreachable"
    return str(hops) if hops < 3 else "3+"
