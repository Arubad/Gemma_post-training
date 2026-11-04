#!/usr/bin/env python
"""Convert `# %%` percent-format scripts to .ipynb (no jupytext needed).

    python scripts/py2ipynb.py notebooks/*.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def convert(src: Path) -> Path:
    cells, cur, kind = [], [], "code"

    def flush():
        nonlocal cur
        text = "\n".join(cur).strip("\n")
        if text:
            if kind == "markdown":
                text = "\n".join(re.sub(r"^# ?", "", ln) for ln in text.splitlines())
            lines = text.splitlines(keepends=True)
            cell = {"cell_type": kind, "metadata": {}, "source": lines}
            if kind == "code":
                cell.update(outputs=[], execution_count=None)
            cells.append(cell)
        cur = []

    for ln in src.read_text().splitlines():
        m = re.match(r"^# %%(.*)$", ln)
        if m:
            flush()
            kind = "markdown" if "[markdown]" in m.group(1) else "code"
            title = m.group(1).replace("[markdown]", "").strip()
            if title and kind == "code":
                cur.append(f"# {title}")
            continue
        cur.append(ln)
    flush()
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = src.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1))
    return out


if __name__ == "__main__":
    if not sys.argv[1:] or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    for p in sys.argv[1:]:
        print(convert(Path(p)))
