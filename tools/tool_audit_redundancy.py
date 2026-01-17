#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repo redundancy audit for Python KGC projects.

Outputs:
  - artifacts/redundancy_report.md
  - artifacts/redundancy_report.json
"""

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    "checkpoints", "data", "cache", "logs", "wandb", "runs",
    "outputs", "artifact", "artifacts", "venv", ".venv", "env",
}

ENTRY_HINT_PATTERNS = [
    re.compile(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:'),
    re.compile(r'argparse\.ArgumentParser\s*\('),
]

SH_REF_PAT = re.compile(r'(?:python|python3)\s+([A-Za-z0-9_./-]+\.py)')

VERSION_PAT = re.compile(
    r'(?P<base>.*?)(?:'
    r'(_v\d+)|'
    r'(_fix[a-z0-9_]+)|'
    r'(_topkonly)|'
    r'(_topk_inject)|'
    r'(_tail_only)|'
    r'(_debug)|'
    r'(_old)|'
    r'(_bak)|'
    r')$',
    re.IGNORECASE
)

DEBUG_NAME_PAT = re.compile(r'(debug_|_debug|probe_|scratch_|tmp_|temp_)', re.IGNORECASE)


@dataclass
class FileInfo:
    path: str
    is_entry_hint: bool
    referenced_by_sh_or_md: bool
    imports: List[str]
    family: str
    family_reason: str
    is_debug_named: bool


def iter_py_files(root: Path) -> List[Path]:
    res = []
    for p in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.name == '__init__.py':
            # keep package boundary files
            res.append(p)
            continue
        res.append(p)
    return sorted(res)


def read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def parse_imports(py_text: str) -> List[str]:
    try:
        tree = ast.parse(py_text)
    except Exception:
        return []
    imps = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imps.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imps.append(node.module)
    return imps


def is_entry_hint(py_text: str) -> bool:
    return any(p.search(py_text) for p in ENTRY_HINT_PATTERNS)


def collect_sh_md_refs(root: Path) -> Set[str]:
    refs = set()
    for ext in ("*.sh", "README.md", "*.md"):
        for p in root.rglob(ext):
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            text = read_text_safe(p)
            for m in SH_REF_PAT.finditer(text):
                refs.add(m.group(1))
    refs_norm = set()
    for r in refs:
        refs_norm.add(r.strip().lstrip("./"))
    return refs_norm


def file_family(stem: str) -> Tuple[str, str]:
    m = VERSION_PAT.match(stem)
    if not m:
        return stem, "no_version_suffix"
    base = m.group("base")
    base = base.rstrip("_-")
    if not base:
        base = stem
    return base, "version_suffix"


def module_name_from_path(root: Path, p: Path) -> str:
    rel = p.relative_to(root).as_posix()
    if rel.endswith(".py"):
        rel = rel[:-3]
    return rel.replace("/", ".")


def build_reachability(root: Path, py_files: List[Path], entry_modules: Set[str], mod_to_path: Dict[str, Path]) -> Set[str]:
    graph: Dict[str, Set[str]] = {m: set() for m in mod_to_path.keys()}

    for m, p in mod_to_path.items():
        text = read_text_safe(p)
        for imp in parse_imports(text):
            if imp in graph:
                graph[m].add(imp)

    reachable = set()
    q = list(entry_modules)
    while q:
        cur = q.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        for nxt in graph.get(cur, []):
            if nxt not in reachable:
                q.append(nxt)
    return reachable


def main():
    root = Path(".").resolve()
    (root / "artifacts").mkdir(exist_ok=True)

    py_files = iter_py_files(root)
    sh_md_refs = collect_sh_md_refs(root)

    mod_to_path: Dict[str, Path] = {}
    path_to_mod: Dict[str, str] = {}
    for p in py_files:
        m = module_name_from_path(root, p)
        mod_to_path[m] = p
        path_to_mod[p.relative_to(root).as_posix()] = m

    infos: List[FileInfo] = []
    entry_modules: Set[str] = set()

    for p in py_files:
        rel = p.relative_to(root).as_posix()
        text = read_text_safe(p)
        entry_hint = is_entry_hint(text)
        ref_by_sh = (rel in sh_md_refs) or (rel.lstrip("./") in sh_md_refs) or (p.name in sh_md_refs)

        if entry_hint or ref_by_sh:
            entry_modules.add(path_to_mod[rel])

        fam, reason = file_family(p.stem)
        infos.append(FileInfo(
            path=rel,
            is_entry_hint=entry_hint,
            referenced_by_sh_or_md=ref_by_sh,
            imports=parse_imports(text),
            family=fam,
            family_reason=reason,
            is_debug_named=bool(DEBUG_NAME_PAT.search(p.name)),
        ))

    reachable = build_reachability(root, py_files, entry_modules, mod_to_path)

    fam_to_files: Dict[str, List[FileInfo]] = {}
    for fi in infos:
        fam_to_files.setdefault(fi.family, []).append(fi)

    def score_keep(fi: FileInfo) -> Tuple[int, int, int, int]:
        a = 1 if fi.referenced_by_sh_or_md else 0
        b = 1 if fi.is_entry_hint else 0
        c = 1 if path_to_mod.get(fi.path, "") in reachable else 0
        d = 0 if fi.is_debug_named else 1
        return (a, b, c, d)

    keep: List[str] = []
    archive_candidates: List[str] = []
    delete_candidates: List[str] = []
    manual_review: List[str] = []

    for fam, files in sorted(fam_to_files.items(), key=lambda x: x[0]):
        if any(fi.path.endswith("/__init__.py") or fi.path == "__init__.py" for fi in files):
            for fi in files:
                keep.append(fi.path)
            continue
        if len(files) == 1:
            fi = files[0]
            if fi.path.endswith("/__init__.py") or fi.path == "__init__.py":
                keep.append(fi.path)
                continue
            m = path_to_mod.get(fi.path, "")
            is_reach = m in reachable
            if fi.is_debug_named and not (fi.referenced_by_sh_or_md or fi.is_entry_hint):
                delete_candidates.append(fi.path)
            elif not is_reach and not (fi.referenced_by_sh_or_md or fi.is_entry_hint):
                archive_candidates.append(fi.path)
            else:
                keep.append(fi.path)
            continue

        files_sorted = sorted(files, key=lambda x: score_keep(x), reverse=True)
        best = files_sorted[0]
        if best.path.endswith("/__init__.py") or best.path == "__init__.py":
            keep.append(best.path)
        else:
            keep.append(best.path)

        for fi in files_sorted[1:]:
            m = path_to_mod.get(fi.path, "")
            is_reach = m in reachable
            if fi.is_debug_named and not (fi.referenced_by_sh_or_md or fi.is_entry_hint):
                delete_candidates.append(fi.path)
            elif not is_reach and not (fi.referenced_by_sh_or_md or fi.is_entry_hint):
                archive_candidates.append(fi.path)
            else:
                manual_review.append(fi.path)

    report = {
        "entry_points_modules": sorted(entry_modules),
        "reachable_modules_count": len(reachable),
        "keep": sorted(set(keep)),
        "archive_candidates": sorted(set(archive_candidates)),
        "delete_candidates": sorted(set(delete_candidates)),
        "manual_review": sorted(set(manual_review)),
    }

    out_json = root / "artifacts" / "redundancy_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = root / "artifacts" / "redundancy_report.md"
    md = []
    md.append("# Repo Redundancy Report\n")
    md.append(f"- Reachable modules: **{len(reachable)}**\n")
    md.append(f"- Entry points detected: **{len(entry_modules)}**\n")

    def sec(title: str, items: List[str]):
        md.append(f"\n## {title} ({len(items)})\n")
        for x in items:
            md.append(f"- `{x}`\n")

    sec("KEEP (当前主线/可达/疑似入口)", report["keep"])
    sec("ARCHIVE_CANDIDATES (建议移动到 legacy/)", report["archive_candidates"])
    sec("DELETE_CANDIDATES (高概率可直接删)", report["delete_candidates"])
    sec("MANUAL_REVIEW (有引用/可达但可能是历史分支)", report["manual_review"])

    out_md.write_text("".join(md), encoding="utf-8")

    print(f"[OK] Wrote:\n  - {out_json}\n  - {out_md}")


if __name__ == "__main__":
    main()
