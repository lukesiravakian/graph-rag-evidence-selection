"""Build a multi-gold, multi-repository code-QA benchmark.

Design
------
Passages are function and class definitions extracted with ``ast``, one per
definition, identified by a *qualified* id (``repo/path.py::Outer.inner``) so ids
are unique -- unlike the single-repo benchmark, where 320 chunks carried only 280
distinct ids and collapsed at scoring time.

Questions are built from the call graph. For an *anchor* definition X with
in-repo callees, the gold set is {X} union {callees of X that are themselves
passages}, and the query is the first sentence of X's docstring. The docstring is
then **removed from X's passage text**, so the query is not a substring of any
passage it labels (the CodeSearchNet convention).

The point of this construction is that the query overlaps the anchor lexically
but not its callees. A pointwise retriever can find the anchor from the query
alone; it has no signal pointing at the complementary passages. That makes the
benchmark able to reward set-level selection, which a one-gold-per-question
benchmark cannot do even in principle.

Two recall targets are therefore scored downstream: over the full gold set, and
over gold minus the anchor, which is the circularity-free part.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

SITE = Path(".venv/lib/python3.12/site-packages")
OUT = Path("benchmark")

# Pure-Python, permissively licensed, widely read. Held-out repos are chosen in
# split_benchmark(); the list order is not meaningful.
REPOS = [
    "requests", "jinja2", "urllib3", "httpx", "charset_normalizer",
    "yarl", "dateutil", "attr", "tqdm", "packaging", "filelock", "multidict",
]

MIN_DOC_WORDS = 6
MAX_DOC_WORDS = 60
MIN_CALLEES = 2          # so every question has |G| >= 3
MAX_GOLD = 8
MIN_BODY_LINES = 3


def first_sentence(doc: str) -> str:
    """First sentence of a docstring, collapsed to one line."""
    doc = " ".join(doc.strip().split())
    m = re.match(r"(.+?[.!?])(\s|$)", doc)
    return (m.group(1) if m else doc).strip()


def strip_docstring(node: ast.AST, src: str) -> str | None:
    """Source of ``node`` with its own docstring expression removed."""
    seg = ast.get_source_segment(src, node)
    if seg is None:
        return None
    body = getattr(node, "body", None)
    if not body:
        return seg
    first = body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return seg
    doc_seg = ast.get_source_segment(src, first)
    if not doc_seg:
        return seg
    out = seg.replace(doc_seg, "", 1)
    # drop the blank line the removal leaves behind
    return re.sub(r"\n\s*\n", "\n", out, count=1)


def is_overload_stub(node: ast.AST) -> bool:
    """True for @overload / @typing.overload type signatures, which carry no body."""
    for dec in getattr(node, "decorator_list", []):
        name = dec.attr if isinstance(dec, ast.Attribute) else getattr(dec, "id", "")
        if name == "overload":
            return True
    return False


def qualname_map(tree: ast.AST) -> dict[ast.AST, str]:
    """Map each def node to its dotted qualified name."""
    out: dict[ast.AST, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}.{child.name}" if prefix else child.name
                out[child] = q
                walk(child, q)
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def called_names(node: ast.AST) -> set[str]:
    """
    In-repo call targets invoked inside ``node``, resolved conservatively.

    Two call forms are accepted:

    ``foo(...)``        a bare name, which in Python can only be a local, a module
                        global, or a builtin -- so matching it against repo-level
                        definitions is high precision.
    ``self.foo(...)``   an attribute call on ``self`` or ``cls``, which must be a
                        method of the enclosing class hierarchy.

    Every other attribute call is rejected. Without type inference there is no way
    to tell ``url.join(x)`` from ``"".join(x)``, and a repo that happens to define a
    uniquely named ``join`` would otherwise collect a false gold label from every
    ``str.join`` call in the file. Rejecting these trades gold-set size for label
    precision, which is the right trade for a retrieval benchmark: a wrong gold id
    penalizes a method for not retrieving something irrelevant.
    """
    names: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        f = sub.func
        if isinstance(f, ast.Name):
            names.add(f.id)
        elif isinstance(f, ast.Attribute):
            recv = f.value
            if isinstance(recv, ast.Name) and recv.id in ("self", "cls"):
                names.add(f.attr)
    return {n for n in names if not n.startswith("__")}


def extract_repo(repo: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = SITE / repo
    if not root.exists():
        return [], []

    passages: list[dict[str, Any]] = []
    # anchor qualified id -> set of called short names
    calls: dict[str, set[str]] = {}
    # short name -> list of passage ids defining it (for callee resolution)
    by_short: dict[str, list[str]] = {}

    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(SITE).as_posix()
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError):
            continue

        seen_in_file: dict[str, int] = {}
        for node, qual in qualname_map(tree).items():
            if is_overload_stub(node):
                continue
            text = strip_docstring(node, src)
            if not text or len(text.splitlines()) < MIN_BODY_LINES:
                continue
            # A qualified name can still repeat within one file when a definition is
            # guarded by platform or version branches. Both are real, so keep both and
            # make the id unique rather than letting them collapse at scoring time.
            n = seen_in_file.get(qual, 0)
            seen_in_file[qual] = n + 1
            pid = f"{rel}::{qual}" if n == 0 else f"{rel}::{qual}#{n + 1}"
            passages.append({
                "passage_id": pid,
                "repo": repo,
                "title": f"{path.name}::{qual}",
                "file_path": rel,
                "qualname": qual,
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "text": text,
                "docstring": (ast.get_docstring(node) or "").strip(),
            })
            calls[pid] = called_names(node)
            by_short.setdefault(qual.split(".")[-1], []).append(pid)

    # keep only unambiguous short names, so a callee resolves to exactly one passage
    unique_short = {k: v[0] for k, v in by_short.items() if len(v) == 1}

    questions: list[dict[str, Any]] = []
    for p in passages:
        doc = p["docstring"]
        if not doc:
            continue
        q = first_sentence(doc)
        n_words = len(q.split())
        if not (MIN_DOC_WORDS <= n_words <= MAX_DOC_WORDS):
            continue

        anchor = p["passage_id"]
        own_short = p["qualname"].split(".")[-1]
        callees = sorted({
            unique_short[name]
            for name in calls[anchor]
            if name in unique_short and unique_short[name] != anchor
            and name != own_short
        })
        if len(callees) < MIN_CALLEES:
            continue

        gold = [anchor] + callees[: MAX_GOLD - 1]
        questions.append({
            "qid": f"{repo}::{p['qualname']}",
            "repo": repo,
            "question": q,
            "anchor_id": anchor,
            "gold_passage_ids": gold,
            "complementary_ids": gold[1:],
        })

    return passages, questions


def main() -> None:
    all_p: list[dict[str, Any]] = []
    all_q: list[dict[str, Any]] = []
    print(f"{'repo':20s} {'passages':>9} {'questions':>10} {'mean |G|':>9}")
    for repo in REPOS:
        ps, qs = extract_repo(repo)
        if not ps:
            print(f"{repo:20s} {'--- not installed':>9}")
            continue
        all_p += ps
        all_q += qs
        mg = sum(len(q["gold_passage_ids"]) for q in qs) / len(qs) if qs else 0
        print(f"{repo:20s} {len(ps):>9} {len(qs):>10} {mg:>9.2f}")

    # Sync/async and platform variants often share a docstring verbatim. Two questions
    # with identical text but different gold sets are unanswerable by construction, so
    # drop every question whose text is not globally unique.
    from collections import Counter
    text_counts = Counter(q["question"] for q in all_q)
    ambiguous = {t for t, n in text_counts.items() if n > 1}
    dropped = [q for q in all_q if q["question"] in ambiguous]
    all_q = [q for q in all_q if q["question"] not in ambiguous]
    print(f"\ndropped {len(dropped)} questions with non-unique text "
          f"({len(ambiguous)} duplicated strings)")

    ids = {p["passage_id"] for p in all_p}
    assert len(ids) == len(all_p), "passage ids must be unique"
    # every gold id must exist as a passage
    for q in all_q:
        for g in q["gold_passage_ids"]:
            assert g in ids, f"dangling gold id {g}"

    OUT.mkdir(exist_ok=True)
    with (OUT / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in all_p:
            f.write(json.dumps(p) + "\n")
    with (OUT / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in all_q:
            f.write(json.dumps(q) + "\n")

    mg = sum(len(q["gold_passage_ids"]) for q in all_q) / len(all_q)
    print(f"\nTOTAL: {len(all_p)} passages ({len(ids)} unique), {len(all_q)} questions, "
          f"mean |G| = {mg:.2f}")
    from collections import Counter
    print("gold-set sizes:", dict(sorted(Counter(
        len(q['gold_passage_ids']) for q in all_q).items())))


if __name__ == "__main__":
    main()
