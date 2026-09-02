"""Static name-resolution check for an exported notebook.

`src/export_notebook.py` inlines each module into ONE shared namespace and drops
its intra-package imports, which is what makes the export self-contained. The
failure mode that creates is silent: a function body that said
`from src.models import trees` and then `trees.make(...)` still *parses* after the
import is stripped, and only raises `NameError` when that line finally executes --
which for `ionic_term` was 542 seconds into a run.

This walks every code cell with `symtable`, accumulates the names each cell binds,
and reports any name a later cell reads from global scope that nothing has bound.
It costs a second and covers every cell, including branches a given run never
reaches.

    ./.venv/bin/python scripts/lint_notebook.py submissions/final.ipynb
"""
import builtins
import json
import sys
import symtable


def cell_symbols(code: str, name: str):
    """(globals_read, globals_bound) for one cell, locals correctly excluded."""
    st = symtable.symtable(code, name, "exec")
    read, bound = set(), set()

    def walk(tbl, top):
        for sym in tbl.get_symbols():
            n = sym.get_name()
            if top:
                if sym.is_assigned() or sym.is_imported():
                    bound.add(n)
                if sym.is_referenced() and not sym.is_assigned():
                    read.add(n)
            else:
                # inside a function/class: only names resolving to global scope
                if sym.is_global() and sym.is_referenced():
                    read.add(n)
        for child in tbl.get_children():
            walk(child, False)

    walk(st, True)
    return read, bound


def main(path, allow=()):
    nb = json.load(open(path))
    known = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__builtins__"}
    # Names read only inside a runtime-guarded branch (`if "gnn" in MODELS:`).
    # symtable cannot see the guard, so a reduced export needs them excused.
    known |= set(allow)
    problems = []

    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        code = "".join(c["source"])
        if code.lstrip().startswith(("!", "%")):
            continue
        try:
            read, bound = cell_symbols(code, f"<cell {i}>")
        except SyntaxError as e:
            problems.append((i, f"SYNTAX: {e}"))
            continue
        # a cell's own bindings count: order within a cell is the cell's business,
        # and a function defined here may legitimately call one defined below it.
        known |= bound
        for n in sorted(read - known):
            problems.append((i, f"undefined global: {n}"))

    if problems:
        print(f"{len(problems)} problem(s) in {path}:")
        for i, m in problems:
            print(f"  cell {i:>3}  {m}")
        return 1
    print(f"{path}: every global resolves ({len(known)} names bound)")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    allow = []
    if "--allow" in args:
        k = args.index("--allow")
        allow = args[k + 1].split(",")
        args = args[:k] + args[k + 2:]
    sys.exit(main(args[0] if args else "submissions/final.ipynb", allow))
