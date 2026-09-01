"""Execute a notebook end to end the way Jupyter would, and report.

    ./.venv/bin/python scripts/execute_notebook.py submissions/final.ipynb

Runs every code cell in ONE namespace, in order, streaming output and timing
each cell. The only cell skipped is the `!pip install` magic, which is not
Python and whose package is already present locally.

This is the proof that the notebook runs: it exercises the real training path,
writes a real submission.csv, and trips the notebook's own asserts (the
invariance certificate and the compliance audit) if anything is wrong.
"""
import argparse, io, json, os, sys, time, traceback


def main(path, out_dir, stop_after=None):
    nb = json.load(open(path))
    cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
             if c["cell_type"] == "code"]
    ns = {"__name__": "__main__"}
    os.makedirs(out_dir, exist_ok=True)
    prev = os.getcwd()
    os.chdir(out_dir)
    t_all = time.time()
    try:
        for n, (i, src) in enumerate(cells):
            if src.lstrip().startswith("!"):
                print(f"[cell {i:2}] skipped (shell magic): {src.strip()[:50]}", flush=True)
                continue
            t0 = time.time()
            print(f"[cell {i:2}] running...", flush=True)
            try:
                exec(compile(src, f"<cell {i}>", "exec"), ns)
            except Exception:
                print(f"[cell {i:2}] FAILED after {time.time()-t0:.0f}s", flush=True)
                traceback.print_exc()
                return 1
            print(f"[cell {i:2}] ok  {time.time()-t0:.1f}s "
                  f"(total {time.time()-t_all:.0f}s)", flush=True)
            if stop_after is not None and n >= stop_after:
                print("stopping early as requested")
                break
    finally:
        os.chdir(prev)
    print(f"\nNOTEBOOK COMPLETED in {time.time()-t_all:.0f}s")
    sub = os.path.join(out_dir, "submission.csv")
    print("submission.csv exists:", os.path.isfile(sub),
          os.path.getsize(sub) if os.path.isfile(sub) else "")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--out-dir", default="/tmp/nbrun")
    ap.add_argument("--stop-after", type=int, default=None)
    a = ap.parse_args()
    sys.exit(main(os.path.abspath(a.notebook), a.out_dir, a.stop_after))
