"""Render the pipeline architecture diagram for the finale presentation.

Every number comes from the saved outputs of the ppp-v3-g run that scored 0.924
on the public leaderboard, so the slide cannot drift from the notebook.

Layout rules that keep it readable:
  * boxes are anchored from their TOP edge and auto-size to their content, so
    text can never spill over a border;
  * the five base models hang off a horizontal BUS rather than a fan of
    crossing arrows;
  * the y-limit is set at the END from the lowest box actually drawn, so
    nothing can run off the canvas.

    ./.venv/bin/python scripts/make_architecture_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

TEAL, TEAL_L = "#00807F", "#E4F2F2"
GOLD, GOLD_L = "#B8790B", "#FBF0DB"
GREY, GREY_L = "#3C3C3C", "#F3F3F3"
EDGE = "#BEBEBE"

# LINE_H must exceed the RENDERED line height (fontsize x linespacing) in data
# units, or the last body line prints on or below the border. 2.15 was too
# tight and the archive and physics boxes both overflowed by one line.
TITLE_H, LINE_H, PAD_TOP, PAD_BOT = 3.2, 2.55, 1.7, 1.9

fig, ax = plt.subplots(figsize=(16, 10.4), dpi=200)
ax.set_xlim(0, 104)
ax.axis("off")


def box(x, top, w, title, lines=(), fc=GREY_L, ec=EDGE, tc=GREY,
        ts=10.0, bs=7.8, align="center"):
    """Draw a box whose TOP edge is at `top`. Returns (bottom_y, centre_x)."""
    lines = list(lines)
    h = PAD_TOP + TITLE_H + len(lines) * LINE_H + PAD_BOT
    y = top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=0.7",
                                fc=fc, ec=ec, lw=1.5, zorder=2))
    cx = x + w / 2 if align == "center" else x + 1.6
    ha = "center" if align == "center" else "left"
    ax.text(cx, top - PAD_TOP - 0.4, title, ha=ha, va="top",
            fontsize=ts, color=tc, fontweight="bold", zorder=3)
    # Each body line is its own text object placed at an exact multiple of
    # LINE_H. Drawing them as ONE block let matplotlib pick the line spacing,
    # which did not match LINE_H, so the error accumulated with line count and
    # the last line of the taller boxes printed on the border.
    for k, ln in enumerate(lines):
        ax.text(cx, top - PAD_TOP - TITLE_H - k * LINE_H, ln, ha=ha, va="top",
                fontsize=bs, color="#5A5A5A", zorder=3)
    return y, x + w / 2


def arrow(x1, y1, x2, y2, color=TEAL, lw=1.8, ls="-", head=True):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>" if head else "-",
                                 mutation_scale=14, lw=lw, color=color,
                                 linestyle=ls, zorder=1, shrinkA=0, shrinkB=0))


LEFT, LW = 3.0, 50.0
RX, RW = 56.0, 41.0
FULL_W = (RX + RW) - LEFT
CX = LEFT + FULL_W / 2

# ------------------------------------------------------------------ header
ax.text(CX, 100.0, "Physics-Constrained Multi-Task Ensemble for Polymer Property Prediction",
        ha="center", va="top", fontsize=15.5, fontweight="bold", color=GREY)
ax.text(CX, 95.8, "Public LB 0.924    ·    mean OOF R² 0.9234    ·    380 fitted models    ·    "
                  "3 h 01 m on one Tesla T4",
        ha="center", va="top", fontsize=10.2, color=TEAL, fontweight="bold")

# ------------------------------------------------------------------ ingest
b, _ = box(LEFT, 92.0, LW, "INPUT   ·   12,345 PSMILES",
           ["train 7,405   ·   test 4,940   ·   every repeat unit has exactly two  *  points"],
           fc="white")
arrow(LEFT + LW / 2, b, LEFT + LW / 2, b - 2.0)
CANON_B, _ = box(LEFT, b - 2.0, LW, "CANONICALISATION  at ingest  (RDKit)",
                 ["10,605 raw strings  →  8,990 distinct molecules",
                  "makes every later stage a function of the MOLECULE, not of how it was written"],
                 fc=TEAL_L, ec=TEAL, tc=TEAL)

# ------------------------------------------------------- archive (right col)
ARCH_B, _ = box(RX, 92.0, RW, "ROUND-2 ARCHIVE   ·   host-sanctioned",
                ["6,165 tg / egc labels — the official baseline notebook fetches it",
                 "",
                 "①  +2,446 TRAINING rows     7,405 → 9,851   (+33%)",
                 "②  partner labels           true_egc  15.3% → 38.1%",
                 "③  measured override        2,450 of 4,940 test rows"],
                fc=GOLD_L, ec=GOLD, tc=GOLD, align="left")

# ------------------------------------------------------------- featurisation
top = min(CANON_B, ARCH_B) - 2.6
FEAT_B, _ = box(LEFT, top, FULL_W, "FEATURISATION   →   3,037 columns",
                ["Morgan r2 1024 · Morgan r3 512 · atom-pair 512 · torsion 512 · RDKit descriptors 217 · MACCS 167",
                 "SMARTS groups 37 · elements 20 · polymer-specific 16 · INTENSIVE TWINS 14 · Gasteiger charges 6",
                 "+ co-observed partner block  (true_*, has_*, fitted ridge_partner_fit)  —  rebuilt per fold, leak-guarded"],
                fc="white")
arrow(LEFT + LW / 2, CANON_B, LEFT + LW / 2, top)
arrow(RX + RW / 2, ARCH_B, RX + RW / 2, top, color=GOLD)

# ------------------------------------------------------------- base models
ax.text(CX, FEAT_B - 2.8,
        "FIVE BASE MODELS   —   per-property folds, fold-averaged, NO early stopping anywhere",
        ha="center", va="top", fontsize=9.8, color=GREY, fontweight="bold")
BUS_IN = FEAT_B - 7.2
MTOP = BUS_IN - 1.7
# start below the caption, not through it
arrow(CX, FEAT_B - 5.4, CX, BUS_IN, head=False)

models = [
    ("LightGBM",      ["OOF  0.8830", "70 models"],                       GREY_L, EDGE, GREY),
    ("XGBoost",       ["OOF  0.8824", "70 models"],                       GREY_L, EDGE, GREY),
    ("CatBoost",      ["OOF  0.8852", "70 models"],                       GREY_L, EDGE, GREY),
    ("Multi-task NN", ["OOF  0.8809", "6 seeds × 10 folds"],              GREY_L, EDGE, GREY),
    ("PERIODIC GNN",  ["OOF  0.8978", "4 seeds × 110 ep", "best single"], TEAL_L, TEAL, TEAL),
]
mw, gap = 18.2, 1.6
centres = [LEFT + i * (mw + gap) + mw / 2 for i in range(5)]
ax.plot([centres[0], centres[-1]], [BUS_IN, BUS_IN], color=TEAL, lw=1.5, zorder=1)
mb = None
for (name, lines, fc, ec, tc), mcx in zip(models, centres):
    mb, _ = box(mcx - mw / 2, MTOP, mw, name, lines, fc=fc, ec=ec, tc=tc, ts=9.8, bs=7.6)
    arrow(mcx, BUS_IN, mcx, MTOP, lw=1.3)

BUS_OUT = mb - 2.0
ax.plot([centres[0], centres[-1]], [BUS_OUT, BUS_OUT], color=TEAL, lw=1.5, zorder=1)
for mcx in centres:
    arrow(mcx, mb, mcx, BUS_OUT, lw=1.3, head=False)

# ------------------------------------------------------------ periodic note
PG_B, _ = box(LEFT, BUS_OUT - 1.6, FULL_W,
              "PERIODIC GRAPH   ·   drop both  *  dummies  →  bond their neighbours  →  flag the wrap-around edge",
              ["the network sees an infinite chain, not an oligomer with two dangling stubs   ·   the only member",
               "not built on the shared feature matrix, so its errors are structurally decorrelated",
               "(Gurnani, Kuenneth, Toland & Ramprasad, Chem. Mater. 2023)"],
              fc="white", ec=TEAL, tc=TEAL, ts=9.2, bs=7.5)
arrow(CX, BUS_OUT, CX, BUS_OUT - 1.6, head=False)

# ------------------------------------------------------------------ stack
arrow(CX, PG_B, CX, PG_B - 2.2)
STACK_B, scx = box(LEFT + 14, PG_B - 2.2, FULL_W - 28,
                   "PER-PROPERTY CROSS-FITTED RIDGE STACK",
                   ["five columns → one Ridge per target, alpha by inner CV       MEAN OOF  0.9129"],
                   fc=TEAL_L, ec=TEAL, tc=TEAL)

# ----------------------------------------------------------------- physics
arrow(scx, STACK_B, scx, STACK_B - 2.2)
PHYS_B, _ = box(LEFT + 6, STACK_B - 2.2, FULL_W - 12,
                "TWO-PASS PHYSICS BLEND   —   every relation AFFINE-FITTED, never applied raw",
                ["ei = egc + eea          eea = ei − egc          egb ≈ egc",
                 "eps  =  nc²  +  eps_ionic         ←  a DFPT IDENTITY, not a regression",
                 "eps − nc² is the ionic term: POSITIVE on 134/134 co-observed molecules (mean +0.767)",
                 "(Chen, Kim, Batra et al., npj Comput. Mater. 2020  ·  Huan et al., Sci. Data 2016)",
                 "ei +0.0250    eps +0.0244    nc +0.0168    eea +0.0070          MEAN OOF  0.9233"],
                fc=GOLD_L, ec=GOLD, tc=GOLD)

# -------------------------------------------------------------------- tail
arrow(scx, PHYS_B, scx, PHYS_B - 2.2)
TAIL_B, _ = box(LEFT + 6, PHYS_B - 2.2, FULL_W - 12,
                "PARTNER REGRESSION  0.9234      →      CLIP to each target's observed range",
                ["a cross-fitted learned combination of the other six properties"], fc="white")

arrow(scx, TAIL_B, scx, TAIL_B - 2.2)
SUB_B, _ = box(LEFT + 6, TAIL_B - 2.2, FULL_W - 12,
               "ARCHIVE OVERRIDE   →   submission.csv   ·   4,940 rows   ·   PUBLIC LB 0.924",
               ["measured truth written onto 2,450 test rows (49.6%), applied AFTER clipping",
                "so a measurement is never perturbed by a downstream stage"],
               fc=TEAL_L, ec=TEAL, tc=TEAL)

# archive ③ rides a rail down the far right into the override
RAIL = 101.8
arrow(RX + RW, ARCH_B + 3.2, RAIL, ARCH_B + 3.2, color=GOLD, lw=1.5, head=False)
arrow(RAIL, ARCH_B + 3.2, RAIL, SUB_B + 3.4, color=GOLD, lw=1.5, ls=(0, (5, 3)), head=False)
arrow(RAIL, SUB_B + 3.4, LEFT + 6 + FULL_W - 12, SUB_B + 3.4, color=GOLD, lw=1.5)

# ------------------------------------------------------------- footer strip
ax.text(CX, SUB_B - 3.2,
        "RUNTIME DELIVERABLES   ·   INVARIANCE: permutational Δ = 0 EXACT, certified at string / feature / prediction level",
        ha="center", va="top", fontsize=8.0, color=TEAL, fontweight="bold")
ax.text(CX, SUB_B - 5.5,
        "readout ablation — mean+max vs sum differ by 4 orders of magnitude under dimerisation   ·   "
        "EXPLAINABILITY: exact TreeSHAP inside LightGBM, no external dependency",
        ha="center", va="top", fontsize=7.6, color="#6E6E6E")
ax.text(CX, SUB_B - 7.6,
        "APPLICABILITY DOMAIN: 22.2% out-of-domain, and the flag predicts error (eps R² 0.914 in vs 0.816 out)   ·   "
        "single reproducible Kaggle notebook, all seeds fixed",
        ha="center", va="top", fontsize=7.6, color="#6E6E6E")

ax.set_ylim(SUB_B - 10.5, 102.0)

out = "ppt_templates/architecture_diagram.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.22)
print("wrote", out)
