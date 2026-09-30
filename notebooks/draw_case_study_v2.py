"""
Draw publication-quality case-study diagrams in the style of
hand-drawn tree graphs with box nodes, labeled edges, alignment lines,
and explanation boxes.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches, matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.dpi": 200,
})

# ── helpers ──────────────────────────────────────────────────────────

def draw_box(ax, cx, cy, text, w=1.2, h=0.7, fc="#F5F5F5", ec="#333",
             lw=1.5, fontsize=11, fontweight="bold", text_color="#222",
             rounded=True):
    """Draw a rounded-rect box with centered text. Returns (cx, cy)."""
    style = "round,pad=0.08" if rounded else "square,pad=0.0"
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle=style, fc=fc, ec=ec, lw=lw,
                         zorder=3, transform=ax.transData)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=4)
    return cx, cy


def draw_arrow(ax, x1, y1, x2, y2, label="", color="#555", lw=1.5,
               fontsize=9, label_color=None, style="-|>", ls="-"):
    """Draw an arrow with an optional edge label at the midpoint."""
    if label_color is None:
        label_color = color
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, color=color,
                          lw=lw, mutation_scale=14,
                          connectionstyle="arc3,rad=0.0",
                          linestyle=ls, zorder=2)
    ax.add_patch(arr)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # offset label slightly
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx*dx + dy*dy) + 1e-9
        nx_, ny_ = -dy / length, dx / length
        off = 0.15
        ax.text(mx + nx_ * off, my + ny_ * off, label,
                ha="center", va="center", fontsize=fontsize,
                color=label_color, fontstyle="italic", zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))


def draw_dashed_line(ax, x1, y1, x2, y2, label="", color="#888"):
    """Draw a horizontal dashed alignment line with optional label."""
    ax.plot([x1, x2], [y1, y2], ls="--", color=color, lw=1.2, zorder=1)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label,
                ha="center", va="bottom", fontsize=9, color=color)


def draw_title_bar(ax, cx, cy, text, w=4.5, h=0.55, fc="#1976D2"):
    """Draw a colored title bar (like the blue/orange header)."""
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.05", fc=fc, ec="none",
                         lw=0, zorder=3)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=12,
            fontweight="bold", color="white", zorder=4)


def draw_explanation_box(ax, cx, cy, lines, w=5.0, fc="#F5F5F5", ec="#999"):
    """Draw a multi-line explanation box."""
    text = "\n".join(lines)
    h = 0.35 * len(lines) + 0.3
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.15", fc=fc, ec=ec,
                         lw=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10,
            color="#333", zorder=4, linespacing=1.5)


# ── Case 1: Doc 1072 — Intrinsic Error ──────────────────────────────

def draw_case_1072():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-4, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("SAK sees the same syntax;  S2-K catches the semantic mismatch",
                 fontsize=16, fontweight="bold", y=0.96)

    # ── Left: Human summary graph ──
    LX = 3.5
    draw_title_bar(ax, LX, 7.5, "Human summary graph (correct)", w=5.5, fc="#1976D2")

    draw_box(ax, LX, 6.2, "Huawei\nORG", fc="#E3F2FD", ec="#1565C0")
    draw_box(ax, LX, 4.5, "threatened\nVERB", fc="#E0E0E0", ec="#555")
    draw_box(ax, LX + 2.5, 4.5, "file\nVERB", fc="#E0E0E0", ec="#555")
    draw_box(ax, LX + 2.5, 2.8, "action\nNOUN", fc="#E0E0E0", ec="#555")
    draw_box(ax, LX - 2.5, 2.8, "Samsung\nORG", fc="#E3F2FD", ec="#1565C0")
    draw_box(ax, LX, 1.1, "infringement\nNOUN", fc="#E0E0E0", ec="#555")

    draw_arrow(ax, LX, 5.85, LX, 6.2 - 0.35 - 1.0 + 0.35, label="nsubj", color="#555")
    # threatened → Huawei (upward)
    draw_arrow(ax, LX, 4.85, LX, 5.85, label="nsubj", color="#666")
    # threatened → file
    draw_arrow(ax, LX + 0.6, 4.5, LX + 2.5 - 0.6, 4.5, label="xcomp", color="#666")
    # file → action
    draw_arrow(ax, LX + 2.5, 4.15, LX + 2.5, 3.15, label="dobj", color="#666")
    # file → Samsung
    draw_arrow(ax, LX + 2.5 - 0.6, 4.15, LX - 2.5 + 0.6, 3.15, label="prep:against", color="#666")
    # action → infringement
    draw_arrow(ax, LX + 2.5 - 0.6, 2.45, LX + 0.6, 1.45, label="prep:over", color="#666")

    # ── Right: Candidate graph ──
    RX = 12.5
    draw_title_bar(ax, RX, 7.5, "Candidate graph (with error)", w=5.5, fc="#E65100")

    draw_box(ax, RX, 6.2, "Huawei\nORG", fc="#E3F2FD", ec="#1565C0")
    draw_box(ax, RX, 4.5, "filed\nVERB", fc="#FFCDD2", ec="#C62828", lw=2.5)
    draw_box(ax, RX + 2.5, 2.8, "action\nNOUN", fc="#E0E0E0", ec="#555")
    draw_box(ax, RX - 2.5, 2.8, "Samsung\nORG", fc="#E3F2FD", ec="#1565C0")
    draw_box(ax, RX, 1.1, "infringement\nNOUN", fc="#E0E0E0", ec="#555")

    # filed → Huawei
    draw_arrow(ax, RX, 4.85, RX, 5.85, label="nsubj", color="#666")
    # filed → action (direct dobj, no xcomp)
    draw_arrow(ax, RX + 0.6, 4.15, RX + 2.5 - 0.6, 3.15, label="dobj", color="#666")
    # filed → Samsung
    draw_arrow(ax, RX - 0.6, 4.15, RX - 2.5 + 0.6, 3.15, label="prep:against", color="#666")
    # action → infringement
    draw_arrow(ax, RX + 2.5 - 0.6, 2.45, RX + 0.6, 1.45, label="prep:over", color="#666")

    # ── Alignment dashed lines ──
    draw_dashed_line(ax, LX + 0.6, 6.2, RX - 0.6, 6.2, label="aligned nodes")
    draw_dashed_line(ax, LX + 0.6, 4.5, RX - 0.6, 4.5, label="aligned nodes")
    draw_dashed_line(ax, LX - 2.5 + 0.6, 2.8, RX - 2.5 - 0.6, 2.8)
    draw_dashed_line(ax, LX + 2.5 + 0.6, 2.8, RX + 2.5 - 0.6, 2.8)
    draw_dashed_line(ax, LX + 0.6, 1.1, RX - 0.6, 1.1)

    # ── Sentence text ──
    ax.text(LX, -0.5,
            'Huawei has threatened to file legal action\nagainst Samsung over alleged patent infringement.',
            ha="center", va="top", fontsize=10, fontstyle="italic", color="#333")
    ax.text(RX, -0.5,
            'Huawei has filed legal action\nagainst Samsung over alleged patent infringement.',
            ha="center", va="top", fontsize=10, fontstyle="italic", color="#C62828")

    # ── Explanation box ──
    draw_explanation_box(ax, 8, -2.5, [
        'SAK:  isomorphic dep-tree structure  →  match',
        '        (nsubj → root → dobj, same relations)',
        '',
        'S2-K: embed("threatened") ≠ embed("filed")',
        '        intent vs. completed action  →  factual error',
    ], w=8.5, fc="#FFF9C4", ec="#F9A825")

    outpath = os.path.join(OUT_DIR, "case_study_1072_v2.png")
    fig.savefig(outpath, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Case 2: Doc 1051 — Extrinsic Error ──────────────────────────────

def draw_case_1051():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-5, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("SAK sees the same syntax;  S2-K catches the extrinsic entity",
                 fontsize=16, fontweight="bold", y=0.96)

    # ── Left: Human summary ──
    LX = 3.5
    draw_title_bar(ax, LX, 7.8, "Human summary graph (correct)", w=5.5, fc="#1976D2")

    draw_box(ax, LX, 6.5, "woman\nNOUN", fc="#E0E0E0", ec="#555")
    draw_box(ax, LX, 4.8, "jailed\nVERB", fc="#E0E0E0", ec="#555")
    draw_box(ax, LX - 2.2, 3.1, "part\nNOUN", fc="#E0E0E0", ec="#555")
    draw_box(ax, LX + 2.2, 3.1, "killing\nNOUN", fc="#E0E0E0", ec="#555")
    draw_box(ax, LX + 2.2, 1.4, "player\nNOUN", fc="#E0E0E0", ec="#555")
    draw_box(ax, LX + 0.5, 1.4, "poker\nNOUN", fc="#E0E0E0", ec="#555", w=1.0)

    # Edges
    draw_arrow(ax, LX, 5.15, LX, 5.85, label="nsubjpass", color="#666")
    draw_arrow(ax, LX - 0.6, 4.45, LX - 2.2 + 0.6, 3.45, label="prep:for", color="#666")
    draw_arrow(ax, LX + 0.6, 4.45, LX + 2.2 - 0.6, 3.45, label="prep:in", color="#666")
    draw_arrow(ax, LX + 2.2, 2.75, LX + 2.2, 1.75, label="prep:of", color="#666")
    draw_arrow(ax, LX + 2.2 - 0.6, 1.4, LX + 0.5 + 0.5, 1.4, label="compound", color="#666")

    # ── Right: Candidate ──
    RX = 12.5
    draw_title_bar(ax, RX, 7.8, "Candidate graph (with error)", w=5.5, fc="#E65100")

    draw_box(ax, RX, 6.5, "woman\nNOUN", fc="#E0E0E0", ec="#555")
    draw_box(ax, RX, 4.8, "jailed\nVERB", fc="#E0E0E0", ec="#555")
    draw_box(ax, RX - 2.2, 3.1, "part\nNOUN", fc="#E0E0E0", ec="#555")
    draw_box(ax, RX + 2.2, 3.1, "killing\nNOUN", fc="#E0E0E0", ec="#555")
    draw_box(ax, RX + 2.2, 1.4, "player\nNOUN", fc="#E0E0E0", ec="#555")
    draw_box(ax, RX + 0.5, 1.4, "poker\nNOUN", fc="#E0E0E0", ec="#555", w=1.0)

    # Error nodes
    draw_box(ax, RX + 2.2, -0.3, "London\nPROPN", fc="#FFCDD2", ec="#C62828", lw=2.5)
    draw_box(ax, RX + 0.5, -0.3, "north\nNOUN", fc="#FFCDD2", ec="#C62828", lw=2.5, w=1.0)
    draw_box(ax, RX - 2.0, -0.3, "«London»\nMention", fc="#FFE0B2", ec="#E65100", lw=2.0)
    draw_box(ax, RX - 2.0, -1.8, "[Q84]\nEntity", fc="#F8BBD0", ec="#AD1457", lw=2.0)

    # Edges (same structure as left)
    draw_arrow(ax, RX, 5.15, RX, 5.85, label="nsubjpass", color="#666")
    draw_arrow(ax, RX - 0.6, 4.45, RX - 2.2 + 0.6, 3.45, label="prep:for", color="#666")
    draw_arrow(ax, RX + 0.6, 4.45, RX + 2.2 - 0.6, 3.45, label="prep:in", color="#666")
    draw_arrow(ax, RX + 2.2, 2.75, RX + 2.2, 1.75, label="prep:of", color="#666")
    draw_arrow(ax, RX + 2.2 - 0.6, 1.4, RX + 0.5 + 0.5, 1.4, label="compound", color="#666")

    # Extra edges from error nodes
    draw_arrow(ax, RX + 2.2, 1.05, RX + 2.2, 0.05, label="prep:in", color="#C62828")
    draw_arrow(ax, RX + 2.2 - 0.6, -0.3, RX + 0.5 + 0.5, -0.3, label="compound", color="#C62828")
    draw_arrow(ax, RX - 0.5, -0.3, RX + 0.5 - 0.5, -0.3, label="RELATED_TO", color="#1976D2",
               fontsize=8, label_color="#1976D2")
    draw_arrow(ax, RX - 2.0, -0.65, RX - 2.0, -1.45, label="BELONGS_TO", color="#F57C00",
               fontsize=8, label_color="#F57C00")

    # Alignment lines
    draw_dashed_line(ax, LX + 0.6, 6.5, RX - 0.6, 6.5, label="aligned nodes")
    draw_dashed_line(ax, LX + 0.6, 4.8, RX - 0.6, 4.8)
    draw_dashed_line(ax, LX - 2.2 + 0.6, 3.1, RX - 2.2 - 0.6, 3.1)
    draw_dashed_line(ax, LX + 2.2 + 0.6, 3.1, RX + 2.2 - 0.6, 3.1)
    draw_dashed_line(ax, LX + 2.2 + 0.6, 1.4, RX + 0.5 - 0.5, 1.4)

    # Sentences
    ax.text(LX, -2.5,
            'A woman has been jailed for her part\nin the "wicked" killing of a poker player.',
            ha="center", va="top", fontsize=10, fontstyle="italic", color="#333")
    ax.text(RX, -2.5,
            'A woman has been jailed ... killing of\na poker player in north London.',
            ha="center", va="top", fontsize=10, fontstyle="italic", color="#C62828")

    # Explanation
    draw_explanation_box(ax, 8, -4.0, [
        'SAK:  same dep-tree topology for shared nodes  →  match',
        '        ("in north London" = ordinary PP, same as any other PP)',
        '',
        'S2-K: "London" entity has no matching node in source graph',
        '        (source says "Mayfair")  →  lower similarity  →  factual error',
    ], w=9.5, fc="#FFF9C4", ec="#F9A825")

    outpath = os.path.join(OUT_DIR, "case_study_1051_v2.png")
    fig.savefig(outpath, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Case 3: Doc 1079 — Extrinsic Error ──────────────────────────────

def draw_case_1079():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-4, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("SAK sees the same syntax;  S2-K catches the unsupported entity",
                 fontsize=16, fontweight="bold", y=0.96)

    LX = 3.5
    draw_title_bar(ax, LX, 7.5, "Human summary graph (correct)", w=5.5, fc="#1976D2")

    draw_box(ax, LX, 6.2, "Lichfield\nPROPN", fc="#E3F2FD", ec="#1565C0")
    draw_box(ax, LX, 4.5, "left\nVERB", fc="#E0E0E0", ec="#555")
    draw_box(ax, LX + 2.5, 2.8, "competition\nNOUN", fc="#E0E0E0", ec="#555")
    draw_box(ax, LX, 2.8, "Rugby\nPROPN", fc="#E0E0E0", ec="#555", w=1.0)
    draw_box(ax, LX - 2.5, 2.8, "Women\nNOUN", fc="#E0E0E0", ec="#555", w=1.1)

    draw_arrow(ax, LX, 5.85, LX, 5.55, label="nsubjpass", color="#666")
    draw_arrow(ax, LX, 4.85, LX, 5.85, label="nsubjpass", color="#666")
    draw_arrow(ax, LX + 0.6, 4.15, LX + 2.5 - 0.6, 3.15, label="prep:of", color="#666")
    draw_arrow(ax, LX + 2.5 - 0.6, 2.8, LX + 0.5, 2.8, label="compound", color="#666")
    draw_arrow(ax, LX - 0.6, 4.15, LX - 2.5 + 0.55, 3.15, label="poss", color="#666")

    # Right
    RX = 12.5
    draw_title_bar(ax, RX, 7.5, "Candidate graph (with error)", w=5.5, fc="#E65100")

    draw_box(ax, RX, 6.2, "Lichfield\nPROPN", fc="#E3F2FD", ec="#1565C0")
    draw_box(ax, RX, 4.5, "left\nVERB", fc="#E0E0E0", ec="#555")
    draw_box(ax, RX + 2.5, 2.8, "competition\nNOUN", fc="#E0E0E0", ec="#555")
    draw_box(ax, RX, 2.8, "Rugby\nPROPN", fc="#E0E0E0", ec="#555", w=1.0)
    draw_box(ax, RX - 2.5, 2.8, "Women\nNOUN", fc="#E0E0E0", ec="#555", w=1.1)

    # Error nodes
    draw_box(ax, RX - 2.5, 1.0, "«RFU»\nMention", fc="#FFCDD2", ec="#C62828", lw=2.5, w=1.3)
    draw_box(ax, RX, 1.0, "Football\nPROPN", fc="#FFCDD2", ec="#C62828", lw=2.0)
    draw_box(ax, RX + 2.5, 1.0, "Union\nPROPN", fc="#FFCDD2", ec="#C62828", lw=2.0, w=1.0)

    draw_arrow(ax, RX, 5.85, RX, 5.55, label="nsubjpass", color="#666")
    draw_arrow(ax, RX, 4.85, RX, 5.85, label="nsubjpass", color="#666")
    draw_arrow(ax, RX + 0.6, 4.15, RX + 2.5 - 0.6, 3.15, label="prep:of", color="#666")
    draw_arrow(ax, RX + 2.5 - 0.6, 2.8, RX + 0.5, 2.8, label="compound", color="#666")
    draw_arrow(ax, RX - 0.6, 4.15, RX - 2.5 + 0.55, 3.15, label="poss", color="#666")

    draw_arrow(ax, RX + 2.5, 2.45, RX + 2.5, 1.35, label="poss", color="#C62828")
    draw_arrow(ax, RX + 2.5 - 0.5, 1.0, RX + 0.6, 1.0, label="compound", color="#C62828")
    draw_arrow(ax, RX - 0.6, 1.0, RX - 2.5 + 0.65, 1.0, label="RELATED_TO", color="#1976D2",
               fontsize=8, label_color="#1976D2")

    # Alignment
    draw_dashed_line(ax, LX + 0.6, 6.2, RX - 0.6, 6.2, label="aligned nodes")
    draw_dashed_line(ax, LX + 0.6, 4.5, RX - 0.6, 4.5)
    draw_dashed_line(ax, LX + 2.5 + 0.6, 2.8, RX + 2.5 - 0.6, 2.8)

    # Sentences
    ax.text(LX, -0.5,
            "Lichfield have been left out of the new\nWomen's Super Rugby competition.",
            ha="center", va="top", fontsize=10, fontstyle="italic", color="#333")
    ax.text(RX, -0.5,
            "Lichfield have been left out of the\nRugby Football Union's new ... competition.",
            ha="center", va="top", fontsize=10, fontstyle="italic", color="#C62828")

    draw_explanation_box(ax, 8, -2.5, [
        'SAK:  "Rugby Football Union\'s" is just a possessive modifier',
        '        — same syntactic role as "Women\'s"  →  match',
        '',
        'S2-K: "Rugby Football Union" entity absent in source graph',
        '        → unmatched Mention/Entity nodes  →  lower similarity',
    ], w=9.0, fc="#FFF9C4", ec="#F9A825")

    outpath = os.path.join(OUT_DIR, "case_study_1079_v2.png")
    fig.savefig(outpath, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {outpath}")


OUT_DIR = "/root/autodl-tmp/FactGraphGAWL/experiments/figures"
os.makedirs(OUT_DIR, exist_ok=True)

draw_case_1072()
draw_case_1051()
draw_case_1079()
print("All done.")
