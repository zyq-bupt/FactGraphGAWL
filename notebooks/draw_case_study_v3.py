"""
Draw publication-quality case-study diagrams — perfect isomorphic-tree cases
where SAK fails (identical syntax) but S2-K succeeds (different semantics).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "figure.dpi": 200,
})

OUT_DIR = "/root/autodl-tmp/FactGraphGAWL/experiments/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── shared sizes ──
BOX_FONT = 13
EDGE_FONT = 11
LABEL_FONT = 12
SENT_FONT = 12
EXPL_FONT = 12
TITLE_BAR_FONT = 14
SUPTITLE_FONT = 18
TICK_FONT = 11


def draw_box(ax, cx, cy, text, w=1.5, h=0.8, fc="#F5F5F5", ec="#555",
             lw=1.5, fontsize=BOX_FONT, fontweight="bold", text_color="#222"):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, label="", color="#555", lw=1.5,
               fontsize=EDGE_FONT, label_color=None, head_w=10):
    if label_color is None:
        label_color = color
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle="-|>", color=color, lw=lw,
                          mutation_scale=head_w, zorder=2)
    ax.add_patch(arr)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        ln = np.sqrt(dx*dx + dy*dy) + 1e-9
        nx_, ny_ = -dy / ln, dx / ln
        off = 0.20
        ax.text(mx + nx_ * off, my + ny_ * off, label,
                ha="center", va="center", fontsize=fontsize,
                color=label_color, fontstyle="italic", zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))


def draw_dashed(ax, x1, y1, x2, y2, label="", color="#999"):
    ax.plot([x1, x2], [y1, y2], ls="--", color=color, lw=1.3, zorder=1)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.20, label,
                ha="center", va="bottom", fontsize=TICK_FONT, color=color)


def draw_title_bar(ax, cx, cy, text, w=5.5, h=0.7, fc="#1976D2"):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.06", fc=fc, ec="none", zorder=3)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=TITLE_BAR_FONT,
            fontweight="bold", color="white", zorder=4)


def draw_expl_box(ax, cx, cy, lines, w=9.0, fc="#FFF9C4", ec="#F9A825"):
    text = "\n".join(lines)
    h = 0.42 * len(lines) + 0.4
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.18", fc=fc, ec=ec, lw=1.3, zorder=3)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=EXPL_FONT,
            color="#333", zorder=4, linespacing=1.6, family="DejaVu Sans")


# ═══════════════════════════════════════════════════════════════════════
# Case A: Doc 136
# ═══════════════════════════════════════════════════════════════════════

def draw_case_136():
    fig, ax = plt.subplots(figsize=(20, 13))
    ax.set_xlim(-3, 21)
    ax.set_ylim(-6, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("DeFacto — Doc 136  (Intrinsic Error)",
                 fontsize=SUPTITLE_FONT, fontweight="bold", y=0.95)

    LX = 4.0
    RX = 16.0
    MID = (LX + RX) / 2

    draw_title_bar(ax, LX, 9.3, "Human summary graph (correct)", w=6.5, fc="#1976D2")
    draw_title_bar(ax, RX, 9.3, "Candidate graph (with error)", w=6.5, fc="#E65100")

    y_top = 7.8
    y_root = 6.0
    y_mid = 4.2
    y_bot = 2.3
    y_mention = 0.5

    for X, quant, q_fc, q_ec, q_lw, m_text, m_label, m_color in [
        (LX, "Hundreds\nNOUN", "#C8E6C9", "#2E7D32", 2.5,
         "«Hundreds of\nthousands»", "✓ correct", "#2E7D32"),
        (RX, "Tens\nNOUN", "#FFCDD2", "#C62828", 2.5,
         "«Tens of\nthousands»", "✗ error", "#C62828"),
    ]:
        draw_box(ax, X, y_root, "taken\nVERB")
        draw_box(ax, X - 1.5, y_top, "thousands\nNOUN")
        draw_box(ax, X - 3.0, y_bot, quant, fc=q_fc, ec=q_ec, lw=q_lw)
        draw_box(ax, X - 1.0, y_mid, "part\nNOUN", w=1.1)
        draw_box(ax, X + 2.0, y_mid, "rallies\nNOUN")
        draw_box(ax, X + 2.0, y_bot, "independence\nNOUN", w=1.8)
        draw_box(ax, X - 0.5, y_bot, "Catalan\nPROPN", w=1.3)

        draw_box(ax, X - 3.0, y_mention, m_text + "\nMention",
                 fc="#FFE0B2", ec="#E65100", w=2.2, h=1.1, fontsize=11)

        draw_arrow(ax, X - 0.6, y_root, X - 1.5 + 0.75, y_top - 0.4,
                   label="nsubj")
        draw_arrow(ax, X - 1.5, y_top - 0.4, X - 3.0 + 0.75, y_bot + 0.4,
                   label="quantmod")
        draw_arrow(ax, X - 0.2, y_root - 0.4, X - 1.0, y_mid + 0.4,
                   label="dobj")
        draw_arrow(ax, X + 0.75, y_root - 0.4, X + 2.0 - 0.75, y_mid + 0.4,
                   label="prep:in")
        draw_arrow(ax, X + 2.0, y_mid - 0.4, X + 2.0, y_bot + 0.4,
                   label="compound")
        draw_arrow(ax, X + 2.0 - 0.75, y_mid - 0.4, X - 0.5 + 0.65, y_bot + 0.4,
                   label="amod")

        draw_arrow(ax, X - 3.0, y_mention + 0.55, X - 3.0, y_bot - 0.4,
                   label="RELATED_TO", color="#1976D2", fontsize=10, label_color="#1565C0")

        ax.text(X - 3.0, y_mention - 0.8, m_label,
                ha="center", fontsize=LABEL_FONT, color=m_color, fontweight="bold")

    # Alignment
    draw_dashed(ax, LX + 0.75, y_root, RX - 0.75, y_root, label="aligned nodes")
    draw_dashed(ax, LX - 1.5 + 0.75, y_top, RX - 1.5 - 0.75, y_top)
    draw_dashed(ax, LX + 2.0 + 0.9, y_mid, RX + 2.0 - 0.9, y_mid)

    y_route = y_bot - 0.7
    ax.plot([LX - 3.0 + 0.75, LX - 3.0 + 0.75, RX - 3.0 - 0.75, RX - 3.0 - 0.75],
            [y_bot - 0.4, y_route, y_route, y_bot - 0.4],
            ls="--", color="#C62828", lw=1.5, zorder=1)
    ax.text(MID, y_route - 0.22, "aligned nodes", ha="center", va="top",
            fontsize=TICK_FONT, color="#C62828")

    # Sentences
    ax.text(LX, -1.5,
            'Hundreds of thousands of people have taken part\n'
            'in pro-Catalan independence rallies across the region.',
            ha="center", va="top", fontsize=SENT_FONT, fontstyle="italic", color="#333")
    ax.text(RX, -1.5,
            'Tens of thousands of people have taken part\n'
            'in pro-Catalan independence rallies across the region.',
            ha="center", va="top", fontsize=SENT_FONT, fontstyle="italic", color="#C62828")

    draw_expl_box(ax, MID, -4.2, [
        'SAK:  isomorphic trees (same nodes, same edges)  →  match',
        '        "Hundreds" and "Tens" occupy identical syntactic positions (quantmod)',
        '',
        'S2-K:  embed("Hundreds") ≠ embed("Tens")',
        '        quantifier mismatch  →  lower similarity  →  factual error detected',
    ], w=11.5, fc="#FFF9C4", ec="#F9A825")

    outpath = os.path.join(OUT_DIR, "case_study_136_v3.png")
    fig.savefig(outpath, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════════════
# Case B: Doc 1821
# ═══════════════════════════════════════════════════════════════════════

def draw_case_1821():
    fig, ax = plt.subplots(figsize=(20, 13))
    ax.set_xlim(-2, 20)
    ax.set_ylim(-5.5, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("DeFacto — Doc 1821  (Intrinsic Error)",
                 fontsize=SUPTITLE_FONT, fontweight="bold", y=0.95)

    LX, RX = 4.0, 15.0
    MID = (LX + RX) / 2
    draw_title_bar(ax, LX, 9.3, "Human summary graph (correct)", w=6.5, fc="#1976D2")
    draw_title_bar(ax, RX, 9.3, "Candidate graph (with error)", w=6.5, fc="#E65100")

    y_fans = 7.8
    y_react = 6.3
    y_news = 4.8
    y_played = 3.3
    y_bot = 1.6

    for X, aux_word, aux_fc, aux_ec, aux_lw in [
        (LX, "could\nMOD", "#C8E6C9", "#2E7D32", 2.5),
        (RX, "will\nMOD", "#FFCDD2", "#C62828", 2.5),
    ]:
        draw_box(ax, X, y_fans, "fans\nNOUN")
        draw_box(ax, X, y_react, "reacting\nVERB")
        draw_box(ax, X, y_news, "news\nNOUN", w=1.2)
        draw_box(ax, X, y_played, "played\nVERB")
        draw_box(ax, X - 2.8, y_bot, "Hermione\nPROPN", fc="#E3F2FD", ec="#1565C0", w=1.6)
        draw_box(ax, X, y_bot, aux_word, fc=aux_fc, ec=aux_ec, lw=aux_lw, w=1.2)
        draw_box(ax, X + 2.8, y_bot, "actress\nNOUN")

        draw_arrow(ax, X, y_react + 0.4, X, y_fans - 0.4, label="nsubj")
        draw_arrow(ax, X, y_react - 0.4, X, y_news + 0.4, label="prep:to")
        draw_arrow(ax, X, y_news - 0.4, X, y_played + 0.4, label="ccomp")
        draw_arrow(ax, X - 0.7, y_played - 0.4, X - 2.8 + 0.8, y_bot + 0.4,
                   label="nsubjpass")
        draw_arrow(ax, X, y_played - 0.4, X, y_bot + 0.4, label="aux")
        draw_arrow(ax, X + 0.7, y_played - 0.4, X + 2.8 - 0.75, y_bot + 0.4,
                   label="prep:by")

    # Alignment
    for y in [y_fans, y_react, y_news, y_played]:
        draw_dashed(ax, LX + 0.75, y, RX - 0.75, y,
                    label="aligned nodes" if y == y_fans else "")
    draw_dashed(ax, LX - 2.8 + 0.8, y_bot, RX - 2.8 - 0.8, y_bot)
    draw_dashed(ax, LX + 2.8 + 0.75, y_bot, RX + 2.8 - 0.75, y_bot)
    draw_dashed(ax, LX + 0.6, y_bot, RX - 0.6, y_bot, label="aligned nodes", color="#C62828")

    ax.text(LX, y_bot - 0.9, "✓ correct", ha="center", fontsize=LABEL_FONT,
            color="#2E7D32", fontweight="bold")
    ax.text(RX, y_bot - 0.9, "✗ error", ha="center", fontsize=LABEL_FONT,
            color="#C62828", fontweight="bold")

    ax.text(LX, -0.2,
            'Harry Potter fans have been reacting to the news\nthat Hermione could be played by a black actress.',
            ha="center", va="top", fontsize=SENT_FONT, fontstyle="italic", color="#333")
    ax.text(RX, -0.2,
            'Harry Potter fans have been reacting to the news\nthat Hermione will be played by a black actress.',
            ha="center", va="top", fontsize=SENT_FONT, fontstyle="italic", color="#C62828")

    draw_expl_box(ax, MID, -3.0, [
        'SAK:  isomorphic trees (same structure, same dep relations)  →  match',
        '        "could" and "will" both serve as aux modifiers of "played"',
        '',
        'S2-K:  embed("could") ≠ embed("will")',
        '        possibility vs certainty  →  lower similarity  →  factual error detected',
    ], w=11.5, fc="#FFF9C4", ec="#F9A825")

    outpath = os.path.join(OUT_DIR, "case_study_1821_v3.png")
    fig.savefig(outpath, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════════════
# Case C: Doc 1511
# ═══════════════════════════════════════════════════════════════════════

def draw_case_1511():
    fig, ax = plt.subplots(figsize=(20, 13))
    ax.set_xlim(-2, 20)
    ax.set_ylim(-5.5, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("DeFacto — Doc 1511  (Intrinsic Error)",
                 fontsize=SUPTITLE_FONT, fontweight="bold", y=0.95)

    LX, RX = 4.0, 15.0
    MID = (LX + RX) / 2
    draw_title_bar(ax, LX, 9.3, "Human summary graph (correct)", w=6.5, fc="#1976D2")
    draw_title_bar(ax, RX, 9.3, "Candidate graph (with error)", w=6.5, fc="#E65100")

    y0 = 7.5
    y1 = 5.8
    y2 = 4.1
    y3 = 2.3

    for X, adj, adj_fc, adj_ec, adj_lw in [
        (LX, "Deprived\nADJ", "#C8E6C9", "#2E7D32", 2.5),
        (RX, "Poorer\nADJ", "#FFCDD2", "#C62828", 2.5),
    ]:
        draw_box(ax, X, y1, "children\nNOUN")
        draw_box(ax, X - 2.8, y0, adj, fc=adj_fc, ec=adj_ec, lw=adj_lw)
        draw_box(ax, X + 2.8, y0, "Scotland\nPROPN", fc="#E3F2FD", ec="#1565C0")
        draw_box(ax, X, y2, "likely\nADJ", w=1.2)
        draw_box(ax, X - 2.3, y3, "involved\nVERB")
        draw_box(ax, X + 2.3, y3, "study\nNOUN", w=1.2)

        draw_arrow(ax, X - 0.7, y1, X - 2.8 + 0.75, y0, label="amod")
        draw_arrow(ax, X + 0.7, y1, X + 2.8 - 0.75, y0, label="prep:in")
        draw_arrow(ax, X, y2 + 0.4, X, y1 - 0.4, label="nsubj")
        draw_arrow(ax, X - 0.6, y2 - 0.4, X - 2.3 + 0.75, y3 + 0.4, label="xcomp")
        draw_arrow(ax, X + 0.6, y2 - 0.4, X + 2.3 - 0.6, y3 + 0.4, label="prep:to")

    # Alignment
    for y in [y1, y2]:
        draw_dashed(ax, LX + 0.75, y, RX - 0.75, y,
                    label="aligned nodes" if y == y1 else "")
    draw_dashed(ax, LX + 2.8 + 0.75, y0, RX + 2.8 - 0.75, y0)
    draw_dashed(ax, LX - 2.3 + 0.75, y3, RX - 2.3 - 0.75, y3)
    draw_dashed(ax, LX + 2.3 + 0.6, y3, RX + 2.3 - 0.6, y3)
    draw_dashed(ax, LX - 2.8 + 0.75, y0, RX - 2.8 - 0.75, y0,
                label="aligned nodes", color="#C62828")

    ax.text(LX - 2.8, y0 - 0.9, "✓ correct", ha="center", fontsize=LABEL_FONT,
            color="#2E7D32", fontweight="bold")
    ax.text(RX - 2.8, y0 - 0.9, "✗ error", ha="center", fontsize=LABEL_FONT,
            color="#C62828", fontweight="bold")

    ax.text(LX, 0.6,
            'Deprived children in Scotland are more likely\nto be involved in crime, according to a new study.',
            ha="center", va="top", fontsize=SENT_FONT, fontstyle="italic", color="#333")
    ax.text(RX, 0.6,
            'Poorer children in Scotland are more likely\nto be involved in crime, according to a new study.',
            ha="center", va="top", fontsize=SENT_FONT, fontstyle="italic", color="#C62828")

    draw_expl_box(ax, MID, -2.5, [
        'SAK:  isomorphic trees (same nodes, same edges)  →  match',
        '        "Deprived" and "Poorer" both serve as amod of "children"',
        '',
        'S2-K:  embed("Deprived") ≠ embed("Poorer")',
        '        different semantic meaning  →  lower similarity  →  factual error detected',
    ], w=11.5, fc="#FFF9C4", ec="#F9A825")

    outpath = os.path.join(OUT_DIR, "case_study_1511_v3.png")
    fig.savefig(outpath, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {outpath}")


draw_case_136()
draw_case_1821()
draw_case_1511()
print("All done.")
