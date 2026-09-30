"""
Draw case-study figures: SAK fails but S2-K succeeds.
Simplified graphs: filter out punctuation and function words for clarity.
"""
import json, os, textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.dpi": 180,
})

STOP_TOKENS = {
    ".", ",", '"', "'", ":", ";", "-", "(", ")", "!", "?",
    "a", "A", "an", "An", "the", "The",
    "has", "have", "had", "been", "be", "is", "was", "were", "are",
    "for", "in", "of", "to", "on", "at", "by", "with", "from",
    "her", "his", "its", "their", "my", "your", "our",
    "that", "which", "who", "whom",
    "and", "or", "but", "not", "no",
    "it", "this", "these", "those",
}

EDGE_TYPE_COLORS = {
    "token-token": "#78909C",
    "token-mention": "#1976D2",
    "mention-entity": "#F57C00",
    "entity-entity": "#C62828",
}
EDGE_TYPE_STYLES = {
    "token-token": "-",
    "token-mention": "--",
    "mention-entity": "-.",
    "entity-entity": "-",
}

NODE_FILL = {"Token": "#BBDEFB", "Mention": "#FFE0B2", "Entity": "#F8BBD0"}
NODE_BORDER = {"Token": "#1565C0", "Mention": "#E65100", "Entity": "#AD1457"}
NODE_SIZE = {"Token": 900, "Mention": 1400, "Entity": 1600}
NODE_SHAPE = {"Token": "o", "Mention": "s", "Entity": "D"}


def classify_edge(src_label, tgt_label):
    pair = (src_label, tgt_label)
    if pair == ("Token", "Token"):
        return "token-token"
    if "Mention" in pair and "Token" in pair:
        return "token-mention"
    if "Mention" in pair and "Entity" in pair:
        return "mention-entity"
    if pair == ("Entity", "Entity"):
        return "entity-entity"
    return "token-token"


def build_nx_graph(graph_data, filter_stop=True):
    """Build networkx graph, optionally filtering stop/punctuation Token nodes."""
    node_map = {}
    for n in graph_data["nodes"]:
        node_map[n["id"]] = n

    keep_ids = set()
    for n in graph_data["nodes"]:
        if n["label"] != "Token":
            keep_ids.add(n["id"])
        elif not filter_stop:
            keep_ids.add(n["id"])
        else:
            name = str(n["name"])
            if name not in STOP_TOKENS and len(name) > 1:
                keep_ids.add(n["id"])

    G = nx.DiGraph()
    for nid in keep_ids:
        n = node_map[nid]
        G.add_node(nid, name=str(n["name"]), label=n["label"])

    for e in graph_data["links"]:
        src, tgt = e["source"], e["target"]
        if src == tgt or src not in keep_ids or tgt not in keep_ids:
            continue
        sl = node_map[src]["label"]
        tl = node_map[tgt]["label"]
        G.add_edge(src, tgt, rel=e["relationship"], etype=classify_edge(sl, tl))

    # Remove isolated nodes
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    return G


def draw_graph(ax, G, title, highlight_nodes=None):
    if highlight_nodes is None:
        highlight_nodes = set()
    if len(G.nodes) == 0:
        ax.set_title(title)
        return

    k_val = max(3.0, 8.0 / np.sqrt(len(G.nodes) + 1))
    pos = nx.spring_layout(G, seed=42, k=k_val, iterations=80)

    # Edges
    for etype, color in EDGE_TYPE_COLORS.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == etype]
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                                   edge_color=color, alpha=0.45, width=1.5,
                                   arrows=True, arrowsize=12,
                                   connectionstyle="arc3,rad=0.12")
            # Edge labels for non-token-token
            if etype != "token-token":
                elabels = {(u, v): d["rel"] for u, v, d in G.edges(data=True) if d.get("etype") == etype}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels, ax=ax,
                                             font_size=6, font_color=color, alpha=0.7,
                                             bbox=dict(alpha=0))

    # Nodes
    for ntype in ["Token", "Mention", "Entity"]:
        nodes = [n for n in G.nodes if G.nodes[n].get("label") == ntype]
        if not nodes:
            continue
        fc = []
        ec = []
        for n in nodes:
            if n in highlight_nodes:
                fc.append("#FF1744")
                ec.append("#B71C1C")
            else:
                fc.append(NODE_FILL.get(ntype, "#E0E0E0"))
                ec.append(NODE_BORDER.get(ntype, "#333"))
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax,
                               node_color=fc, edgecolors=ec,
                               node_size=NODE_SIZE.get(ntype, 900),
                               node_shape=NODE_SHAPE.get(ntype, "o"),
                               linewidths=2.0, alpha=0.92)

    # Labels
    labels = {}
    for n in G.nodes:
        name = str(G.nodes[n].get("name", n))
        lbl = G.nodes[n].get("label", "Token")
        if lbl == "Entity":
            labels[n] = f"[{name[:10]}]"
        elif lbl == "Mention":
            labels[n] = f"«{name}»"
        else:
            labels[n] = name
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9, font_weight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")


def load_case(doc_id, split="test"):
    for base_dir in [
        "/root/autodl-fs/zyq/defacto_data_gawl/factgraph_result",
    ]:
        for sp in ([split] if split else ["test", "train", "val"]):
            path = os.path.join(base_dir, sp, f"{doc_id}.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                result = {}
                for key in ["humman_summary", "candidate"]:
                    part = data[key]
                    if isinstance(part, str):
                        part = json.loads(part)
                    result[key] = {
                        "text": part["text"],
                        "graph": part["graph_without_emb"],
                    }
                result["intrinsic_error"] = data["intrinsic_error"]
                result["extrinsic_error"] = data["extrinsic_error"]
                return result
    raise FileNotFoundError(f"Cannot find doc {doc_id}")


def find_extra_nodes(human_graph, cand_graph):
    human_names = {str(n["name"]).lower() for n in human_graph["nodes"]
                   if str(n["name"]).lower() not in {s.lower() for s in STOP_TOKENS}}
    extra = set()
    for n in cand_graph["nodes"]:
        name = str(n["name"]).lower()
        if name in {s.lower() for s in STOP_TOKENS}:
            continue
        if name not in human_names:
            extra.add(n["id"])
    return extra


CASES = [
    {
        "doc_id": "1051", "split": "test", "error_type": "Extrinsic",
        "human_text": 'A woman has been jailed for her part in the "wicked" killing of a poker player.',
        "cand_text": 'A 23-year-old woman has been jailed for 16 years for her part\nin the "wicked" killing of a poker player in north London.',
        "error_desc": 'Candidate adds "in north London" — source says Mayfair.\n'
                      'SAK: syntactically equivalent PP attachment → cannot distinguish.\n'
                      'S2-K: semantic embeddings of "north London" ≠ "Mayfair" → lower score.',
    },
    {
        "doc_id": "1072", "split": "test", "error_type": "Intrinsic",
        "human_text": 'Huawei has threatened to file legal action in the US and China\nagainst Samsung over alleged patent infringement.',
        "cand_text": 'Huawei has filed legal action in the US and China\nagainst Samsung over alleged patent infringement.',
        "error_desc": '"threatened to file" → "filed" (intent vs completed action).\n'
                      'SAK: same dep-tree structure (nsubj→root) → cannot distinguish.\n'
                      'S2-K: embedding("threatened to file") ≠ embedding("filed") → lower score.',
    },
    {
        "doc_id": "1079", "split": "test", "error_type": "Extrinsic",
        "human_text": "Lichfield have been left out of the new Women's Super Rugby competition.",
        "cand_text": "Lichfield have been left out of the Rugby Football Union's\nnew Women's Super Rugby competition.",
        "error_desc": 'Candidate adds "Rugby Football Union" — not in source.\n'
                      'SAK: extra possessive modifier, same syntactic role → cannot distinguish.\n'
                      'S2-K: no matching entity node in source graph → lower score.',
    },
]


def main():
    out_dir = "/root/autodl-tmp/FactGraphGAWL/experiments/figures"
    os.makedirs(out_dir, exist_ok=True)

    for ci, case in enumerate(CASES):
        data = load_case(case["doc_id"], case["split"])
        h_graph = build_nx_graph(data["humman_summary"]["graph"], filter_stop=True)
        c_graph = build_nx_graph(data["candidate"]["graph"], filter_stop=True)
        extra = find_extra_nodes(data["humman_summary"]["graph"], data["candidate"]["graph"])

        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        fig.suptitle(
            f'Case {ci+1} (Doc {case["doc_id"]}):  {case["error_type"]} Error  —  SAK ✗   S2-K ✓',
            fontsize=15, fontweight="bold", y=0.97
        )

        draw_graph(axes[0], h_graph, "Human Summary (correct)")
        draw_graph(axes[1], c_graph, "Candidate Summary (with error)", highlight_nodes=extra)

        # Text boxes at bottom
        fig.text(0.25, -0.01,
                 f"Human: {case['human_text']}",
                 ha="center", fontsize=9, va="top",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#E8F5E9", ec="#43A047", lw=1.5))
        fig.text(0.75, -0.01,
                 f"Candidate: {case['cand_text']}",
                 ha="center", fontsize=9, va="top",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#FFEBEE", ec="#E53935", lw=1.5))

        # Error explanation box
        fig.text(0.50, -0.08,
                 case["error_desc"],
                 ha="center", fontsize=9, va="top",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9C4", ec="#F9A825", lw=1.5),
                 fontstyle="italic")

        # Legend
        legend_handles = [
            mpatches.Patch(fc=NODE_FILL["Token"], ec=NODE_BORDER["Token"], lw=1.5, label="Token (content word)"),
            mpatches.Patch(fc=NODE_FILL["Mention"], ec=NODE_BORDER["Mention"], lw=1.5, label="Mention (named entity span)"),
            mpatches.Patch(fc=NODE_FILL["Entity"], ec=NODE_BORDER["Entity"], lw=1.5, label="Entity (KB-linked)"),
            mpatches.Patch(fc="#FF1744", ec="#B71C1C", lw=1.5, label="Error node (extra in candidate)"),
            plt.Line2D([], [], color=EDGE_TYPE_COLORS["token-token"], lw=1.5, label="Dep edge (token–token)"),
            plt.Line2D([], [], color=EDGE_TYPE_COLORS["token-mention"], lw=1.5, ls="--", label="RELATED_TO (token–mention)"),
            plt.Line2D([], [], color=EDGE_TYPE_COLORS["mention-entity"], lw=1.5, ls="-.", label="BELONGS_TO (mention–entity)"),
        ]
        fig.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.95,
                   edgecolor="#999", fancybox=True)

        fig.subplots_adjust(bottom=0.15, top=0.92, wspace=0.05)
        outpath = os.path.join(out_dir, f"case_study_{case['doc_id']}.png")
        fig.savefig(outpath, bbox_inches="tight", facecolor="white", pad_inches=0.5)
        plt.close(fig)
        print(f"Saved: {outpath}")

    # Bar chart
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    categories = ["Intrinsic\n(S2-K only)", "Extrinsic\n(S2-K only)",
                   "Intrinsic\n(SAK only)", "Extrinsic\n(SAK only)"]
    values = [28, 64, 22, 44]
    bar_colors = ["#66BB6A", "#43A047", "#EF5350", "#E53935"]
    bars = ax2.bar(categories, values, color=bar_colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, str(val),
                 ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Correctly judged samples", fontsize=12)
    ax2.set_title("Uniquely correct samples: S2-K vs SAK  (DeFacto dataset)",
                  fontsize=13, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(0, 78)

    legend2 = [
        mpatches.Patch(color="#43A047", label="S2-K correct, SAK wrong  (net +26)"),
        mpatches.Patch(color="#E53935", label="SAK correct, S2-K wrong"),
    ]
    ax2.legend(handles=legend2, fontsize=10, loc="upper left")

    outpath2 = os.path.join(out_dir, "sak_vs_s2k_comparison.png")
    fig2.savefig(outpath2, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved: {outpath2}")


if __name__ == "__main__":
    main()
