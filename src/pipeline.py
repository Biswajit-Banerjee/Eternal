import torch
import matplotlib.pyplot as plt
import networkx as nx


def predict_structure(
    sequence, model, sequence_encoder, structure_decoder, device=None, plot=True
):
    """
    Predicts the secondary structure for a given RNA sequence.

    Args:
        sequence (str): Raw RNA sequence (e.g., "GGGAAACCC")
        model: Trained PyTorch model
        device: torch device (CPU/GPU)
        sequence_encoder (function): Function to map the sequence
        structure_decoder (function): Function to decode the prediction

    Returns:
        str: Predicted secondary structure notation
    """

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # Convert sequence to one-hot encoding
    seq_indices = sequence_encoder(sequence).to(device)

    seq_tensor = seq_indices.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        logits = model(seq_tensor)
        preds = torch.argmax(logits, dim=-1)

    # Convert prediction indices to structure notation
    pred_structure = "".join(
        [
            structure_decoder[idx.item()]
            for idx in preds[0][
                : len(sequence)
            ]  # Remove batch dimension and trim to sequence length
        ]
    )

    clean_structure = remove_unmatched_brackets(pred_structure)
    print(pred_structure)
    print(clean_structure)
    
    if plot:
        plot_rna_structure(sequence, clean_structure, title="RNA Secondary Structure")
        
    return clean_structure


def remove_unmatched_brackets(dot_bracket):
    stack = []
    illegal_indexes = []
    for index, symbol in enumerate(dot_bracket):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if len(stack) > 0:
                stack.pop()
            else:
                illegal_indexes.append(index)

    illegal_indexes.extend(stack)
    dot_bracket = list(dot_bracket)

    for i in illegal_indexes:
        dot_bracket[i] = "."

    return "".join(dot_bracket)


def plot_rna_structure(sequence, dot_bracket, title="RNA Secondary Structure"):
    """
    Visualizes RNA secondary structure as a graph.

    Args:
        sequence (str): RNA sequence (e.g., "GGGAAACCC")
        dot_bracket (str): Dot-bracket notation (e.g., "(((...)))")
        title (str): Title for the plot
    """
    # Input validation
    if len(sequence) != len(dot_bracket):
        raise ValueError("Sequence and structure lengths must match")

    if not all(c in "AUGC-" for c in sequence.upper()):
        raise ValueError("Sequence must contain only A, U, G, C, or -")

    if not all(c in "().[]{}" for c in dot_bracket):
        raise ValueError("Structure must be in dot-bracket notation")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create graph
    G = nx.Graph()

    # Define colors for different bases
    base_colors = {
        "A": "#FFA07A",  # Light Salmon
        "C": "#98FB98",  # Pale Green
        "G": "#87CEFA",  # Light Sky Blue
        "U": "#DDA0DD",  # Plum
        "-": "#757575",  # Gray
    }

    # Add nodes (nucleotides)
    for i, (dot, base) in enumerate(zip(dot_bracket, sequence.upper())):
        label = f"{base}{i+1}" if base != "-" else str(i + 1)
        G.add_node(i, base=base, label=label)

    # Add backbone edges
    for i in range(len(sequence) - 1):
        G.add_edge(i, i + 1, color="gray", weight=1, style="solid")

    # Add base pair edges
    pairs = {"(": ")", "[": "]", "{": "}"}
    for bracket, complement in pairs.items():
        stack = []
        for i, dot in enumerate(dot_bracket):
            if dot == bracket:
                stack.append(i)
            elif dot == complement and stack:
                j = stack.pop()
                if sequence[i].upper() == "-" and sequence[j].upper() == "-":
                    G.add_edge(i, j, color="#FFFFFF", weight=2, style="dashed")
                else:
                    G.add_edge(i, j, color="#8BDDBB", weight=2, style="solid")

    # Calculate layout
    pos = nx.kamada_kawai_layout(G)

    # Draw nodes
    node_colors = [base_colors[data["base"]] for _, data in G.nodes(data=True)]
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=False,
        node_color=node_colors,
        node_size=500,
        font_size=6,
        font_weight="bold",
        linewidths=1.5,
        edgecolors="black",
    )

    # Draw labels
    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight="bold")

    # Draw edges with different styles
    solid_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d["style"] == "solid"]
    dashed_edges = [
        (u, v) for (u, v, d) in G.edges(data=True) if d["style"] == "dashed"
    ]

    # Draw solid edges
    if solid_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=solid_edges,
            edge_color=[G[u][v]["color"] for u, v in solid_edges],
            width=[G[u][v]["weight"] for u, v in solid_edges],
        )

    # Draw dashed edges
    if dashed_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=dashed_edges,
            edge_color=[G[u][v]["color"] for u, v in dashed_edges],
            width=[G[u][v]["weight"] for u, v in dashed_edges],
            style="dashed",
        )

    # Customize plot
    ax.axis("off")
    ax.text(
        0.5,
        -0.1,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=20,
        wrap=True,
    )

    plt.tight_layout()
    return fig, ax
