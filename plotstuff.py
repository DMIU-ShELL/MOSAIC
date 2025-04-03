import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict
import numpy as np
import plotly.graph_objects as go
import ast
from bisect import bisect_right
import matplotlib.gridspec as gridspec

def clean_betas_column(text):
    if not isinstance(text, str):
        return {}
    text = re.sub(r"tensor\(\s*\[(.*?)\)", r"[\1]", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*", " ", text)
    text = re.sub(r", device='cuda:0', grad_fn=<SliceBackward0>]", "", text)
    text = re.sub(r", device='cuda:0', grad_fn=<SoftmaxBackward0>]", "", text)
    try:
        return ast.literal_eval(text.strip())
    except Exception:
        return {}

def extract_betas_df(path):
    # Load and clean beta logs
    beta_paths = []
    for root, dirs, files in os.walk(path):
        if 'betas.csv' in files:
            beta_paths.append(os.path.join(root, 'betas.csv'))

    beta_dfs = []
    for beta_path in beta_paths:
        beta_df = pd.read_csv(beta_path)
        print(beta_df['betas'].apply(clean_betas_column))
        beta_df['betas'] = beta_df['betas'].apply(clean_betas_column)
        beta_df['agent_id'] = extract_agent_task_id(beta_path)
        beta_dfs.append(beta_df)

    if not beta_dfs:
        return pd.DataFrame()

    beta_df = pd.concat(beta_dfs, ignore_index=True)

    return beta_df  # Return the wide-format DataFrame for alignment

def extract_aligned_beta_df(beta_df, exchanges_df, layer_name='network.phi_body.layers.0'):
    from collections import defaultdict
    beta_influence_records = []

    exchange_map = defaultdict(lambda: defaultdict(list))
    for _, row in exchanges_df.iterrows():
        agent_id = int(row['port']) - 29500
        iteration = int(row['iteration'])
        source_task = int(row['task_id'])
        exchange_map[iteration][agent_id].append(source_task)

    for _, row in beta_df.iterrows():
        itr = int(row['iteration'])
        agent_id = row['agent_id']
        betas_dict = row['betas']

        if layer_name not in betas_dict:
            continue

        coeffs = betas_dict[layer_name]
        if len(coeffs) < 1:
            continue

        target_task = agent_id
        beta_influence_records.append({
            "iteration": itr,
            "agent_id": agent_id,
            "target_task": target_task,
            "source_task": target_task,
            "beta_value": coeffs[0]
        })

        source_tasks = exchange_map.get(itr, {}).get(agent_id, [])
        for i, src in enumerate(source_tasks):
            if i + 1 < len(coeffs):
                beta_influence_records.append({
                    "iteration": itr,
                    "agent_id": agent_id,
                    "target_task": target_task,
                    "source_task": src,
                    "beta_value": coeffs[i + 1]
                })

    return pd.DataFrame(beta_influence_records)

def load_exchanges_from_directory(base_dir):
    exchange_paths = []
    for root, dirs, files in os.walk(base_dir):
        if 'exchanges.csv' in files:
            exchange_paths.append(os.path.join(root, 'exchanges.csv'))
    return exchange_paths

def extract_agent_task_id(path):
    while path != os.path.dirname(path):
        folder_name = os.path.basename(path)
        match = re.match(r"agent_(\d+)", folder_name)
        if match:
            return int(match.group(1))
        path = os.path.dirname(path)
    return None

def build_dependency_graph(exchange_paths):
    edge_weights = defaultdict(int)
    for path in exchange_paths:
        try:
            df = pd.read_csv(path)
            agent_path = os.path.dirname(path)
            agent_task = extract_agent_task_id(agent_path)
            if agent_task is None:
                continue
            for _, row in df.iterrows():
                source_task = row['port'] - 29500
                if source_task != agent_task:
                    edge_weights[(source_task, agent_task)] += 1
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return edge_weights

def visualize_task_dependency_graph(edge_weights, title="Emergent Task Hierarchy"):
    G = nx.DiGraph()
    all_tasks = set()
    for (src, tgt), weight in edge_weights.items():
        G.add_edge(f"Task {src}", f"Task {tgt}", weight=weight)
        all_tasks.update([src, tgt])

    task_colors = {}
    group_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for task in all_tasks:
        group_id = task // 7
        color = group_colors[group_id % len(group_colors)]
        task_colors[f"Task {task}"] = color

    pos = nx.circular_layout(G)
    edge_widths = [0.2 * d['weight'] for _, _, d in G.edges(data=True)]

    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos, with_labels=True,
        node_color=[task_colors[n] for n in G.nodes()],
        node_size=1000,
        edge_color="gray",
        width=edge_widths,
        arrows=True,
        font_size=7
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig('dag_colored_groups.pdf')
    plt.close()

def structured_staggered_layout(tasks, row_width=7):
    x, y = [], []
    for i, task in enumerate(tasks):
        row = i // row_width
        col = i % row_width
        x_pos = (col + 0.5 * (row % 2)) / row_width
        y_pos = 1.0 - (row / ((len(tasks) // row_width) + 1))
        x.append(x_pos)
        y.append(y_pos)
    return x, y

def visualize_sankey_from_edges(edge_weights, min_weight=3):
    filtered_edges = {k: v for k, v in edge_weights.items() if v >= min_weight}
    if not filtered_edges:
        print("No edges to display in Sankey (after filtering).")
        return

    tasks = sorted(set([src for src, _ in filtered_edges] + [tgt for _, tgt in filtered_edges]))
    task_to_idx = {task: i for i, task in enumerate(tasks)}
    sources = [task_to_idx[src] for (src, tgt) in filtered_edges]
    targets = [task_to_idx[tgt] for (src, tgt) in filtered_edges]
    values = [weight for weight in filtered_edges.values()]
    x, y = structured_staggered_layout(tasks)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=5,
            line=dict(color="black", width=0.5),
            label=[f"Task{t}" for t in tasks],
            x=x, y=y
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    fig.update_layout(
        title_text="Sankey Diagram of Task Knowledge Flow",
        font_size=10,
        width=1024,
        height=512
    )
    fig.write_image("sankey_task_flow.pdf")

def visualize_heatmap_matrix(edge_weights, num_tasks=28):
    matrix = np.zeros((num_tasks, num_tasks))
    for (src, tgt), weight in edge_weights.items():
        if 0 <= src < num_tasks and 0 <= tgt < num_tasks:
            matrix[src, tgt] = weight

    df_matrix = pd.DataFrame(
        matrix,
        index=[f"Task {i}" for i in range(num_tasks)],
        columns=[f"Task {i}" for i in range(num_tasks)]
    )

    plt.figure(figsize=(14, 12))
    sns.heatmap(df_matrix, annot=False, cmap="Blues", linewidths=0.5, square=True)
    plt.title("28x28 Task-to-Task Knowledge Flow Heatmap")
    plt.xlabel("Target Task")
    plt.ylabel("Source Task")
    plt.tight_layout()
    plt.savefig("heatmap_28x28.pdf")
    plt.close()


def visualize_temporal_sliced_heatmaps(exchange_paths, num_tasks=28):
    """
    Creates 3 heatmaps (early, mid, late) with shared color scale and a single aligned colorbar.
    Adds group dividers between each block of 7 tasks.
    """
    phases = ['Early', 'Mid', 'Late']
    phase_matrices = {phase: np.zeros((num_tasks, num_tasks)) for phase in phases}
    all_iterations = []

    # First pass: get max iteration
    for path in exchange_paths:
        try:
            df = pd.read_csv(path)
            if 'iteration' in df.columns:
                all_iterations.extend(df['iteration'].dropna().astype(int).tolist())
        except Exception as e:
            print(f"[WARNING] Couldn't read {path}: {e}")

    if not all_iterations:
        print("No valid iteration data found.")
        return

    max_iter = max(all_iterations)
    bins = [0, max_iter // 3, 2 * max_iter // 3, max_iter + 1]

    # Second pass: accumulate data into bins
    for path in exchange_paths:
        try:
            df = pd.read_csv(path)
            agent_path = os.path.dirname(path)
            agent_task = extract_agent_task_id(agent_path)
            if agent_task is None:
                continue

            df = df.dropna(subset=['iteration', 'port'])
            df['iteration'] = df['iteration'].astype(int)
            df['source_task'] = df['port'].astype(int) - 29500
            df['target_task'] = agent_task

            df['phase'] = pd.cut(df['iteration'], bins=bins, labels=phases, right=False)

            for _, row in df.iterrows():
                src = int(row['source_task'])
                tgt = int(row['target_task'])
                phase = row['phase']
                if phase in phases and 0 <= src < num_tasks and 0 <= tgt < num_tasks:
                    phase_matrices[phase][src, tgt] += 1

        except Exception as e:
            print(f"[ERROR] Parsing {path}: {e}")

    # Global color scale across all phases
    vmin = min(np.min(mat) for mat in phase_matrices.values())
    vmax = max(np.max(mat) for mat in phase_matrices.values())

    # Grid layout with space for shared colorbar
    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
    axes = [plt.subplot(gs[i]) for i in range(3)]
    cbar_ax = plt.subplot(gs[3])

    group_size = 7
    group_lines = [group_size * i for i in range(1, num_tasks // group_size)]

    for i, phase in enumerate(phases):
        ax = axes[i]
        sns.heatmap(
            phase_matrices[phase],
            ax=ax,
            cmap="Blues",
            cbar=(i == 2),
            cbar_ax=cbar_ax if i == 2 else None,
            square=True,
            linewidths=0.5,
            xticklabels=[f"{i}" for i in range(num_tasks)],
            yticklabels=[f"{i}" for i in range(num_tasks)],
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(f"{phase} Communication")
        ax.set_xlabel("Target Task")
        ax.set_ylabel("Source Task" if i == 0 else "")

        for line in group_lines:
            ax.axhline(line, color='#929da8', linewidth=1.2)
            ax.axvline(line, color='#929da8', linewidth=1.2)

    plt.tight_layout(rect=[0, 0, 0.96, 1])  # reserve space for colorbar
    plt.savefig("heatmap_3_slices.pdf")
    plt.close()

def compute_phase_matrices(exchange_paths, num_tasks=28):
    """
    Reusable function to return phase_matrices = {'Early': ..., 'Mid': ..., 'Late': ...}
    """
    phases = ['Early', 'Mid', 'Late']
    phase_matrices = {phase: np.zeros((num_tasks, num_tasks)) for phase in phases}
    all_iterations = []

    for path in exchange_paths:
        try:
            df = pd.read_csv(path)
            if 'iteration' in df.columns:
                all_iterations.extend(df['iteration'].dropna().astype(int).tolist())
        except: continue

    if not all_iterations:
        return phase_matrices

    max_iter = max(all_iterations)
    bins = [0, max_iter // 3, 2 * max_iter // 3, max_iter + 1]

    for path in exchange_paths:
        try:
            df = pd.read_csv(path)
            agent_path = os.path.dirname(path)
            agent_task = extract_agent_task_id(agent_path)
            if agent_task is None:
                continue

            df = df.dropna(subset=['iteration', 'port'])
            df['iteration'] = df['iteration'].astype(int)
            df['source_task'] = df['port'].astype(int) - 29500
            df['target_task'] = agent_task
            df['phase'] = pd.cut(df['iteration'], bins=bins, labels=phases, right=False)

            for _, row in df.iterrows():
                src = int(row['source_task'])
                tgt = int(row['target_task'])
                phase = row['phase']
                if phase in phases and 0 <= src < num_tasks and 0 <= tgt < num_tasks:
                    phase_matrices[phase][src, tgt] += 1

        except: continue

    return phase_matrices

def plot_phase_deltas(phase_matrices):
    """
    Plot Mid−Early, Late−Mid, and Late−Early delta heatmaps with shared colorbar and axis labels.
    """
    deltas = {
        "Early ~ Mid": phase_matrices['Mid'] - phase_matrices['Early'],
        "Mid ~ Late": phase_matrices['Late'] - phase_matrices['Mid'],
        "Early ~ Late": phase_matrices['Late'] - phase_matrices['Early']
    }

    # Compute global color scale
    all_values = np.concatenate([mat.flatten() for mat in deltas.values()])
    abs_max = np.max(np.abs(all_values))

    # Use GridSpec for shared colorbar
    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
    axes = [plt.subplot(gs[i]) for i in range(3)]
    cbar_ax = plt.subplot(gs[3])

    for i, (title, delta_matrix) in enumerate(deltas.items()):
        ax = axes[i]
        sns.heatmap(
            delta_matrix,
            ax=ax,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            xticklabels=[f"{i}" for i in range(28)],
            yticklabels=[f"{i}" for i in range(28)],
            vmin=-abs_max,
            vmax=abs_max,
            cbar=(i == 2),
            cbar_ax=cbar_ax if i == 2 else None
        )
        ax.set_title(f"Δ {title}")
        ax.set_xlabel("Target Task")
        if i == 0:
            ax.set_ylabel("Source Task")

    plt.tight_layout(rect=[0, 0, 0.96, 1])
    plt.savefig("heatmap_phase_deltas.pdf")
    plt.close()


def plot_normalized_heatmaps(phase_matrices, normalize='row'):
    """
    Plot normalized heatmaps with a shared colorbar and axis labels.
    """
    assert normalize in ['row', 'column'], "normalize must be 'row' or 'column'"

    phases = ['Early', 'Mid', 'Late']
    norm_data = {}

    for phase, matrix in phase_matrices.items():
        if normalize == 'row':
            norm_matrix = matrix / matrix.sum(axis=1, keepdims=True)
        else:
            norm_matrix = matrix / matrix.sum(axis=0, keepdims=True)
        norm_data[phase] = np.nan_to_num(norm_matrix)

    # Setup grid layout for shared colorbar
    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
    axes = [plt.subplot(gs[i]) for i in range(3)]
    cbar_ax = plt.subplot(gs[3])

    for i, phase in enumerate(phases):
        ax = axes[i]
        sns.heatmap(
            norm_data[phase],
            ax=ax,
            cmap="YlGnBu",
            cbar=(i == 2),
            cbar_ax=cbar_ax if i == 2 else None,
            square=True,
            linewidths=0.5,
            xticklabels=[f"{i}" for i in range(28)],
            yticklabels=[f"{i}" for i in range(28)],
            vmin=0,
            vmax=1
        )
        ax.set_title(f"{phase} ({normalize}-normalized)")
        ax.set_xlabel("Target Task")
        if i == 0:
            ax.set_ylabel("Source Task")
        else:
            ax.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 0.96, 1])
    plt.savefig(f"heatmap_{normalize}_normalized.pdf")
    plt.close()

def plot_row_entropy_heatmap(matrix, title, filename):
    row_probs = matrix / matrix.sum(axis=1, keepdims=True)
    row_entropy = -np.nansum(row_probs * np.log2(row_probs + 1e-12), axis=1)  # Avoid log(0)
    entropy_matrix = np.tile(row_entropy[:, None], (1, matrix.shape[1]))  # Broadcast to 28x28

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        entropy_matrix,
        cmap="YlOrRd",
        square=True,
        linewidths=0.25,
        linecolor="#eeeeee",
        xticklabels=[str(i) for i in range(matrix.shape[1])],
        yticklabels=[str(i) for i in range(matrix.shape[0])]
    )
    plt.title(title)
    plt.xlabel("Target Task")
    plt.ylabel("Source Task")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_column_entropy_heatmap(matrix, title, filename):
    col_probs = matrix / matrix.sum(axis=0, keepdims=True)
    col_entropy = -np.nansum(col_probs * np.log2(col_probs + 1e-12), axis=0)
    entropy_matrix = np.tile(col_entropy[None, :], (matrix.shape[0], 1))  # Broadcast to 28x28

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        entropy_matrix,
        cmap="YlGnBu",
        square=True,
        linewidths=0.25,
        linecolor="#eeeeee",
        xticklabels=[str(i) for i in range(matrix.shape[1])],
        yticklabels=[str(i) for i in range(matrix.shape[0])]
    )
    plt.title(title)
    plt.xlabel("Target Task")
    plt.ylabel("Source Task")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_sparsity_heatmap(matrix, title, filename):
    binary_matrix = (matrix > 0).astype(int)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        binary_matrix,
        cmap="Greys",
        square=True,
        linewidths=0.25,
        linecolor="#dddddd",
        xticklabels=[str(i) for i in range(matrix.shape[1])],
        yticklabels=[str(i) for i in range(matrix.shape[0])],
        cbar=False
    )
    plt.title(title)
    plt.xlabel("Target Task")
    plt.ylabel("Source Task")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_hierarchy_flow_heatmap(matrix, title, filename):
    hierarchy_matrix = np.triu(matrix) - np.tril(matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        hierarchy_matrix,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.25,
        linecolor="#eeeeee",
        xticklabels=[str(i) for i in range(matrix.shape[1])],
        yticklabels=[str(i) for i in range(matrix.shape[0])]
    )
    plt.title(title)
    plt.xlabel("Target Task")
    plt.ylabel("Source Task")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def compute_entropy_table_latex(matrix, phase="Early", output_path=None):
    row_probs = matrix / matrix.sum(axis=1, keepdims=True)
    col_probs = matrix / matrix.sum(axis=0, keepdims=True)

    row_entropy = -np.nansum(row_probs * np.log2(row_probs + 1e-12), axis=1)
    col_entropy = -np.nansum(col_probs * np.log2(col_probs + 1e-12), axis=0)

    df = pd.DataFrame({
        f"Task {i}": [row_entropy[i], col_entropy[i]] for i in range(matrix.shape[0])
    }, index=["Row Entropy (outgoing)", "Column Entropy (incoming)"])

    latex_table = df.to_latex(
        index=True,
        float_format="%.2f",
        caption=f"Entropy Table for {phase} Phase (Horizontal)",
        label=f"tab:entropy_{phase.lower()}_horizontal"
    )

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_table)
        print(f"[✓] Saved horizontal LaTeX entropy table to: {output_path}")

    return latex_table


def plot_average_influence_heatmap(beta_df, target_task, num_tasks=28, save_path="influence_heatmap.pdf"):
    influence_matrix = np.zeros((num_tasks, num_tasks))
    counts_matrix = np.zeros((num_tasks, num_tasks))

    for _, row in beta_df.iterrows():
        src = int(row["source_task"])
        if 0 <= src < num_tasks and 0 <= target_task < num_tasks:
            influence_matrix[src, target_task] += row["beta_value"]
            counts_matrix[src, target_task] += 1

    with np.errstate(invalid='ignore'):
        avg_matrix = np.divide(influence_matrix, counts_matrix, where=counts_matrix > 0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_matrix, cmap="YlGnBu", linewidths=0.5, square=True,
                xticklabels=np.arange(num_tasks), yticklabels=np.arange(num_tasks))
    plt.title("Average Beta Influence (Source → Target)")
    plt.xlabel("Target Task")
    plt.ylabel("Source Task")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_temporal_beta_trajectories(beta_df, top_n=28, save_path="temporal_beta_trajectories.pdf"):
    top_sources = beta_df["source_task"].value_counts().head(top_n).index.tolist()
    print(top_sources)

    plt.figure(figsize=(10, 5))
    for src_task in top_sources:
        subset = beta_df[beta_df["source_task"] == src_task]
        grouped = subset.groupby("iteration")["beta_value"].mean().reset_index()
        plt.plot(grouped["iteration"], grouped["beta_value"], label=f"Task {src_task}")

    plt.xlabel("Iteration")
    plt.ylabel("Beta Coefficient")
    plt.title("Temporal Influence of Top Source Tasks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_influence_vs_frequency(beta_df, save_path="influence_vs_frequency.pdf"):
    source_counts = beta_df["source_task"].value_counts().sort_index()
    source_means = beta_df.groupby("source_task")["beta_value"].mean()

    influence_vs_freq = pd.DataFrame({
        "Task": source_means.index.astype(int),
        "Mean Beta": source_means.values,
        "Frequency": source_counts.reindex(source_means.index).fillna(0).values
    })

    plt.figure(figsize=(8, 6))
    plt.scatter(influence_vs_freq["Frequency"], influence_vs_freq["Mean Beta"])
    plt.xlabel("Reuse Frequency (times selected)")
    plt.ylabel("Mean Beta Coefficient")
    plt.title("Influence vs Frequency of Reuse")
    for _, row in influence_vs_freq.iterrows():
        plt.text(row["Frequency"] + 0.2, row["Mean Beta"], str(int(row["Task"])), fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_weighted_frequency_heatmap(betas_df, exchanges_df, comm_interval=10, num_tasks=28, layer="network.phi_body.layers.0", output_path="weighted_frequency_heatmap.pdf"):
    """
    Combine exchange frequency and beta coefficients to build a weighted heatmap of task reuse.
    """
    from collections import defaultdict

    # Preprocess exchanges: iteration → agent_id → source_tasks
    exchange_map = defaultdict(lambda: defaultdict(list))
    for _, row in exchanges_df.iterrows():
        iteration = int(row['iteration'])
        agent_id = int(row['port']) - 29500
        source_task = int(row['task_id'])
        exchange_map[iteration][agent_id].append(source_task)

    # Build the matrix
    weighted_matrix = np.zeros((num_tasks, num_tasks))
    agents = betas_df['agent_id'].unique()

    for agent in agents:
        agent_betas = betas_df[betas_df['agent_id'] == agent]
        iterations = sorted(agent_betas['iteration'].unique())
        comm_points = list(range(min(iterations), max(iterations), comm_interval))

        for start in comm_points:
            end = start + comm_interval
            round_betas = agent_betas[(agent_betas['iteration'] >= start) & (agent_betas['iteration'] < end)]

            if round_betas.empty:
                continue

            final_row = round_betas.sort_values("iteration").iloc[-1]
            coeffs = final_row['betas'].get(layer, [])
            if not coeffs:
                continue

            target_task = agent
            weighted_matrix[target_task, target_task] += coeffs[0]

            source_tasks = exchange_map.get(start, {}).get(agent, [])
            for i, src in enumerate(source_tasks):
                if i + 1 < len(coeffs):
                    weighted_matrix[src, target_task] += coeffs[i + 1]

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(weighted_matrix, cmap="YlGnBu", linewidths=0.5, square=True,
                xticklabels=np.arange(num_tasks), yticklabels=np.arange(num_tasks))
    plt.title("Weighted Task Reuse Heatmap (Beta-Scaled)")
    plt.xlabel("Target Task")
    plt.ylabel("Source Task")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Example usage
if __name__ == "__main__":
    base_path = "log/FINAL/mctgraph/fullcomm/seed1"
    exchange_paths = load_exchanges_from_directory(base_path)
    edge_weights = build_dependency_graph(exchange_paths)

    visualize_task_dependency_graph(edge_weights)
    visualize_sankey_from_edges(edge_weights, min_weight=3)
    #visualize_heatmap_matrix(edge_weights)
    visualize_temporal_sliced_heatmaps(exchange_paths)


    # Build once
    phase_matrices = compute_phase_matrices(exchange_paths)

    for phase, matrix in phase_matrices.items():
        # Save entropy tables to LaTeX files
        filename = f"entropy_table_{phase.lower()}.tex"
        compute_entropy_table_latex(matrix, phase=phase, output_path=filename)

    for phase_name, matrix in phase_matrices.items():
        '''plot_row_entropy_heatmap(
            matrix,
            title=f"{phase_name} – Row-wise Entropy",
            filename=f"entropy_row_{phase_name}.pdf"
        )

        plot_column_entropy_heatmap(
            matrix,
            title=f"{phase_name} – Column-wise Entropy",
            filename=f"entropy_col_{phase_name}.pdf"
        )'''

        plot_sparsity_heatmap(
            matrix,
            title=f"{phase_name} – Sparsity Mask",
            filename=f"sparsity_{phase_name}.pdf"
        )

        '''plot_hierarchy_flow_heatmap(
            matrix,
            title=f"{phase_name} – Curriculum Flow",
            filename=f"hierarchy_flow_{phase_name}.pdf"
        )'''

    # Plot deltas
    plot_phase_deltas(phase_matrices)

    # Plot row- and column-normalized variants
    plot_normalized_heatmaps(phase_matrices, normalize='row')
    plot_normalized_heatmaps(phase_matrices, normalize='column')


    betas_df = extract_betas_df(base_path)
    all_exchanges_df = pd.concat([pd.read_csv(p) for p in exchange_paths], ignore_index=True)
    beta_influence_df = extract_aligned_beta_df(betas_df, all_exchanges_df)

    print(betas_df)

    #plot_average_influence_heatmap(beta_influence_df, target_task=6)
    #plot_temporal_beta_trajectories(beta_influence_df)
    #plot_influence_vs_frequency(beta_influence_df)
    plot_weighted_frequency_heatmap(betas_df, all_exchanges_df)