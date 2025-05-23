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
from matplotlib.colors import PowerNorm

# HEATMAPS for beta coefficients
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
    beta_paths = []
    for root, dirs, files in os.walk(path):
        if 'betas.csv' in files:
            beta_paths.append(os.path.join(root, 'betas.csv'))

    beta_dfs = []
    for beta_path in beta_paths:
        beta_df = pd.read_csv(beta_path)
        beta_df['betas'] = beta_df['betas'].apply(clean_betas_column)
        beta_df['agent_id'] = extract_agent_task_id(beta_path)
        beta_dfs.append(beta_df)

    if not beta_dfs:
        return pd.DataFrame()

    beta_df = pd.concat(beta_dfs, ignore_index=True)

    expanded_records = []
    for _, row in beta_df.iterrows():
        iteration = int(row['iteration'])
        agent_id = row['agent_id']
        betas_dict = row['betas']
        for layer, coeffs in betas_dict.items():
            for idx, beta_val in enumerate(coeffs):
                expanded_records.append({
                    "iteration": iteration,
                    "layer": layer,
                    "source_index": idx,
                    "beta_value": beta_val,
                    "agent_id": agent_id
                })

    beta_df = pd.DataFrame(expanded_records)
    beta_df = beta_df.sort_values(by='iteration').reset_index(drop=True)
    return beta_df

def extract_agent_task_id(path):
    while path != os.path.dirname(path):
        folder_name = os.path.basename(path)
        match = re.match(r"agent_(\d+)", folder_name)
        if match:
            return int(match.group(1))
        path = os.path.dirname(path)
    return None

def extract_exchange_df(exchange_paths):
    dfs = []
    for path in exchange_paths:
        try:
            df = pd.read_csv(path)
            agent_path = os.path.dirname(path)
            agent_id = extract_agent_task_id(agent_path)
            df['agent_id'] = agent_id
            df['source_task'] = df['port'] - 29500
            dfs.append(df)
        except Exception as e:
            print(f"[!] Failed to read {path}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def compute_phased_weighted_heatmaps(beta_df, exchange_df, layer, num_tasks=28, num_phases=3): # Added num_phases
    """
    Computes influence heatmaps weighted by beta values, dividing the run into
    a specified number of phases.

    Args:
        beta_df (pd.DataFrame): Expanded dataframe from extract_betas_df.
        exchange_df (pd.DataFrame): Dataframe from extract_exchange_df.
        layer (str): The specific layer name to compute the heatmap for.
        num_tasks (int): The total number of tasks (for heatmap dimensions).
        num_phases (int): The number of phases (bins) to divide the iterations into.

    Returns:
        dict: A dictionary mapping phase names (e.g., 'Phase 1', 'Phase 2') to
              numpy arrays representing the heatmaps.
    """
    if num_phases <= 0:
        raise ValueError("num_phases must be a positive integer")

    # Generate phase names dynamically
    phases = [f"Phase {i+1}" for i in range(num_phases)]
    # Track both sums and counts for averaging later
    phase_beta_sums = {phase: defaultdict(float) for phase in phases}
    phase_beta_counts = {phase: defaultdict(int) for phase in phases}
    final_heatmaps = {phase: np.zeros((num_tasks, num_tasks)) for phase in phases}

    all_exchange_iterations = sorted(exchange_df['iteration'].dropna().astype(int).unique())

    if not all_exchange_iterations:
        print("No exchange iteration data found.")
        # Return empty heatmaps matching the phase structure
        return {phase: np.zeros((num_tasks, num_tasks)) for phase in phases}

    # Determine iteration range for binning
    min_iter = min(all_exchange_iterations) if all_exchange_iterations else 0
    max_iter = max(all_exchange_iterations) if all_exchange_iterations else 1 # Use 1 if only one iter

    # Create bin edges using linspace for equal intervals
    # Add a small epsilon to max_iter to ensure the last iteration falls into the last bin
    bin_edges = np.linspace(min_iter, max_iter + 1, num_phases + 1)

    # Filter beta_df once and prepare for lookup
    layer_beta_df = beta_df[beta_df['layer'] == layer].copy()
    if layer_beta_df.empty:
        print(f"Warning: No beta data found for layer {layer}. Returning empty heatmaps.")
        return {phase: np.zeros((num_tasks, num_tasks)) for phase in phases}

    layer_beta_df['iteration'] = layer_beta_df['iteration'].astype(int)
    try:
        layer_beta_df.set_index(['agent_id', 'iteration', 'source_index'], inplace=True)
        layer_beta_df.sort_index(inplace=True)
        agent_ids_with_betas = set(layer_beta_df.index.get_level_values('agent_id').unique())
    except KeyError as e:
         print(f"Error setting index on layer_beta_df, missing column? {e}")
         return {phase: np.zeros((num_tasks, num_tasks)) for phase in phases}


    exchanges_grouped_by_receiver = exchange_df.groupby('agent_id')

    for agent_id, received_exchanges in exchanges_grouped_by_receiver:
        if agent_id not in agent_ids_with_betas:
            # print(f"Debug: Agent {agent_id} has exchanges but no betas for layer {layer}. Skipping.")
            continue # Skip agents with no betas for this layer

        target_task = int(agent_id)
        agent_exchange_iters = sorted(received_exchanges['iteration'].dropna().astype(int).unique())

        if not agent_exchange_iters:
            continue

        # Determine beta sampling iterations
        beta_sampling_iters = {}
        agent_beta_iters = None # Cache agent's available beta iterations
        try:
            # Get all beta iterations available for this agent_id ONCE. Check if index exists.
             if agent_id in layer_beta_df.index.get_level_values('agent_id'):
                agent_beta_iters = layer_beta_df.loc[agent_id].index.get_level_values('iteration').unique()
             else:
                 # This agent ID is not in the filtered beta index
                 print(f"Debug: Agent {agent_id} not found in beta index after filtering for layer {layer}.")
                 continue # Skip this agent if it has no betas at all for this layer
        except KeyError:
             print(f"Debug: KeyError looking up agent {agent_id} in beta index.") # Should not happen if agent_id in agent_ids_with_betas
             continue
        except AttributeError: # Handle case where .loc[agent_id] might return a Series if only one iter/index exists
             if isinstance(layer_beta_df.loc[agent_id], pd.Series): # Check if it's a Series (less common case)
                 # If loc returns a Series, index structure is different. Handle appropriately.
                 # This part might need adjustment based on actual data structure in edge cases.
                 print(f"Warning: Unexpected data structure for agent {agent_id} betas. Skipping.")
                 continue
             else: # Some other AttributeError
                 print(f"Warning: AttributeError accessing beta iterations for agent {agent_id}. Skipping.")
                 continue


        if agent_beta_iters is None or agent_beta_iters.empty:
            # print(f"Debug: No beta iterations found for agent {agent_id} after attempting lookup. Skipping.")
            continue


        max_beta_iter_for_agent = agent_beta_iters.max()

        for i, t_exchange in enumerate(agent_exchange_iters):
            beta_iter_to_use = -1
            if i + 1 < len(agent_exchange_iters):
                t_next_exchange = agent_exchange_iters[i+1]
                # Find the highest available beta iteration <= t_next_exchange - 1
                valid_beta_iters = agent_beta_iters[agent_beta_iters <= t_next_exchange - 1]
                if len(valid_beta_iters) > 0:
                     beta_iter_to_use = valid_beta_iters.max()

            else: # Last exchange round
                 # Use the latest beta available that's >= t_exchange
                 valid_beta_iters = agent_beta_iters[agent_beta_iters >= t_exchange]
                 if len(valid_beta_iters) > 0:
                    beta_iter_to_use = valid_beta_iters.max() # Or min() depending on exact definition needed


            # Ensure we found a valid beta iter that is not before the exchange itself
            if beta_iter_to_use != -1 and beta_iter_to_use >= t_exchange:
                beta_sampling_iters[t_exchange] = beta_iter_to_use
            #else: print(f"Debug: No valid t_beta found for agent {agent_id}, t_exchange {t_exchange}")


        # Process each exchange round
        for t_exchange, t_beta in beta_sampling_iters.items():
            current_exchanges = received_exchanges[received_exchanges['iteration'] == t_exchange].sort_values(by='port').reset_index()
            if current_exchanges.empty: continue

            # Determine the phase using the calculated bin edges
            # Use side='right' so iteration `t` falls into bin `[edge_i, edge_{i+1})`
            phase_idx = np.digitize([t_exchange], bin_edges, right=False)[0] - 1
            # Clamp index to valid range [0, num_phases-1]
            phase_idx = max(0, min(phase_idx, num_phases - 1))
            phase = phases[phase_idx]

            try:
                # Betas for this agent at the specific t_beta iteration
                betas_at_t_beta = layer_beta_df.loc[(agent_id, t_beta)]
                beta_lookup = betas_at_t_beta['beta_value'].to_dict()
            except KeyError:
                # print(f"Debug: No betas found for agent {agent_id} at t_beta {t_beta} (for t_exchange {t_exchange}).")
                continue
            except TypeError as e: # Handle potential multiindex type issues
                 print(f"Debug: TypeError accessing betas for agent {agent_id}, t_beta {t_beta}. Error: {e}")
                 continue


            # Update self-influence
            self_beta = beta_lookup.get(0, 0.0)
            if 0 <= target_task < num_tasks:
                phase_beta_sums[phase][(target_task, target_task)] += self_beta
                phase_beta_counts[phase][(target_task, target_task)] += 1

            # Update influence from others
            for i, exchange_row in enumerate(current_exchanges.itertuples(), 1):
                source_task = int(exchange_row.source_task)
                beta_val = beta_lookup.get(i, 0.0)
                if 0 <= source_task < num_tasks and 0 <= target_task < num_tasks:
                    phase_beta_sums[phase][(source_task, target_task)] += beta_val
                    phase_beta_counts[phase][(source_task, target_task)] += 1

    # Convert sums and counts to averages in the final numpy arrays
    for phase in phases:
        for (source, target), beta_sum in phase_beta_sums[phase].items():
            count = phase_beta_counts[phase].get((source, target), 1)
            if 0 <= source < num_tasks and 0 <= target < num_tasks:
                final_heatmaps[phase][source, target] = beta_sum / count

    return final_heatmaps

def convert_summed_to_averaged_heatmaps(phase_heatmaps, num_tasks):
    """
    Converts a dict of summed beta influence heatmaps into averaged heatmaps
    by keeping a count matrix and dividing at the end.

    Args:
        phase_heatmaps (dict): Dictionary mapping phase names to heatmap defaultdicts (source, target): beta_sum
        num_tasks (int): Total number of tasks (matrix dimensions)

    Returns:
        dict: Dictionary mapping phase names to averaged numpy arrays
    """
    averaged_heatmaps = {}
    for phase, influence_dict in phase_heatmaps.items():
        value_matrix = np.zeros((num_tasks, num_tasks))
        count_matrix = np.zeros((num_tasks, num_tasks))

        for (source, target), beta_sum in influence_dict.items():
            if 0 <= source < num_tasks and 0 <= target < num_tasks:
                value_matrix[source, target] += beta_sum
                count_matrix[source, target] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            averaged_matrix = np.divide(value_matrix, count_matrix)
            averaged_matrix[np.isnan(averaged_matrix)] = 0.0

        averaged_heatmaps[phase] = averaged_matrix

    return averaged_heatmaps


def plot_phased_heatmaps(phase_heatmaps, save_path_prefix="phased_beta_heatmap",
                         hide_diagonal=False, manual_masks=None):
    """
    Plots phased heatmaps, with options to hide the diagonal, manually mask
    specific source rows, target columns, or individual cells, and adapt to the
    number of phases provided.

    Args:
        phase_heatmaps (dict): Dictionary mapping phase names to heatmap numpy arrays.
        save_path_prefix (str): Prefix for the saved file name.
        hide_diagonal (bool): If True, sets diagonal elements to NaN and adjusts
                              the color scale to off-diagonal values.
        manual_masks (dict | None): Dictionary to specify manual masks.
            Format: {'PhaseName': {'sources': [id1, id2],
                                    'targets': [id3, id4],
                                    'cells': [(r1, c1), (r2, c2)]}}
            Allows masking entire rows (sources), columns (targets), or specific cells.
    """

    phases = list(phase_heatmaps.keys())
    if not phases:
        print("Warning: No phases found in phase_heatmaps dictionary. Cannot plot.")
        return

    num_phases = len(phases)
    num_tasks = phase_heatmaps[phases[0]].shape[0]

    # --- 1. Prepare matrices with ALL masking applied ---
    masked_matrices = {}
    for phase in phases:
        matrix = phase_heatmaps[phase].copy()

        if hide_diagonal:
            np.fill_diagonal(matrix, np.nan)

        if manual_masks and phase in manual_masks:
            phase_mask_info = manual_masks[phase]
            sources_to_mask = phase_mask_info.get('sources', [])
            for source_id in sources_to_mask:
                if 0 <= source_id < num_tasks:
                    matrix[source_id, :] = np.nan
                else:
                    print(f"Warning: Manual mask source_id {source_id} out of bounds [0, {num_tasks-1}] for phase '{phase}'.")

            targets_to_mask = phase_mask_info.get('targets', [])
            for target_id in targets_to_mask:
                if 0 <= target_id < num_tasks:
                    matrix[:, target_id] = np.nan
                else:
                    print(f"Warning: Manual mask target_id {target_id} out of bounds [0, {num_tasks-1}] for phase '{phase}'.")

            cells_to_mask = phase_mask_info.get('cells', [])
            for r, c in cells_to_mask:
                if 0 <= r < num_tasks and 0 <= c < num_tasks:
                    matrix[r, c] = np.nan
                else:
                    print(f"Warning: Manual mask cell ({r}, {c}) out of bounds [0, {num_tasks-1}] for phase '{phase}'.")

        masked_matrices[phase] = matrix

    all_visible_values = []
    for phase in phases:
        visible_in_phase = masked_matrices[phase][~np.isnan(masked_matrices[phase])]
        if visible_in_phase.size > 0:
            all_visible_values.append(visible_in_phase)

    if all_visible_values:
        all_visible_values = np.concatenate(all_visible_values)
        if len(all_visible_values) > 0:
            vmin_plot = np.min(all_visible_values)
            vmax_plot = np.max(all_visible_values)
        else:
            vmin_plot = 0
            vmax_plot = 1
    else:
        vmin_plot = 0
        vmax_plot = 1

    if vmin_plot == vmax_plot:
        if vmin_plot == 0: vmax_plot = 1e-9
        else: vmax_plot += abs(vmin_plot) * 1e-9 + 1e-9

    fig_width = max(15, 5 * num_phases)
    fig = plt.figure(figsize=(fig_width + 2, 5))
    gs = gridspec.GridSpec(1, num_phases + 1, width_ratios=[1]*num_phases + [0.05])
    axes = [plt.subplot(gs[i]) for i in range(num_phases)]
    cbar_ax = plt.subplot(gs[-1])

    for i, phase in enumerate(phases):
        ax = axes[i]
        matrix_to_plot = masked_matrices[phase]

        sns.heatmap(
            matrix_to_plot,
            ax=ax,
            cmap="YlGnBu",
            vmin=vmin_plot,
            vmax=vmax_plot,
            square=True,
            linewidths=0.5,
            linecolor='lightgrey',
            cbar=(i == num_phases - 1),
            cbar_ax=cbar_ax if i == num_phases - 1 else None,
            xticklabels=[str(j) for j in range(num_tasks)],
            yticklabels=[str(j) for j in range(num_tasks)] if i == 0 else False,
            #norm=PowerNorm(gamma=0.5)
        )
        ax.set_title(f"{phase}")
        ax.set_xlabel("Target Task")
        if i == 0:
            ax.set_ylabel("Source Task")

    #plt.suptitle(f"Influence Heatmap ({save_path_prefix.split('/')[-1]})", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 1])

    suffix = ""
    if hide_diagonal: suffix += "_noDiag"
    if manual_masks: suffix += "_masked"
    if hide_diagonal or manual_masks: suffix += "_rescaled"

    plt.savefig(f"{save_path_prefix}{suffix}.pdf")
    print(f"Saved heatmap to {save_path_prefix}{suffix}.pdf")
    plt.close(fig)

def plot_normalized_phased_heatmaps(phase_heatmaps, save_path_prefix="phased_beta_heatmap_norm", normalize='row', hide_diagonal=False, manual_masks=None):
    """
    Plot row- or column-normalized phased heatmaps with shared colorbar and optional masking.

    Args:
        phase_heatmaps (dict): Dict mapping phase names to heatmap matrices.
        save_path_prefix (str): Output file name prefix.
        normalize (str): 'row' or 'column' normalization.
        hide_diagonal (bool): Optionally hide diagonal.
        manual_masks (dict): Optional manual masking per phase.
    """
    phases = list(phase_heatmaps.keys())
    if not phases:
        print("Warning: No phases found in phase_heatmaps dictionary. Cannot plot.")
        return

    num_phases = len(phases)
    num_tasks = phase_heatmaps[phases[0]].shape[0]

    norm_matrices = {}
    for phase in phases:
        matrix = phase_heatmaps[phase].copy()

        if normalize == 'row':
            with np.errstate(invalid='ignore', divide='ignore'):
                matrix = matrix / matrix.sum(axis=1, keepdims=True)
        elif normalize == 'column':
            with np.errstate(invalid='ignore', divide='ignore'):
                matrix = matrix / matrix.sum(axis=0, keepdims=True)
        matrix = np.nan_to_num(matrix)

        if hide_diagonal:
            np.fill_diagonal(matrix, np.nan)

        if manual_masks and phase in manual_masks:
            mask_info = manual_masks[phase]
            for r in mask_info.get('sources', []):
                if 0 <= r < num_tasks:
                    matrix[r, :] = np.nan
            for c in mask_info.get('targets', []):
                if 0 <= c < num_tasks:
                    matrix[:, c] = np.nan
            for r, c in mask_info.get('cells', []):
                if 0 <= r < num_tasks and 0 <= c < num_tasks:
                    matrix[r, c] = np.nan

        norm_matrices[phase] = matrix

    all_values = np.concatenate([m[~np.isnan(m)] for m in norm_matrices.values()])
    vmin_plot = np.nanmin(all_values)
    vmax_plot = np.nanmax(all_values)
    if vmin_plot == vmax_plot:
        vmax_plot += 1e-9

    fig_width = max(15, 5 * num_phases)
    fig = plt.figure(figsize=(fig_width + 2, 5))
    gs = gridspec.GridSpec(1, num_phases + 1, width_ratios=[1]*num_phases + [0.05])
    axes = [plt.subplot(gs[i]) for i in range(num_phases)]
    cbar_ax = plt.subplot(gs[-1])

    for i, phase in enumerate(phases):
        ax = axes[i]
        matrix = norm_matrices[phase]

        sns.heatmap(
            matrix,
            ax=ax,
            cmap="YlGnBu",
            vmin=vmin_plot,
            vmax=vmax_plot,
            square=True,
            linewidths=0.5,
            linecolor='lightgrey',
            cbar=(i == num_phases - 1),
            cbar_ax=cbar_ax if i == num_phases - 1 else None,
            xticklabels=[str(j) for j in range(num_tasks)],
            yticklabels=[str(j) for j in range(num_tasks)] if i == 0 else False,
            #norm=PowerNorm(gamma=2)
        )
        ax.set_title(f"{phase} ({normalize}-norm)")
        ax.set_xlabel("Target Task")
        if i == 0:
            ax.set_ylabel("Source Task")

    plt.tight_layout(rect=[0, 0, 1, 1])
    suffix = f"_{normalize}Norm"
    if hide_diagonal: suffix += "_noDiag"
    if manual_masks: suffix += "_masked"
    plt.savefig(f"{save_path_prefix}{suffix}.pdf")
    print(f"Saved {normalize}-normalized heatmap to {save_path_prefix}{suffix}.pdf")
    plt.close(fig)


# HEATMAPS for frequency 
def load_exchanges_from_directory(base_dir):
    exchange_paths = []
    for root, dirs, files in os.walk(base_dir):
        if 'exchanges.csv' in files:
            exchange_paths.append(os.path.join(root, 'exchanges.csv'))
    return exchange_paths

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

def visualize_temporal_sliced_heatmaps(exchange_paths, num_tasks=28, num_phases=3):
    """
    Creates 3 heatmaps (early, mid, late) with shared color scale and a single aligned colorbar.
    Adds group dividers between each block of 7 tasks.
    """
    if num_phases <= 0:
        raise ValueError("num_phases must be a positive integer")
    
    phases = [f"Phase {i+1}" for i in range(num_phases)]
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

    #max_iter = max(all_iterations)
    # Determine iteration range for binning
    min_iter = min(all_iterations) if all_iterations else 0
    max_iter = max(all_iterations) if all_iterations else 1 # Use 1 if only one iter
    bins = np.linspace(min_iter, max_iter + 1, num_phases + 1)#[0, max_iter // 3, 2 * max_iter // 3, max_iter + 1]
    print(f'visualize_temporal_sliced_heatmaps: {bins}')

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
    fig_width = max(15, 5 * num_phases)
    fig = plt.figure(figsize=(fig_width + 2, 5))
    gs = gridspec.GridSpec(1, num_phases + 1, width_ratios=[1]*num_phases + [0.05])
    axes = [plt.subplot(gs[i]) for i in range(num_phases)]
    cbar_ax = plt.subplot(gs[-1])

    group_size = 7
    group_lines = [group_size * i for i in range(1, num_tasks // group_size)]

    for i, phase in enumerate(phases):
        ax = axes[i]
        sns.heatmap(
            phase_matrices[phase],
            ax=ax,
            cmap="Blues",
            cbar=(i == num_phases - 1),
            cbar_ax=cbar_ax if i == num_phases - 1 else None,
            square=True,
            linewidths=0.5,
            xticklabels=[str(j) for j in range(num_tasks)],
            yticklabels=[str(j) for j in range(num_tasks)] if i == 0 else False,
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(f"{phase} Communication")
        ax.set_xlabel("Target Task")
        if i == 0:
            ax.set_ylabel("Source Task")

        for line in group_lines:
            ax.axhline(line, color='#929da8', linewidth=1.2)
            ax.axvline(line, color='#929da8', linewidth=1.2)

    plt.suptitle("Communication Heatmap (Dynamic Phases)", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("heatmap_dynamic_phases.pdf")
    plt.close(fig)

def compute_phase_matrices(exchange_paths, num_tasks=28, num_phases=3):
    """
    Reusable function to return phase_matrices = {'Early': ..., 'Mid': ..., 'Late': ...}
    """
    phases = [f"Phase {i+1}" for i in range(num_phases)]
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

    #max_iter = max(all_iterations)
    #bins = [0, max_iter // 3, 2 * max_iter // 3, max_iter + 1]
    # Determine iteration range for binning
    min_iter = min(all_iterations) if all_iterations else 0
    max_iter = max(all_iterations) if all_iterations else 1 # Use 1 if only one iter
    bins = np.linspace(min_iter, max_iter + 1, num_phases + 1)#[0, max_iter // 3, 2 * max_iter // 3, max_iter + 1]

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

def plot_phase_deltas(phase_matrices, num_tasks=28, num_phases=3):
    """
    Plot pairwise delta heatmaps for sequential phases with shared colorbar and axis labels.
    Handles arbitrary number of phases.
    """
    phase_names = list(phase_matrices.keys())

    # Generate delta pairs: (Early ~ Mid), (Mid ~ Late), (Early ~ Late) or more if applicable
    delta_titles = [f"{a} ~ {b}" for a, b in zip(phase_names[:-1], phase_names[1:])] + [f"{phase_names[0]} ~ {phase_names[-1]}"]
    delta_keys = [(phase_names[i], phase_names[i+1]) for i in range(num_phases - 1)] + [(phase_names[0], phase_names[-1])]

    # Compute deltas
    deltas = {title: phase_matrices[b] - phase_matrices[a] for title, (a, b) in zip(delta_titles, delta_keys)}

    # Compute global color scale
    all_values = np.concatenate([mat.flatten() for mat in deltas.values()])
    abs_max = np.max(np.abs(all_values))

    # Plot
    num_deltas = len(deltas)
    fig_width = max(15, 5 * num_phases)
    fig = plt.figure(figsize=(fig_width + 2, 5))
    gs = gridspec.GridSpec(1, num_deltas + 1, width_ratios=[1]*num_deltas + [0.05])
    axes = [plt.subplot(gs[i]) for i in range(num_deltas)]
    cbar_ax = plt.subplot(gs[-1])

    for i, (title, delta_matrix) in enumerate(deltas.items()):
        ax = axes[i]
        sns.heatmap(
            delta_matrix,
            ax=ax,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            xticklabels=[f"{j}" for j in range(num_tasks)],
            yticklabels=[f"{j}" for j in range(num_tasks)] if i == 0 else False,
            vmin=-abs_max,
            vmax=abs_max,
            cbar=(i == num_deltas - 1),
            cbar_ax=cbar_ax if i == num_deltas - 1 else None
        )
        ax.set_title(f"Δ {title}")
        ax.set_xlabel("Target Task")
        if i == 0:
            ax.set_ylabel("Source Task")

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("heatmap_phase_deltas.pdf")
    plt.close(fig)

def plot_normalized_heatmaps(phase_matrices, normalize='row', num_tasks=28, num_phases=3):
    """
    Plot normalized heatmaps with a shared colorbar and axis labels.
    """
    assert normalize in ['row', 'column'], "normalize must be 'row' or 'column'"

    phases = [f"Phase {i+1}" for i in range(num_phases)]
    norm_data = {}

    for phase, matrix in phase_matrices.items():
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_matrix = matrix / matrix.sum(axis=1 if normalize == 'row' else 0, keepdims=True)
            norm_matrix = np.nan_to_num(norm_matrix)
        norm_data[phase] = norm_matrix

    # Setup grid layout for shared colorbar
    fig_width = max(15, 5 * num_phases)
    fig = plt.figure(figsize=(fig_width + 2, 5))
    gs = gridspec.GridSpec(1, num_phases + 1, width_ratios=[1]*num_phases + [0.05])
    axes = [plt.subplot(gs[i]) for i in range(num_phases)]
    cbar_ax = plt.subplot(gs[-1])

    for i, phase in enumerate(phases):
        ax = axes[i]
        sns.heatmap(
            norm_data[phase],
            ax=ax,
            cmap="YlGnBu",
            cbar=(i == num_phases - 1),
            cbar_ax=cbar_ax if i == num_phases - 1 else None,
            square=True,
            linewidths=0.5,
            xticklabels=[str(j) for j in range(num_tasks)],
            yticklabels=[str(j) for j in range(num_tasks)] if i == 0 else False,
            vmin=0,
            vmax=1
        )
        ax.set_title(f"{phase} ({normalize}-normalized)")
        ax.set_xlabel("Target Task")
        if i == 0:
            ax.set_ylabel("Source Task")
        else:
            ax.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f"heatmap_{normalize}_normalized.pdf")
    plt.close(fig)

def compute_phase_frequency_matrices(exchange_paths, num_tasks=28, num_phases=3):
    if num_phases <= 0:
        raise ValueError("num_phases must be a positive integer")

    phases = [f"Phase {i+1}" for i in range(num_phases)]
    phase_matrices = {phase: np.zeros((num_tasks, num_tasks)) for phase in phases}
    all_iterations = []

    for path in exchange_paths:
        try:
            df = pd.read_csv(path)
            if "iteration" in df.columns:
                all_iterations.extend(df["iteration"].dropna().astype(int).tolist())
        except Exception as e:
            print(f"[WARNING] Failed to read {path}: {e}")

    if not all_iterations:
        print("No valid iterations found.")
        return phase_matrices

    min_iter, max_iter = min(all_iterations), max(all_iterations)
    bins = np.linspace(min_iter, max_iter + 1, num_phases + 1)

    for path in exchange_paths:
        try:
            df = pd.read_csv(path)
            agent_path = os.path.dirname(path)
            agent_task = extract_agent_task_id(agent_path)
            if agent_task is None:
                continue

            df = df.dropna(subset=["iteration", "port"])
            df["iteration"] = df["iteration"].astype(int)
            df["source_task"] = df["port"].astype(int) - 29500
            df["target_task"] = agent_task
            df["phase"] = pd.cut(df["iteration"], bins=bins, labels=phases, right=False)

            for _, row in df.iterrows():
                src = int(row["source_task"])
                tgt = int(row["target_task"])
                phase = row["phase"]
                if phase in phases and 0 <= src < num_tasks and 0 <= tgt < num_tasks:
                    phase_matrices[phase][src, tgt] += 1

        except Exception as e:
            print(f"[ERROR] Parsing {path}: {e}")

    return phase_matrices

def plot_frequency_heatmaps(phase_matrices, save_prefix="freq_heatmap"):
    phases = list(phase_matrices.keys())
    num_phases = len(phases)
    num_tasks = phase_matrices[phases[0]].shape[0]

    vmin = min(np.min(m) for m in phase_matrices.values())
    vmax = max(np.max(m) for m in phase_matrices.values())

    fig_width = max(15, 5 * num_phases)
    fig = plt.figure(figsize=(fig_width + 2, 5))
    gs = gridspec.GridSpec(1, num_phases + 1, width_ratios=[1] * num_phases + [0.05])
    axes = [plt.subplot(gs[i]) for i in range(num_phases)]
    cbar_ax = plt.subplot(gs[-1])

    for i, phase in enumerate(phases):
        ax = axes[i]
        matrix = phase_matrices[phase]

        sns.heatmap(
            matrix,
            ax=ax,
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            linecolor="gray",
            cbar=(i == num_phases - 1),
            cbar_ax=cbar_ax if i == num_phases - 1 else None,
            xticklabels=[str(j) for j in range(num_tasks)],
            yticklabels=[str(j) for j in range(num_tasks)] if i == 0 else False
        )
        ax.set_title(f"{phase}")
        ax.set_xlabel("Target Task")
        if i == 0:
            ax.set_ylabel("Source Task")

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f"{save_prefix}.pdf")
    print(f"Saved frequency heatmap to {save_prefix}.pdf")
    plt.close(fig)


# HEATMAPS for similarity metric

def load_all_metadata_files(base_path):
    metadata_paths = []
    for root, dirs, files in os.walk(base_path):
        if "metadata.csv" in files:
            metadata_paths.append(os.path.join(root, "metadata.csv"))
    return metadata_paths

def extract_agent_task_id(path):
    while path != os.path.dirname(path):  # While not at root
        folder = os.path.basename(path)
        match = re.match(r"agent_(\d+)", folder)
        if match:
            return int(match.group(1))
        path = os.path.dirname(path)
    return None

def build_similarity_dataframe(metadata_paths):
    records = []

    for path in metadata_paths:
        try:
            df = pd.read_csv(path)
            target_task = extract_agent_task_id(path)
            if target_task is None:
                continue

            df = df.dropna(subset=["iteration", "sender_port", "sender_similarity"])
            df["iteration"] = df["iteration"].astype(int)
            df["source_task"] = df["sender_port"].astype(int) - 29500
            df["target_task"] = target_task
            df["similarity"] = df["sender_similarity"].astype(float)

            records.extend(df[["iteration", "source_task", "target_task", "similarity"]].to_dict("records"))

        except Exception as e:
            print(f"[ERROR] reading {path}: {e}")

    return pd.DataFrame(records)

def compute_similarity_heatmaps(sim_df, num_tasks=28, num_phases=3):
    sim_df = sim_df.copy()
    if sim_df.empty:
        return {}

    min_iter = sim_df["iteration"].min()
    max_iter = sim_df["iteration"].max()
    bins = np.linspace(min_iter, max_iter + 1, num_phases + 1)
    print(bins)
    phases = [f"Phase {i+1}" for i in range(num_phases)]

    sim_df["phase"] = pd.cut(sim_df["iteration"], bins=bins, labels=phases, right=False)

    phase_matrices = {phase: np.full((num_tasks, num_tasks), np.nan) for phase in phases}
    count_matrices = {phase: np.zeros((num_tasks, num_tasks)) for phase in phases}

    for _, row in sim_df.iterrows():
        src = int(row["source_task"])
        tgt = int(row["target_task"])
        sim = float(row["similarity"])
        phase = row["phase"]
        if phase in phases and 0 <= src < num_tasks and 0 <= tgt < num_tasks:
            if np.isnan(phase_matrices[phase][src, tgt]):
                phase_matrices[phase][src, tgt] = sim
            else:
                phase_matrices[phase][src, tgt] += sim
            count_matrices[phase][src, tgt] += 1

    for phase in phases:
        with np.errstate(invalid='ignore'):
            phase_matrices[phase] = np.divide(phase_matrices[phase], count_matrices[phase])

    return phase_matrices

def plot_similarity_heatmaps(phase_matrices, save_prefix="similarity_heatmap"):
    phases = list(phase_matrices.keys())
    num_phases = len(phases)
    num_tasks = next(iter(phase_matrices.values())).shape[0]

    vmin = np.nanmin([np.nanmin(m) for m in phase_matrices.values()])
    vmax = np.nanmax([np.nanmax(m) for m in phase_matrices.values()])

    fig_width = max(15, 5 * num_phases)
    fig = plt.figure(figsize=(fig_width + 2, 5))
    gs = gridspec.GridSpec(1, num_phases + 1, width_ratios=[1]*num_phases + [0.05])
    axes = [plt.subplot(gs[i]) for i in range(num_phases)]
    cbar_ax = plt.subplot(gs[-1])

    for i, phase in enumerate(phases):
        ax = axes[i]
        matrix = phase_matrices[phase]

        sns.heatmap(
            matrix,
            ax=ax,
            cmap="YlOrBr",
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            linecolor="gray",
            cbar=(i == num_phases - 1),
            cbar_ax=cbar_ax if i == num_phases - 1 else None,
            xticklabels=[str(j) for j in range(num_tasks)],
            yticklabels=[str(j) for j in range(num_tasks)] if i == 0 else False,
            norm=PowerNorm(5)
        )
        ax.set_title(phase)
        ax.set_xlabel("Target Task")
        if i == 0:
            ax.set_ylabel("Source Task")

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f"{save_prefix}.pdf")
    print(f"Saved similarity heatmap to {save_prefix}.pdf")
    plt.close(fig)



# EXTRA ANALYSIS
import networkx as nx
import matplotlib.pyplot as plt

def build_similarity_graph(matrix, k=3, threshold=None):
    num_tasks = matrix.shape[0]
    G = nx.DiGraph()
    for tgt in range(num_tasks):
        similarities = matrix[:, tgt]
        if threshold is not None:
            top_sources = np.where(similarities >= threshold)[0]
        else:
            top_sources = similarities.argsort()[-k:][::-1]
        for src in top_sources:
            if similarities[src] > 0:
                G.add_edge(src, tgt, weight=similarities[src])
    return G

def plot_similarity_graph(G, save_path="similarity_graph.pdf"):
    pos = nx.circular_layout(G)
    weights = [G[u][v]["weight"] * 5 for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", width=weights, arrows=True)
    plt.title("Top-K Similarity Curriculum Graph")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

import seaborn as sns

def plot_clustered_similarity_heatmap(matrix, save_path="clustered_similarity_heatmap.pdf"):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    # Replace NaNs or infinities with zeros (or optionally the mean of finite entries)
    clean_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    sns.clustermap(
        clean_matrix,
        method="average",
        metric="cosine",
        cmap="YlOrBr",
        linewidths=0.5
    )
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved clustered heatmap to {save_path}")

from scipy.cluster.hierarchy import linkage, dendrogram


def plot_similarity_dendrogram(matrix, save_path="similarity_dendrogram.pdf"):
    # Clean the matrix
    clean_matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    clean_matrix = (clean_matrix + clean_matrix.T) / 2  # enforce symmetry

    # Compute linkage
    linked = linkage(clean_matrix, method="average", metric="cosine")

    # Sorted task labels (e.g., 0 to N)
    num_tasks = matrix.shape[0]
    labels = [str(i) for i in range(num_tasks)]

    plt.figure(figsize=(10, 2))
    dendrogram(
        linked,
        labels=labels,
        leaf_rotation=0,
        leaf_font_size=10,
        orientation="top",
        distance_sort=False,  # important: disables reordering
        show_leaf_counts=False
    )
    plt.title("Curriculum Tree (Dendrogram)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved dendrogram to {save_path}")

def plot_similarity_over_time(phase_matrices, target_task, top_k=3, save_path="similarity_traces.pdf"):
    phases = list(phase_matrices.keys())
    num_tasks = next(iter(phase_matrices.values())).shape[0]

    sim_traces = {src: [] for src in range(num_tasks)}
    for phase in phases:
        for src in range(num_tasks):
            sim_traces[src].append(phase_matrices[phase][src, target_task])

    # Choose top-K sources by average similarity
    avg_sims = {src: np.mean(values) for src, values in sim_traces.items()}
    top_sources = sorted(avg_sims, key=avg_sims.get, reverse=True)[:top_k]

    plt.figure(figsize=(10, 6))
    for src in top_sources:
        plt.plot(phases, sim_traces[src], label=f"Task {src}")
    plt.xlabel("Phase")
    plt.ylabel("Similarity")
    plt.title(f"Agent Perception Over Time for Target Task {target_task}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_similarity_frequency(sim_matrix, freq_matrix, threshold=0.01, save_path="similarity_vs_frequency_overlay.pdf"):
    # Mask similarities where frequency is too low
    mask = freq_matrix < threshold
    masked = np.where(mask, np.nan, sim_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(masked, cmap="YlGnBu", linewidths=0.5, square=True)
    plt.title("Similarity Masked by Low Frequency")
    plt.xlabel("Target Task")
    plt.ylabel("Source Task")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def scatter_similarity_vs_beta(sim_matrix, beta_matrix, save_path="similarity_vs_beta.pdf"):
    similarities = []
    betas = []
    num_tasks = sim_matrix.shape[0]

    for i in range(num_tasks):
        for j in range(num_tasks):
            similarities.append(sim_matrix[i, j])
            betas.append(beta_matrix[i, j])

    plt.figure(figsize=(8, 6))
    plt.scatter(similarities, betas, alpha=0.6)
    plt.xlabel("Similarity")
    plt.ylabel("Beta Coefficient")
    plt.title("Similarity vs. Influence (Beta)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



# BAYESIAN DAG
def build_bayesian_network_from_similarity(similarity_matrix, phase_name="Phase 3", top_k=5):
    """
    Build a Bayesian Network DAG based on top-k similarity edges.

    Args:
        similarity_matrix (np.ndarray): A 2D numpy array representing similarity values.
        phase_name (str): Name of the phase (for annotation).
        top_k (int): Number of top sources to include for each target.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    from pgmpy.estimators import HillClimbSearch, BicScore

    num_tasks = similarity_matrix.shape[0]
    records = []

    for tgt in range(num_tasks):
        sim_vector = similarity_matrix[:, tgt]
        top_sources = np.argsort(sim_vector)[-top_k:]
        record = {f"Task_{src}": sim_vector[src] for src in top_sources}
        record["Target"] = tgt  # Optional for reference
        records.append(record)

    df = pd.DataFrame(records).fillna(0.0)
    if "Target" in df.columns:
        df = df.drop(columns=["Target"])

    est = HillClimbSearch(df)
    model = est.estimate(scoring_method=BicScore(df))

    print(f"[✓] Learned Bayesian DAG ({phase_name}):")
    for edge in model.edges():
        print(f"  {edge[0]} → {edge[1]}")

    # Visualize
    G = nx.DiGraph(model.edges())
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", width=1.5, arrows=True)
    plt.title(f"Bayesian DAG ({phase_name})")
    plt.tight_layout()
    plt.savefig(f"bayesian_dag_{phase_name.replace(' ', '_').lower()}.pdf")
    plt.close()


# BAYESIAN POSTERIOR APPROXIMATION HEATMAP
def compute_bayesian_posterior_heatmap(similarity_matrix, frequency_matrix, normalize_axis=0):
    """
    Compute an approximate Bayesian posterior heatmap using similarity as the likelihood
    and frequency as the prior. Normalizes across the specified axis.

    Args:
        similarity_matrix (np.ndarray): Similarity matrix (likelihood).
        frequency_matrix (np.ndarray): Frequency matrix (prior).
        normalize_axis (int): 0 to normalize columns (target-wise), 1 to normalize rows (source-wise).

    Returns:
        np.ndarray: Normalized posterior matrix.
    """
    posterior_matrix = similarity_matrix * frequency_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = posterior_matrix.sum(axis=normalize_axis, keepdims=True)
        normalized_posterior = np.divide(posterior_matrix, norm)
        normalized_posterior[np.isnan(normalized_posterior)] = 0.0
    return normalized_posterior

def plot_bayesian_posterior_heatmap(posterior_matrix, save_path="heatmaps/bayesian_posterior_heatmap.pdf"):
    """
    Plot the Bayesian posterior matrix as a heatmap and save it to a file.

    Args:
        posterior_matrix (np.ndarray): Posterior matrix to plot.
        save_path (str): Path to save the heatmap image.
    """
    num_tasks = posterior_matrix.shape[0]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        posterior_matrix,
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="gray",
        square=True,
        xticklabels=[str(i) for i in range(num_tasks)],
        yticklabels=[str(i) for i in range(num_tasks)]
    )
    plt.title("Bayesian Posterior Approximation")
    plt.xlabel("Target Task")
    plt.ylabel("Source Task")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Bayesian posterior heatmap to {save_path}")


if __name__ == "__main__":
    OUTPUT_DIR = "heatmaps"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_path = "log/FINAL/mctgraph/fullcomm2/seed1"
    exchange_paths = load_exchanges_from_directory(base_path)
    edge_weights = build_dependency_graph(exchange_paths)
    num_tasks = 28
    num_phases = 1


    #visualize_temporal_sliced_heatmaps(exchange_paths, num_tasks=num_tasks, num_phases=num_phases)

    freq_matrices = compute_phase_frequency_matrices(exchange_paths, num_tasks=num_tasks, num_phases=num_phases)
    plot_frequency_heatmaps(freq_matrices)

    # Build once
    phase_matrices = compute_phase_matrices(exchange_paths, num_tasks=num_tasks, num_phases=num_phases)

    # Plot deltas
    plot_phase_deltas(phase_matrices, num_tasks=num_tasks, num_phases=num_phases)

    # Plot row- and column-normalized variants
    plot_normalized_heatmaps(phase_matrices, normalize='row', num_tasks=num_tasks, num_phases=num_phases)
    plot_normalized_heatmaps(phase_matrices, normalize='column', num_tasks=num_tasks, num_phases=num_phases)



    # Step 2: Load and process data
    exchange_df = extract_exchange_df(exchange_paths)
    beta_df = extract_betas_df(base_path)


    layers = beta_df['layer'].unique().tolist()
    for layer in layers:
        # Step 3: Compute per-phase weighted beta heatmaps
        phased_heatmaps = compute_phased_weighted_heatmaps(beta_df, exchange_df, layer=layer, num_tasks=num_tasks, num_phases=num_phases)
        print(phased_heatmaps)

        #averaged_heatmaps = convert_summed_to_averaged_heatmaps(phased_heatmaps, num_tasks)

        #plot_phased_heatmaps(phased_heatmaps, save_path_prefix=f"avg_phased_beta_heatmap_{layer}")




        # Define the specific cells to mask: (row, column)
        cells_to_mask_mid_late = []
        for r in range(14, 19): # Rows 14, 15, 16, 17, 18
            for c in [19, 20]: # Columns 19, 20
                cells_to_mask_mid_late.append((r, c))

        # Create the manual_masks dictionary
        masks_to_apply = {
            'Phase 2': { # Assuming 'Mid' phase is Phase 2
                'cells': cells_to_mask_mid_late
            },
            'Phase 3': { # Assuming 'Late' phase is Phase 3
                'cells': cells_to_mask_mid_late
                # You could also add other masks here for Phase 3 if needed, e.g.
                # 'sources': [some_other_source_to_mask_only_in_late_phase]
            }
            # Add entries for other phases if needed, otherwise they won't be masked
        }

        # Step 4: Plot and save
        plot_phased_heatmaps(
            phased_heatmaps,
            save_path_prefix=f"phased_beta_heatmap_{layer}",
            hide_diagonal=True, 
            #manual_masks=masks_to_apply
        )

        plot_normalized_phased_heatmaps(
            phased_heatmaps,
            save_path_prefix=f"phased_beta_heatmap_{layer}",
            hide_diagonal=True,
            normalize='column'
        )

        plot_similarity_dendrogram(phased_heatmaps[list(phased_heatmaps)[-1]], f'beta_dendrogram_{layer}.pdf')


    metadata_paths = load_all_metadata_files(base_path)
    sim_df = build_similarity_dataframe(metadata_paths)
    sim_heatmaps = compute_similarity_heatmaps(sim_df, num_tasks=num_tasks, num_phases=num_phases)
    sim_heatmaps[list(sim_heatmaps)[-1]] = np.nan_to_num(sim_heatmaps[list(sim_heatmaps)[-1]], nan=1.0, posinf=1.0, neginf=1.0)
    plot_similarity_heatmaps(sim_heatmaps, save_prefix="similarity_heatmap")


    # EXTRA ANALYSIS
    similarity_matrix = sim_heatmaps[list(sim_heatmaps)[-1]]#phase_matrices['Phase 3']  # use latest for hierarchy
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=1.0, posinf=1.0, neginf=1.0)
    #similarity_matrix = similarity_matrix ** 2
    # 1A: Top-K Similarity Graph
    G = build_similarity_graph(similarity_matrix, k=3)
    plot_similarity_graph(G)

    # 1B: Clustered Heatmap
    plot_clustered_similarity_heatmap(similarity_matrix)

    # 1C: Dendrogram
    plot_similarity_dendrogram(similarity_matrix)

    # 2: Perception over time
    plot_similarity_over_time(phase_matrices, target_task=6, top_k=3)

    # 3: Similarity vs Frequency
    compare_similarity_frequency(similarity_matrix, freq_matrices[list(freq_matrices)[-1]])

    # 4: Similarity vs Beta
    #scatter_similarity_vs_beta(similarity_matrix, phase_matrices)

    # Optional: Normalized heatmaps
    #plot_normalized_phased_heatmaps(phase_matrices, normalize='row')
    #plot_normalized_phased_heatmaps(phase_matrices, normalize='column')

    # Compute Bayesian posterior using similarity and frequency matrices
    similarity_matrix = sim_heatmaps[list(sim_heatmaps)[-1]]
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=1.0, posinf=1.0, neginf=1.0)
    np.save("similarity_matrix.npy", similarity_matrix)
    frequency_matrix = freq_matrices[list(freq_matrices)[-1]]
    np.save("frequency_matrix.npy", frequency_matrix)
    print(similarity_matrix)
    bayesian_posterior_freq = compute_bayesian_posterior_heatmap(similarity_matrix, frequency_matrix)
    np.save("bayesian_posterior_freq.npy", bayesian_posterior_freq)
    plot_bayesian_posterior_heatmap(
        bayesian_posterior_freq,
        save_path="bayesian_posterior_freq.pdf"
    )

    # Compute Bayesian posterior using similarity and last-phase beta matrix
    last_beta_heatmap = phased_heatmaps[list(phased_heatmaps)[-1]]
    def zero_diagonal(matrix):
        """
        Set the diagonal of a square matrix to zero.
        Args:
            matrix (np.ndarray): Input 2D matrix.
        Returns:
            np.ndarray: Matrix with diagonal set to 0.
        """
        matrix = matrix.copy()
        np.fill_diagonal(matrix, 0.0)
        return matrix
    #last_beta_heatmap = zero_diagonal(last_beta_heatmap)
    np.save("last_beta_heatmap.npy", last_beta_heatmap)
    bayesian_posterior_beta = compute_bayesian_posterior_heatmap(similarity_matrix, last_beta_heatmap)
    np.save("bayesian_posterior_beta.npy", bayesian_posterior_beta)
    plot_bayesian_posterior_heatmap(
        bayesian_posterior_beta,
        save_path="bayesian_posterior_beta.pdf"
    )

    plot_similarity_dendrogram(bayesian_posterior_freq, f'bayesian_dendrogram_freq.pdf')
    plot_similarity_dendrogram(bayesian_posterior_beta, f'bayesian_dendrogram_beta.pdf')




    print("Similarity matrix stats:")
    print("Min:", np.min(similarity_matrix), "Max:", np.max(similarity_matrix), "Mean:", np.mean(similarity_matrix))

    print("Prior matrix stats (frequency or beta):")
    print("Min:", np.min(frequency_matrix), "Max:", np.max(frequency_matrix), "Mean:", np.mean(frequency_matrix))

    build_bayesian_network_from_similarity(similarity_matrix, phase_name='190')