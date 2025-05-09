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
    


'''def extract_betas_df(path):
    # Load and clean beta logs
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
            expanded_records.append({
                "iteration": iteration,
                "layer": layer,
                "betas": coeffs,
                "agent_id": agent_id
            })
    beta_df = pd.DataFrame(expanded_records)
    beta_df = beta_df.sort_values(by='iteration').reset_index(drop=True)
    
    return beta_df  # Return the flattened DataFrame with columns: iteration, layer, betas, and agent_id'''

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

"""def compute_phased_weighted_heatmaps(beta_df, exchange_df, comm_interval=10, layer='network.phi_body.layers.0', num_tasks=28):
    phases = ['Early', 'Mid', 'Late']
    phase_heatmaps = {phase: np.zeros((num_tasks, num_tasks)) for phase in phases}
    all_iterations = exchange_df['iteration'].dropna().astype(int).tolist()

    if not all_iterations:
        print("No valid iteration data found.")
        return phase_heatmaps

    max_iter = max(all_iterations)
    bins = [0, max_iter // 3, 2 * max_iter // 3, max_iter + 1]

    grouped = beta_df[beta_df['layer'] == layer].groupby(['agent_id'])

    for agent_id, group in grouped:
        group = group.sort_values('iteration')
        iters = group['iteration'].unique()
        comm_points = list(range(min(iters), max(iters), comm_interval))

        print('comm_points')
        print(comm_points)
        for start in comm_points:
            end = start + comm_interval
            round_betas = group[(group['iteration'] >= start) & (group['iteration'] < end)]
            if round_betas.empty:
                continue

            final_row = round_betas[round_betas['iteration'] == round_betas['iteration'].max()]
            final_coeffs = final_row[['source_index', 'beta_value']].values
            #print(final_row)
            #print(final_coeffs)
            print()
            received = exchange_df[(exchange_df['iteration'] == start) & (exchange_df['agent_id'] == agent_id)]

            target = agent_id
            phase_idx = np.digitize([start], bins, right=False)[0] - 1
            phase = phases[phase_idx] if 0 <= phase_idx < len(phases) else None
            if phase is None:
                continue

            phase_heatmaps[phase][target, target] += float(final_coeffs[0][1])

            '''max_betas_to_use = min(len(final_coeffs), 1 + len(received))  # 1 for self + # received
            for i in range(max_betas_to_use):
                print(i)
                source = agent_id if i == 0 else received.iloc[i - 1]['source_task']
                beta_val = float(final_coeffs[i][1])
                print(source, agent_id, beta_val)
                phase_heatmaps[phase][source, agent_id] += beta_val'''
            
            #print(received)
            for i, row in enumerate(received.itertuples(), 1):
                print(i, row)
                if i >= len(final_coeffs):
                    print('oh dear')
                    break
                source = getattr(row, 'source_task')
                beta_val = float(final_coeffs[i][1])
                print(source, target, beta_val)
                phase_heatmaps[phase][source, target] += beta_val

    return phase_heatmaps"""

'''def compute_phased_weighted_heatmaps(beta_df, exchange_df, layer='network.phi_body.layers.0', num_tasks=28):
    """
    Computes influence heatmaps weighted by beta values from the iteration
    before the *next* exchange round.

    Args:
        beta_df (pd.DataFrame): Expanded dataframe from extract_betas_df.
                                Must contain columns: 'iteration', 'layer',
                                'source_index', 'beta_value', 'agent_id'.
        exchange_df (pd.DataFrame): Dataframe from extract_exchange_df.
                                    Must contain columns: 'iteration', 'agent_id',
                                    'source_task', 'port' (or others if needed
                                    for ordering).
        layer (str): The specific layer name to compute the heatmap for.
        num_tasks (int): The total number of tasks (for heatmap dimensions).

    Returns:
        dict: A dictionary mapping phase names ('Early', 'Mid', 'Late') to
              numpy arrays representing the heatmaps.
    """
    phases = ['Early', 'Mid', 'Late']
    # Use defaultdict for easier accumulation
    phase_heatmaps = {phase: defaultdict(float) for phase in phases}
    final_heatmaps = {phase: np.zeros((num_tasks, num_tasks)) for phase in phases}

    all_exchange_iterations = sorted(exchange_df['iteration'].dropna().astype(int).unique())

    if not all_exchange_iterations:
        print("No exchange iteration data found.")
        return final_heatmaps

    max_iter = max(all_exchange_iterations) if all_exchange_iterations else 1
    # Ensure bins cover the entire range, handle max_iter=0 or small values
    bin_edges = [0, max(1, max_iter // 3), max(2, (2 * max_iter) // 3), max(3, max_iter + 1)]

    # Filter beta_df once for the relevant layer
    layer_beta_df = beta_df[beta_df['layer'] == layer].copy()
    # Ensure iteration is integer type
    layer_beta_df['iteration'] = layer_beta_df['iteration'].astype(int)
    # Create a multi-index for faster lookups later
    layer_beta_df.set_index(['agent_id', 'iteration', 'source_index'], inplace=True)
    layer_beta_df.sort_index(inplace=True)

    # Group exchanges by the agent who RECEIVED them
    exchanges_grouped_by_receiver = exchange_df.groupby('agent_id')

    for agent_id, received_exchanges in exchanges_grouped_by_receiver:
        target_task = int(agent_id) # Ensure target_task is an integer index

        # Find the unique iterations where THIS agent received exchanges
        agent_exchange_iters = sorted(received_exchanges['iteration'].dropna().astype(int).unique())

        if not agent_exchange_iters:
            continue

        # Determine the beta iteration to use for each exchange round
        beta_sampling_iters = {}
        for i, t_exchange in enumerate(agent_exchange_iters):
            if i + 1 < len(agent_exchange_iters):
                t_next_exchange = agent_exchange_iters[i+1]
                # Use beta from iteration right before the next exchange
                beta_iter_to_use = t_next_exchange - 1
            else:
                # For the last exchange round, find the max beta iteration available for this agent
                try:
                    max_beta_iter_for_agent = layer_beta_df.loc[agent_id].index.get_level_values('iteration').max()
                    beta_iter_to_use = max_beta_iter_for_agent
                except KeyError: # Agent might not have betas if index is empty
                     print(f"Warning: Agent {agent_id} has exchanges but no betas for layer {layer}. Skipping.")
                     beta_iter_to_use = -1 # Indicate no valid beta iteration
                except IndexError: # Handle cases where MultiIndex slicing results in empty DataFrame/Index
                     print(f"Warning: Could not determine max beta iteration for agent {agent_id} (IndexError). Skipping.")
                     beta_iter_to_use = -1


            if beta_iter_to_use >= t_exchange : # Beta must be from *after* the exchange happened
                 beta_sampling_iters[t_exchange] = beta_iter_to_use
            # else: No valid beta iteration found or it's before the exchange


        # Process each exchange round for this agent
        for t_exchange, t_beta in beta_sampling_iters.items():
            # Get exchanges for this agent at this specific iteration
            # IMPORTANT: Ensure consistent order if multiple exchanges happen in the same iteration for the same agent.
            # Add sorting by 'port' or another column if 'read order' isn't reliable enough.
            current_exchanges = received_exchanges[received_exchanges['iteration'] == t_exchange].sort_values(by='port').reset_index()

            if current_exchanges.empty:
                continue # Should not happen based on logic, but safe check

            # Determine the phase based on the exchange iteration
            phase_idx = np.digitize([t_exchange], bin_edges, right=False)[0] - 1
            phase = phases[phase_idx] if 0 <= phase_idx < len(phases) else None
            if phase is None:
                continue

            # --- Retrieve the relevant beta values for t_beta ---
            try:
                # Select all betas for this agent at the target beta iteration
                betas_at_t_beta = layer_beta_df.loc[(agent_id, t_beta)]
                # Create a dictionary for quick lookup: source_index -> beta_value
                beta_lookup = betas_at_t_beta['beta_value'].to_dict()
                #print(f"Agent {agent_id}, t_exchange={t_exchange}, t_beta={t_beta}: Found {len(beta_lookup)} betas.")

            except KeyError:
                # No betas found for this agent at t_beta
                #print(f"Warning: No betas found for agent {agent_id} at iteration {t_beta} (for exchanges at {t_exchange}). Skipping round.")
                continue # Skip this exchange round if no corresponding betas exist

            # --- Update heatmap based on these exchanges and betas ---

            # Handle self-influence (source_index 0)
            self_beta = beta_lookup.get(0, 0.0) # Use .get for safety
            if target_task < num_tasks:
                 phase_heatmaps[phase][(target_task, target_task)] += self_beta
            # else: print(f"Warning: Target task {target_task} out of bounds ({num_tasks}).")


            # Handle influence from others (source_index 1 onwards)
            for i, exchange_row in enumerate(current_exchanges.itertuples(), 1):
                source_task = int(exchange_row.source_task)
                # Index 'i' corresponds to source_index 'i' in the beta tensor
                beta_val = beta_lookup.get(i, 0.0) # Use .get for safety

                #print(f"  Exchange {i}: Source={source_task}, Target={target_task}, Beta Idx={i}, Beta Val={beta_val}")


                # Add to the heatmap dictionary
                if 0 <= source_task < num_tasks and 0 <= target_task < num_tasks:
                     phase_heatmaps[phase][(source_task, target_task)] += beta_val
                # else: print(f"Warning: Task index out of bounds. Source={source_task}, Target={target_task} (max={num_tasks}).")


    # Convert defaultdicts to final numpy arrays
    for phase in phases:
        for (source, target), value in phase_heatmaps[phase].items():
             if 0 <= source < num_tasks and 0 <= target < num_tasks:
                 final_heatmaps[phase][source, target] = value
             #else: Already warned above potentially

    return final_heatmaps'''

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
    phase_heatmaps = {phase: defaultdict(float) for phase in phases}
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
                 phase_heatmaps[phase][(target_task, target_task)] += self_beta

            # Update influence from others
            for i, exchange_row in enumerate(current_exchanges.itertuples(), 1):
                source_task = int(exchange_row.source_task)
                beta_val = beta_lookup.get(i, 0.0)
                if 0 <= source_task < num_tasks and 0 <= target_task < num_tasks:
                     phase_heatmaps[phase][(source_task, target_task)] += beta_val

    # Convert defaultdicts to final numpy arrays
    for phase in phases:
        for (source, target), value in phase_heatmaps[phase].items():
             if 0 <= source < num_tasks and 0 <= target < num_tasks:
                 final_heatmaps[phase][source, target] = value

    return final_heatmaps

'''def plot_phased_heatmaps(phase_heatmaps, save_path_prefix="phased_beta_heatmap"):
    phases = list(phase_heatmaps.keys())
    vmin = min(np.min(mat) for mat in phase_heatmaps.values())
    vmax = max(np.max(mat) for mat in phase_heatmaps.values())

    fig, axes = plt.subplots(1, len(phases), figsize=(20, 6))
    for i, phase in enumerate(phases):
        ax = axes[i]
        sns.heatmap(
            phase_heatmaps[phase],
            ax=ax,
            cmap="YlGnBu",
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            xticklabels=[str(i) for i in range(phase_heatmaps[phase].shape[1])],
            yticklabels=[str(i) for i in range(phase_heatmaps[phase].shape[0])]
        )
        ax.set_title(f"{phase} Phase")
        ax.set_xlabel("Target Task")
        ax.set_ylabel("Source Task")

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}.pdf")
    plt.close()'''

'''def plot_phased_heatmaps(phase_heatmaps, save_path_prefix="phased_beta_heatmap", hide_diagonal=False):
    """
    Plots phased heatmaps, with an option to hide the main diagonal and
    optionally rescale the colorbar based on off-diagonal values.

    Args:
        phase_heatmaps (dict): Dictionary mapping phase names to heatmap numpy arrays.
        save_path_prefix (str): Prefix for the saved file name.
        hide_diagonal (bool): If True, sets the diagonal elements to NaN before plotting
                              AND adjusts the color scale to off-diagonal values.
    """
    phases = list(phase_heatmaps.keys())
    num_tasks = phase_heatmaps[phases[0]].shape[0] if phases else 0

    # --- Determine color scale limits ---
    all_values_orig = np.concatenate([mat.flatten() for mat in phase_heatmaps.values()])
    all_values_orig = all_values_orig[~np.isnan(all_values_orig)]
    vmin_orig = np.min(all_values_orig) if len(all_values_orig) > 0 else 0
    vmax_orig = np.max(all_values_orig) if len(all_values_orig) > 0 else 1

    vmin_plot = vmin_orig
    vmax_plot = vmax_orig

    if hide_diagonal:
        all_values_offdiag = []
        for phase in phases:
            mat = phase_heatmaps[phase].copy()
            np.fill_diagonal(mat, np.nan)
            all_values_offdiag.append(mat.flatten())

        if all_values_offdiag:
             all_values_offdiag = np.concatenate(all_values_offdiag)
             all_values_offdiag = all_values_offdiag[~np.isnan(all_values_offdiag)]

             if len(all_values_offdiag) > 0:
                  vmin_plot = np.min(all_values_offdiag)
                  vmax_plot = np.max(all_values_offdiag)
             else:
                  vmin_plot = 0
                  vmax_plot = 1

    # --- Create plot ---
    fig, axes = plt.subplots(1, len(phases), figsize=(7 * len(phases), 6), squeeze=False, sharey=True)

    for i, phase in enumerate(phases):
        ax = axes[0, i]
        matrix_to_plot = phase_heatmaps[phase].copy()

        if hide_diagonal:
            np.fill_diagonal(matrix_to_plot, np.nan)

        if vmin_plot == vmax_plot:
             vmax_plot += 1e-9

        sns.heatmap(
            matrix_to_plot,
            ax=ax,
            cmap="YlGnBu",
            vmin=vmin_plot,
            vmax=vmax_plot,
            square=True,
            linewidths=0.5,
            cbar=(i == len(phases) - 1),
            # na_color='lightgrey', # <--- REMOVED THIS LINE
            xticklabels=[str(j) for j in range(num_tasks)],
            yticklabels=[str(j) for j in range(num_tasks)]
        )
        ax.set_title(f"{phase} Phase")
        ax.set_xlabel("Target Task")
        if i == 0:
             ax.set_ylabel("Source Task")

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    suffix = "_no_diagonal_rescaled" if hide_diagonal else ""
    plt.savefig(f"{save_path_prefix}{suffix}.pdf")
    print(f"Saved heatmap to {save_path_prefix}{suffix}.pdf")
    plt.close(fig)'''

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

        # Apply diagonal masking FIRST if requested
        if hide_diagonal:
            np.fill_diagonal(matrix, np.nan)

        # Apply manual masks for this phase
        if manual_masks and phase in manual_masks:
            phase_mask_info = manual_masks[phase]

            # Mask source rows
            sources_to_mask = phase_mask_info.get('sources', [])
            for source_id in sources_to_mask:
                if 0 <= source_id < num_tasks:
                    matrix[source_id, :] = np.nan
                else:
                    print(f"Warning: Manual mask source_id {source_id} out of bounds [0, {num_tasks-1}] for phase '{phase}'.")

            # Mask target columns
            targets_to_mask = phase_mask_info.get('targets', [])
            for target_id in targets_to_mask:
                if 0 <= target_id < num_tasks:
                    matrix[:, target_id] = np.nan
                else:
                    print(f"Warning: Manual mask target_id {target_id} out of bounds [0, {num_tasks-1}] for phase '{phase}'.")

            # *** NEW: Mask specific cells ***
            cells_to_mask = phase_mask_info.get('cells', [])
            for r, c in cells_to_mask:
                if 0 <= r < num_tasks and 0 <= c < num_tasks:
                    matrix[r, c] = np.nan
                else:
                     print(f"Warning: Manual mask cell ({r}, {c}) out of bounds [0, {num_tasks-1}] for phase '{phase}'.")
            # *** End NEW ***

        masked_matrices[phase] = matrix # Store the fully masked matrix

    # --- 2. Calculate vmin/vmax from ALL *visible* data across phases ---
    # (Calculation logic remains the same as before)
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


    # --- 3. Create plot ---
    # (Plotting logic remains the same as before)
    fig_width = max(15, 5 * num_phases)
    fig, axes = plt.subplots(1, num_phases, figsize=(fig_width, 5), squeeze=False, sharey=True)

    for i, phase in enumerate(phases):
        ax = axes[0, i]
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
            cbar_kws={"shrink": 0.7},
            xticklabels=[str(j) for j in range(num_tasks)],
            yticklabels=[str(j) for j in range(num_tasks)]
        )
        ax.set_title(f"{phase}")
        ax.set_xlabel("Target Task")
        if i == 0:
             ax.set_ylabel("Source Task")

    plt.suptitle(f"Influence Heatmap ({save_path_prefix.split('/')[-1]})", y=1.02)
    plt.tight_layout(rect=[0, 0, 0.95, 1])

    suffix = ""
    if hide_diagonal: suffix += "_noDiag"
    if manual_masks: suffix += "_masked"
    if hide_diagonal or manual_masks: suffix += "_rescaled"

    plt.savefig(f"{save_path_prefix}{suffix}.pdf")
    print(f"Saved heatmap to {save_path_prefix}{suffix}.pdf")
    plt.close(fig)

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

    #visualize_task_dependency_graph(edge_weights)
    #visualize_sankey_from_edges(edge_weights, min_weight=3)
    #visualize_heatmap_matrix(edge_weights)

    visualize_temporal_sliced_heatmaps(exchange_paths)


    # Build once
    phase_matrices = compute_phase_matrices(exchange_paths)

    #for phase, matrix in phase_matrices.items():
    #    # Save entropy tables to LaTeX files
    #    filename = f"entropy_table_{phase.lower()}.tex"
    #    compute_entropy_table_latex(matrix, phase=phase, output_path=filename)

    #for phase_name, matrix in phase_matrices.items():
    #    plot_sparsity_heatmap(
    #        matrix,
    #        title=f"{phase_name} – Sparsity Mask",
    #        filename=f"sparsity_{phase_name}.pdf"
    #    )

    # Plot deltas
    plot_phase_deltas(phase_matrices)

    # Plot row- and column-normalized variants
    plot_normalized_heatmaps(phase_matrices, normalize='row')
    plot_normalized_heatmaps(phase_matrices, normalize='column')


    #betas_df = extract_betas_df(base_path)
    #all_exchanges_df = pd.concat([pd.read_csv(p) for p in exchange_paths], ignore_index=True)
    #beta_influence_df = extract_aligned_beta_df(betas_df, all_exchanges_df)

    #plot_average_influence_heatmap(beta_influence_df, target_task=6)
    #plot_temporal_beta_trajectories(beta_influence_df)
    #plot_influence_vs_frequency(beta_influence_df)
    #plot_weighted_frequency_heatmap(betas_df, all_exchanges_df)


    # Step 2: Load and process data
    exchange_df = extract_exchange_df(exchange_paths)
    beta_df = extract_betas_df(base_path)

    # Step 3: Compute per-phase weighted beta heatmaps
    phased_heatmaps = compute_phased_weighted_heatmaps(beta_df, exchange_df, layer='network.phi_body.layers.0')


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
    plot_phased_heatmaps(phased_heatmaps, save_path_prefix="phased_beta_heatmap", hide_diagonal=True, manual_masks=masks_to_apply)