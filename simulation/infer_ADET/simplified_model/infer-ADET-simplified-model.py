import networkx as nx
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import statsmodels.api as sm
import time


def gen_graph(num_nodes=None, graph_type='er', **kwargs):
    """
    Generates a random graph based on the specified type and parameters.

    Parameters:
    - num_nodes: Number of nodes in the graph.
    - graph_type: Type of the random graph to generate. Supported types are 'er' for Erdős-Rényi,
                  'ws' for Watts-Strogatz, and 'ba' for Barabási-Albert.
    - **kwargs: Additional parameters required by specific graph models.
                For 'er': p (float) - Probability of edge creation.
                For 'ws': k (int) - Each node is connected to k nearest neighbors in ring topology,
                         p (float) - Probability of rewiring each edge.
                For 'ba': m (int) - Number of edges to attach from a new node to existing nodes.

    Returns:
    - network: The generated graph (NetworkX graph object) or None if parameters are invalid.
    """
    if num_nodes is None or num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer")

    if graph_type == 'er':
        p = kwargs.get('p')
        if p is None or not (0 <= p <= 1):
            raise ValueError("For 'er' graph type, 'p' must be provided and between 0 and 1.")
        graph = nx.random_graphs.erdos_renyi_graph(num_nodes, p)

    elif graph_type == 'ws':
        k = kwargs.get('k')
        p = kwargs.get('p')
        if k is None or p is None or not (0 <= p <= 1):
            raise ValueError("For 'ws' graph type, 'k' and 'p' must be provided, and 'p' must be between 0 and 1.")
        graph = nx.random_graphs.watts_strogatz_graph(num_nodes, k, p)

    elif graph_type == 'ba':
        m = kwargs.get('m')
        if m is None or m <= 0 or m >= num_nodes:
            raise ValueError("For 'ba' graph type, 'm' must be provided and be a positive integer less than num_nodes.")
        graph = nx.random_graphs.barabasi_albert_graph(num_nodes, m)

    elif graph_type == 'sbm':
        blocks = kwargs.get('blocks')
        probs = kwargs.get('probs')
        if blocks is None or probs is None:
            raise ValueError("For 'sbm' graph type, 'blocks' and 'probs' must be provided.")
        graph = nx.stochastic_block_model(blocks, probs)

    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    return graph



# Count the m=hop total neighbor and treated neighbor
def gamma_treated_neighbor(graph, treatment, m=None):
    """
    Calculates m-hop neighbor counts for each node in the graph. Two matrices are returned:
    one for the total neighbor count and another for the treated neighbor count based on a treatment vector.

    Parameters:
    - graph: The input graph (NetworkX graph object).
    - treatment: A NumPy array indicating the treatment status of each node (1 for treated, 0 for untreated).
    - m: The maximum number of hops to consider for neighborhood counts.

    Returns:
    - treated_neighbor_count: A matrix indicating the count of treated neighbors within m hops for each node.
    - total_neighbor_count: A matrix indicating the total count of neighbors within m hops for each node.
    """
    num_nodes = len(graph)

    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
    distance_matrix = [[shortest_path_lengths[i].get(j, float('inf'))
                        for j in range(len(graph))] for i in range(len(graph))]
    distance_matrix_array = np.array(distance_matrix)
    distance_matrix_array = distance_matrix_array.astype(np.float64)
    distance_matrix_array[np.isinf(distance_matrix_array)] = -np.inf

    value_range = range(1, m + 1)

    # Initialize the matrix total_neighbor_count with zeros
    total_neighbor_count = np.zeros((num_nodes, len(value_range) + 1), dtype=int)

    # Count m-hop neighborhoods for each node
    for i in range(num_nodes):
        counts = np.histogram(distance_matrix_array[i, ~np.isinf(distance_matrix_array[i])],
                              bins=list(value_range) + [np.inf])[0]
        total_neighbor_count[i, :] = np.insert(counts, 0, 0)

    # Initialize the matrix treated_neighbor_count with zeros
    treated_neighbor_count = np.zeros((num_nodes, len(value_range) + 1), dtype=int)

    # Count m-hop treated neighborhoods for each node
    distance_matrix_array_mask = distance_matrix_array.copy()
    distance_matrix_array_mask[:, treatment == 0] = 0
    for i in range(num_nodes):
        counts = np.histogram(distance_matrix_array_mask[i, ~np.isinf(distance_matrix_array_mask[i])],
                              bins=list(value_range) + [np.inf])[0]
        treated_neighbor_count[i, :] = np.insert(counts, 0, 0)

    return treated_neighbor_count, total_neighbor_count


def gen_outcome(k0, gamma_full, treatment, gen_type='mod1', tau=0.6, coef_p=0.5, c0=6.0, sigma0=1.0, sigma1=1.0):
    """
    Generate potential outcomes based on treatment, and ture k0.

    Parameters:
    - k0: ture #-hop neighbors.
    - gamma_full: ndarray, n*n.
    - treatment: ndarray, n*1.
    - tau: float, treatment effect.
    - coef_p: float, better to set in (0, 1) for interpretation.
    - sigma0: float, standard deviation of the noise for control group.
    - sigma1: float, standard deviation of the noise for treatment group.

    Returns:
    - outcome: ndarray, generated outcomes with shape (n_samples,).
    """

    if gen_type == 'mod1':
        outcome = treatment * tau + c0 * sum([coef_p ** i * gamma_full[:, i] for i in range(k0 + 1)]) + \
                  np.random.normal(0, treatment * (sigma1 - sigma0) + sigma0, len(treatment))

    elif gen_type == 'mod2':
        outcome = treatment * tau + c0 + sum([coef_p ** (i - 1) * gamma_full[:, i] for i in range(1, k0 + 1)]) + \
                  np.random.normal(0, sigma0, len(treatment))

    else:
        raise ValueError(f"Unsupported graph type: {gen_type}")

    return outcome


def group_node_k_hops(gamma_mat, k):
    """
    Group the nodes by k-hops. (for all nodes / control nodes / treated nodes), etc.

    Parameters:
    - gamma_full: ndarray, n*n.
    - k: int, the number of hops to consider.

    Returns:
    - groups_k: dict, groups of identical rows, using the row's hashable representation as key.
    """
    groups_k = {}
    for control_node_i, row in enumerate(gamma_mat[:, 0:(k + 1)]):
        # Convert the row to a tuple to use as a dictionary key
        row_key = tuple(row)
        if row_key not in groups_k:
            groups_k[row_key] = []
        groups_k[row_key].append(control_node_i)
    return groups_k


def design_matrix_given_groups(gamma_mat, groups):
    """
    Constructs a design matrix for k-hops based on the input matrix and groupings.

    Parameters:
    - gamma_mat: The input matrix (e.g., gamma_matrix).
    - groups: A dictionary where keys are group identifiers and values are lists of row indices belonging to each group.

    Returns:
    - A matrix where each row is marked with 1s for the groups it belongs to, and 0s otherwise.
    """
    # Initialize the matrix with zeros
    group_matrix = np.zeros((len(gamma_mat), len(groups.keys())))

    # Iterate through the groups and mark corresponding entries
    for group_index, (group_key, row_indices) in enumerate(groups.items()):
        for row_index in row_indices:
            group_matrix[row_index, group_index] = 1

    return group_matrix


def infer_tau_same_k0_same_tau_same_sigma01(y_outcomes, gamma_full, treatment, k0_list, sig_level=0.05):
    """
    Inference for tau.
    Setting: k_{i,0}=k_0 for all nodes; tau_i=tau for all treated nodes; sigma_0=sigma_1 for treated and control group

    Parameters:
    y_outcomes: ndarray, shape (n,). The outcomes associated with each node.
    gamma_full: ndarray, shape (n, n). The adjacency matrix representing the network.
    treatment: ndarray, shape (n,). Indicates the treatment status of each node.
    k0_list: list. A list of candidate values for the number of hops (k_0).
    sig_level: float, optional. Significance level.

    Returns:
    ci_list: list. Different CI corresponds to k0 in k0_list
    """
    treatment_reshaped = treatment.reshape(-1, 1)
    ci_list = []
    for k_candi in k0_list:
        groups_k_candi = group_node_k_hops(gamma_full, k_candi)
        x_k_candi = design_matrix_given_groups(gamma_full, groups_k_candi)

        # Add constant column to the design matrix for the intercept
        x_k_candi_treatment_indicator = np.concatenate((treatment_reshaped, x_k_candi), axis=1)

        # Fit the multiple linear regression model
        model = sm.OLS(y_outcomes, x_k_candi_treatment_indicator)
        results = model.fit()

        # Get the confidence interval for the intercept
        conf_interval = results.conf_int(alpha=sig_level)  
        tau_ci = conf_interval[0]
        print(tau_ci)
        ci_list.append(tau_ci)
    return ci_list


n = 1000  
p_er = 0.005  
sig_examp = 0.2
k0_examp = 3
iter_num = 1000

result_array = np.zeros((14, 1))
for iter_gen in range(iter_num):
    print("iter_num: ", iter_gen)

    er_graph = gen_graph(num_nodes=n, graph_type='er', p=p_er)
    Z = np.random.binomial(1, 0.2, n)
    treated_neighbor_count, total_neighbor_count = gamma_treated_neighbor(er_graph, Z, m=n)
    total_neighbor_count_inflate = total_neighbor_count.copy()
    total_neighbor_count_inflate = np.where(total_neighbor_count_inflate == 0, 0.1, total_neighbor_count_inflate)
    treated_proportion = treated_neighbor_count / total_neighbor_count_inflate
    gamma_matrix = treated_proportion.copy()
    gamma_matrix = gamma_matrix // 0.05
    max_values = np.where(gamma_matrix.max(axis=0) == 0, 0.1, gamma_matrix.max(axis=0))
    gamma_matrix = gamma_matrix / max_values

    Y_iter = gen_outcome(k0=k0_examp, gamma_full=gamma_matrix, gen_type='mod1', treatment=Z,
                         c0=10, sigma0=sig_examp, sigma1=sig_examp)
    cis_for_tau = infer_tau_same_k0_same_tau_same_sigma01(y_outcomes=Y_iter, gamma_full=gamma_matrix,
                                                          treatment=Z, k0_list=[0, 1, 2, 3, 4, 5, 6],
                                                          sig_level=0.05)
    R = np.array(cis_for_tau)
    R_reshaped = R.reshape(-1, 1)
    result_array = np.hstack((result_array, R_reshaped))

result_array = np.delete(result_array, 0, axis=1)


# Plot for each K v.s. k0, K=0,1,2,4,5,6, k0=3 
result_array = result_array[:, :100]
colors = ['gold', 'darkorange', 'red', 'black', 'blue', 'dodgerblue', 'cyan']

fig, axs = plt.subplots(3, 2, figsize=(12, 6))  

for j in range(2):  
    for i in range(3): 
        index = 3 * j + i 
        if j > 0:
            index = 3 * j + i + 1
        axs[i, j].fill_between(np.arange(100), result_array[2*index], result_array[2*index+1], color=colors[index], alpha=0.3, label=f'k = {index}')
        axs[i, j].fill_between(np.arange(100), result_array[6], result_array[7], color='black', alpha=0.3, label='k0 = 3')
        axs[i, j].set_ylabel('Inference of $\\tau$')
        axs[i, j].set_ylim(0, 1.2)
        axs[i, j].legend(loc='upper left')
        axs[i, j].axhline(y=0.6, color='dimgray', linestyle='--')

axs[-1, 0].set_xlabel('Number of iterations')
axs[-1, 1].set_xlabel('Number of iterations')
plt.show()



