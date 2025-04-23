import networkx as nx
import numpy as np
import cvxpy as cp
from scipy.stats import norm

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

    elif graph_type == 'sbm':
        x = np.random.uniform(size=num_nodes)
        l = np.floor(x * 6).astype(int)
        P = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            retval = (l + 1) / 40
            retval[l != l[i]] = 0.3 / 40
            P[i, :] = retval
        A = np.random.rand(num_nodes, num_nodes) < P
        A = np.triu(A, 1)
        A = A + A.T
        graph = nx.from_numpy_array(A)

    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    return graph



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

    total_neighbor_count = np.zeros((num_nodes, len(value_range) + 1), dtype=int)

    for i in range(num_nodes):
        counts = np.histogram(distance_matrix_array[i, ~np.isinf(distance_matrix_array[i])],
                              bins=list(value_range) + [np.inf])[0]
        total_neighbor_count[i, :] = np.insert(counts, 0, 0)

    treated_neighbor_count = np.zeros((num_nodes, len(value_range) + 1), dtype=int)

    distance_matrix_array_mask = distance_matrix_array.copy()
    distance_matrix_array_mask[:, treatment == 0] = 0
    for i in range(num_nodes):
        counts = np.histogram(distance_matrix_array_mask[i, ~np.isinf(distance_matrix_array_mask[i])],
                              bins=list(value_range) + [np.inf])[0]
        treated_neighbor_count[i, :] = np.insert(counts, 0, 0)

    return treated_neighbor_count, total_neighbor_count



def find_smallest_k(gamma_full):
    """
    Finds the smallest i such that len(np.unique(gamma_matrix[:, 0:i], axis=0))
    equals len(np.unique(gamma_matrix, axis=0)).

    Parameters:
    - gamma_full: A n*n NumPy ndarray.

    Returns:
    - k: The smallest k satisfying the condition. If no such k is found, returns the number of columns in gamma_matrix.
    """

    total_unique_rows = len(np.unique(gamma_full, axis=0))

    for col_i in range(gamma_full.shape[1] + 1):
        unique_rows_i = len(np.unique(gamma_full[:, :col_i], axis=0))
        if unique_rows_i == total_unique_rows:
            return col_i - 1
    return gamma_full.shape[1] - 1



def find_smallest_k_untreated(gamma_full, treatment, k_upper_b):
    """
    Finds the smallest i such that len(np.unique(gamma_matrix[:, 0:i], axis=0))
    equals len(np.unique(gamma_matrix, axis=0)).

    Parameters:
    - gamma_full: A n*n NumPy ndarray.
    - treatment: ndarray, shape (n,). Indicates the treatment status of each node.
    - k_upper_b: upper bound of k based on smallest_k

    Returns:
    - k: The smallest k satisfying the condition. If no such k is found, returns the number of columns in gamma_matrix.
    """

    # untreated nodes should cover all features
    gamma_full_control = gamma_full[treatment == 0]

    for col_i in range(k_upper_b + 1):
        len_1 = len(np.unique(gamma_full[:, :(k_upper_b+1-col_i)], axis=0))
        len_2 = len(np.unique(gamma_full_control[:, :(k_upper_b+1-col_i)], axis=0))
        if len_1 == len_2:
            return k_upper_b-col_i

    return "untreated nodes can not cover the features"


def gen_outcome(k0, gamma_full, treatment, tau, coef_p=0.5, c0=1, sigma0=1, sigma1=1, f_type='linear'):
    """
    Generate potential outcomes based on treatment, and ture k0.

    Parameters:
    - k0: ture #-hop neighbors.
    - gamma_full: ndarray, n*n.
    - treatment: ndarray, n*1.
    - tau: float or vector, treatment effect.
    - coef_p: float, better to set in (0, 1) for interpretation.
    - sigma0: float, standard deviation of the noise for control group.
    - sigma1: float, standard deviation of the noise for treatment group.

    Returns:
    - outcome: ndarray, generated outcomes with shape (n_samples,).
    """

    if f_type == 'linear':
        outcome = treatment * tau + c0 * sum([coef_p ** i * gamma_full[:, i] for i in range(k0 + 1)]) + \
                  np.random.normal(0, treatment * (sigma1 - sigma0) + sigma0, len(treatment))

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

    group_matrix = np.zeros((len(gamma_mat), len(groups.keys())))

    for group_index, (group_key, row_indices) in enumerate(groups.items()):
        for row_index in row_indices:
            group_matrix[row_index, group_index] = 1

    return group_matrix


def gen_outcome_mismatch(k0, gamma_full, treatment, tau, coef_p=0.5, c0=10, sigma0=1, sigma1=1, f_type='linear'):
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
    groups_full_k = group_node_k_hops(gamma_full, k=k0)
    x_full_k_raw = design_matrix_given_groups(gamma_full, groups_full_k)
    mismatch_sigma = np.random.uniform(0, 0.5, size=(len(groups_full_k.keys()), 1))
    mismatch_sigma_all = x_full_k_raw @ mismatch_sigma

    if f_type == 'linear':
        outcome = treatment * tau + c0 * sum([coef_p ** i * gamma_full[:, i] for i in range(k0 + 1)]) + \
                  np.random.normal(0, treatment * (sigma1 - sigma0) + sigma0, len(treatment)) + \
                  np.random.normal(0, mismatch_sigma_all.flatten())

    return outcome



def infer_tau(y_outcomes, gamma_full, treatment, k_input, propen_vec, beta_group, reweight=False, alpha=0.05, est_type='SFL'):
    """
    Construct the confidence interval of tau.

    Parameters:
    y_outcomes: ndarray, shape (n,). The outcomes associated with each node.
    gamma_full: ndarray, shape (n, n). The adjacency matrix representing the network.
    treatment: ndarray, shape (n,). Indicates the treatment status of each node.
    propen_vec: ndarray, shape (n,). propensity vector for all nodes (propen_vec=np.full((n,), 0.1))
    k_input: float, input neighborhood size
    reweight: for binary setting
    alpha: significance level
    beta_group: ndarray of arrays, shape (|hat Grp|,).
                e.g. beta_group = np.array([np.r_[0, 3, 4], np.r_[1, 2, 5], np.r_[6:19]], dtype=object)

    Returns:
    confidence interval of tau
    """

    groups_full_k = group_node_k_hops(gamma_full, k=k_input)
    x_full_k_raw = design_matrix_given_groups(gamma_full, groups_full_k)

    if est_type == 'OLS':
        x_full_k = x_full_k_raw
    elif est_type == 'SFL':
        flat_grp = np.zeros((x_full_k_raw.shape[1], len(beta_group)))
        for i, indices in enumerate(beta_group):
            flat_grp[indices, i] = 1
        x_full_k = x_full_k_raw @ flat_grp

    x_control_k = x_full_k[treatment == 0]

    y_control_outcomes = y_outcomes[treatment == 0]
    beta_ols = np.linalg.inv(x_control_k.T @ x_control_k) @ x_control_k.T @ y_control_outcomes
    f_full_ols = x_full_k @ beta_ols
    residuals = y_control_outcomes - x_control_k @ beta_ols

    if reweight:
        for j in range(10):
            x_control_k = x_control_k / (residuals.reshape(-1, 1) + 1e-4)
            y_control_outcomes = y_control_outcomes / (residuals + 1e-4)
            beta_ols = np.linalg.inv(x_control_k.T @ x_control_k) @ x_control_k.T @ y_control_outcomes
            f_full_ols = x_full_k @ beta_ols
            residuals = y_control_outcomes - x_control_k @ beta_ols

    sigma2 = (residuals.T @ residuals) / (x_control_k.shape[0] - x_control_k.shape[1])

    tau_or = np.mean(y_outcomes[treatment == 1] - f_full_ols[treatment == 1])
    tau_dr = tau_or + np.sum((y_outcomes[treatment == 0] - f_full_ols[treatment == 0]) *
             propen_vec[treatment == 0] / (1 - propen_vec[treatment == 0])) / np.sum(treatment)

    v_vec = np.sum(x_full_k[treatment == 1], axis=0) / np.sum(treatment)
    w_or = np.sqrt((v_vec.T @ np.linalg.inv(x_control_k.T @ x_control_k) @ v_vec + 1 / np.sum(treatment)) * sigma2)

    lower_bound_or = tau_or - norm.ppf(1 - alpha / 2) * w_or
    upper_bound_or = tau_or + norm.ppf(1 - alpha / 2) * w_or

    one_over_q_vec = 1 / (1 - propen_vec)
    x_temp_with_propen = x_full_k * propen_vec[:, np.newaxis] * one_over_q_vec[:, np.newaxis]
    u_vec = v_vec - np.sum(x_temp_with_propen[treatment == 0], axis=0) / np.sum(treatment)
    w_dr = np.sqrt((u_vec.T @ np.linalg.inv(x_control_k.T @ x_control_k) @ u_vec + 1 / np.sum(treatment) +
                    np.sum((propen_vec[treatment == 0] / (1 - propen_vec[treatment == 0]))**2) / np.sum(treatment)**2) * sigma2)

    lower_bound_dr = tau_dr - norm.ppf(1 - alpha / 2) * w_dr
    upper_bound_dr = tau_dr + norm.ppf(1 - alpha / 2) * w_dr

    return lower_bound_or, upper_bound_or, lower_bound_dr, upper_bound_dr


def infer_smry(result, tau_true_value):

    # OR
    lower_bounds_OR, upper_bounds_OR = result[:, 0], result[:, 1]
    coverage_prob_OR = np.sum((lower_bounds_OR <= tau_true_value) & (upper_bounds_OR >= tau_true_value)) / len(result)
    mean_width_OR = np.mean(upper_bounds_OR - lower_bounds_OR)
    sd_width_OR = np.std(upper_bounds_OR - lower_bounds_OR)

    # DR
    lower_bounds_DR, upper_bounds_DR = result[:, 2], result[:, 3]
    coverage_prob_DR = np.sum((lower_bounds_DR <= tau_true_value) & (upper_bounds_DR >= tau_true_value)) / len(result)
    mean_width_DR = np.mean(upper_bounds_DR - lower_bounds_DR)
    sd_width_DR = np.std(upper_bounds_DR - lower_bounds_DR)

    return coverage_prob_OR, coverage_prob_DR, mean_width_OR, mean_width_DR, sd_width_OR, sd_width_DR



def construct_T(beta):

    n = len(beta)
    pairs = np.transpose(np.triu_indices(n, k=1))
    num_pairs = len(pairs)

    T = np.zeros((num_pairs, n))

    rows = np.arange(num_pairs)
    T[rows, pairs[:, 0]] = 1
    T[rows, pairs[:, 1]] = -1

    return T, num_pairs


def compute_beta(X, y_obs, b, n0, T, nu, rho, p, c):


    norm_y_obs_Xb = np.linalg.norm(y_obs - np.dot(X, b), 2)

    factor = 1 / (np.sqrt(2 * n0) * norm_y_obs_Xb)

    # Compute the matrix to be inverted
    A = factor * np.dot(X.T, X) + rho * np.dot(T.T, T) + np.diag([1e-4] * X.shape[1])

    # Compute the vector part of the equation
    vec_part = c - np.dot(T.T, nu) + factor * np.dot(X.T, y_obs) + rho * np.dot(T.T, p)
    beta = np.linalg.solve(A, vec_part)
    return beta


def soft_thresholding(x, lam):

    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def compute_c_k(beta_k, lambda_1, lambda_2):

    beta_dim = len(beta_k)
    c_k = np.zeros(beta_dim)
    beta_reshaped = beta_k.reshape(beta_dim, )
    for i in range(beta_dim):
        sum_term = 0
        for j in range(beta_dim):
            if j != i:
                if beta_reshaped[i] - beta_reshaped[j] > lambda_2:
                    sum_term += 1
                elif beta_reshaped[i] - beta_reshaped[j] < -lambda_2:
                    sum_term -= 1
        c_k[i] = lambda_1 * sum_term

    return c_k


def update_beta(X, y_obs, n0, T, nu, rho, p, c, beta_k):
    """
    beta_init = beta_k
    beta_solution, info, ier, msg = fsolve(lambda beta: nonlinear_equation(beta.reshape(-1, 1), X, y_obs, c, T, nu, rho, p, n0).flatten(), beta_init.flatten(), full_output=True)

    return beta_solution.reshape(-1, 1)
    """

    b = beta_k
    maxiter_sub = 100
    tolstop = 1e-3

    for i in range(0, maxiter_sub):

        beta_subloop = compute_beta(X, y_obs, b, n0, T, nu, rho, p, c)

        numerator = np.linalg.norm(beta_subloop - b, 2)
        denominator = np.linalg.norm(b, 2)
        tol = numerator / denominator

        if tol < tolstop:
            break
        b = beta_subloop

    return b


def update_p(T, beta, nu_k, rho, lambda_1):

    argument = np.dot(T, beta) + (1 / rho) * nu_k
    threshold = lambda_1 / rho
    p_k_plus_1 = soft_thresholding(argument, threshold)
    return p_k_plus_1


def update_nu(nu_k, T, beta_k_plus_1, p_k_plus_1, rho):

    nu_k_plus_1 = nu_k + rho * (np.dot(T, beta_k_plus_1) - p_k_plus_1)
    return nu_k_plus_1


def nonlinear_equation(beta, X, y_obs, c, T, nu, rho, p, n0):

    term1 = (1 / np.sqrt(2 * n0 * np.linalg.norm(y_obs - X @ beta, 2))) * (X.T @ (X @ beta - y_obs))
    term2 = c
    term3 = T.T @ nu
    term4 = rho * (T.T @ (T @ beta - p))
    return term1 - term2 + term3 + term4


def smooth_beta_vector(beta, eps):

    smoothed_beta = beta.copy()

    for i in range(len(beta)):
        close_indices = np.where(np.square(beta - beta[i]) <= eps)[0]
        mean_value = np.mean(beta[close_indices])
        smoothed_beta[i] = mean_value
    return smoothed_beta



def sfl_grp(num_nodes, y_outcomes, gamma_full, treatment, k_input, scale_tune=1/30, smoothing_tune=0.05):

    lambda_1_eg = scale_tune / np.sqrt(num_nodes)
    lambda_2_eg = scale_tune / np.sqrt(num_nodes)
    gamma_eg = 1.01  

    y_obs_eg = y_outcomes[treatment == 0]
    n0_eg = len(y_obs_eg)
    y_obs_eg = y_obs_eg.reshape(len(y_obs_eg), 1)

    groups_full_k_eg = group_node_k_hops(gamma_full, k=k_input)
    x_full_k_eg = design_matrix_given_groups(gamma_full, groups_full_k_eg)
    x_design = x_full_k_eg[treatment == 0]


    beta_eg = np.linalg.inv(np.dot(x_design.T, x_design)).dot(np.dot(x_design.T, y_obs_eg))
    t_eg, dim_nu_p_eg = construct_T(beta_eg)
    p_eg = np.ones((dim_nu_p_eg, 1))
    nu_eg = np.ones((dim_nu_p_eg, 1))

    beta_k = beta_eg
    b = beta_k
    nu_k = nu_eg
    p_k = p_eg

    tolstop = 1e-8

    for i in range(0, 1000):
        c_k = compute_c_k(beta_k, lambda_1_eg, lambda_2_eg)
        c_k = c_k.reshape((len(c_k), 1))

        rho_eg = scale_tune / np.sqrt(num_nodes)

        for j in range(0, 10):
            beta_k = update_beta(x_design, y_obs_eg, n0_eg, t_eg, nu_k, rho_eg, p_k, c_k, beta_k)
            p_k = update_p(t_eg, beta_k, nu_k, rho_eg, lambda_1_eg)
            nu_k = update_nu(nu_k, t_eg, beta_k, p_k, rho_eg)
            rho_eg *= gamma_eg

        numerator = np.linalg.norm(beta_k - b, 2)
        denominator = np.linalg.norm(b, 2)
        tol = numerator / denominator

        if tol < tolstop:
            break
        b = beta_k

    beta_smooth = beta_k.reshape(-1)
    smoothing_threshold_ini = smoothing_tune
    beta_smooth_old = beta_smooth

    for i in range(10):
        beta_smooth = smooth_beta_vector(beta_smooth, smoothing_threshold_ini)
        smoothing_threshold_ini = 2 * np.mean(np.square(beta_smooth - beta_smooth_old))

    unique_values = np.unique(beta_smooth)
    beta_group_eg = np.empty(len(unique_values), dtype=object)
    for i, value in enumerate(unique_values):
        beta_group_eg[i] = np.where(beta_smooth == value)[0]

    return beta_group_eg



def optim_given_ub_lambda(y_control, gamma_control, maximum_k, lambda_b, u_star):

    optimized_value_list = [np.inf] * (maximum_k + 1)

    for k_prime in range(maximum_k + 1):
        groups_k_prime = group_node_k_hops(gamma_control, k_prime)
        x_k_prime = design_matrix_given_groups(gamma_control, groups_k_prime)

        x_k_prime_add_u_front = np.hstack((u_star, x_k_prime))

        beta_tune = cp.Variable(len(groups_k_prime) + 1)
        # Define the constraints with a non-strict inequality
        epsilon = 1e-6 
        constraints = [beta_tune[0] >= epsilon]
        k_penalty = lambda_b * k_prime

        # Define the objective function
        residual_k_prime = y_control - x_k_prime_add_u_front @ beta_tune
        l2_norm_squared = cp.sum_squares(residual_k_prime)
        objective = cp.Minimize(l2_norm_squared + k_penalty)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        optimized_value_list[k_prime] = problem.value

    opt_k_prime = optimized_value_list.index(min(optimized_value_list))
    return opt_k_prime



def calculate_lambda_alg1(lambda_values, y_control, gamma_control, maximum_k, u_star):
    """
    Constuct the confidence set of k.

    Parameters:
    lambda_values: list. A list of potential lambda values for lam.k in algorithm 1
    maximum_k: upper bound for the number of hops.

    Returns:
    bic_list: list. a list of lambda values chosen from u_star
    """

    bic_list = []
    n0 = len(y_control)

    for lambda_val in lambda_values:
        optimized_value_list = [np.inf] * (maximum_k + 1)
        optimized_value_list_rss_for_bic = [np.inf] * (maximum_k + 1)
        for k_prime in range(maximum_k+1):
            groups_k_prime = group_node_k_hops(gamma_control, k_prime)
            x_k_prime = design_matrix_given_groups(gamma_control, groups_k_prime)

            x_k_prime_add_u_front = np.hstack((u_star, x_k_prime))

            beta_tune = cp.Variable(len(groups_k_prime) + 1)
            epsilon = 1e-6
            constraints = [beta_tune[0] >= epsilon]
            k_penalty = lambda_val * k_prime

            residual_k_prime = y_control - x_k_prime_add_u_front @ beta_tune
            l2_norm_squared = cp.sum_squares(residual_k_prime)
            objective = cp.Minimize(l2_norm_squared + k_penalty)
            problem = cp.Problem(objective, constraints)
            problem.solve()
            optimized_value_list[k_prime] = problem.value
            estimated_beta_values = beta_tune.value[1:]
            residuals = y_control - x_k_prime @ estimated_beta_values
            optimized_value_list_rss_for_bic[k_prime] = np.sum(residuals**2)

        opt_k_prime = optimized_value_list.index(min(optimized_value_list))
        groups_opt_k_prime = group_node_k_hops(gamma_control, opt_k_prime)

        n_samples = n0
        n_features = len(groups_opt_k_prime)
        rss = optimized_value_list_rss_for_bic[opt_k_prime]
        bic = n_samples * np.log(rss / n_samples) + n_features * np.log(n_samples)

        bic_list.append(bic)

    min_bic_indices = [i for i, x in enumerate(bic_list) if x == min(bic_list)]
    lambda_by_ustar = [lambda_values[i] for i in min_bic_indices]
    return lambda_by_ustar


def candidate_set_k_bic(num_repo_b, y_outcomes, gamma_full, treatment, lambda_list, m=None):
    """
    Construct the candidate set of k.

    Parameters:
    num_repo_b: int. The number of repetitions or iterations for the algorithm.
    y_outcomes: ndarray, shape (n,). The outcomes associated with each node.
    gamma_full: ndarray, shape (n, n). The adjacency matrix representing the network.
    treatment: ndarray, shape (n,). Indicates the treatment status of each node.
    lambda_list: list. float, optional. Regularization parameter for optimization.
    m: The maximum number of hops to consider for neighborhood counts, set as "smallest_k" in practice.

    Returns:
    candidate_set for k0
    """

    candidate_set = set()
    y_control_outcomes = y_outcomes[treatment == 0]
    gamma_full_control = gamma_full[treatment == 0]
    n0 = len(y_control_outcomes)

    for iter_num in range(num_repo_b):
        u_star_iter = np.random.normal(0, 1, (n0, 1))
        lambda_list_ustar = calculate_lambda_alg1(lambda_values=lambda_list, y_control=y_control_outcomes,
                                                  gamma_control=gamma_full_control, maximum_k=m,
                                                  u_star=u_star_iter)
        for lambda_candi in lambda_list_ustar:
            opt_k_iter = optim_given_ub_lambda(y_control=y_control_outcomes, gamma_control=gamma_full_control,
                                               maximum_k=m, lambda_b=lambda_candi, u_star=u_star_iter)
            candidate_set.add(opt_k_iter)

    return list(candidate_set)



def calculate_lambda_alg2(lambda_values, y_control_sample, gamma_control, candidate_list):
    """
    Constuct the confidence set of k.

    Parameters:
    lambda_values: list. A list of potential lambda values for lam.k in algorithm 2
    candidate_list: list. A list of candidate values for the number of hops.

    Returns:
    bic_list: list. optim bic value for each lambda value. Same dimension as len(lambda_values)
    """

    bic_list = []
    n0 = len(y_control_sample)

    for lambda_val in lambda_values:
        optimized_value_list = [np.inf] * (max(candidate_list) + 1)
        for k_prime in candidate_list:
            groups_k_prime = group_node_k_hops(gamma_control, k_prime)
            x_k_prime = design_matrix_given_groups(gamma_control, groups_k_prime)
            beta_tune = cp.Variable(len(groups_k_prime))
            k_penalty = lambda_val * k_prime
            residual_k_prime = y_control_sample - x_k_prime @ beta_tune
            l2_norm_squared = cp.sum_squares(residual_k_prime)
            objective = cp.Minimize(l2_norm_squared + k_penalty)
            problem = cp.Problem(objective)
            problem.solve()
            optimized_value_list[k_prime] = problem.value

        opt_k_prime = optimized_value_list.index(min(optimized_value_list))
        groups_opt_k_prime = group_node_k_hops(gamma_control, opt_k_prime)
        opt_rss_add_penalty = min(optimized_value_list)

        n_samples = n0
        n_features = len(groups_opt_k_prime) 
        rss = opt_rss_add_penalty - lambda_val * opt_k_prime 
        bic = n_samples * np.log(rss / n_samples) + n_features * np.log(n_samples) 

        bic_list.append(bic)

    return bic_list



def optim_alg2_penal_k(y_control_sample, gamma_control, candidate_list, n_control, lambda_conf=1):
    """
    Optimization criteria in algorithm 2.

    Parameters:
    - y_control_sample: ndarray, shape (n_control,). The control sample.
    - gamma_control: ndarray, shape (n_control, n_control). Adjacency matrix for the control group.
    - k: int. The number of candidate hops to consider.
    - n_control: int. The number of control nodes.
    - lambda_conf: float, optional. Regularization parameter.

    Returns:
    - int. Index of the minimum optimized value.
    """
    optimized_value_list = [np.inf] * (max(candidate_list) + 1)

    for k_prime in candidate_list:
        groups_k_prime = group_node_k_hops(gamma_control, k_prime)
        x_k_prime = design_matrix_given_groups(gamma_control, groups_k_prime)

        beta_tune = cp.Variable(len(groups_k_prime))
        k_penalty = lambda_conf * len(groups_k_prime)

        # Define the objective function
        residual_k_prime = y_control_sample - x_k_prime @ beta_tune
        l2_norm_squared = cp.sum_squares(residual_k_prime)
        objective = cp.Minimize(l2_norm_squared + k_penalty)
        problem = cp.Problem(objective)
        problem.solve()
        optimized_value_list[k_prime] = problem.value

    opt_k_prime = optimized_value_list.index(min(optimized_value_list))
    return opt_k_prime


def confident_set_k(num_repo, y_outcomes, gamma_full, treatment, candidates, lambda_tune=1.0, alpha=0.05, m=None):
    """
    Constuct the confidence set of k.

    Parameters:
    num_repo: int. The number of repetitions or iterations for the algorithm.
    y_outcomes: ndarray, shape (n,). The outcomes associated with each node.
    gamma_full: ndarray, shape (n, n). The adjacency matrix representing the network.
    treatment: ndarray, shape (n,). Indicates the treatment status of each node.
    candidates: list. A list of candidate values for the number of hops.
    lambda_tune: float, optional. Regularization parameter for optimization.
    alpha: float, optional. Significance level.
    m: The maximum number of hops to consider for neighborhood counts, set as "smallest_k" in practice.

    Returns: confidence set for k0
    """

    confidence_set = []
    for k_candidate in candidates:

        y_control_outcomes = y_outcomes[treatment == 0]
        gamma_full_control = gamma_full[treatment == 0]
        n0 = len(y_control_outcomes)
        opt_k_prob_distribution = {}

        for iter_num in range(num_repo):
            groups_k_candidate = group_node_k_hops(gamma_full_control, k_candidate)
            x_k_candidate = design_matrix_given_groups(gamma_full_control, groups_k_candidate)
            h_k_candidate = x_k_candidate @ np.linalg.inv(x_k_candidate.T @ x_k_candidate) @ x_k_candidate.T
            a_obs = h_k_candidate @ y_control_outcomes
            b_obs = np.linalg.norm((np.eye(n0) - h_k_candidate) @ y_control_outcomes, ord=None)
            u_star_iter = np.random.normal(0, 1, n0)
            y_star_iter = a_obs + b_obs * ((np.eye(n0) - h_k_candidate) @ u_star_iter) / \
                          np.linalg.norm((np.eye(n0) - h_k_candidate) @ u_star_iter, ord=None)

            opt_k_iter = optim_alg2_penal_k(y_control_sample=y_star_iter, gamma_control=gamma_full_control,
                                            candidate_list=candidates, n_control=n0, lambda_conf=lambda_tune)

            if opt_k_iter in opt_k_prob_distribution:
                opt_k_prob_distribution[opt_k_iter] = opt_k_prob_distribution[opt_k_iter] + 1
            else:
                opt_k_prob_distribution[opt_k_iter] = 1

        opt_k_obs = optim_alg2_penal_k(y_control_sample=y_control_outcomes, gamma_control=gamma_full_control,
                                       candidate_list=candidates, n_control=n0, lambda_conf=lambda_tune)

        if opt_k_obs in opt_k_prob_distribution:
            th_value = opt_k_prob_distribution[opt_k_obs]
            f_hat = sum(value for key, value in opt_k_prob_distribution.items() if value <= th_value) / num_repo
            if f_hat > alpha:
                confidence_set.append(k_candidate)

    return confidence_set



