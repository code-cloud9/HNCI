from network_utils import *

wave = 2
adj_matrix = np.loadtxt('glasgow/network_w3.txt', delimiter=',')
graph_eg = nx.from_numpy_array(adj_matrix)


Z123 = np.loadtxt('glasgow/romantic_w123.txt', delimiter=',')
Z = Z123[:, wave]
n = 160

treated_neighbor_count, total_neighbor_count = gamma_treated_neighbor(graph_eg, Z, m=n)
total_neighbor_count_inflate = total_neighbor_count.copy()
total_neighbor_count_inflate = np.where(total_neighbor_count_inflate == 0, 0.1, total_neighbor_count_inflate)
treated_proportion = treated_neighbor_count // 2 
gamma_matrix = treated_proportion.copy()
max_values = np.where(gamma_matrix.max(axis=0) == 0, 0.1, gamma_matrix.max(axis=0))
gamma_matrix = gamma_matrix / max_values

# choose one outcome of interest
alcohol123 = np.loadtxt('glasgow/alcohol_w123.txt', delimiter=',')
Y = alcohol123[:, wave]
tobacco123 = np.loadtxt('glasgow/tobacco_w123.txt', delimiter=',')
Y = tobacco123[:, wave]
cannabis123 = np.loadtxt('glasgow/cannabis_w123.txt', delimiter=',')
Y = cannabis123[:, wave]

# BIC choose lambda
lambda_candidates = [0, 0.1, 1, 5, 10, 20, 50, 100, 200, 1000]
bic_res = calculate_lambda_alg2(lambda_values=lambda_candidates, y_control_sample=Y[Z == 0],
                                gamma_control=gamma_matrix[Z == 0], candidate_list=[0, 1, 2])
bic_indices = [i for i, x in enumerate(bic_res) if x == min(bic_res)]
lambda_choose = [lambda_candidates[i] for i in bic_indices]

# infer k0
cs = confident_set_k(num_repo=100, y_outcomes=Y, gamma_full=gamma_matrix, treatment=Z,
                     candidates=[0, 1, 2], lambda_tune=5, alpha=0.1, m=None)

k_eg = 2

# OLS
CIs_1 = infer_tau(Y,
                  gamma_matrix,
                  Z,
                  k_input=k_eg,
                  propen_vec=np.full((n,), np.mean(Z)),
                  beta_group=np.array([None, None, None], dtype=object),
                  reweight=False, alpha=0.05, est_type='OLS')
# SFL
beta_grp = sfl_grp(num_nodes=n,
                        y_outcomes=Y,
                        gamma_full=gamma_matrix,
                        treatment=Z,
                        k_input=k_eg,
                        scale_tune=1/30,
                        smoothing_tune=0.05)
CIs_2 = infer_tau(Y,
                  gamma_matrix,
                  Z,
                  k_input=k_eg,
                  propen_vec=np.full((n,), np.mean(Z)),
                  beta_group=beta_grp,
                  reweight=False,
                  alpha=0.05,
                  est_type='SFL')

