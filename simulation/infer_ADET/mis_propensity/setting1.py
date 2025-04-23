from network_utils import *
from joblib import Parallel, delayed

np.random.seed(123)
design_rep = 100
infer_results = np.empty((2, design_rep*5, 6))
k0_true = 2
k_input_set = 5

for ii in range(design_rep):
    print(ii, flush=True)

    smallest_k_refine = 1
    while smallest_k_refine < 4:
        n = 1000  
        p_er = 0.02 
        graph_eg = gen_graph(num_nodes=n, graph_type='er', p=p_er)

        prop_score = np.random.uniform(0.03, 0.06, n)
        Z = np.random.binomial(1, prop_score) 

        tau_vec = np.random.uniform(0.6, 0.8, n)
        tau_true = np.sum(tau_vec * Z) / np.sum(Z)

        treated_neighbor_count, total_neighbor_count = gamma_treated_neighbor(graph_eg, Z, m=n)
        total_neighbor_count_inflate = total_neighbor_count.copy()
        total_neighbor_count_inflate = np.where(total_neighbor_count_inflate == 0, 0.1, total_neighbor_count_inflate)
        treated_proportion = treated_neighbor_count // 4
        gamma_matrix = treated_proportion.copy()
        max_values = np.where(gamma_matrix.max(axis=0) == 0, 0.1, gamma_matrix.max(axis=0))
        gamma_matrix = gamma_matrix / max_values

        smallest_k = find_smallest_k(gamma_matrix)
        smallest_k_refine = find_smallest_k_untreated(gamma_matrix, Z, smallest_k)

    for k_eg in range(k_input_set):
        print(f"input k = {k_eg}", flush=True)

        def process_single_iteration():

            Y = gen_outcome(k0=k0_true,
                            gamma_full=gamma_matrix,
                            treatment=Z,
                            tau=tau_vec,
                            coef_p=0.5,
                            c0=10,
                            sigma0=0.5,
                            sigma1=0.5,
                            f_type='linear')

            CIs_1 = infer_tau(Y,
                              gamma_matrix,
                              Z,
                              k_input=k_eg,
                              propen_vec=np.full((1000,), np.mean(Z)),
                              beta_group=np.array([None, None, None], dtype=object),
                              reweight=False, alpha=0.05, est_type='OLS')

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
                              propen_vec=np.full((1000,), np.mean(Z)),
                              beta_group=beta_grp,
                              reweight=False,
                              alpha=0.05,
                              est_type='SFL')

            return CIs_1, CIs_2

        results = Parallel(n_jobs=20)(delayed(process_single_iteration)() for _ in range(1000))
        results_1 = [res[0] for res in results]
        results_2 = [res[1] for res in results]

        results_array_1 = np.array(results_1)
        results_array_2 = np.array(results_2)

        infer_results[0, design_rep * k_eg + ii] = infer_smry(result=results_array_1,
                                                              tau_true_value=tau_true)
        infer_results[1, design_rep * k_eg + ii] = infer_smry(result=results_array_2,
                                                              tau_true_value=tau_true)
