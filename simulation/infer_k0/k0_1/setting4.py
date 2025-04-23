from network_utils import *
from joblib import Parallel, delayed

n = 1000 
p_er = 0.02
iteration_time = 500

neigh_set_results = np.empty((1, iteration_time, 3),dtype = object)
for i in range(1):
    for j in range(iteration_time):
        for k in range(3):
            neigh_set_results[i,j,k] = []

def process_each_iteration():

    smallest_k_refine = 0
    while smallest_k_refine < 4:
        graph_eg = gen_graph(num_nodes=n, graph_type='sbm')

        prop_score = np.random.uniform(0.03, 0.06, n)
        Z = np.random.binomial(1, prop_score) 

        tau_vec = np.random.uniform(0.6, 0.8, n)
        tau_true = np.sum(tau_vec * Z) / np.sum(Z) 

        treated_neighbor_count, total_neighbor_count = gamma_treated_neighbor(graph_eg, Z, m=n)
        total_neighbor_count_inflate = total_neighbor_count.copy()
        total_neighbor_count_inflate = np.where(total_neighbor_count_inflate == 0, 0.1, total_neighbor_count_inflate)
        treated_proportion = treated_neighbor_count / total_neighbor_count_inflate
        gamma_matrix = treated_proportion.copy()
        gamma_matrix = gamma_matrix // 0.05
        max_values = np.where(gamma_matrix.max(axis=0) == 0, 0.1, gamma_matrix.max(axis=0))
        gamma_matrix = gamma_matrix / max_values

        smallest_k = find_smallest_k(gamma_matrix)
        smallest_k_refine = find_smallest_k_untreated(gamma_matrix, Z, smallest_k)

    k0_interval_collection = []
    for k0_true in [1]:

        Y = gen_outcome(k0=k0_true,
                        gamma_full=gamma_matrix,
                        treatment=Z,
                        tau=tau_vec,
                        coef_p=0.5,
                        c0=10,
                        sigma0=0.5,
                        sigma1=0.5,
                        f_type='linear')

        lambdas_generated = [0.001, 0.005, 0.01, 0.005, 0.1, 1, 5, 10, 20, 30, 40, 50, 100, 1000]
        # Alg1 - infer k0 - with BIC
        candi_set_bic = candidate_set_k_bic(num_repo_b=200, y_outcomes=Y, gamma_full=gamma_matrix,
                                            treatment=Z, lambda_list=lambdas_generated, m=smallest_k_refine)
        k0_interval_collection.append(candi_set_bic)
        # Alg2 - BIC choose lambda
        lambda_candidates = [0.0001,0.001,0.01, 0.1, 1, 5, 10, 20, 50, 100, 200, 1000]
        bic_res = calculate_lambda_alg2(lambda_values=lambda_candidates, y_control_sample=Y[Z == 0],
                                        gamma_control=gamma_matrix[Z == 0], candidate_list=np.arange(smallest_k_refine+1))

        bic_indices = [i for i, x in enumerate(bic_res) if x == min(bic_res)]
        lambda_choose = [lambda_candidates[i] for i in bic_indices]

        cs_without_filtration = confident_set_k(num_repo=100, y_outcomes=Y, gamma_full=gamma_matrix, treatment=Z,
                             candidates=np.arange(smallest_k_refine+1), lambda_tune=np.mean(lambda_choose), alpha=0.05, m=None)

        k0_interval_collection.append(cs_without_filtration)
        # Alg2 - infer k0 - using lambda chosen by BIC
        bic_res_12 = calculate_lambda_alg2(lambda_values=lambda_candidates, y_control_sample=Y[Z == 0],
                                        gamma_control=gamma_matrix[Z == 0],
                                        candidate_list=candi_set_bic)
        bic_indices_12 = [i for i, x in enumerate(bic_res_12) if x == min(bic_res_12)]
        lambda_choose_12 = [lambda_candidates[i] for i in bic_indices_12]
        lambda_choose_12 = [lambda_candidates[i] for i in bic_indices_12]
        cs = confident_set_k(num_repo=100, y_outcomes=Y, gamma_full=gamma_matrix, treatment=Z,
                             candidates=candi_set_bic, lambda_tune=np.mean(lambda_choose_12), alpha=0.05, m=None)
        k0_interval_collection.append(cs)

    return k0_interval_collection

results = Parallel(n_jobs=20)(delayed(process_each_iteration)() for _ in range(iteration_time))

num_algs = 3
matrices = [[[row[i:i+num_algs] for row in results] for i in range(0, len(results[0]), num_algs)]]
data = matrices[0]

Alg1_length = 0
Alg1_coverage_prob = 0
Alg2_length = 0
Alg2_coverage_prob = 0
Alg12_length = 0
Alg12_coverage_prob = 0

iteration_time = len(data[0])

k0_true = 1

for i in range(0,iteration_time):

    Alg1_length = Alg1_length + len(data[0][i][0])
    if k0_true in data[0][i][0]:
        Alg1_coverage_prob = Alg1_coverage_prob + 1

    Alg2_length = Alg2_length + len(data[0][i][1])
    if k0_true in data[0][i][1]:
        Alg2_coverage_prob = Alg2_coverage_prob + 1

    Alg12_length = Alg12_length + len(data[0][i][2])
    if k0_true in data[0][i][2]:
        Alg12_coverage_prob = Alg12_coverage_prob + 1


print("average length of Alg1 is:", Alg1_length/ iteration_time)
print("coverage probability of Alg1 is", Alg1_coverage_prob / iteration_time)

print("average length of Alg2 is:", Alg2_length/ iteration_time)
print("coverage probability of Alg2 is", Alg2_coverage_prob / iteration_time)

print("average length of Alg12 is :", Alg12_length / iteration_time)
print("coverage probability of Alg12 is:", Alg12_coverage_prob / iteration_time)
