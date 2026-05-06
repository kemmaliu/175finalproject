import numpy as np
from learning import (
    K, ALPHA, Q_SBM, THETA_COM,
    get_kappa, sample_network, network_equilibrium, l2_distance,
    extract_features_level2, extract_features_level3, extract_features_level4,
    plugin_estimator, train_nn, predict_nn, evaluate_per_network,
)

S_GRAPHON = np.maximum(np.linalg.solve(np.eye(K) - ALPHA * (Q_SBM / K), THETA_COM), 0)


def extract_features_level3_norm(A, community, theta):
    # lvl 3 norm
    N = A.shape[0]
    kappa = get_kappa(N)
    expected_deg = kappa * N * Q_SBM.mean()
    degrees = A.sum(axis=1)
    features = np.zeros((N, 7))

    for i in range(N):
        neighbors = np.where(A[i] > 0)[0]
        deg_i = len(neighbors)

        features[i, 0] = theta[i]
        features[i, 1] = deg_i / expected_deg

        if deg_i > 0:
            nd = degrees[neighbors]
            features[i, 2] = nd.mean() / expected_deg
            features[i, 3] = (nd.std() / expected_deg) if deg_i > 1 else 0
            features[i, 6] = nd.max() / expected_deg
            features[i, 5] = np.mean(theta[neighbors] == theta[i])

            if deg_i >= 2:
                subgraph = A[np.ix_(neighbors, neighbors)]
                features[i, 4] = np.sum(subgraph) / (deg_i * (deg_i - 1))

    return features


def exploitability(A, s_pred, theta, kappa):
    N = A.shape[0]
    z = (A @ s_pred) / (kappa * N)
    s_br = np.maximum(ALPHA * z + theta, 0)
    return np.sqrt(np.mean((s_pred - s_br) ** 2))


def _build_dataset(N, n_networks, rng):
    feats_l2, feats_l3, feats_l3n, feats_l4 = [], [], [], []
    targets, naive, oracle, plugin = [], [], [], []

    for _ in range(n_networks):
        A, community, theta, kappa = sample_network(N, rng)
        s_true = network_equilibrium(A, theta, kappa)
        targets.append(s_true)

        feats_l2.append(extract_features_level2(A, community, theta))
        feats_l3.append(extract_features_level3(A, community, theta))
        feats_l3n.append(extract_features_level3_norm(A, community, theta))
        feats_l4.append(extract_features_level4(A, community, theta))

        naive.append(theta.copy())
        oracle.append(S_GRAPHON[community])
        plugin.append(plugin_estimator(A, community, theta, kappa))

    return {
        'feats_l2': feats_l2, 'feats_l3': feats_l3,
        'feats_l3n': feats_l3n, 'feats_l4': feats_l4,
        'targets': targets,
        'naive': naive, 'oracle': oracle, 'plugin': plugin,
    }


def run_cross_N_experiment(N_train=300, n_train=80, n_test=30,
                            N_test_list=(200, 300, 600, 1000), seed=42):
    rng = np.random.default_rng(seed)
    train = _build_dataset(N_train, n_train, rng)

    y = np.concatenate(train['targets'])
    X_l2 = np.vstack(train['feats_l2'])
    X_l3 = np.vstack(train['feats_l3'])
    X_l3n = np.vstack(train['feats_l3n'])
    X_l4 = np.vstack(train['feats_l4'])

    mlp_l2, sc_l2 = train_nn(X_l2, y)
    mlp_l3, sc_l3 = train_nn(X_l3, y)
    mlp_l3n, sc_l3n = train_nn(X_l3n, y)
    mlp_l4, sc_l4 = train_nn(X_l4, y)

    methods = ['Naive', 'Graphon oracle', 'Plugin estimator',
               'Level 4 NN', 'Level 2 NN', 'Level 3 NN', 'Level 3-Norm NN']
    results = {m: [] for m in methods}

    for N in N_test_list:
        rng_te = np.random.default_rng(seed + 9999 + N)
        test = _build_dataset(N, n_test, rng_te)
        targets = test['targets']

        results['Naive'].append(evaluate_per_network(test['naive'], targets)[0])
        results['Graphon oracle'].append(evaluate_per_network(test['oracle'], targets)[0])
        results['Plugin estimator'].append(evaluate_per_network(test['plugin'], targets)[0])
        results['Level 4 NN'].append(evaluate_per_network(
            [predict_nn(mlp_l4, sc_l4, f) for f in test['feats_l4']], targets)[0])
        results['Level 2 NN'].append(evaluate_per_network(
            [predict_nn(mlp_l2, sc_l2, f) for f in test['feats_l2']], targets)[0])
        results['Level 3 NN'].append(evaluate_per_network(
            [predict_nn(mlp_l3, sc_l3, f) for f in test['feats_l3']], targets)[0])
        results['Level 3-Norm NN'].append(evaluate_per_network(
            [predict_nn(mlp_l3n, sc_l3n, f) for f in test['feats_l3n']], targets)[0])

    print()
    print("OUT-OF-DISTRIBUTION L2 ERROR (Table 3)")
    header = f"{'Method':<26}"
    for N in N_test_list:
        header += f"  N={N:>5}"
    print(header)
    print("-" * 70)
    for m in methods:
        row = f"{m:<26}"
        for v in results[m]:
            row += f"  {v:7.4f}"
        print(row)

    return results, (mlp_l2, sc_l2, mlp_l3, sc_l3, mlp_l3n, sc_l3n, mlp_l4, sc_l4)


def run_exploitability_experiment(mlps, N=300, n_test=30, seed=5555):
    mlp_l2, sc_l2, mlp_l3, sc_l3, mlp_l3n, sc_l3n, mlp_l4, sc_l4 = mlps
    rng = np.random.default_rng(seed)

    methods = ['Naive', 'Graphon oracle', 'Plugin estimator',
               'Level 4 NN', 'Level 2 NN', 'Level 3 NN', 'Level 3-Norm NN']
    results = {m: {'l2': [], 'expl': []} for m in methods}

    for _ in range(n_test):
        A, community, theta, kappa = sample_network(N, rng)
        s_true = network_equilibrium(A, theta, kappa)

        preds = {
            'Naive': theta.copy(),
            'Graphon oracle': S_GRAPHON[community],
            'Plugin estimator': plugin_estimator(A, community, theta, kappa),
            'Level 4 NN': predict_nn(mlp_l4, sc_l4, extract_features_level4(A, community, theta)),
            'Level 2 NN': predict_nn(mlp_l2, sc_l2, extract_features_level2(A, community, theta)),
            'Level 3 NN': predict_nn(mlp_l3, sc_l3, extract_features_level3(A, community, theta)),
            'Level 3-Norm NN': predict_nn(mlp_l3n, sc_l3n, extract_features_level3_norm(A, community, theta)),
        }

        for name, sp in preds.items():
            results[name]['l2'].append(l2_distance(sp, s_true))
            results[name]['expl'].append(exploitability(A, sp, theta, kappa))

    print()
    print(f"EXPLOITABILITY at N={N}")
    print(f"{'Method':<28} {'L2 dist':>10} {'Expl':>10}")
    print("-" * 60)
    for name in methods:
        r = results[name]
        print(f"{name:<28} {np.mean(r['l2']):8.4f}   {np.mean(r['expl']):8.4f}")

    return results


if __name__ == "__main__":
    cross_results, mlps = run_cross_N_experiment()
    expl_results = run_exploitability_experiment(mlps)
