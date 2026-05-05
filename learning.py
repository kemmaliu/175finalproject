"""
Learning the graphon: comparing observation levels with neural networks.

Setup: agents DON'T know the graphon Q, but observe local information
and use a NN to predict their equilibrium strategy.

Observation levels (most → least information):
  Level 1: Full adjacency matrix A → can solve exact NE (benchmark)
  Level 2: Own community label + neighbor counts per community + θ_i
  Level 3: Own neighbors (no community labels) → local topology features + θ_i
  Level 4: Just own community + total degree + θ_i (minimal info)

Additional baselines:
  Graphon oracle: knows Q exactly → plays graphon equilibrium s̄_com[c_i]
  Naive: ignores network → plays θ_i

Ground truth: exact network NE  s* = (I - α/(κN)*A)^{-1} θ

Metric: L² distance  sqrt( (1/N) Σ_i (s_pred_i - s*_i)² )
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt

# Set up

K = 4
ALPHA = 2.65
THETA_COM = np.array([0.1, 0.1, 0.1, 0.25])

Q_SBM = np.array([
    [0.90, 0.05, 0.00, 0.00],
    [0.05, 0.20, 0.05, 0.00],
    [0.00, 0.05, 0.20, 0.05],
    [0.00, 0.00, 0.05, 0.80],
])

E_MAT = Q_SBM / K
C_PER_AGENT = 0.02

# Graphon equilibrium (ground truth for infinite population)
S_GRAPHON = np.maximum(np.linalg.solve(np.eye(K) - ALPHA * E_MAT, THETA_COM), 0)


def get_kappa(N, gamma=0.8):
    return 40.0 / (N ** gamma)


def sample_network(N, rng, gamma=0.8, Q_SBM=Q_SBM, THETA_COM = THETA_COM): # modify to take kappa
    """Sample sparse SBM network. Returns A, community, theta, kappa."""
    kappa = get_kappa(N, gamma)
    t = rng.uniform(0, 1, size=N)
    community = np.clip((t * K).astype(int), 0, K - 1)
    prob = np.clip(kappa * Q_SBM[community[:, None], community[None, :]], 0, 1)
    U = rng.uniform(0, 1, (N, N))
    A = np.triu((U < prob).astype(float), 1)
    A = A + A.T
    theta = THETA_COM[community]
    return A, community, theta, kappa


def network_equilibrium(A, theta, kappa):
    """Exact NE: s* = (I - α/(κN) * A)^{-1} θ."""
    N = A.shape[0]
    M = np.eye(N) - (ALPHA / (kappa * N)) * A
    return np.maximum(np.linalg.solve(M, theta), 0)


def l2_distance(s_pred, s_true):
    """Per-network L² distance: sqrt(mean((s_pred - s_true)²))."""
    return np.sqrt(np.mean((s_pred - s_true) ** 2))


# getting features

def extract_features_level2(A, community, theta):
    """
    Level 2: community label + neighbor counts per community + θ_i.
    
    Features per agent (9-dim):
      - community one-hot  (4)
      - n_neighbors in C1, C2, C3, C4  (4)
      - θ_i  (1)
    """
    N = A.shape[0]
    features = np.zeros((N, 4 + K + 1))
    
    for i in range(N):
        # one-hot community
        features[i, community[i]] = 1.0
        # neighbor counts per community
        neighbors = np.where(A[i] > 0)[0]
        for c in range(K):
            features[i, 4 + c] = np.sum(community[neighbors] == c)
        # theta
        features[i, -1] = theta[i]
    
    return features


def extract_features_level3(A, community, theta):
    """
    Level 3: own neighbors (NO community labels), local topology + θ_i.
    
    Features per agent (7-dim):
      - θ_i  (1)
      - degree  (1)
      - mean neighbor degree  (1)
      - std neighbor degree  (1)
      - local clustering coefficient  (1)
      - fraction of neighbors with same θ as self  (1)
      - max neighbor degree  (1)
    """
    N = A.shape[0]
    degrees = A.sum(axis=1)
    features = np.zeros((N, 7))
    
    for i in range(N):
        neighbors = np.where(A[i] > 0)[0]
        deg_i = len(neighbors)
        
        features[i, 0] = theta[i]
        features[i, 1] = deg_i
        
        if deg_i > 0:
            neighbor_degs = degrees[neighbors]
            features[i, 2] = np.mean(neighbor_degs)
            features[i, 3] = np.std(neighbor_degs) if deg_i > 1 else 0
            features[i, 6] = np.max(neighbor_degs)
            
            # fraction of neighbors with same theta
            features[i, 5] = np.mean(theta[neighbors] == theta[i])
            
            # local clustering coefficient
            if deg_i >= 2:
                subgraph = A[np.ix_(neighbors, neighbors)]
                n_triangles = np.sum(subgraph) / 2
                max_triangles = deg_i * (deg_i - 1) / 2
                features[i, 4] = n_triangles / max_triangles
    
    return features


def extract_features_level4(A, community, theta):
    """
    Level 4: own community + total degree + θ_i (minimal info).
    
    Features per agent (6-dim):
      - community one-hot  (4)
      - degree  (1)
      - θ_i  (1)
    """
    N = A.shape[0]
    degrees = A.sum(axis=1)
    features = np.zeros((N, 4 + 2))
    
    for i in range(N):
        features[i, community[i]] = 1.0
        features[i, 4] = degrees[i]
        features[i, 5] = theta[i]
    
    return features


# non-NN estimators

def plugin_estimator(A, community, theta, kappa):
    """
    Level 2 model-based: estimate Q from neighbor counts,
    solve K×K graphon equilibrium, play s̄_com[c_i].
    
    This is the "smart" non-NN alternative for Level 2.
    """
    N = A.shape[0]
    
    # Count neighbors per community pair
    Q_hat = np.zeros((K, K))
    for c1 in range(K):
        mask1 = community == c1
        n1 = mask1.sum()
        if n1 == 0:
            continue
        for c2 in range(K):
            mask2 = community == c2
            n2 = mask2.sum()
            if n2 == 0:
                continue
            # average edge probability estimate
            edges = A[np.ix_(mask1, mask2)].sum()
            if c1 == c2:
                # self-community: exclude diagonal
                denom = kappa * n1 * (n1 - 1) if n1 > 1 else 1
            else:
                denom = kappa * n1 * n2
            Q_hat[c1, c2] = edges / denom if denom > 0 else 0
    
    Q_hat = (Q_hat + Q_hat.T) / 2
    Q_hat = np.clip(Q_hat, 0, 1)
    
    # Solve graphon equilibrium with estimated Q
    E_hat = Q_hat / K
    try:
        s_com_hat = np.linalg.solve(np.eye(K) - ALPHA * E_hat, THETA_COM)
        s_com_hat = np.maximum(s_com_hat, 0)
    except np.linalg.LinAlgError:
        s_com_hat = THETA_COM.copy()
    
    return s_com_hat[community]


# generate graphs

def generate_dataset(N, n_networks, rng, gamma=0.8, Q_SBM=Q_SBM, THETA_COM = THETA_COM):
    """
    Generate training/test data: sample networks, compute exact NE,
    extract features at all levels.
    
    Returns dict with features and targets.
    """
    all_features_l2 = []
    all_features_l3 = []
    all_features_l4 = []
    all_targets = []
    all_communities = []
    all_graphon_pred = []     # graphon oracle predictions
    all_naive_pred = []       # naive predictions (just theta)
    all_plugin_pred = []      # model-based estimator
    
    for net_idx in range(n_networks):
        A, community, theta, kappa = sample_network(N, rng, gamma, Q_SBM, THETA_COM) # modify to take kappa
        s_true = network_equilibrium(A, theta, kappa)
        
        # Features at each level
        feat_l2 = extract_features_level2(A, community, theta)
        feat_l3 = extract_features_level3(A, community, theta)
        feat_l4 = extract_features_level4(A, community, theta)
        
        # Baselines
        K = len(Q_SBM)
        E_MAT = Q_SBM / K
        # keep ALPHA equal to our global ALPHA, for simplicity
        S_GRAPHON = np.maximum(np.linalg.solve(np.eye(K) - ALPHA * E_MAT, THETA_COM), 0)
        graphon_pred = S_GRAPHON[community]
        naive_pred = theta.copy()
        plugin_pred = plugin_estimator(A, community, theta, kappa)
        
        all_features_l2.append(feat_l2)
        all_features_l3.append(feat_l3)
        all_features_l4.append(feat_l4)
        all_targets.append(s_true)
        all_communities.append(community)
        all_graphon_pred.append(graphon_pred)
        all_naive_pred.append(naive_pred)
        all_plugin_pred.append(plugin_pred)
    
    return {
        'feat_l2': np.vstack(all_features_l2),
        'feat_l3': np.vstack(all_features_l3),
        'feat_l4': np.vstack(all_features_l4),
        'targets': np.concatenate(all_targets),
        'communities': np.concatenate(all_communities),
        'graphon_pred': np.concatenate(all_graphon_pred),
        'naive_pred': np.concatenate(all_naive_pred),
        'plugin_pred': np.concatenate(all_plugin_pred),
        # Keep per-network targets for per-network L2
        'targets_by_net': all_targets,
        'graphon_by_net': all_graphon_pred,
        'naive_by_net': all_naive_pred,
        'plugin_by_net': all_plugin_pred,
        'feat_l2_by_net': all_features_l2,
        'feat_l3_by_net': all_features_l3,
        'feat_l4_by_net': all_features_l4,
    }


# Train NN

def train_nn(X_train, y_train, hidden=(128, 64, 32), max_iter=500):
    """Train a simple MLP regressor with standardized inputs."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
    )
    mlp.fit(X_scaled, y_train)
    return mlp, scaler


def predict_nn(mlp, scaler, X):
    """Predict with trained NN."""
    X_scaled = scaler.transform(X)
    return np.maximum(mlp.predict(X_scaled), 0)


def evaluate_per_network(predictions_by_net, targets_by_net):
    """Compute per-network L² distances and return mean + std."""
    dists = []
    for s_pred, s_true in zip(predictions_by_net, targets_by_net):
        dists.append(l2_distance(s_pred, s_true))
    return np.mean(dists), np.std(dists)


# experiment functions

def run_learning_experiment(N=300, n_train=100, n_test=50, seed=42, gamma = 0.8, Q_SBM = Q_SBM, THETA_COM = THETA_COM):
    """Run the full comparison across all observation levels."""
    
    print("=" * 65)
    print(f"LEARNING EXPERIMENT: N={N}, {n_train} train nets, {n_test} test nets, gamma = {gamma}")
    print("=" * 65)
    
    # Generate data
    print("\nGenerating training data...")
    rng_train = np.random.default_rng(seed)
    train_data = generate_dataset(N, n_train, rng_train, gamma, Q_SBM, THETA_COM) # MODIFY TO TAKE GAMMA
    
    print("Generating test data...")
    rng_test = np.random.default_rng(seed + 9999)
    test_data = generate_dataset(N, n_test, rng_test, gamma, Q_SBM, THETA_COM) # MODIFY TO TAKE GAMMA
    
    print(f"Training samples: {len(train_data['targets'])}")
    print(f"Test samples: {len(test_data['targets'])}")
    
    # ── Train NNs ──
    results = {}
    
    # Naive baseline
    print("\nEvaluating baselines...")
    naive_dist, naive_std = evaluate_per_network(
        test_data['naive_by_net'], test_data['targets_by_net'])
    results['Naive (play θ_i)'] = (naive_dist, naive_std)
    
    # Graphon oracle
    graphon_dist, graphon_std = evaluate_per_network(
        test_data['graphon_by_net'], test_data['targets_by_net'])
    results['Graphon oracle'] = (graphon_dist, graphon_std)
    
    # Plugin estimator (model-based Level 2)
    plugin_dist, plugin_std = evaluate_per_network(
        test_data['plugin_by_net'], test_data['targets_by_net'])
    results['Plugin estimator'] = (plugin_dist, plugin_std)
    
    # Level 4 NN (minimal: community + degree)
    print("Training Level 4 NN (community + degree)...")
    mlp4, scl4 = train_nn(train_data['feat_l4'], train_data['targets'])
    pred4_by_net = [predict_nn(mlp4, scl4, f) for f in test_data['feat_l4_by_net']]
    l4_dist, l4_std = evaluate_per_network(pred4_by_net, test_data['targets_by_net'])
    results['Level 4 NN'] = (l4_dist, l4_std)
    
    # Level 3 NN (topology features, no community labels)
    print("Training Level 3 NN (local topology, no labels)...")
    mlp3, scl3 = train_nn(train_data['feat_l3'], train_data['targets'])
    pred3_by_net = [predict_nn(mlp3, scl3, f) for f in test_data['feat_l3_by_net']]
    l3_dist, l3_std = evaluate_per_network(pred3_by_net, test_data['targets_by_net'])
    results['Level 3 NN'] = (l3_dist, l3_std)
    
    # Level 2 NN (community + neighbor counts per community)
    print("Training Level 2 NN (community + neighbor counts)...")
    mlp2, scl2 = train_nn(train_data['feat_l2'], train_data['targets'])
    pred2_by_net = [predict_nn(mlp2, scl2, f) for f in test_data['feat_l2_by_net']]
    l2_dist, l2_std = evaluate_per_network(pred2_by_net, test_data['targets_by_net'])
    results['Level 2 NN'] = (l2_dist, l2_std)
    
    # ── Print results ──
    print("\n" + "=" * 65)
    print(f"{'Method':<28} {'L² distance':>14} {'± std':>10}")
    print("-" * 65)
    
    # Sort by distance (worst to best)
    for name, (dist, std) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"  {name:<26} {dist:10.4f}     ±{std:.4f}")
    
    # Level 1 is just the exact NE (distance = 0 by definition)
    print(f"  {'Level 1 (exact NE)':<26} {'0.0000':>10}     ±0.0000")
    print("=" * 65)
    
    # ── Per-community breakdown ──
    print(f"\n{'Per-community L² (test set)':}")
    print(f"{'Method':<28}", end="")
    for c in range(K):
        print(f"  {'C'+str(c+1):>6}", end="")
    print()
    print("-" * 60)
    
    all_preds = {
        'Naive': test_data['naive_pred'],
        'Graphon oracle': test_data['graphon_pred'],
        'Plugin estimator': test_data['plugin_pred'],
        'Level 4 NN': np.concatenate(pred4_by_net),
        'Level 3 NN': np.concatenate(pred3_by_net),
        'Level 2 NN': np.concatenate(pred2_by_net),
    }
    
    comms = test_data['communities']
    targets = test_data['targets']
    
    for name, pred in all_preds.items():
        print(f"  {name:<26}", end="")
        for c in range(K):
            mask = comms == c
            d = np.sqrt(np.mean((pred[mask] - targets[mask]) ** 2))
            print(f"  {d:6.4f}", end="")
        print()
    
    return results, test_data, all_preds


def run_stress_tests():
    # --- Stress Test 1: Sparsity Stress Test ---
    # We test different 'gammas' in kappa = 40 / N^gamma
    # Higher gamma = sparser graph (less information for agents)
    gammas = [0.9, 1.0, 1.1, 1.2, 1.3]
    sparsity_results = []

    print("Running Sparsity Stress Test...")
    for g in gammas:
        run_learning_experiment(
            N=300, n_train=100, n_test=50, seed=42, gamma=g
        )

    # --- Stress Test 2: Structural Stress Test (Disassortative/Dissociative SBM) ---
    # High q (between community), Low p (within community)
    # This is the "Opposite" of a standard social network
    Q_DISASSORTATIVE = np.array([
        [0.05, 0.80, 0.80, 0.80],
        [0.80, 0.05, 0.80, 0.80],
        [0.80, 0.80, 0.05, 0.80],
        [0.80, 0.80, 0.80, 0.05],
    ])
    
    print("\nRunning Disassortative SBM Test...")
    dis_res = run_learning_experiment(
        N=300, n_train=100, n_test=50, seed=42,
        Q_SBM=Q_DISASSORTATIVE,
    )
    
    return sparsity_results, dis_res


# For debugging and figuring out what the impact of gamma is on the actual graphs 

def calculate_average_degrees(N=300, gammas=[0.9, 1.0, 1.1, 1.2, 1.3]):
    Q_SBM = np.array([
        [0.90, 0.05, 0.00, 0.00],
        [0.05, 0.20, 0.05, 0.00],
        [0.00, 0.05, 0.20, 0.05],
        [0.00, 0.00, 0.05, 0.80],
    ])
    
    Q_bar = np.mean(Q_SBM)
    
    print(f"{'Gamma':<10} | {'Avg Degree (Expected)':<20}")
    print("-" * 35)
    
    for g in gammas:
        # kappa = 40 / N^gamma
        kappa = 40 / (N**g)
        
        # Expected degree = kappa * N * Q_bar
        avg_degree = kappa * N * Q_bar
        
        print(f"{g:<10} | {avg_degree:<20.2f}")


if __name__ == "__main__":
    results, test_data, all_preds = run_learning_experiment(
        N=300, n_train=100, n_test=50, seed=42
    )

    sparsity_data, disassortative_results = run_stress_tests()

    # calculate_average_degrees() 
    