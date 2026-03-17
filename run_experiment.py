import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
import pickle
import requests
import zipfile
import io
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def jester():
    combined_path = os.path.join(DATA_DIR, 'jester_combined.npy')
    if os.path.exists(combined_path):
        return combined_path
    urls = [
        "https://eigentaste.berkeley.edu/dataset/jester_dataset_1_1.zip",
        "https://eigentaste.berkeley.edu/dataset/jester_dataset_1_2.zip",
        "https://eigentaste.berkeley.edu/dataset/jester_dataset_1_3.zip",
    ]
    all_data = []
    for url in urls:
        print(f"Downloading {url}...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for name in z.namelist():
                if name.endswith('.xls'):
                    z.extract(name, DATA_DIR)
                    df = pd.read_excel(os.path.join(DATA_DIR, name), header=None)
                    all_data.append(df.values)
    combined = np.vstack(all_data)
    np.save(combined_path, combined)
    return combined_path


def load_jester():
    combined_path = os.path.join(DATA_DIR, 'jester_combined.npy')
    if not os.path.exists(combined_path):
        jester()
    data = np.load(combined_path, allow_pickle=True)
    ratings = data[:, 1:101].astype(np.float64)
    mask = ratings != 99.0
    valid = mask.sum(axis=1) >= 20
    ratings = ratings[valid]
    mask = mask[valid]
    ratings[~mask] = 0.0
    print(f"Loaded {ratings.shape[0]} users, {ratings.shape[1]} items")
    return ratings, mask


def split_data(ratings, mask, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    train_mask = np.zeros_like(mask)
    test_mask = np.zeros_like(mask)
    for i in range(ratings.shape[0]):
        rated = np.where(mask[i])[0]
        n_test = max(1, int(len(rated) * test_frac))
        test_idx = rng.choice(rated, n_test, replace=False)
        train_idx = np.setdiff1d(rated, test_idx)
        train_mask[i, train_idx] = True
        test_mask[i, test_idx] = True
    return train_mask, test_mask


def als(ratings, train_mask, d=20, lam=0.1, n_iter=15):
    print(f"ALS: d={d}, λ={lam}, {n_iter} iters")
    n_users, n_items = ratings.shape
    rng = np.random.RandomState(42)
    U = rng.randn(n_users, d) * 0.1
    V = rng.randn(n_items, d) * 0.1
    for it in range(n_iter):
        for i in range(n_users):
            idx = np.where(train_mask[i])[0]
            if len(idx) == 0: continue
            Vi = V[idx]
            ri = ratings[i, idx]
            U[i] = np.linalg.solve(Vi.T @ Vi + lam * np.eye(d), Vi.T @ ri)
        for j in range(n_items):
            idx = np.where(train_mask[:, j])[0]
            if len(idx) == 0: continue
            Uj = U[idx]
            rj = ratings[idx, j]
            V[j] = np.linalg.solve(Uj.T @ Uj + lam * np.eye(d), Uj.T @ rj)
        if (it + 1) % 5 == 0:
            err = (U @ V.T - ratings) * train_mask
            rmse = np.sqrt(np.sum(err**2) / train_mask.sum())
            print(f"  Iter {it+1}: RMSE = {rmse:.4f}")
    return U, V



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class JesterEnv:
    """Environment: Bernoulli rewards from MF factors."""
    def __init__(self, U, V, scale=1.0, seed=42):
        self.U = U
        self.V = V
        self.K = V.shape[0]
        self.d = V.shape[1]
        self.scale = scale
        self.rng = np.random.RandomState(seed)
        # Precompute reward probs
        self.P = sigmoid(U @ V.T / scale)  # (n_users, K)

    def pull(self, user_idx, arm):
        return self.rng.binomial(1, self.P[user_idx, arm])

    def expected_reward(self, user_idx, arm):
        return self.P[user_idx, arm]

    def best_arm(self, user_idx):
        return np.argmax(self.P[user_idx])

    def oracle_reward(self, user_idx):
        return self.P[user_idx, self.best_arm(user_idx)]



def phi_full(x_t, v_a, d):
    """φ(x_t, a) = [x_t ∥ v_a ∥ x_t[:d] ⊙ v_a] ∈ ℝ^{5d}
    where x_t ∈ ℝ^{2d} = [u_i ∥ v̄]"""
    interaction = x_t[:d] * v_a  # u_i ⊙ v_a
    # Combine: [x_t (2d), v_a (d), u_i⊙v_a (d)] => 4d total
    return np.concatenate([x_t, v_a, interaction])


def phi_user_only(x_t, v_a, d):
    """φ(x_t, a) = [u_i ∥ v_a ∥ u_i ⊙ v_a] ∈ ℝ^{3d}
    where x_t = u_i ∈ ℝ^d (no session embedding)"""
    interaction = x_t * v_a
    return np.concatenate([x_t, v_a, interaction])  # d + d + d = 3d


def phi_no_context(x_t, v_a, d):
    """φ(x_t, a) = v_a ∈ ℝ^d (no user context at all — like TF-IDF)"""
    return v_a.copy()


# Precomputed feature matrices for all arms (vectorized)
def build_phi_matrix_full(x_t, V, d):
    """Returns (K, 4d) matrix of features for all arms."""
    K = V.shape[0]
    x_broadcast = np.tile(x_t, (K, 1))        # (K, 2d)
    interaction = x_t[:d] * V                   # (K, d) — u_i ⊙ v_a
    return np.hstack([x_broadcast, V, interaction])  # (K, 4d)


def build_phi_matrix_user_only(x_t, V, d):
    """Returns (K, 3d) matrix."""
    K = V.shape[0]
    x_broadcast = np.tile(x_t, (K, 1))  # (K, d)
    interaction = x_t * V                 # (K, d)
    return np.hstack([x_broadcast, V, interaction])  # (K, 3d)


def build_phi_matrix_no_context(x_t, V, d):
    """Returns (K, d) matrix — just arm features."""
    return V.copy()


class ContextualBandit:
    """Base for contextual bandits with Sherman-Morrison updates."""
    def __init__(self, p, gamma=1.0):
        self.p = p
        self.gamma = gamma
        self.V_inv = np.eye(p) / gamma
        self.b = np.zeros(p)
        self.theta = np.zeros(p)

    def update(self, phi, reward):
        Vp = self.V_inv @ phi
        self.V_inv -= np.outer(Vp, Vp) / (1.0 + phi @ Vp)
        self.b += reward * phi
        self.theta = self.V_inv @ self.b


class FrequencyBaseline:
    """Select arm with highest global positive rate. Ignores context."""
    def __init__(self, pos_rate):
        self.pos_rate = pos_rate

    def select(self, phi_matrix):
        return np.argmax(self.pos_rate)

    def update(self, phi, reward):
        pass


class EGreedy(ContextualBandit):
    def __init__(self, p, gamma=1.0, epsilon=0.05, seed=123):
        super().__init__(p, gamma)
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)

    def select(self, phi_matrix):
        if self.rng.random() < self.epsilon:
            return self.rng.randint(phi_matrix.shape[0])
        scores = phi_matrix @ self.theta
        return np.argmax(scores)


class LinUCB(ContextualBandit):
    def __init__(self, p, gamma=1.0, alpha=1.0):
        super().__init__(p, gamma)
        self.alpha = alpha

    def select(self, phi_matrix):
        means = phi_matrix @ self.theta                          # (K,)
        Vphi = phi_matrix @ self.V_inv                           # (K, p)
        widths = self.alpha * np.sqrt(np.sum(Vphi * phi_matrix, axis=1))  # (K,)
        return np.argmax(means + widths)


class ThompsonSampling(ContextualBandit):
    def __init__(self, p, gamma=1.0, nu=0.1, seed=456):
        super().__init__(p, gamma)
        self.nu = nu
        self.rng = np.random.RandomState(seed)

    def select(self, phi_matrix):
        try:
            theta_sample = self.rng.multivariate_normal(
                self.theta, self.nu**2 * self.V_inv
            )
        except np.linalg.LinAlgError:
            theta_sample = self.theta + self.nu * self.rng.randn(self.p)
        scores = phi_matrix @ theta_sample
        return np.argmax(scores)


def contextual_stream(env, algo, user_sequence, V, d,
                           build_phi_fn, n_rounds_per_user=10, k=5):
    """
    user_sequence: list of user indices in order of arrival
    Session history resets per user (different "sessions").
    returns: cumulative regret array of length len(user_sequence) * n_rounds_per_user
    """
    T_total = len(user_sequence) * n_rounds_per_user
    cum_regret = np.zeros(T_total)
    cum_reward = np.zeros(T_total)
    instant_regrets = np.zeros(T_total)

    t = 0
    for user_idx in user_sequence:
        u_i = env.U[user_idx]
        oracle_r = env.oracle_reward(user_idx)

        # Per-user session state
        session_vecs = []
        session_rews = []

        for step in range(n_rounds_per_user):
            # Build session embedding
            if len(session_vecs) == 0:
                v_bar = np.zeros(d)
            else:
                recent_v = np.array(session_vecs[-k:])
                recent_r = np.array(session_rews[-k:])
                if sum(recent_r) > 0:
                    v_bar = np.average(recent_v, axis=0, weights=recent_r)
                else:
                    v_bar = recent_v.mean(axis=0)

            # Build context x_t according to feature mode (handled by build_phi_fn)
            if build_phi_fn == build_phi_matrix_full:
                x_t = np.concatenate([u_i, v_bar])
            elif build_phi_fn == build_phi_matrix_user_only:
                x_t = u_i.copy()
            else:  # no_context
                x_t = np.zeros(0)  # unused

            # Build feature matrix for all arms
            phi_matrix = build_phi_fn(x_t, V, d)

            # Select arm
            selected = algo.select(phi_matrix)

            # Get reward
            reward = env.pull(user_idx, selected)
            exp_reward = env.expected_reward(user_idx, selected)
            inst_regret = oracle_r - env.expected_reward(user_idx, selected)

            # Record
            instant_regrets[t] = inst_regret
            cum_regret[t] = (cum_regret[t-1] if t > 0 else 0) + inst_regret
            cum_reward[t] = (cum_reward[t-1] if t > 0 else 0) + reward

            # Update bandit
            phi_selected = phi_matrix[selected]
            algo.update(phi_selected, reward)

            # Update session
            session_vecs.append(V[selected])
            session_rews.append(reward)

            t += 1

    return cum_regret, cum_reward, instant_regrets


def diagnostic_misspecification(env, V, user_ids, d):
    """Compare linear vs. logistic prediction MSE."""
    rng = np.random.RandomState(999)
    X, y = [], []
    for uid in user_ids[:2000]:
        for _ in range(5):
            a = rng.randint(env.K)
            x_t = np.concatenate([env.U[uid], np.zeros(d)])
            phi = phi_full(x_t, V[a], d)
            r = env.pull(uid, a)
            X.append(phi)
            y.append(r)
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(y))

    ridge = Ridge(alpha=1.0).fit(X[:split], y[:split])
    mse_lin = mean_squared_error(y[split:], ridge.predict(X[split:]))

    logit = LogisticRegression(max_iter=1000, C=1.0)
    if len(np.unique(y[:split])) > 1:
        logit.fit(X[:split], y[:split])
        mse_log = mean_squared_error(y[split:], logit.predict_proba(X[split:])[:, 1])
    else:
        mse_log = mse_lin
    return mse_lin, mse_log


def diagnostic_effective_rank(env, V, user_ids, d, n_rounds=10, k=5):
    """Compute effective rank of the context stream and per-user sessions."""
    rng = np.random.RandomState(888)
    # Collect contexts from a random stream
    contexts = []
    for uid in user_ids[:500]:
        session_vecs = []
        session_rews = []
        for step in range(n_rounds):
            if len(session_vecs) == 0:
                v_bar = np.zeros(d)
            else:
                recent_v = np.array(session_vecs[-k:])
                recent_r = np.array(session_rews[-k:])
                if sum(recent_r) > 0:
                    v_bar = np.average(recent_v, axis=0, weights=recent_r)
                else:
                    v_bar = recent_v.mean(axis=0)
            x_t = np.concatenate([env.U[uid], v_bar])
            contexts.append(x_t)
            a = rng.randint(env.K)
            r = env.pull(uid, a)
            session_vecs.append(V[a])
            session_rews.append(r)

    X_ctx = np.array(contexts)
    s = np.linalg.svd(X_ctx, compute_uv=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    global_eff_rank = np.exp(-np.sum(p * np.log(p + 1e-15)))

    # Also per-user effective ranks (of session embeddings only)
    per_user_ranks = []
    for uid in user_ids[:500]:
        session_vecs = []
        session_rews = []
        user_contexts = []
        for step in range(n_rounds):
            if len(session_vecs) == 0:
                v_bar = np.zeros(d)
            else:
                recent_v = np.array(session_vecs[-k:])
                v_bar = recent_v.mean(axis=0)
            user_contexts.append(v_bar)
            a = rng.randint(env.K)
            r = env.pull(uid, a)
            session_vecs.append(V[a])
            session_rews.append(r)
        X_u = np.array(user_contexts)
        try:
            s_u = np.linalg.svd(X_u, compute_uv=False)
            s_u = s_u[s_u > 1e-10]
            if len(s_u) > 0:
                p_u = s_u / s_u.sum()
                per_user_ranks.append(np.exp(-np.sum(p_u * np.log(p_u + 1e-15))))
            else:
                per_user_ranks.append(0)
        except:
            per_user_ranks.append(0)

    return global_eff_rank, np.array(per_user_ranks)


def main():
    t0 = time.time()

    ratings, mask = load_jester()
    train_mask, test_mask = split_data(ratings, mask)
    n_users, n_items = ratings.shape
    binary = (ratings > 0).astype(float)
    pos_rate = (binary * train_mask).sum(axis=0) / (train_mask.sum(axis=0) + 1e-10)
    print(f"  Train: {train_mask.sum():.0f}, Test: {test_mask.sum():.0f}")

    d = 20
    U, V = als(ratings, train_mask, d=d, lam=0.1, n_iter=15)

    # Scale: we want reward probabilities in a reasonable range
    raw_scores = U @ V.T
    scale = np.std(raw_scores[mask]) * 0.5
    env = JesterEnv(U, V, scale=scale, seed=42)
    print(f"\nEnvironment: scale={scale:.2f}")
    print(f"  Mean reward prob: {env.P[mask].mean():.3f}")

    # Draw users in a random order, each gets n_rounds interactions
    N_USERS = 2000        # distinct users in stream
    N_ROUNDS_PER_USER = 10  # interactions per user per session
    T_TOTAL = N_USERS * N_ROUNDS_PER_USER  # 20,000 total interactions

    rng_users = np.random.RandomState(42)
    user_ids = rng_users.choice(n_users, size=N_USERS, replace=False)

    print(f"\nstream: {N_USERS} users × {N_ROUNDS_PER_USER} rounds = {T_TOTAL} interactions")

    # Feature dims
    p_full = 4 * d       # [x_t(2d), v_a(d), u_i⊙v_a(d)] = 80
    p_user_only = 3 * d  # [u_i(d), v_a(d), u_i⊙v_a(d)] = 60
    p_no_context = d      # [v_a(d)] = 20

    print("hyperparameter search (500 users)")

    hp_users = user_ids[:500]

    best_alpha, best_alpha_reg = 0.5, np.inf
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        algo = LinUCB(p_full, gamma=1.0, alpha=alpha)
        reg, _, _ = contextual_stream(env, algo, hp_users, V, d,
                                           build_phi_matrix_full, N_ROUNDS_PER_USER)
        final_reg = reg[-1]
        print(f"  LinUCB α={alpha}: regret={final_reg:.1f}")
        if final_reg < best_alpha_reg:
            best_alpha, best_alpha_reg = alpha, final_reg

    best_nu, best_nu_reg = 0.1, np.inf
    for nu in [0.01, 0.1, 1.0]:
        algo = ThompsonSampling(p_full, gamma=1.0, nu=nu)
        reg, _, _ = contextual_stream(env, algo, hp_users, V, d,
                                           build_phi_matrix_full, N_ROUNDS_PER_USER)
        final_reg = reg[-1]
        print(f"  TS ν={nu}: regret={final_reg:.1f}")
        if final_reg < best_nu_reg:
            best_nu, best_nu_reg = nu, final_reg

    best_eps, best_eps_reg = 0.05, np.inf
    for eps in [0.01, 0.05, 0.1]:
        algo = EGreedy(p_full, gamma=1.0, epsilon=eps)
        reg, _, _ = contextual_stream(env, algo, hp_users, V, d,
                                           build_phi_matrix_full, N_ROUNDS_PER_USER)
        final_reg = reg[-1]
        print(f"  ε-greedy ε={eps}: regret={final_reg:.1f}")
        if final_reg < best_eps_reg:
            best_eps, best_eps_reg = eps, final_reg

    print(f"\nBest: α={best_alpha}, ν={best_nu}, ε={best_eps}")

    print()
    print(f"multiple seeds for CIs: ({N_USERS} users, {N_ROUNDS_PER_USER} rounds/user)")

    N_SEEDS = 10  # Run with different user orderings for confidence intervals

    algo_configs = {
        'Frequency': lambda: FrequencyBaseline(pos_rate),
        f'ε-greedy (ε={best_eps})': lambda: EGreedy(p_full, gamma=1.0, epsilon=best_eps, seed=123),
        f'LinUCB (α={best_alpha})': lambda: LinUCB(p_full, gamma=1.0, alpha=best_alpha),
        f'Thompson (ν={best_nu})': lambda: ThompsonSampling(p_full, gamma=1.0, nu=best_nu, seed=456),
    }

    results = {}
    for name, make_algo in algo_configs.items():
        print(f"\n  {name}...")
        t1 = time.time()
        seed_regrets = []
        for seed in range(N_SEEDS):
            rng_s = np.random.RandomState(42 + seed)
            user_order = rng_s.choice(n_users, size=N_USERS, replace=False)
            # Fresh environment RNG per seed
            env_s = JesterEnv(U, V, scale=scale, seed=100 + seed)
            algo = make_algo()
            reg, rew, inst = contextual_stream(
                env_s, algo, user_order, V, d,
                build_phi_matrix_full, N_ROUNDS_PER_USER
            )
            seed_regrets.append(reg)

        elapsed = time.time() - t1
        # Regret curves: average across seeds
        regret_matrix = np.array(seed_regrets)  # (N_SEEDS, T_TOTAL)
        mean_final = regret_matrix[:, -1].mean()
        se_final = regret_matrix[:, -1].std() / np.sqrt(N_SEEDS)
        print(f"    Final regret: {mean_final:.1f} ± {se_final:.1f}  ({elapsed:.1f}s)")
        results[name] = regret_matrix

    print("BPR Headroom (hit@k):")

    hit_at = {k: [] for k in [1, 3, 5, 10]}
    for uid in user_ids:
        scores = U[uid] @ V.T
        ranked = np.argsort(-scores)
        test_items = np.where(test_mask[uid])[0]
        for item_j in test_items:
            if binary[uid, item_j] == 1:
                pos = np.where(ranked == item_j)[0][0] + 1
                for k in [1, 3, 5, 10]:
                    hit_at[k].append(1.0 if pos <= k else 0.0)

    for k in [1, 3, 5, 10]:
        vals = np.array(hit_at[k])
        print(f"  hit@{k}: {vals.mean():.3f} ± {vals.std()/np.sqrt(len(vals)):.3f}")


    print("Context Ablation (LinUCB):")

    ablation_configs = {
        'Full [u_i ∥ v̄, v_a, u_i⊙v_a]': (build_phi_matrix_full, p_full),
        'User-only [u_i, v_a, u_i⊙v_a]': (build_phi_matrix_user_only, p_user_only),
        'No context [v_a]': (build_phi_matrix_no_context, p_no_context),
    }

    ablation_results = {}
    for name, (build_fn, p) in ablation_configs.items():
        print(f"  {name} (p={p})...")
        seed_regrets = []
        for seed in range(N_SEEDS):
            rng_s = np.random.RandomState(42 + seed)
            user_order = rng_s.choice(n_users, size=N_USERS, replace=False)
            env_s = JesterEnv(U, V, scale=scale, seed=100 + seed)
            algo = LinUCB(p, gamma=1.0, alpha=best_alpha)
            reg, _, _ = contextual_stream(
                env_s, algo, user_order, V, d,
                build_fn, N_ROUNDS_PER_USER
            )
            seed_regrets.append(reg)

        regret_matrix = np.array(seed_regrets)
        mean_final = regret_matrix[:, -1].mean()
        se_final = regret_matrix[:, -1].std() / np.sqrt(N_SEEDS)
        print(f"    Final regret: {mean_final:.1f} ± {se_final:.1f}")
        ablation_results[name] = regret_matrix

    print("diagnostics...")

    mse_lin, mse_log = diagnostic_misspecification(env, V, user_ids, d)
    print(f"  (a) Linear MSE={mse_lin:.4f}, Logistic MSE={mse_log:.4f}, "
          f"gap={mse_lin - mse_log:.4f}")

    ess = T_TOTAL / n_items
    print(f"  (b) Total interactions={T_TOTAL}, ESS={ess:.1f}")

    global_rank, per_user_ranks = diagnostic_effective_rank(env, V, user_ids, d, N_ROUNDS_PER_USER)
    print(f"  (c) Global effective rank of context stream: {global_rank:.1f} / {2*d}")
    print(f"      Per-user session rank: median={np.median(per_user_ranks):.2f}, "
          f"mean={np.mean(per_user_ranks):.2f}")

    print("95% CIs (across seeds)")

    bootstrap = {}
    for name, regret_matrix in results.items():
        finals = regret_matrix[:, -1]
        mean_r = finals.mean()
        ci = (np.percentile(finals, 2.5), np.percentile(finals, 97.5))
        # With few seeds, CI calculation is approximate
        se = finals.std() / np.sqrt(len(finals))
        ci = (mean_r - 1.96 * se, mean_r + 1.96 * se)
        bootstrap[name] = {'mean': mean_r, 'se': se, 'ci': ci}
        print(f"  {name}: {mean_r:.1f} ± {se:.1f}  [{ci[0]:.1f}, {ci[1]:.1f}]")

    ablation_bootstrap = {}
    for name, regret_matrix in ablation_results.items():
        finals = regret_matrix[:, -1]
        mean_r = finals.mean()
        se = finals.std() / np.sqrt(len(finals))
        ci = (mean_r - 1.96 * se, mean_r + 1.96 * se)
        ablation_bootstrap[name] = {'mean': mean_r, 'se': se, 'ci': ci}
        print(f"  [ablation] {name}: {mean_r:.1f} ± {se:.1f}  [{ci[0]:.1f}, {ci[1]:.1f}]")

    all_results = {
        'results': {k: v.tolist() for k, v in results.items()},
        'ablation': {k: v.tolist() for k, v in ablation_results.items()},
        'bootstrap': bootstrap,
        'ablation_bootstrap': ablation_bootstrap,
        'hit_at_k': {k: np.array(v) for k, v in hit_at.items()},
        'diagnostics': {
            'mse_linear': mse_lin, 'mse_logistic': mse_log,
            'ess': ess, 'total_interactions': T_TOTAL,
            'global_eff_rank': global_rank,
            'per_user_eff_ranks': per_user_ranks,
        },
        'hyperparams': {
            'best_alpha': best_alpha, 'best_nu': best_nu, 'best_eps': best_eps,
            'd': d, 'N_USERS': N_USERS, 'N_ROUNDS_PER_USER': N_ROUNDS_PER_USER,
            'T_TOTAL': T_TOTAL, 'scale': scale, 'N_SEEDS': N_SEEDS,
        },
    }

    path = os.path.join(RESULTS_DIR, 'experiment_results.pkl')
    with open(path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved to {path}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
