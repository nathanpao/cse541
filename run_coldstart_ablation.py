import os
import time
import numpy as np
import pickle

from run_experiment import (
    load_jester, split_data, als, sigmoid,
    JesterEnv, LinUCB,
    build_phi_matrix_user_only,
    RESULTS_DIR, DATA_DIR,
)


def stream_oracle(env, algo, user_sequence, V, d, n_rounds_per_user):
    """original: x_t = env.U[user_idx]."""
    T = len(user_sequence) * n_rounds_per_user
    cum_regret = np.zeros(T)
    t = 0
    for user_idx in user_sequence:
        u_i = env.U[user_idx]
        oracle_r = env.oracle_reward(user_idx)
        for step in range(n_rounds_per_user):
            phi_matrix = build_phi_matrix_user_only(u_i, V, d)
            selected = algo.select(phi_matrix)
            reward = env.pull(user_idx, selected)
            inst_regret = oracle_r - env.expected_reward(user_idx, selected)
            cum_regret[t] = (cum_regret[t - 1] if t > 0 else 0) + inst_regret
            algo.update(phi_matrix[selected], reward)
            t += 1
    return cum_regret


def stream_progressive_linearized(env, algo, user_sequence, V, d,
                                      n_rounds_per_user, lam_user=1.0):
    """progressive cold-start with a linearized sigmoid correction
    uses the transform r_tilde = 4 * s * (r - 0.5) and fits via ridge.
    """
    s = env.scale
    T = len(user_sequence) * n_rounds_per_user
    cum_regret = np.zeros(T)
    t = 0
    for user_idx in user_sequence:
        oracle_r = env.oracle_reward(user_idx)
        VtV = lam_user * np.eye(d)
        Vtr = np.zeros(d)
        u_hat = np.zeros(d)

        for step in range(n_rounds_per_user):
            phi_matrix = build_phi_matrix_user_only(u_hat, V, d)
            selected = algo.select(phi_matrix)
            reward = env.pull(user_idx, selected)
            inst_regret = oracle_r - env.expected_reward(user_idx, selected)
            cum_regret[t] = (cum_regret[t - 1] if t > 0 else 0) + inst_regret
            algo.update(phi_matrix[selected], reward)

            v_sel = V[selected]
            VtV += np.outer(v_sel, v_sel)
            r_tilde = 4.0 * s * (reward - 0.5)  # linearized transform
            Vtr += r_tilde * v_sel
            u_hat = np.linalg.solve(VtV, Vtr)
            t += 1
    return cum_regret


def stream_progressive_logistic(env, algo, user_sequence, V, d,
                                    n_rounds_per_user, lam_user=1.0,
                                    n_newton=3):
        """progressive cold-start using logistic ridge (Newton updates)
        runs a few Newton steps per update to fit the logistic model
        """
    s = env.scale
    T = len(user_sequence) * n_rounds_per_user
    cum_regret = np.zeros(T)
    t = 0
    for user_idx in user_sequence:
        oracle_r = env.oracle_reward(user_idx)
        u_hat = np.zeros(d)
        obs_v = []  # list of v_a vectors seen
        obs_r = []  # list of rewards

        for step in range(n_rounds_per_user):
            phi_matrix = build_phi_matrix_user_only(u_hat, V, d)
            selected = algo.select(phi_matrix)
            reward = env.pull(user_idx, selected)
            inst_regret = oracle_r - env.expected_reward(user_idx, selected)
            cum_regret[t] = (cum_regret[t - 1] if t > 0 else 0) + inst_regret
            algo.update(phi_matrix[selected], reward)

            obs_v.append(V[selected].copy())
            obs_r.append(reward)

            # Newton steps for logistic ridge
            if len(obs_v) >= 2:  # need ≥2 obs for Newton to be stable
                Vm = np.array(obs_v)   # (n_obs, d)
                rm = np.array(obs_r)   # (n_obs,)
                u_cur = u_hat.copy()
                for _ in range(n_newton):
                    z = Vm @ u_cur / s          # (n_obs,)
                    p = sigmoid(z)               # (n_obs,)
                    residual = rm - p            # (n_obs,)
                    w = p * (1 - p)              # (n_obs,)
                    # Gradient
                    g = (Vm.T @ residual) / s - lam_user * u_cur
                    # Hessian (negative definite)
                    H = -(Vm.T * w) @ Vm / (s * s) - lam_user * np.eye(d)
                    # Newton step: u ← u - H⁻¹ g
                    try:
                        delta = np.linalg.solve(H, g)
                        u_cur = u_cur - delta
                    except np.linalg.LinAlgError:
                        break
                u_hat = u_cur

            t += 1
    return cum_regret


def stream_noisy_oracle(env, algo, user_sequence, V, d,
                            n_rounds_per_user, noise_scale=2.0):
    """diagnostic: noisy oracle using true u_i plus decaying Gaussian noise."""
    T = len(user_sequence) * n_rounds_per_user
    cum_regret = np.zeros(T)
    rng = np.random.RandomState(999)
    t = 0
    for user_idx in user_sequence:
        u_i = env.U[user_idx]
        oracle_r = env.oracle_reward(user_idx)

        for step in range(n_rounds_per_user):
            noise_std = noise_scale / np.sqrt(step + 1)
            u_noisy = u_i + rng.randn(d) * noise_std

            phi_matrix = build_phi_matrix_user_only(u_noisy, V, d)
            selected = algo.select(phi_matrix)
            reward = env.pull(user_idx, selected)
            inst_regret = oracle_r - env.expected_reward(user_idx, selected)
            cum_regret[t] = (cum_regret[t - 1] if t > 0 else 0) + inst_regret
            algo.update(phi_matrix[selected], reward)
            t += 1
    return cum_regret


def run_stream_per_user(run_fn, env, algo, user_sequence, V, d,
                        n_rounds_per_user, **kwargs):
    # per-user decomposition requires re-running
    # 
    #handled in main().
    pass

def main():
    t0 = time.time()

    ratings, mask = load_jester()
    train_mask, test_mask = split_data(ratings, mask)
    n_users, n_items = ratings.shape
    d = 20
    U, V = als(ratings, train_mask, d=d, lam=0.1, n_iter=15)
    raw_scores = U @ V.T
    scale = np.std(raw_scores[mask]) * 0.5

    best_alpha = 2.0
    p_user_only = 3 * d  # 60
    N_SEEDS = 10
    T_FIXED = 20000

    print(f"\nSetup: T={T_FIXED}, d={d}, α={best_alpha}, {N_SEEDS} seeds")
    print(f"Scale={scale:.2f}")


    print("experiment 1: Estimation methods")

    methods = {
        'Oracle': stream_oracle,
        'Progressive (linearized)': stream_progressive_linearized,
        'Progressive (logistic MLE)': stream_progressive_logistic,
        'Noisy oracle (c=2.0)': lambda env, algo, us, V, d, nr:
            stream_noisy_oracle(env, algo, us, V, d, nr, noise_scale=2.0),
        'Noisy oracle (c=1.0)': lambda env, algo, us, V, d, nr:
            stream_noisy_oracle(env, algo, us, V, d, nr, noise_scale=1.0),
        'Noisy oracle (c=0.5)': lambda env, algo, us, V, d, nr:
            stream_noisy_oracle(env, algo, us, V, d, nr, noise_scale=0.5),
    }

    curve_results = {}
    for name, run_fn in methods.items():
        print(f"\n  {name}...")
        seed_curves = []
        for seed in range(N_SEEDS):
            rng_s = np.random.RandomState(42 + seed)
            user_order = rng_s.choice(n_users, size=2000, replace=False)
            env_s = JesterEnv(U, V, scale=scale, seed=100 + seed)
            algo = LinUCB(p_user_only, gamma=1.0, alpha=best_alpha)
            reg = run_fn(env_s, algo, user_order, V, d, 10)
            seed_curves.append(reg)
        rm = np.array(seed_curves)
        curve_results[name] = rm
        mean_f = rm[:, -1].mean()
        se_f = rm[:, -1].std() / np.sqrt(N_SEEDS)
        print(f"    Final regret: {mean_f:.1f} ± {se_f:.1f}")

    # Rate verification
    print("\n  Rate verification (β):")
    for name, rm in curve_results.items():
        mean_curve = rm.mean(axis=0)
        T_arr = np.arange(1, len(mean_curve) + 1)
        half = len(T_arr) // 2
        log_T = np.log(T_arr[half:])
        log_R = np.log(np.maximum(mean_curve[half:], 1e-10))
        beta = np.polyfit(log_T, log_R, 1)[0]
        print(f"    {name}: β = {beta:.3f}")

    horizons = [10, 25, 50, 100]

    print("experiment 2: Horizon Sweep (T=20,000)")

    sweep_methods = {
        'Oracle': stream_oracle,
        'Linearized': stream_progressive_linearized,
        'Logistic MLE': stream_progressive_logistic,
    }

    sweep_results = {}
    for n_rounds in horizons:
        n_users_this = T_FIXED // n_rounds
        print(f"\n  --- {n_users_this} users × {n_rounds} rounds ---")

        for mode_name, run_fn in sweep_methods.items():
            seed_regrets = []
            for seed in range(N_SEEDS):
                rng_s = np.random.RandomState(42 + seed)
                user_order = rng_s.choice(n_users, size=n_users_this,
                                          replace=False)
                env_s = JesterEnv(U, V, scale=scale, seed=100 + seed)
                algo = LinUCB(p_user_only, gamma=1.0, alpha=best_alpha)
                reg = run_fn(env_s, algo, user_order, V, d, n_rounds)
                seed_regrets.append(reg[-1])

            finals = np.array(seed_regrets)
            mean_r = finals.mean()
            se_r = finals.std() / np.sqrt(N_SEEDS)
            key = (mode_name, n_rounds)
            sweep_results[key] = {'mean': mean_r, 'se': se_r,
                                  'n_users': n_users_this}
            print(f"    {mode_name}: {mean_r:.1f} ± {se_r:.1f}")

    # Ratios
    print("\n  --- Progressive / Oracle ratio ---")
    for method in ['Linearized', 'Logistic MLE']:
        print(f"  {method}:")
        for n_rounds in horizons:
            o = sweep_results[('Oracle', n_rounds)]['mean']
            p = sweep_results[(method, n_rounds)]['mean']
            ratio = p / o if o > 0 else float('inf')
            print(f"    {n_rounds} rounds: {ratio:.2f}x  "
                  f"(Oracle={o:.0f}, {method}={p:.0f})")

    print("experiment 3: û estimation quality (100 rounds, single seed)")

    env_diag = JesterEnv(U, V, scale=scale, seed=42)
    rng_diag = np.random.RandomState(42)
    test_users = rng_diag.choice(n_users, size=200, replace=False)

    for method_name, n_rounds in [('Linearized', 10), ('Linearized', 50),
                                   ('Linearized', 100),
                                   ('Logistic MLE', 10), ('Logistic MLE', 100)]:
        cosine_sims = []
        l2_errors = []
        for uid in test_users:
            u_true = env_diag.U[uid]
            # Simulate n_rounds of random pulls to estimate u
            rng_est = np.random.RandomState(uid)
            obs_v = []
            obs_r = []
            for s in range(n_rounds):
                a = rng_est.randint(env_diag.K)
                r = env_diag.pull(uid, a)
                obs_v.append(V[a])
                obs_r.append(r)

            Vm = np.array(obs_v)
            rm = np.array(obs_r)

            if method_name == 'Linearized':
                r_tilde = 4.0 * scale * (rm - 0.5)
                VtV = Vm.T @ Vm + 1.0 * np.eye(d)
                u_hat = np.linalg.solve(VtV, Vm.T @ r_tilde)
            else:  # Logistic MLE
                u_hat = np.zeros(d)
                for _ in range(5):
                    z = Vm @ u_hat / scale
                    p = sigmoid(z)
                    residual = rm - p
                    w = p * (1 - p)
                    g = (Vm.T @ residual) / scale - 1.0 * u_hat
                    H = -(Vm.T * w) @ Vm / (scale * scale) - 1.0 * np.eye(d)
                    try:
                        delta = np.linalg.solve(H, g)
                        u_hat = u_hat - delta
                    except np.linalg.LinAlgError:
                        break

            # Metrics
            cos = (u_true @ u_hat) / (np.linalg.norm(u_true) *
                                       np.linalg.norm(u_hat) + 1e-10)
            l2 = np.linalg.norm(u_hat - u_true)
            cosine_sims.append(cos)
            l2_errors.append(l2)

        cos_arr = np.array(cosine_sims)
        l2_arr = np.array(l2_errors)
        u_norms = np.array([np.linalg.norm(env_diag.U[uid])
                            for uid in test_users])
        print(f"  {method_name} ({n_rounds} rounds): "
              f"cosine={cos_arr.mean():.3f}±{cos_arr.std():.3f}, "
              f"||û-u||={l2_arr.mean():.2f}±{l2_arr.std():.2f}, "
              f"||u||={u_norms.mean():.2f}")

    save_data = {
        'curves': {k: v.tolist() for k, v in curve_results.items()},
        'sweep': {str(k): v for k, v in sweep_results.items()},
        'horizons': horizons,
        'T_fixed': T_FIXED,
        'd': d,
        'alpha': best_alpha,
        'scale': scale,
    }
    path = os.path.join(RESULTS_DIR, 'coldstart_ablation.pkl')
    with open(path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved to {path}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        T_arr = np.arange(1, T_FIXED + 1)
        plot_methods = [
            ('Oracle', '#1f77b4', '-', 2.0),
            ('Progressive (linearized)', '#2ca02c', '--', 2.0),
            ('Progressive (logistic MLE)', '#d62728', '-.', 2.0),
            ('Noisy oracle (c=1.0)', '#9467bd', ':', 1.5),
        ]
        for name, color, ls, lw in plot_methods:
            if name in curve_results:
                rm = curve_results[name]
                mean_c = rm.mean(axis=0)
                se_c = rm.std(axis=0) / np.sqrt(N_SEEDS)
                ax.plot(T_arr, mean_c, label=name, color=color, ls=ls, lw=lw)
                ax.fill_between(T_arr, mean_c - 1.96 * se_c,
                                mean_c + 1.96 * se_c,
                                alpha=0.12, color=color)
        ax.set_xlabel('Interaction $t$', fontsize=12)
        ax.set_ylabel('Cumulative Pseudo-Regret', fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, T_FIXED)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        p3 = os.path.join(RESULTS_DIR, 'figure3_coldstart.pdf')
        fig.savefig(p3, dpi=150, bbox_inches='tight')
        fig.savefig(p3.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Figure 3 saved to {p3}")
        plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        oracle_means = [sweep_results[('Oracle', h)]['mean'] for h in horizons]
        oracle_ses = [sweep_results[('Oracle', h)]['se'] for h in horizons]

        ax1.errorbar(horizons, oracle_means,
                     yerr=[1.96 * s for s in oracle_ses],
                     marker='o', ls='-', color='#1f77b4', lw=2,
                     label='Oracle', capsize=4)

        for method, color, marker in [('Linearized', '#2ca02c', 's'),
                                       ('Logistic MLE', '#d62728', 'D')]:
            means = [sweep_results[(method, h)]['mean'] for h in horizons]
            ses = [sweep_results[(method, h)]['se'] for h in horizons]
            ax1.errorbar(horizons, means, yerr=[1.96 * s for s in ses],
                         marker=marker, ls='--', color=color, lw=2,
                         label=method, capsize=4)

        ax1.set_xlabel('Rounds per user', fontsize=12)
        ax1.set_ylabel('Final cumulative regret ($T$=20,000)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.set_xticks(horizons)

        # Right: ratio
        for method, color, marker in [('Linearized', '#2ca02c', 's'),
                                       ('Logistic MLE', '#d62728', 'D')]:
            ratios = [sweep_results[(method, h)]['mean'] /
                      sweep_results[('Oracle', h)]['mean']
                      for h in horizons]
            ax2.plot(horizons, ratios, marker=marker, ls='-', color=color,
                     lw=2, label=method)

        ax2.axhline(1.0, ls=':', color='gray', alpha=0.7)
        ax2.set_xlabel('Rounds per user', fontsize=12)
        ax2.set_ylabel('Progressive / Oracle regret ratio', fontsize=12)
        ax2.set_xticks(horizons)
        ax2.set_ylim(bottom=0.5)
        ax2.legend(fontsize=10)
        ax2.annotate('parity', xy=(horizons[-1], 1.05), fontsize=10,
                     color='gray', ha='right')

        fig.tight_layout()
        p4 = os.path.join(RESULTS_DIR, 'figure4_horizon_sweep.pdf')
        fig.savefig(p4, dpi=150, bbox_inches='tight')
        fig.savefig(p4.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        print(f"Figure 4 saved to {p4}")
        plt.close(fig)

    except ImportError:
        print("matplotlib not available — skipping plots")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()