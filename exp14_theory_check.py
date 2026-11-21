"""EXP14 — numerical verification of Proposition 1 / 1' in THEORY_NOTES.md.

Linear SCM: x = [c; s], y = beta_c . c + eps (Var(eps)=s2e), s = rho_e*gamma*y + eta
(Var(eta)=s2n). Predictor w in R^{K+1}, objective (population V-REx)
    J(w) = sum_e MSE_e(w) + lam * Var_e(MSE_e(w)).

Theory (Prop 1'): at delta = w_c - beta small, the first-order escape from
w* = (beta, 0) is
    w_s* ~ sum(rho_e) / (4 * lam * gamma * s2e * Var_env(rho))
so (i) finite lam leaves a residual shortcut leak proportional to the training
rho-imbalance sum(rho_e), vanishing as 1/lam; (ii) with sum(rho_e)=0 (symmetric
2-env flip) the first-order leak vanishes entirely. OOD flip change
    dR = R(w; rho_test=-0.9) - R(w; rho_bar_train)
       ~ -2*gamma*w_s*s2e*(rho_test - rho_bar)
         + w_s^2*gamma^2*Var(y)*(rho_test^2 - rho_bar^2).

We minimize the ANALYTIC population objective (exact 5x5 covariance matrices,
deterministic) and cross-check one config with the sample-based objective.
"""

import math

import torch

K = 4
S2E = 0.25          # Var(eps)
S2N = 0.04          # Var(eta)
GAMMA = 1.0
N = 400_000         # samples per env (finite-sample cross-check only)
SEEDS = (0, 1, 2)

BETA = torch.tensor([1.0, 0.8, -1.2, 0.6])
STD_C = 0.5 + torch.rand(K, generator=torch.Generator().manual_seed(7))
VAR_C = STD_C ** 2
VAR_Y = float((BETA ** 2 * VAR_C).sum()) + S2E


def env_moments(rho):
    q = torch.cat([BETA * VAR_C, torch.tensor([rho * GAMMA * VAR_Y])])
    S = torch.zeros(K + 1, K + 1)
    S[:K, :K] = torch.diag(VAR_C)
    S[:K, K] = rho * GAMMA * BETA * VAR_C
    S[K, :K] = S[:K, K]
    S[K, K] = rho ** 2 * GAMMA ** 2 * VAR_Y + S2N
    return S, q


def analytic_ws_star(env_rhos, lam):
    ws = []
    keep = []
    for init in range(8):
        g = torch.Generator().manual_seed(init)
        w = (torch.randn(K + 1, generator=g) * 0.1)
        w[:K] += BETA
        w = w.requires_grad_(True)
        opt = torch.optim.LBFGS([w], max_iter=2000, tolerance_grad=1e-14,
                                tolerance_change=1e-18, history_size=100,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            ls = torch.stack([VAR_Y - 2 * w @ q + w @ S @ w
                              for S, q in (env_moments(r) for r in env_rhos)])
            (ls.sum() + lam * ls.var(unbiased=False)).backward()
            return ls.sum() + lam * ls.var(unbiased=False)

        opt.step(closure)
        obj = float(closure())
        keep.append((obj, w.detach()))
    def j(w):
        ls = torch.stack([VAR_Y - 2 * w @ q + w @ S @ w
                          for S, q in (env_moments(r) for r in env_rhos)])
        return float(ls.sum() + lam * ls.var(unbiased=False))
    return min((w for _, w in keep), key=j)


def predict_ws(env_rhos, lam):
    r = torch.tensor(env_rhos, dtype=torch.float64)
    rho_bar = r.mean()
    var_rho = float(((r - rho_bar) ** 2).mean())
    return float(r.sum()) / (4 * lam * GAMMA * S2E * var_rho)


def dR_closed(w, env_rhos, rho_test):
    w_s = float(w[-1])
    r = torch.tensor(env_rhos, dtype=torch.float64)
    rho_bar = float(r.mean())
    delta = w[:K] - BETA
    lin = -2 * GAMMA * w_s * (S2E + float(delta @ (BETA * VAR_C))) * (rho_test - rho_bar)
    quad = w_s ** 2 * GAMMA ** 2 * VAR_Y * (rho_test ** 2 - rho_bar ** 2)
    return lin + quad


def exact_risks(w, env_rhos, rho_test):
    out = {}
    for tag, r in [("bar", float(torch.tensor(env_rhos).mean())), ("flip", rho_test)]:
        S, q = env_moments(r)
        out[tag] = float(VAR_Y - 2 * w @ q + w @ S @ w)
    return out


def sample_fit_w_s(env_rhos, lam, seed):
    xs, ys = [], []
    for i, r in enumerate(env_rhos):
        g = torch.Generator().manual_seed(10_000 + 1_000 * seed + i)
        c = torch.randn(N, K, generator=g) * STD_C
        eps = math.sqrt(S2E) * torch.randn(N, generator=g)
        y = c @ BETA + eps
        eta = math.sqrt(S2N) * torch.randn(N, generator=g)
        s_mat = (r * GAMMA * y + eta)[:, None]
        xs.append(torch.cat([c, s_mat], dim=1))
        ys.append(y)
    w = torch.zeros(K + 1, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=2000, tolerance_grad=1e-12,
                            tolerance_change=1e-16, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        ls = torch.stack([(b - a @ w).pow(2).mean() for a, b in zip(xs, ys)])
        (ls.sum() + lam * ls.var(unbiased=False)).backward()
        return ls.sum() + lam * ls.var(unbiased=False)

    opt.step(closure)
    return w.detach()


def show(tag, env_rhos, lam, w_s, dr_pred, dr_meas):
    pred = predict_ws(env_rhos, lam)
    print(f"{tag:<28} lam={lam:<5} sumRho={sum(env_rhos):+.2f} | "
          f"w_s pred={pred:.4f} exact={w_s:.4f} | dR(pred)={dr_pred:+.4f} dR(exact)={dr_meas:+.4f}")


if __name__ == "__main__":
    print(f"VAR_Y={VAR_Y:.4f}  noise floor R_id={S2E}\n")
    cases = [
        ("A symmetric 2-env", [0.9, -0.9], 200),
        ("B diverse 3-env", [0.9, 0.3, 0.0], 200),
        ("C diverse high-rho", [0.9, 0.6, 0.2], 200),
    ]
    for name, rhos, lam in cases:
        w = analytic_ws_star(rhos, lam)
        rr = exact_risks(w, rhos, -0.9)
        show(name, rhos, lam, float(w[-1]), dR_closed(w, rhos, -0.9),
             rr["flip"] - rr["bar"])

    print("\n== 1/lam scaling, rho=[0.9,0.3,0.0] (exact population) ==")
    for lam in (50, 100, 200, 400, 800, 1600):
        w = analytic_ws_star([0.9, 0.3, 0.0], lam)
        rr = exact_risks(w, [0.9, 0.3, 0.0], -0.9)
        ws = float(w[-1])
        print(f"lam={lam:<5} w_s pred={predict_ws([0.9, 0.3, 0.0], lam):.4f} "
              f"exact={ws:.4f} w_s*lam={ws * lam:.3f} "
              f"dR(exact)={rr['flip'] - rr['bar']:+.4f}")

    print("\n== finite-sample cross-check (rho=[0.9,0.3,0.0], lam=200) ==")
    for s in SEEDS:
        w = sample_fit_w_s([0.9, 0.3, 0.0], 200, s)
        print(f"seed {s}: w_s sample={float(w[-1]):+.4f} (exact analytic 4-env ref)")
