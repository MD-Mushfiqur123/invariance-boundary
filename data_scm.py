"""Kill-experiment data: synthetic SCM with causal latents, spurious latents, nonlinear mixing.

Ground truth:
  z_c (4 dims)  : causal variables. y depends ONLY on z_c.
  z_s (2 dims)  : spurious variables. Correlated with y, correlation flips across environments.
  z_n (16 dims) : pure nuisance noise.
  x = g(z_c, z_s, z_n) via a fixed random nonlinear mixing network, observed dim = 64.

Environments:
  e0 (train): corr(spurious, label) = +0.9   <- shortcut available
  e1 (train): corr(spurious, label) = +0.3   <- partial shortcut
  e_test_id  (held out from e0 distribution, rho=+0.9)
  e_test_ood (rho = -0.9, flipped shortcut)
"""
import numpy as np

N_CAUSAL, N_SPURIOUS, N_NUISANCE = 4, 2, 16
N_LATENT = N_CAUSAL + N_SPURIOUS + N_NUISANCE
OBS_DIM = 64

ENV_RHO = {"e0": 0.9, "e1": 0.3, "test_id": 0.9, "test_ood": -0.9}


class MixingNetwork:
    """Fixed random 2-layer nonlinear mixing g: R^22 -> R^64 (not trained)."""

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        W1 = rng.normal(0, 1.0 / np.sqrt(N_LATENT), size=(N_LATENT, 32))
        W2 = rng.normal(0, 1.0 / np.sqrt(32), size=(32, OBS_DIM))
        W2b = rng.normal(0, 1.0 / np.sqrt(32), size=(32, OBS_DIM))
        self.W1, self.W2, self.W2b = W1.astype(np.float32), W2.astype(np.float32), W2b.astype(np.float32)

    def __call__(self, z):
        t = np.tanh(z @ self.W1)
        return t @ self.W2 + 0.05 * (t * t) @ self.W2b


def sample_latents(rho, n, rng):
    z_c = rng.normal(0, 1, size=(n, N_CAUSAL))
    logits = z_c[:, 0] * z_c[:, 1] + 0.8 * z_c[:, 2] - 1.0 * z_c[:, 3] + 0.3 * rng.normal(0, 1, n)
    y = (logits > 0).astype(np.float64)
    t = 2 * y - 1
    z_s1 = rho * 1.2 * t + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)
    z_s2 = 0.5 * rho * t + 0.3 * rng.normal(0, 1, n)
    z_n = rng.normal(0, 1, size=(n, N_NUISANCE))
    return np.concatenate([z_c, np.stack([z_s1, z_s2], 1), z_n], 1), y


def make_env(mix, rho, n, seed):
    rng = np.random.default_rng(seed)
    z, y = sample_latents(rho, n, rng)
    x = mix(z).astype(np.float32)
    return x, y.astype(np.int64), z


def get_data(n_train=8192, n_test=4096):
    mix = MixingNetwork()
    data = {}
    for k, rho in ENV_RHO.items():
        n = n_train if k in ("e0", "e1") else n_test
        data[k] = make_env(mix, rho, n, seed=hash(k) % 10_000)
    return data, mix


if __name__ == "__main__":
    data, _ = get_data()
    for k, (x, y, z) in data.items():
        corr = np.corrcoef(z[:, N_CAUSAL], y)[0, 1]
        print(f"{k:9s} x:{x.shape} y:{y.shape} spurious-label corr: {corr:+.2f}")
