"""
Distances between sets of embeddings, their uncertainty and their calibration.

Notation used in this module:

    R, G     real and generated embeddings, one row per image [N, D]
    Rl, Gl   local embeddings grouped by image [N, 16, D]
    units    independent unit of each real image (original micrograph);
             images derived from the same unit by augmentation or
             overlapping crops share the same id
    tr       trace of the covariance of the real embeddings, used to make
             the Frechet distance independent of the scale of each space

Metrics:

    nFID     Frechet distance divided by tr (normalized FID)
    MMD2     unbiased squared MMD with a Gaussian (RBF) kernel. The bandwidth
             sigma is the median distance between real embeddings, so the
             same kernel is used for every generator. Negative values are kept.
    KID      unbiased squared MMD with the cubic kernel (x.y / D + 1)^3
    P, R     k-NN precision and recall (Kynkaanniemi et al., 2019), k = 3

Pairs of real images from the same unit are excluded from the within set
terms of MMD2 and KID, and from the k-NN radii, because they are not
independent observations.
"""

import numpy as np

EPS = 1e-6      # regularization added to the covariance diagonal


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def covariance(X):
    """Sample covariance with a small regularization for numerical stability."""
    return np.cov(X, rowvar=False) + EPS * np.eye(X.shape[1])


def fid(X, Y):
    """
    Frechet distance between Gaussians fitted to X and Y:

        |mx - my|^2 + tr(Sx) + tr(Sy) - 2 tr((Sx Sy)^(1/2))

    The last trace is computed from the eigenvalues of Sx^(1/2) Sy Sx^(1/2),
    which is symmetric, so the result has no imaginary part.
    """
    mx, my = X.mean(0), Y.mean(0)
    Sx, Sy = covariance(X), covariance(Y)
    w, V = np.linalg.eigh(Sx)
    sqrt_Sx = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh(sqrt_Sx @ Sy @ sqrt_Sx), 0, None)).sum()
    d = mx - my
    return float(d @ d + np.trace(Sx) + np.trace(Sy) - 2 * tr_sqrt)


def sq_dist(A, B):
    """Squared Euclidean distances between the rows of A and the rows of B."""
    D = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
    return np.clip(D, 0, None)


def median_sigma(Z, max_n=1000, seed=0):
    """Median heuristic for the kernel bandwidth, from real embeddings only."""
    rng = np.random.default_rng(seed)
    if len(Z) > max_n:
        Z = Z[rng.choice(len(Z), max_n, replace=False)]
    D = np.sqrt(sq_dist(Z, Z))
    return float(np.median(D[np.triu_indices(len(Z), 1)]))


def within_mean(K, ids):
    """Mean of a kernel matrix over pairs that come from different units."""
    return K[ids[:, None] != ids[None, :]].mean()


def mmd2(X, Y, sigma, idx, idy):
    """Unbiased squared MMD with a Gaussian kernel of bandwidth sigma."""
    gamma = 1.0 / (2 * sigma ** 2)
    k = lambda A, B: np.exp(-gamma * sq_dist(A, B))
    return float(within_mean(k(X, X), idx) + within_mean(k(Y, Y), idy) - 2 * k(X, Y).mean())


def kid(X, Y, idx, idy):
    """Unbiased squared MMD with the cubic polynomial kernel of KID."""
    d = X.shape[1]
    k = lambda A, B: (A @ B.T / d + 1) ** 3
    return float(within_mean(k(X, X), idx) + within_mean(k(Y, Y), idy) - 2 * k(X, Y).mean())


def precision_recall(R, G, ids_r, k=3):
    """
    Precision: fraction of generated samples inside the k-NN ball of some
    real sample. Recall: fraction of real samples inside the k-NN ball of
    some generated sample. Neighbors from the same unit do not count for
    the radius of a real sample.
    """
    def radii(Z, ids):
        D = np.sqrt(sq_dist(Z, Z))
        D[ids[:, None] == ids[None, :]] = np.inf        # also removes the diagonal
        return np.sort(D, 1)[:, k - 1]

    r_real, r_gen = radii(R, ids_r), radii(G, np.arange(len(G)))
    D = np.sqrt(sq_dist(G, R))                          # [n_gen, n_real]
    precision = float(np.mean((D <= r_real[None, :]).any(1)))
    recall = float(np.mean((D.T <= r_gen[None, :]).any(1)))
    return precision, recall


def subsample(Z, ids, n, rng):
    """Random subset of n rows, used to keep the local MMD affordable."""
    if len(Z) <= n:
        return Z, ids
    s = rng.choice(len(Z), n, replace=False)
    return Z[s], ids[s]


# ---------------------------------------------------------------------------
# Scores of one generator
# ---------------------------------------------------------------------------

def scores(R, G, Rl, Gl, units, sig, tr, n_local=2000, seed=0):
    """
    All scores of one generator at both scales.

    sig, tr   dicts with the bandwidth and the covariance trace of the real
              embeddings for the global ("g") and local ("l") scales
    n_local   number of patch embeddings used for the local MMD2
    """
    rng = np.random.default_rng(seed)
    N, K, D = Rl.shape
    M = len(G)
    ids_g = np.arange(M)                                # each generated image is its own unit
    rl_all, gl_all = Rl.reshape(-1, D), Gl.reshape(-1, D)
    rl, ids_rl = subsample(rl_all, np.repeat(units, K), n_local, rng)
    gl, ids_gl = subsample(gl_all, np.repeat(ids_g, K), n_local, rng)

    out = {
        "nFID_g": fid(R, G) / tr["g"],
        "nFID_l": fid(rl_all, gl_all) / tr["l"],
        "MMD2_g": mmd2(R, G, sig["g"], units, ids_g),
        "MMD2_l": mmd2(rl, gl, sig["l"], ids_rl, ids_gl),
        "KID_g": kid(R, G, units, ids_g),
    }
    # Sensitivity of the global MMD2 to the bandwidth
    for f in (0.5, 2.0):
        out[f"MMD2_g_sigma_x{f}"] = mmd2(R, G, sig["g"] * f, units, ids_g)
    out["P_g"], out["R_g"] = precision_recall(R, G, units)
    out["P_l"], out["R_l"] = precision_recall(rl, gl, ids_rl)
    return out


# ---------------------------------------------------------------------------
# Uncertainty and calibration
# ---------------------------------------------------------------------------

def bootstrap(R, G, Rl, Gl, units, sig, tr, B=200, n_local=1000, seed=0):
    """
    Finite sample uncertainty of nFID and MMD2 at both scales.

    Each replicate resamples with replacement the real units, with all
    their images, and the generated images. The 16 patches of an image
    always stay together. Returns a dict of arrays with B values each.
    """
    rng = np.random.default_rng(seed)
    N, K, D = Rl.shape
    M = len(G)
    groups = [np.where(units == u)[0] for u in np.unique(units)]
    keys = ["nFID_g", "nFID_l", "MMD2_g", "MMD2_l"]
    out = {k: np.empty(B) for k in keys}
    for b in range(B):
        ir = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        ig = rng.integers(0, M, M)
        rl, gl = Rl[ir].reshape(-1, D), Gl[ig].reshape(-1, D)
        out["nFID_g"][b] = fid(R[ir], G[ig]) / tr["g"]
        out["nFID_l"][b] = fid(rl, gl) / tr["l"]
        # Repeated draws of the same image share the id, so they are excluded too
        out["MMD2_g"][b] = mmd2(R[ir], G[ig], sig["g"], units[ir], ig)
        a, ia = subsample(rl, np.repeat(units[ir], K), n_local, rng)
        c, ic = subsample(gl, np.repeat(ig, K), n_local, rng)
        out["MMD2_l"][b] = mmd2(a, c, sig["l"], ia, ic)
    return out


def real_halves(units, n_splits=10, seed=0):
    """Random splits of the real images into two halves of units."""
    rng = np.random.default_rng(seed)
    u = np.unique(units)
    for _ in range(n_splits):
        p = rng.permutation(u)
        yield np.isin(units, p[: len(u) // 2])


def reference(R, Rl, units, tr, seed=0):
    """
    Real vs real reference: nFID between two halves of the real images,
    averaged over 10 splits. A generator cannot be told apart from real
    data when its nFID at the same sample size is close to this value.
    """
    D = Rl.shape[-1]
    g, l = [], []
    for A in real_halves(units, seed=seed):
        g.append(fid(R[A], R[~A]) / tr["g"])
        l.append(fid(Rl[A].reshape(-1, D), Rl[~A].reshape(-1, D)) / tr["l"])
    return {"g": float(np.mean(g)), "l": float(np.mean(l))}


def half_vs_generated(R, G, Rl, Gl, units, tr, seed=0):
    """
    nFID between one real half and as many generated images as the other
    half has. It uses the same splits as reference(), so both values have
    the same sample size and the same finite sample bias.
    """
    rng = np.random.default_rng(seed + 1)
    D = Rl.shape[-1]
    g, l = [], []
    for A in real_halves(units, seed=seed):
        j = rng.choice(len(G), min(len(G), int((~A).sum())), replace=False)
        g.append(fid(R[A], G[j]) / tr["g"])
        l.append(fid(Rl[A].reshape(-1, D), Gl[j].reshape(-1, D)) / tr["l"])
    return {"g": float(np.mean(g)), "l": float(np.mean(l))}
