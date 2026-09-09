"""
Verification script for `coupling_note.md`.

Subject: the 2-sparse confusability clique of a COUPLED linear row --- the
object that controls the PRE-TAMPER certificate budget, as opposed to the
1-sparse candidate lists that `interval_audit.py` measures AFTER damage.

Setting. One row of a linear head. The record holds
    code_i = Q( <w, h(x_i)> ),   i = 1..N
for a uniform quantiser of cell width W. Fix a column pair (j,k). A
perturbation (dj,dk) supported on {j,k} is INVISIBLE iff every sample stays
in the cell it was recorded in:

    a_i <= dj*h_j(x_i) + dk*h_k(x_i) < b_i      for all i,

with b_i = (upper cell edge - true logit) and a_i = b_i - W. Geometrically:
an intersection of N slabs in the plane, all containing the origin.

Why it is the right object. By Theorem B of `proof_and_validity.md`, two
originals need distinct certificates iff they differ in at most 2s
coordinates. With one change per row (s=1) that is: differ in at most 2
columns. Every invisible (dj,dk) therefore yields an original in the same
transcript fiber pairwise-confusable with every other, i.e. a CLIQUE. So

    L_row  >=  max_{j<k} log2 |invisible set on {j,k}|.

Run:  python3 coupling_verify.py
Requires numpy only.
"""

import numpy as np

W_CELL = 0.5        # quantiser cell width, matching the Llama record
LATTICE = 1e-3      # weight-alphabet spacing (uniform stand-in for BF16)
UNBOUNDED = 1e7     # stands in for "no magnitude cap"


# --------------------------------------------------------------------------
# exact convex-polygon machinery
# --------------------------------------------------------------------------

def clip(poly, nx, ny, c):
    """Keep {(x,y): nx*x + ny*y <= c}. Sutherland-Hodgman."""
    out = []
    M = len(poly)
    for i in range(M):
        p, q = poly[i], poly[(i + 1) % M]
        dp = nx * p[0] + ny * p[1] - c
        dq = nx * q[0] + ny * q[1] - c
        if dp <= 0:
            out.append(p)
        if (dp < 0 < dq) or (dq < 0 < dp):
            t = dp / (dp - dq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def region(hj, hk, a, b, R):
    """The invisible region as an explicit convex polygon."""
    poly = [(-R, -R), (R, -R), (R, R), (-R, R)]
    for hjj, hkk, ai, bi in zip(hj, hk, a, b):
        poly = clip(poly, hjj, hkk, bi)
        if not poly:
            return poly
        poly = clip(poly, -hjj, -hkk, -ai)
        if not poly:
            return poly
    return poly


def area(poly):
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def lattice_count(hj, hk, a, b, R, lat=LATTICE):
    """Exact lattice-point count, scanning only the polygon's x-extent."""
    poly = region(hj, hk, a, b, R)
    if len(poly) < 3:
        return max(1.0, float(len(poly)))
    xs = [p[0] for p in poly]
    k0, k1 = int(np.ceil(min(xs) / lat)), int(np.floor(max(xs) / lat))
    if k1 < k0:
        return 1.0
    dj = np.arange(k0, k1 + 1) * lat
    lo_r = a[None, :] - dj[:, None] * hj[None, :]
    hi_r = b[None, :] - dj[:, None] * hj[None, :]
    lo = np.full(dj.shape, -R)
    hi = np.full(dj.shape, R)
    pos, neg = hk > 0, hk < 0
    if pos.any():
        lo = np.maximum(lo, (lo_r[:, pos] / hk[pos]).max(axis=1))
        hi = np.minimum(hi, (hi_r[:, pos] / hk[pos]).min(axis=1))
    if neg.any():
        lo = np.maximum(lo, (hi_r[:, neg] / hk[neg]).max(axis=1))
        hi = np.minimum(hi, (lo_r[:, neg] / hk[neg]).min(axis=1))
    return float(np.maximum(0.0, np.floor(hi / lat) - np.ceil(lo / lat) + 1.0).sum())


def corr(r):
    return np.array([[1.0, r], [r, 1.0]])


def draw(N, rng, dist="gauss", df=3.0):
    """Unit-scale feature pairs; 't' gives genuinely heavy tails."""
    z = rng.standard_normal((N, 2))
    if dist == "t":
        z = z / np.sqrt(rng.chisquare(df, size=(N, 1)) / df)
        z /= np.sqrt(df / (df - 2))
    return z, rng.uniform(0.0, W_CELL, N)


def bits_from_area(hj, hk, a, b, R=UNBOUNDED, lat=LATTICE):
    return np.log2(max(area(region(hj, hk, a, b, R)) / lat ** 2, 1.0))


def one_sparse_bits(h, a, b, R=UNBOUNDED, lat=LATTICE):
    """Post-tamper single-column count: what interval_audit.py measures."""
    lo, hi = -R, R
    for hi_, a_, b_ in zip(h, a, b):
        if hi_ > 0:
            lo, hi = max(lo, a_ / hi_), min(hi, b_ / hi_)
        elif hi_ < 0:
            lo, hi = max(lo, b_ / hi_), min(hi, a_ / hi_)
    return np.log2(max(np.floor(hi / lat) - np.ceil(lo / lat) + 1.0, 1.0))


def fit_slope(xs, ys):
    A = np.vstack([np.log2(np.array(xs, float)), np.ones(len(xs))]).T
    return np.linalg.lstsq(A, np.array(ys), rcond=None)[0][0]


# --------------------------------------------------------------------------
# feature-Gram route: the O(d^2) computation that needs no simulation
# --------------------------------------------------------------------------

def gram(H):
    """H is (N,d) cached features. Returns energies and correlations."""
    G = H.T @ H / H.shape[0]
    e = np.diag(G).copy()
    s = np.sqrt(e)
    C = G / np.outer(s, s)
    np.fill_diagonal(C, 1.0)
    return e, C


def coupling_penalty(C):
    """-0.5 log2 (1 - rho^2), the pure coupling term, in bits."""
    with np.errstate(divide="ignore"):
        P = -0.5 * np.log2(np.clip(1.0 - C ** 2, 1e-300, None))
    np.fill_diagonal(P, -np.inf)
    return P


def n_star(e_j, e_k, rho, W=W_CELL, lat=LATTICE):
    """Samples needed to shrink the 2-sparse region below one lattice cell."""
    det = e_j * e_k * (1.0 - rho ** 2)
    return np.pi * W / (lat * det ** 0.25)


# --------------------------------------------------------------------------

def exp_identity(rng):
    print("=" * 76)
    print("1. AFFINE EQUIVARIANCE  (the exact result)")
    print("=" * 76)
    print("   <d, Lz> = <L^T d, z>  =>  P(Lz) = L^-T P(z)  =>  area scales by")
    print("   1/|det L|. So the whole dependence on feature scale/correlation")
    print("   is exactly -0.5*log2 det Sigma_jk. No distributional assumption.")
    print()
    print(f"   {'N':>5} {'dist':>6} {'areaZ/areaG':>15} {'|det L|':>13} {'rel err':>10}")
    worst = 0.0
    for t in range(8):
        N = int(rng.integers(8, 80))
        z, b = draw(N, rng, "t" if t % 2 else "gauss")
        a = b - W_CELL
        L = rng.standard_normal((2, 2))
        while abs(np.linalg.det(L)) < 0.05:
            L = rng.standard_normal((2, 2))
        g = z @ L.T
        aZ = area(region(z[:, 0], z[:, 1], a, b, UNBOUNDED))
        aG = area(region(g[:, 0], g[:, 1], a, b, UNBOUNDED))
        ratio, dl = aZ / aG, abs(np.linalg.det(L))
        err = abs(ratio - dl) / dl
        worst = max(worst, err)
        print(f"   {N:>5} {'t' if t%2 else 'gauss':>6} {ratio:>15.8f} "
              f"{dl:>13.8f} {err:>10.1e}")
    print(f"\n   worst relative error: {worst:.1e}  (machine precision)")


def exp_n_sweep(rng):
    print()
    print("=" * 76)
    print("2. THE PRE-TAMPER CLIQUE DECAYS AT TWICE THE POST-TAMPER RATE")
    print("=" * 76)
    Ns = [16, 32, 64, 128, 256, 512]
    for dist in ["gauss", "t"]:
        two, one = [], []
        for N in Ns:
            b2 = [];  b1 = []
            for _ in range(200):
                z, b = draw(N, rng, dist)
                a = b - W_CELL
                b2.append(bits_from_area(z[:, 0], z[:, 1], a, b))
                b1.append(one_sparse_bits(z[:, 0], a, b))
            two.append(np.mean(b2))
            one.append(np.mean(b1))
        print(f"\n   features = {dist}")
        print(f"   {'N':>6} {'2-sparse bits':>14} {'1-sparse bits':>14} {'ratio':>7}")
        for N, t2, t1 in zip(Ns, two, one):
            print(f"   {N:>6} {t2:>14.2f} {t1:>14.2f} {t2/t1:>7.2f}")
        print(f"   slope 2-sparse {fit_slope(Ns, two):+.3f} (theory -2)   "
              f"slope 1-sparse {fit_slope(Ns, one):+.3f} (theory -1)")


def exp_penalty(rng):
    print()
    print("=" * 76)
    print("3. THE COUPLING PENALTY IS EXACTLY -0.5*log2(1 - rho^2)")
    print("=" * 76)
    rs = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]
    for dist in ["gauss", "t"]:
        draws = [draw(64, rng, dist) for _ in range(200)]
        print(f"\n   features = {dist}, N = 64, 200 paired trials")
        print(f"   {'rho':>9} {'mean bits':>11} {'excess':>8} {'SE':>7} "
              f"{'theory':>8} {'err':>8}")
        base = None
        for r in rs:
            L = np.linalg.cholesky(corr(r))
            lg = np.array([bits_from_area(*(z @ L.T).T, b - W_CELL, b)
                           for z, b in draws])
            if base is None:
                base = lg.copy()
            exc = lg - base
            th = -0.5 * np.log2(1 - r ** 2)
            print(f"   {r:>9.4f} {lg.mean():>11.2f} {exc.mean():>8.2f} "
                  f"{exc.std(ddof=1)/np.sqrt(len(lg)):>7.4f} {th:>8.2f} "
                  f"{exc.mean()-th:>8.4f}")


def exp_lattice_caveat(rng):
    print()
    print("=" * 76)
    print("4. CAVEAT: the identity is exact for AREA, approximate for COUNTS")
    print("=" * 76)
    print("   L^-T does not map the weight lattice to itself. The continuum")
    print("   step is fine while counts are large and degrades exactly where")
    print("   the interesting regime is -- a handful of candidates at large N.")
    print()
    print(f"   {'N':>6} {'area bits':>11} {'count bits':>12} {'diff':>8}")
    for N in [16, 64, 256, 1024, 4096]:
        da, dc = [], []
        for _ in range(60):
            z, b = draw(N, rng)
            a = b - W_CELL
            da.append(bits_from_area(z[:, 0], z[:, 1], a, b))
            dc.append(np.log2(max(lattice_count(z[:, 0], z[:, 1], a, b, 1e3), 1)))
        print(f"   {N:>6} {np.mean(da):>11.3f} {np.mean(dc):>12.3f} "
              f"{np.mean(dc)-np.mean(da):>8.3f}")


def exp_gram(rng):
    print()
    print("=" * 76)
    print("5. THE O(d^2) ROUTE: worst pair straight from the feature Gram")
    print("=" * 76)
    print("   penalty(j,k) = -0.5*log2(e_j e_k (1-rho_jk^2)); no simulation.")
    print()
    print("   (a) independent features -- is the worst of millions of pairs")
    print("       large just from finite-sample noise?")
    print(f"   {'d':>6} {'N':>6} {'pairs':>10} {'max|rho|':>9} {'max penalty':>12}")
    for d in [512, 2048]:
        for N in [256, 2048]:
            H = rng.standard_normal((N, d))
            _, C = gram(H)
            P = coupling_penalty(C)
            iu = np.triu_indices(d, 1)
            print(f"   {d:>6} {N:>6} {len(iu[0]):>10} {np.abs(C[iu]).max():>9.4f} "
                  f"{P[iu].max():>12.3f}")
    print("   -> no. Coupling needs REAL correlation, not just many pairs.")

    print()
    print("   (b) block-correlated features -- where does the worst pair live?")
    d, N, nb = 512, 2048, 16
    m = d // nb
    blk = np.arange(d) // m
    iu = np.triu_indices(d, 1)
    same = (blk[:, None] == blk[None, :])[iu]
    print(f"   {'within rho':>11} {'worst within':>14} {'worst across':>14}")
    for wr in [0.3, 0.7, 0.9, 0.98, 0.999]:
        H = np.empty((N, d))
        for bi in range(nb):
            sh = rng.standard_normal((N, 1))
            H[:, bi*m:(bi+1)*m] = (np.sqrt(wr) * sh
                                   + np.sqrt(1 - wr) * rng.standard_normal((N, m)))
        _, C = gram(H)
        P = coupling_penalty(C)[iu]
        print(f"   {wr:>11.3f} {P[same].max():>14.2f} {P[~same].max():>14.2f}")
    print(f"   -> coupling is confined to blocks (here of size {m}).")


def exp_threshold():
    print()
    print("=" * 76)
    print("6. ZERO-LOCATION-BITS THRESHOLD")
    print("=" * 76)
    print("   The decoder gets the column for free exactly when the 2-sparse")
    print("   region holds no lattice point but the origin:")
    print()
    print("       N* = pi*W / ( D * (e_j e_k (1-rho^2))^(1/4) )")
    print()
    print("   Sample complexity to kill 2-sparse ambiguity grows as")
    print("   (1-rho^2)^(-1/4) -- a WEAK dependence. That is the good news.")
    print()
    print("   Illustrative, with W=0.5 and a BF16 step D=1e-4 near |w|~0.02;")
    print("   e is the median feature energy reported in RESULTS_REPORT.md 7.1.")
    print(f"   {'e':>7} {'rho':>8} {'N*':>10} {'bits/pair at N=2048':>21}")
    for e in [10.52, 2.82]:
        for r in [0.0, 0.9, 0.99, 0.999]:
            ns = n_star(e, e, r, lat=1e-4)
            at = max(0.0, 2 * np.log2(ns / 2048.0))
            print(f"   {e:>7.2f} {r:>8.3f} {ns:>10.0f} {at:>21.2f}")
    print()
    print("   These use a stand-in lattice step, NOT measured BF16 spacing in")
    print("   the real head. Treat as an order of magnitude, not a result.")


if __name__ == "__main__":
    rng = np.random.default_rng(20260909)
    exp_identity(rng)
    exp_n_sweep(rng)
    exp_penalty(rng)
    exp_lattice_caveat(rng)
    exp_gram(rng)
    exp_threshold()
    print()
    print("=" * 76)
    print("NOT ESTABLISHED HERE: any upper bound. This makes the pre-tamper")
    print("LOWER bound computable and names its structure. It does not")
    print("exhibit an encoder that spends only that much.")
    print("=" * 76)
