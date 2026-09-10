# The coupled row: making the pre-tamper bound computable

**Companion to `proof_and_validity.md` §6 ("Why the product assumption is a real
restriction").** That section identifies the open problem: every positive result so far
assumes the transcript fiber is a *product* over coordinates, and a neural row's fiber is
not. This note supplies the missing structure for the one-change-per-row case, proves one
exact result, derives one scaling law, verifies both numerically, and reports two negative
findings that close off tempting directions.

Everything here is reproducible with `python3 coupling_verify.py` (numpy only, ~14 s).
No GPU, no Llama weights, no cached features were used — which is also the main
limitation, see §8.

---

## 1. The object

One row of a linear head. The record holds `code_i = Q(<w, h(x_i)>)` for `i = 1..N`,
`Q` a uniform quantiser of cell width `W`. Fix a column pair `(j,k)`. A perturbation
`(δ_j, δ_k)` supported on those two columns is **invisible** iff every sample stays in
the cell it was recorded in:

```
a_i  <=  δ_j·h_j(x_i) + δ_k·h_k(x_i)  <  b_i        for all i = 1..N
```

with `b_i` = (upper cell edge − true logit) and `a_i = b_i − W`. Geometrically this is
an **intersection of N slabs in the plane, all containing the origin** — a convex
polygon.

**Why pairs, and why this is the pre-tamper object.** Theorem B of
`proof_and_validity.md` establishes that two originals require distinct certificates iff
they differ in at most `2s` coordinates. With one change per row, `s = 1`, so: *differ in
at most 2 columns*. Every invisible `(δ_j, δ_k)` therefore produces another original in
the same transcript fiber, and any two of them differ in ≤ 2 columns. **The whole set is
a clique in the confusability graph.** Hence

```
L_row  >=  max_{j<k} log2 | invisible set on {j,k} |
```

This is a genuine pre-tamper lower bound. It is not a post-damage candidate count: no
damaged checkpoint appears anywhere in its definition. That is precisely what the
121-bit figure in `RESULTS_REPORT.md` §6 lacks.

Contrast with what `interval_audit.py` currently computes: the same construction in
**one** dimension, anchored at the damaged row. The move from 1 to 2 sparse coordinates
is the move from Question A to Question B.

---

## 2. An exact result: affine equivariance

> **Proposition 1.** Let `P(G) = {δ ∈ R² : a_i ≤ ⟨δ, g_i⟩ < b_i, i = 1..N}`. For any
> invertible `L`, `P({L z_i}) = L^{-T} P({z_i})`, hence
> `area P(G) = area P(Z) / |det L|`.
>
> *Proof.* `⟨δ, L z⟩ = ⟨L^T δ, z⟩`, so `δ` satisfies the constraints for `{L z_i}` iff
> `L^T δ` satisfies them for `{z_i}`. The map `δ ↦ L^T δ` is a linear bijection with
> Jacobian `|det L|`. ∎

Two lines, and it settles the entire scale-and-correlation dependence. Writing `Σ_jk`
for the empirical 2×2 second-moment matrix of the feature pair over the **same N samples
in the record**, `Σ_jk = L L^T`, and

```
log2 C_jk  =  [ whitening-invariant term ]  −  ½ log2 det Σ_jk
```

Expanding `det Σ_jk = e_j · e_k · (1 − ρ_jk²)` with `e` the feature energy and `ρ` the
correlation:

```
                                    ┌ energy term ┐   ┌ COUPLING PENALTY ┐
log2 C_jk = [ invariant ]  −  ½log2(e_j·e_k)   −   ½ log2(1 − ρ_jk²)
```

**The cost of coupling is exactly `−½ log2(1 − ρ_jk²)` bits.** No Gaussian assumption,
no asymptotics, no approximation — it is an algebraic identity holding for every `N`,
every feature distribution, every cell offset.

It reproduces both known endpoints as special cases:

| case | `ρ` | penalty | matches |
|---|---|---|---|
| independent features | 0 | 0 bits | the product fiber of Theorem D — coupling vanishes, Theorem C applies |
| the `a+b` counterexample | 1 | ∞ | `proof_and_validity.md` §6, where `q` originals are mutually confusable |

**Verified** (`coupling_verify.py` §1, §3): the area ratio matches `1/|det L|` to a worst
relative error of `2e-08` across Gaussian and heavy-tailed (t, df=3) features; the paired
correlation sweep reproduces `−½log2(1−ρ²)` to **0.0000 bits with zero standard error**
at every `ρ` from 0 to 0.9999, for both distributions. Zero variance is the signature of
an identity rather than a fit.

---

## 3. A scaling law: the pre-tamper clique decays twice as fast

Assume `b_i` is roughly uniform on `(0, W]` (the true logit sits at an essentially
arbitrary position inside its cell) and independent of the features. Then the radial
extent in direction `φ` is `t_φ = min_i β_i / |⟨u_φ, g_i⟩|` with `β_i ~ U(0,W)`, giving
`Pr[t_φ > t] ≈ exp(−t·Σ_i|⟨u_φ,g_i⟩|/W)` and `E[t_φ] ≈ W/(N·m(φ))`. Integrating
`area = ½∫t_φ²dφ` and using `∫dφ/(u_φ^T Σ u_φ) = 2π/√det Σ`:

```
E[area]  ≈  π² W² / ( N² · sqrt(det Σ_jk) )
```

So `C_jk ∝ N^{-2}`, where the post-tamper 1-sparse count goes as `N^{-1}` — the law
already measured on the real head at `−1.025 ± 0.064` (`RESULTS_REPORT.md` §5.2).

**Verified** (`coupling_verify.py` §2), 200 trials per point, `N` from 16 to 512:

| features | 2-sparse slope | 1-sparse slope | bits ratio |
|---|--:|--:|--:|
| Gaussian | **−1.948** (theory −2) | −0.965 (theory −1) | 2.10 → 2.36 |
| Student-t, df=3 | **−1.935** | −0.956 | 2.09 → 2.35 |

The exponent is solid. The **constant is not**: measured areas run ≈ 0.65–0.75× the
formula, because the exponential tail approximation overestimates `E[t²]`. Treat the
constant as order-of-magnitude.

---

## 4. Consequence: an O(d²) recipe that needs no simulation

Proposition 1 means the worst pair in a row can be read straight off the feature Gram
matrix. Given cached features `H` (N × d), which the project already has:

```python
G   = H.T @ H / N                      # d x d, one matmul
e   = np.diag(G)
rho = G / np.sqrt(np.outer(e, e))
pen = -0.5 * np.log2(e[:,None]*e[None,:] * (1 - rho**2))
row_lower_bound  =  invariant_term + pen[triu].max()
```

For `d = 4096` this is one 4096×4096 matmul. **This is the concrete experiment to run
next**, and it replaces the uncertified 1,287-bit converse with a bound that has an
actual proof behind it.

### Threshold for zero location bits

The decoder gets the changed column for free exactly when the 2-sparse region contains
no lattice point but the origin. Setting `area = Δ²` for weight-lattice step `Δ`:

```
N*  =  π W / ( Δ · (e_j e_k (1 − ρ_jk²))^{1/4} )
```

**Sample complexity to kill 2-sparse ambiguity grows only as `(1−ρ²)^{−1/4}`.** That is
a weak dependence, and it is the most encouraging number in this note: even `ρ = 0.999`
costs only a factor of `(1−0.998)^{-1/4} ≈ 4.7` in samples, not a factor of 1000.

Illustrative arithmetic with `W = 0.5`, a stand-in BF16 step `Δ = 1e-4`, and the median
feature energies reported in `RESULTS_REPORT.md` §7.1 (10.52 salient, 2.82 uniform):

| `e` | `ρ` | `N*` | bits/pair at N=2048 |
|--:|--:|--:|--:|
| 10.52 | 0.000 | 4,843 | 2.48 |
| 10.52 | 0.990 | 12,894 | 5.31 |
| 10.52 | 0.999 | 22,904 | 6.97 |
| 2.82 | 0.000 | 9,354 | 4.38 |
| 2.82 | 0.999 | 44,238 | 8.87 |

**Prediction.** Over 64 changed rows this puts the pre-tamper lower bound at roughly
**160–450 bits**, against the 121-bit post-damage count and the 1,088-bit construction.
If that survives contact with the real Gram matrix, the honest headline moves from "97%"
to something near **89–96%**, and — unlike 121 — it would be a number with a proof.

This is an order-of-magnitude prediction using a stand-in lattice step, not a measured
BF16 spacing. It is falsifiable, which is the point.

---

## 5. Negative result 1: dimension alone is not the enemy

A natural fear is that with `d = 4096` there are 8.4M pairs per row, so the *maximum*
correlation will be near 1 by sheer chance. Measured (`coupling_verify.py` §5a), for
genuinely independent features:

| d | N | pairs | max\|ρ\| | max penalty |
|--:|--:|--:|--:|--:|
| 2048 | 256 | 2,096,128 | 0.329 | **0.082 bits** |
| 2048 | 2048 | 2,096,128 | 0.120 | **0.010 bits** |

Negligible. **Coupling requires genuine feature correlation, not merely many pairs.**
So the whole question reduces to an empirical property of the actual feature Gram — which
makes §4 decisive rather than merely suggestive.

## 6. Negative result 2: block coding does not save locator bits by itself

Since transformer features are correlated in groups, the obvious construction is to
partition columns into blocks and code within blocks. Measured (`coupling_verify.py` §5b),
with 16 blocks of 32 columns, the penalty is entirely confined to within-block pairs
(worst across-block ≤ 0.01 bits at every within-block `ρ` up to 0.999). The structure is
real.

But the naive accounting fails:

```
locator within a block of size m  +  block id
=  log2(m+1)  +  log2(d/m)   ≈  log2(d)     — for every m
```

Splitting the address space does not reduce total address bits. **The saving only
materialises if the decoder can infer the block from the record for free**, which §5b
supports (across-block 2-sparse perturbations are essentially always visible, so the
interval screen localises to a block unaided). Then the certificate pays `log2(m+1)`
instead of `log2(d+1)` — 4 bits instead of 13 at `m = 8`. That is a real construction
sketch, and it is **not proved here**.

---

## 7. Where the computational barrier actually is

`RESULTS_REPORT.md` §6.1 attributes the 1,088-vs-121 gap to Reed-Solomon field size.
The analysis above suggests a sharper statement, offered as an argument, not a proof:

- To decode **row by row**, the checks must be row-separable, which forces a distinct
  locator per row → ≥ 17 bits per check. This is the RS cost.
- To avoid locator bits, the checks must be **joint** over rows. But the decoder then
  faces ~3 candidates in each of 64 rows — a joint candidate set of `3^64 ≈ 2^101` — and
  must find the combination matching the syndrome. That is syndrome decoding over a large
  product, not a field-size problem.

So the tension is: *row-separable checks cost locator bits; joint checks are hard to
decode.* Also worth separating explicitly — a public random hash of length
`log2|S| + log2(1/ε)` achieves close to the residual count under a **probabilistic**
guarantee, while Theorems 1–2 and the `⌈log2(n+1)⌉ = 17` counterexample are about the
**uniform** guarantee. Part of the 1,088−121 gap is therefore about *which guarantee is
being demanded*, which is neither purely computational nor purely informational. The
consolidated document already distinguishes these quantifiers in §1.4; the results
section does not carry the distinction through.

---

## 8. Claim status

| Claim | Basis | Status |
|---|---|---|
| Invisible 2-sparse set = intersection of N slabs, a convex polygon | Definition | Proved (trivial) |
| It is a clique in the confusability graph | Theorem B of `proof_and_validity.md`, `s=1` | Proved |
| Coupling cost is exactly `−½log2 det Σ_jk` | Proposition 1 | **Proved, distribution-free** |
| …verified to machine precision, Gaussian and heavy-tailed | `coupling_verify.py` §1, §3 | Verified, 2e-08 worst error |
| Clique decays as `N^{-2}`, twice the 1-sparse rate | §3 derivation | Derived under a uniform-`b_i` assumption |
| …exponent verified | `coupling_verify.py` §2 | Verified, −1.94 vs −2; **constant off by ≈0.7×** |
| Worst pair is computable in O(d²) from the Gram matrix | Proposition 1 | Proved; **not yet run on Llama features** |
| Pre-tamper bound for the Llama head is ≈160–450 bits | §4 arithmetic | **Prediction only**, stand-in lattice step |
| `N* ∝ (1−ρ²)^{−1/4}` | §4 | Derived; inherits the §3 constant |
| Dimension alone does not create coupling | §5 | Measured on synthetic features |
| Coupling is confined to correlated blocks | §6 | Measured on synthetic features |
| Block coding saves bits | §6 | **Conditional**: only if the block is free from the record. Not proved |
| The barrier is separable-vs-joint decoding, not field size | §7 | **Argued, not proved** |
| Any upper bound / construction for the coupled row | — | **Not established** |

### Limitations, stated plainly

1. **The clique bound is a lower bound on the chromatic number, not the chromatic number.**
   Theorem B brackets them within a factor 2; this note computes only the clique term.
2. **The lattice step is a continuum approximation.** Proposition 1 is exact for *area*;
   `L^{-T}` does not map the weight lattice to itself. Measured drift (`coupling_verify.py`
   §4): 0.000 bits at N=16, 0.036 at N=256, **0.211 at N=1024** — i.e. it degrades exactly
   in the few-candidates regime that matters. BF16 is log-spaced, which this note models as
   a uniform lattice.
3. **The magnitude bound `ρ` clips the region.** For strongly correlated pairs the polygon
   is a long thin sliver and the cap binds, so §4's numbers are conservative in a direction
   that has not been quantified.
4. **No real features were used.** Every number here is synthetic. §4 is the experiment
   that would change that, and it is cheap.
5. **`s = 1` per row throughout.** If the "dispersed" promise is only a test distribution
   rather than a decoder guarantee (`repair_theory_and_evidence_consolidated.md` §9.7), the
   relevant object is `2s`-sparse for larger `s`, and the planar geometry becomes a
   `2s`-dimensional zonotope problem. Nothing here covers that.

---

## 9. Suggested order of work

1. **Run §4 on the cached features.** One matmul. Produces the first pre-tamper bound for
   the real head with a proof behind it, and tests the 160–450 bit prediction.
2. **Re-measure the 1-sparse count with the same lattice convention** so the 2× ratio can
   be checked on the real head rather than on synthetics.
3. **Decide the guarantee** (§7). Uniform or probabilistic — the two produce different
   target numbers, and the current write-up mixes them.
4. Only then attempt the construction of §6. If it collapses to the generic graph bounds,
   `proof_and_validity.md` §8 already prescribes the response: stop expanding the claim.

Items 1 and 2 are hours of work on the existing GPU box. Item 3 is a definition. Item 4 is
the actual research risk.

---

# Addendum — literature check, a correction to my own §4, and the upper bound

Added after a targeted literature check and a re-reading of Theorem B. Two of the
findings below cut against claims made earlier in this note; they are stated first.

## A1. Correction: I computed the weaker of two cliques

Theorem B supplies **two** clique families, and §1–§4 above use only the second:

- `P_2s(m)` — vary the `2s` coordinates with the largest alphabets. This is the
  pair geometry of §1, and it is what §4's 160–450 bit prediction estimates.
- `B_s(m)` — **all fiber originals within `s` changes of a common checkpoint `z`.**

For the Llama head the second dominates, and not narrowly. Take `z = w` (the original
itself, a legal checkpoint since every candidate reaches it in ≤ 64 changes). Members:
choose any 64 of the 128,256 rows, and in each an invisible single-coordinate change.

```
log2 C(128256, 64)                       =   790 bits   (which rows)
64 rows x 20.15 bits/row (their §8)      = 1,290 bits   (which column, which value)
                                   total = 2,080 bits
```

**The reported 1,287-bit converse is missing the 790-bit row-choice term.** It multiplies
the per-row figure by `s` without paying for the choice of rows, though those choices are
what make the family a clique around a common `z`. Against the 3,968-bit baseline this
caps the provable saving under a **uniform** guarantee at **47.6%**, not 97%.

This also deflates §4 of this note: spreading the coordinate budget across 64 rows beats
doubling up in fewer rows, because the row-choice term rewards more rows (32 rows with
2 changes each gives ≈1,900 bits, below 2,080). **Coupling is therefore a second-order
effect on the head-level lower bound, not the driver.** §4's number should be read as the
`P_2s` term, which loses.

A second warning in the same direction: in the audit's own erasure counterexample the
`P_2s` clique gives 2 bits where the true optimum is `⌈log2(n+1)⌉ = 17`. The pair clique
can be loose by a large factor even when the fiber *is* a product.

## A2. What the coupling result actually is: a bridging lemma

Given A1, the honest description of Proposition 1 is not "a better bound". It is this:

> `proof_and_validity.md` §7 rejects applying Theorem C to the Llama head — *"It is valid
> only when those rank sets are defined from the retained record before the tamper. This
> condition is not established for the per-damaged-row Llama lists."*

That objection is now answerable. Define the **public difference alphabet**

```
A_j^diff = { δ : |δ · h_j(x_i)| < W for all i,  |δ| <= ρ }
```

Two originals differing only in coordinate `j` are confusable iff their difference lies
in `A_j^diff`. This set depends only on the cached features and the cell width — **not on
`w`, not on the tamper, not on the damaged checkpoint.** It is computable by the encoder
before anything happens, which is exactly the condition Theorem C requires and the audit
found missing. The coupled row satisfies Theorem C's hypothesis *on differences* even
though it fails it on values.

Proposition 1 then quantifies the residual error in that transfer: the pair correction is
`-½log2 det Σ_jk`, i.e. `-½log2(1 - ρ_jk²)` beyond the product prediction.

**So the contribution is: the product-fiber theorems transfer to a coupled row, with an
explicit and measurable error term.** That is what §6 of the audit asked for. It does not
by itself lower any bit count.

## A3. The correction is mutual coherence, and superposition predicts it is tiny

`max_{j≠k} |ρ_jk|` is the **mutual coherence** `μ(H)` of the feature matrix — the standard
quantity governing sparse recovery. So the worst-case coupling correction over a row is
`-½log2(1 - μ²)`, and coherence is exactly the classical object it should be.

The correction is startlingly insensitive:

| correction | needs \|ρ\| ≥ |
|--:|--:|
| 0.01 bits | 0.117 |
| 0.1 bits | 0.360 |
| **1 bit** | **0.866** |
| 3 bits | 0.992 |

Superposition theory predicts worst-case coherence on the order of `d^{-1/2}` for a
width-`d` network ([Linear-Readout Floors and Threshold Recovery in Computation in
Superposition](https://arxiv.org/abs/2605.01192)). At `d = 4096` that is `μ ≈ 0.0156`, so

```
correction  =  -0.5 * log2(1 - 0.0156²)  =  0.0002 bits
```

**If Llama's output-head features obey the superposition packing bound, coupling costs
essentially nothing and the product theorems transfer almost exactly.** If instead the
known outlier/massive-activation directions produce near-collinear pairs, `μ → 1` and the
correction is unbounded. This is a sharp, cheap, falsifiable test — one Gram matrix — and
it is the single most informative measurement left in the project.

## A4. The upper bound: reformulated, not solved

The public difference set generalises to

```
D = { δ : ||δ||_0 <= 2s, δ ≠ 0, |<δ, h_i>| < W for all i, |δ_j| <= ρ }
```

again independent of `w`. A certificate is valid iff it separates every pair differing by
an element of `D` — i.e. it is a **colouring of the Cayley graph on the weight lattice
with connection set `D`**. That is a cleaner statement of the construction problem than
"colour the confusability graph on the fiber", because `D` is computable up front.

Three levels, all classical, and the gap between them is the whole problem:

| | cost | note |
|---|--:|---|
| clique (lower) | `max` over supports | §1, and A1 shows which clique to use |
| Vandermonde / RS | `2s·log2 p`, `p > n` | the 1,088-bit construction |
| greedy / union bound | `log2 \|D\|` | ≈ the data-free baseline; useless |

A random public `r ∈ F_p^d` with certificate `⟨r, index(w)⟩ mod p` fails only if
`⟨r,δ⟩ ≡ 0` for some `δ ∈ D`, so `log2|D| + log2(1/ε)` bits suffice — and because `D` is
computable, a chosen `r` can be **verified** against it, upgrading the guarantee from
probabilistic to deterministic-uniform. Unfortunately `log2|D|` carries the full support
count and lands near the baseline.

**I did not find a construction beating 1,088 bits.** The one structural handle I would
pursue: `D` is not an arbitrary sparse set, it is the sparse points of a *polytope*. Codes
matched to Hamming structure (RS, BCH) ignore that. A lattice code matched to the dual of
the measurement lattice is the natural object and I am not aware of it having been tried
here. Offered as a direction, not a result.

## A5. Prior art found

The engineering pipeline is occupied. **LM-Fix** ([arXiv 2511.02866](https://arxiv.org/abs/2511.02866))
runs a short test-vector pass, uses hash-guided checks plus **reference outputs** to
localise bit flips in an LLM, and repairs by integer-view weight editing — <1 KB overhead,
>100× faster than reload. **BitFlipScope** ([arXiv 2512.22174](https://arxiv.org/abs/2512.22174))
does scalable fault localisation and recovery for bit-flip corruption in LLMs. Earlier:
RADAR (2-bit checksum signatures over weight groups), HASHTAG (Pearson hashes over weight
tensors), WeightSentry, NeuroPots.

None of these asks how few protected bits are *necessary*. That is the surviving
differentiator, and it means the paper cannot be sold on the pipeline or the byte count —
442 bytes is not a headline when LM-Fix reports <1 KB with a published method.
