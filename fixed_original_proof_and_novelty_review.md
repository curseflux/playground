# Fixed-original model repair: consolidated proof and novelty review

## Decision

Keep fixed-original recovery as the main problem. The central question remains: **how many protected bits must be written before a one-coordinate tamper, given N retained input/output observations?** No further uniform-fiber development is needed for this paper.

This review makes three concrete advances/corrections:

1. A checksum-specific Gaussian bound removes the earlier `log q` counting term. In the two-check regime, the achieved budget has the same leading N-dependence as a valid fixed-original minimax converse, with an explicit additive `O(log log d)` gap at fixed confidence.
2. The earlier statement that training on the recorded features necessarily invalidates the fixed-original upper bound was too restrictive. **Fresh independent dither after training** permits a conditional-on-features theorem for any trained row. Gaussian averaging then gives the sharper sample law even when that row was trained on the same Gaussian design.
3. “Price the insurance” is a relevant application of the paper, but `gamma_2(H)` alone is neither a new geometric notion nor the exact price. A meaningful price is the achieved, certified budget of a declared protocol for a model, record, precision, and tamper class. The actual cell margins matter.

These are proof arguments and a focused primary-source review, not independent peer review, formal verification, or an exhaustive priority search. The surviving novelty claim and the remaining implementation gate are stated at the end.

## 1. Fix the guarantee before stating the rate

Use a full finite uniform grid `a in {-Q,...,Q}^d`, numerical row `w=Delta a`, and `q=2Q+1`. The intact feature matrix is `H in R^{N x d}`. The record is

\[
\ell_i=\left\lfloor\frac{\Delta(Ha)_i+\xi_i}{W}\right\rfloor,
\qquad \xi_i\overset{\rm iid}{\sim}\operatorname{Unif}[0,W).
\tag{1}
\]

Arithmetic in the theorem is exact; the quantizer is unsaturated. The feature matrix, offsets, output codes, numerical grid, and encoder/decoder conventions are available intact to the repair algorithm. The new protected budget counts the certificate, not an already available retained record. Public information must be intact even when it need not be secret.

One public encoder/decoder pair must serve the whole declared class. For every original fixed before the fresh record randomness,

\[
\Pr\{\text{the decoder recovers this original for every legal one-coordinate tamper}\}
\ge 1-\delta.
\tag{2}
\]

The certificate is written after recording and before tampering. The adversary can see the record and certificate. Probability is over recording/setup randomness, not over a benign distribution of tampers. A single successful setup event handles every legal tamper of that original.

This requirement does **not** promise success for every other original sharing the realized record. That distinction is sufficient here; the uniform problem is outside the scope of this document.

Two cautions about “fixed original”:

* The converse is minimax: for any common protocol claiming (2) for every original, some original forces the budget. It does not assign this lower bound to each named trained checkpoint.
* An unconstrained best decoder for one named model could simply hardcode that model. A per-model price is meaningful only after fixing the common protocol and accounting for any model-dependent side information or decoder description.

## 2. Construction: separate only differences that collide under the checksum

Fix a prime `p`. Let

\[
r(p,d)=\min\{r\ge1:1+p+\cdots+p^{r-1}\ge d\}.
\tag{3}
\]

Choose public representatives `b_1,...,b_d in F_p^r` of distinct one-dimensional subspaces. Save

\[
C_p(a)=\sum_j a_jb_j\pmod p.
\tag{4}
\]

The meaningful bit cost is `ceil(r log_2 p)`. If `p>=d-1`, then `r=2`. For `p>=d`, the simple representatives `(1,j)` give the familiar sum and weighted-sum checks.

Every pair of distinct columns is linearly independent. Therefore a nonzero, at-most-two-coordinate label difference has zero checksum **if and only if all its entries are multiples of p**. This equivalence is the useful simplification: the record only needs to eliminate those multiples, not every sufficiently large alternative.

The decoder searches one-coordinate restorations of the damaged row and returns the unique candidate matching the record and checksum, or reports failure if there is no unique candidate. If it fails on a legal tamper of `a`, there is another original `a+v` satisfying

\[
0<|\operatorname{supp}v|\le2,\quad v\in(p\mathbb Z)^d,
\quad \operatorname{Record}(a+v)=\operatorname{Record}(a).
\tag{5}
\]

For the full grid and exact linear record, each changed-column hypothesis reduces to intersecting scalar intervals and a modular congruence. This gives `O(Nd+rd)` arithmetic operations; if a residue class contains multiple admissible values, the unique-candidate decoder can reject without enumerating them. Numerical bit complexity and finite-arithmetic correctness must be implemented explicitly.

## 3. Conditional theorem: trained rows and arbitrary feature matrices

For a fixed `H`, define the finite set

\[
\mathcal V_p=\{v\in(p\mathbb Z)^d:0<|\operatorname{supp}v|\le2,
\ \|v\|_\infty\le2Q\},
\]

and the risk bound

\[
\mathcal R_p(H)=
\sum_{v\in\mathcal V_p}
\prod_{i=1}^N\left(1-\frac{\Delta|(Hv)_i|}{W}\right)_+.
\tag{6}
\]

**Theorem 1 (conditional fixed-original repair).** Fix the trained row and its feature matrix before drawing the independent dithers. Then

\[
\Pr_\xi\{\exists\text{ legal tamper that is not repaired}\mid H,a\}
\le\min\{1,\mathcal R_p(H)\}.
\tag{7}
\]

The row may have been trained on the same examples and may depend arbitrarily on `H`. A prime chosen from `H` before dither is also permitted. The result does not require iid or Gaussian entries of `H`.

**Proof.** Conditional on `H,a`, the phase `(Delta(Ha)_i+xi_i) mod W` is independent uniform in each cell. For a fixed displacement `v`, equality of the two codes has probability exactly `(1-Delta|(Hv)_i|/W)_+`. Independence of dithers gives the product in (6). Union bound the checksum-colliding differences (5), allowing invalid alternatives outside the original box only to enlarge the bound. This excludes all failing tampers on one event. ∎

The bound applies individually to every pre-dither original; it does not create one dither event good for all originals. This is why dependence of the trained row on `H` is allowed. Retraining or choosing the original in response to these dithers is not allowed by this argument.

Equation (6) is a genuine risk-to-budget formula: choose a public `p(H)` with `R_p(H)<=delta`. It is **not a claim of cheap evaluation**; a direct sum can be enormous. The formula also shows why a single maximum-displacement sensitivity is generally insufficient: exposure across all records determines the collision probability.

Fresh dither fixes the dependence issue, not feature degeneracy. For zero or redundant features, the risk bound can remain large. Arbitrary trained H receives the conditional theorem, not an automatic sharp rate in N. The explicit rate in the next section requires the stated Gaussian marginal law for H.

This correction does not retrofit the existing experiments with random dither. To apply the theorem, dither must actually be used during pre-tamper recording. A short pseudorandom seed is not automatically a substitute for independent randomness in an information-theoretic proof. Offsets/randomness and retained-record storage must be described honestly. No extra secrecy from the adversary is required.

## 4. Sharper Gaussian sample–certificate theorem, independent of q

Now assume the rows of `H` are independent standard Gaussian vectors. The numerical row can be any learning rule `a(H)` in the grid, fixed before the independent dithers; additional training randomness can be conditioned on as well.

Let

\[
\kappa=\Pr\{|G|\ge1\}\approx0.31731,\quad
A=2\ln\frac{28d^2}{\delta},\quad G\sim N(0,1).
\tag{8}
\]

**Theorem 2 (Gaussian fixed-original achievability).** Suppose `N>=max(4,A)`. Fix a prime from the public parameters, independently of H and the dithers, satisfying

\[
p\ge\frac{WA}{\kappa N\Delta}.
\tag{9}
\]

Then (4) achieves (2), with

\[
L\le\lceil r(p,d)\log_2p\rceil.
\tag{10}
\]

There is no `q` in (8)–(9). A prime `p>2Q` always supplies the usual data-free fallback. If `WA/(kappa N Delta)<=1`, zero certificate bits suffice by the same probability argument applied to all nonzero sparse integer differences.

### Proof

Write

\[
\phi(t)=\mathbb E(1-t|G|)_+.
\]

For `0<=t<=1`, the event `|G|>=1` gives `phi(t)<=1-kappa t<=(1+kappa t)^{-1}`. For `t>=1`, the half-normal density is at most `sqrt(2/pi)`, hence

\[
\phi(t)\le\sqrt{2/\pi}\int_0^{1/t}(1-tx)dx
=\frac{1}{\sqrt{2\pi}\,t}
\le\frac1{1+\kappa t}.
\tag{11}
\]

The last inequality follows since `c=1/sqrt(2pi)` obeys `c(1+kappa)<1` and `t>=1`.

For a fixed integer vector `v=pu` supported on at most two coordinates, Theorem 1 and Gaussian averaging give

\[
\Pr\{\text{equal records for this difference}\}
=\phi\!\left(\frac{p\Delta\|u\|_2}{W}\right)^N.
\tag{12}
\]

For an invalid alternative, equality of formal quantized outputs still has this probability and remains a valid upper-bound event. The identity holds even for `a(H)`: conditional on `H`, the dither has already removed dependence on the anchor before averaging over `H`.

There are

`2d + binom(d,2)(8m-4) <= 4d^2 m`

integer vectors `u` with support at most two and `||u||_infty=m`. Enlarge the finite alphabet to all such integer vectors. By (11)–(12), the failure probability is at most

\[
4d^2\sum_{m=1}^\infty m(1+bm)^{-N},
\qquad b=\kappa p\Delta/W\ge A/N.
\tag{13}
\]

This enlargement is legitimate: the series converges for `N>2`. Monotonicity in `b` reduces the analysis to `b=A/N`. For `A>=2`, `N>=max(4,A)`, the function `x(1+bx)^{-N}` decreases for `x>=1`. Therefore

\[
\begin{split}
\sum_{m\ge1}m(1+bm)^{-N}
&\le(1+b)^{-N}+\int_1^\infty x(1+bx)^{-N}dx\\
&=(1+b)^{-N}\left[1+
\frac{(1+b)(1+b(N-1))}{b^2(N-1)(N-2)}\right]\\
&\le7(1+A/N)^{-N}\le7e^{-A/2}.
\end{split}
\tag{14}
\]

For the penultimate bound, `(N-1)(N-2)>=N^2/4`, `1+b<=2`, and `1+b(N-1)<=1+A` give a bracket at most `1+8(1+A)/A^2<=7`. The last inequality uses `ln(1+x)>=x/2` on `[0,1]`. Substituting (8) into `28d^2 exp(-A/2)` gives `delta`. With no checksum, the same reasoning uses `p=1` as a lattice spacing, not as a field modulus. ∎

This proof replaces the previous union bound over all neighbor values by a convergent sum over **checksum collisions**. It removes alphabet-size dependence from the probability bound without changing the original class, decoder access, or tamper guarantee. The constants are conservative; (6), (12), or a numerically certified series bound can be tighter. Selecting p adaptively from H requires Theorem 1's conditional bound or a separate selection argument; one must not insert a data-dependent p into the Gaussian averaging identity (12).

## 5. Necessity under the same fixed-original guarantee

Fix `0<delta<3/4`. Put

\[
m_2=2\sqrt{2/\pi},\qquad
k_0=\min\left\{Q,\left\lfloor\frac{W}{8Nm_2\Delta}\right\rfloor\right\}.
\tag{15}
\]

**Theorem 3 (fixed-original minimax converse).** Every common protocol with worst-case fixed budget `L` satisfying (2) for every fixed original obeys

\[
2^L\ge(3/4-\delta)(2k_0+1)^2.
\tag{16}
\]

**Proof.** Before sampling, fix a square of originals centered at zero and varying two coordinates by up to `k_0` grid steps. Its size is `K=(2k_0+1)^2`. On feature row `i`, every square member's prediction lies in `[-s_i,s_i]`, where `s_i=Delta k_0(|H_ij|+|H_ik|)`. Uniform dither puts the entire interval inside the zero cell with failure probability at most `2s_i/W`. Expectation over features and a union bound give probability at least `3/4` that all K originals share the record.

On that event they form a clique: any pair admits a checkpoint within one coordinate of each. At most `2^L` of these originals can be correctly repaired for every tamper, because two successful originals with the same certificate would require different answers to the same decoder input. Outside the event, at most K can be successful. Expected successful originals are thus at most `2^L+K/4`. Requirement (2) makes their expectation at least `(1-delta)K`, yielding (16). ∎

The proof also accommodates independent public protocol randomness by including it in the expectation and retaining the all-tampers requirement for each successful setup. It is a worst-case-length lower bound, not a lower bound on expected or checkpoint-specific variable length.

### What is matched, and what is not

Let `K_N=W/(N Delta)`. In the regime where `k_0` is not clipped by Q, `k_0>=1`, and the chosen prime is at least `d-1`, Theorems 2–3 yield

\[
2\log_2 K_N-O_\delta(1)
\ \le L^*_{\rm fix}\ \le
2\log_2 K_N+2\log_2\ln(d/\delta)+O(1).
\tag{17}
\]

Here `L*_fix` is the minimax fixed-budget optimum for one protocol satisfying (2). The upper bound follows by choosing a prime within a constant factor of (9); the lower follows from (15)–(16). At fixed confidence, the remaining additive gap is `O(log log d)`, independent of q and with no extra `log log N` factor.

This matches the leading N-dependence. It is **not** a constant-gap theorem jointly in all parameters, nor a bit-by-bit optimum for a particular trained row. Below the two-check regime, retain the explicit `r(p,d)` bound; do not extrapolate (17) through field-size transitions or the zero-bit endpoint. One can minimize the bound over available primes and the data-free fallback rather than insisting on the smallest prime.

Doubling N reduces the leading two-check expression by two bits per row. It does not promise exactly two bits of measured savings at every doubling, because budgets are discrete and regime transitions matter.

## 6. The proposed insurance property: relevant, with a different definition

The suggested quantity is

\[
\gamma_2(H)=\min_{\|u\|_2=1,\ |\operatorname{supp}u|\le2}\|Hu\|_\infty.
\tag{18}
\]

“How loudly does the quietest two-weight nudge appear?” is an accurate interpretation for an exact linear row. Small values can arise from nearly cancelling columns, weak columns, or scale imbalance; redundancy is not the only cause.

However, it should be a diagnostic, not the proposed paper's new intrinsic insurance price:

* **It omits the original and its cell margins.** For a final linear head, every output row shares H and thus shares gamma, while the rows' recorded margins and repair budgets can differ.
* **It does not encode the precision or coverage.** W, Delta, the allowed alphabet, tamper size/count, confidence, and decoder matter.
* **It ignores accumulation of boundary opportunities.** Repeating feature rows leaves (18) unchanged. Recording them with fresh independent dithers can still reduce fixed-original ambiguity rapidly.
* **It is not invariant to parameterization.** Feature rescaling and compensating weight rescaling can preserve the function but change gamma. Exact stored-weight repair is parameterization-dependent, so the parameterization and grid must be part of the policy.
* **It is not a new geometric concept.** It is a two-sparse lower gain from Euclidean norm to maximum norm, closely related to established restricted-isometry quantities; see the primary-source review below.

### A concrete same-gamma counterexample

Take `H=I_2`, `Delta=1`, `W=3/2`, no dither, and alphabet `{-4,...,4}`. For both rows below, `gamma_2(H)=1/sqrt(2)`.

* Original `(0,0)` records `(0,0)`; the record-compatible originals are `{0,1}^2`.
* Original `(2,2)` records `(1,1)`; its record-compatible original is uniquely `(2,2)`.

Thus the same feature sensitivity coexists with either three local alternatives or none. For the declared unique-candidate decoder, zero checks fail on an unchanged `(0,0)` but succeed against every legal tamper of `(2,2)`. Two binary checks suffice for the first case. This example demonstrates why gamma cannot determine the achieved price; it is not presented as a per-original minimax lower bound.

### Is gamma cheap to compute?

For each support pair S, form the centrally symmetric convex hull of the `2N` points `+/- H_iS` in the plane. Its support function is `max_i |H_iS dot u|`; its centered inradius is the pair's minimum gain. If it is lower-dimensional the gain is zero; otherwise compute the smallest distance from the origin to a supporting edge. Taking the minimum over pairs computes gamma.

This yields an exact-real-arithmetic geometric algorithm with roughly `O(d^2 N log N)` operations using standard planar hull construction. It is feasible for small d, but **not automatically cheap** for millions of column pairs. A verified implementation with floating-point inputs needs numerical error control.

A cheaper conservative bound uses `G=H^T H/N`:

\[
\underline\gamma^2=\min_{j<k}\lambda_{\min}(G_{\{j,k\},\{j,k\}})
\quad\Longrightarrow\quad \gamma_2(H)\ge\underline\gamma.
\tag{19}
\]

Computing G costs `O(Nd^2)` arithmetic operations, followed by `O(d^2)` scalar pair calculations. This can exploit a matrix multiply, but should be benchmarked rather than advertised as negligible. Sampling a subset of pairs gives an **upper** bound on the full minimum, so plugging that sampled value into a certificate upper bound is unsafe.

For a nonlinear network, replacing H by its parameter Jacobian only measures infinitesimal response. Exact finite-tamper restoration requires control of the nonlinear remainder or a setting, such as an intact-feature final linear head, where the observation equation is exact.

## 7. A price that can legitimately be attached to a trained model

Fix a public family of certificates and the unique-candidate decoder. For a proposed modulus, define the anchored collision set

\[
\mathcal A_p(a,H,\ell)=\{b\in\Theta:\ b\ne a,\ |\operatorname{supp}(b-a)|\le2,
\ \operatorname{Record}(b)=\ell,\ C_p(b)=C_p(a)\}.
\tag{20}
\]

**Proposition 4 (pre-tamper validation).** If this set is empty, the actual original is recoverable against every legal one-coordinate tamper using (4). For the declared unique-candidate decoder, emptiness is also necessary: a colliding b gives a shared damaged checkpoint with two admissible candidates.

This is an audit of the actual original, performed while it is available before damage. It requires no statistical assumption and applies to trained H. It does not require comparing every pair of possible originals. Under the projective checks, the test becomes exclusion of nonzero multiples-of-p lattice points in the one- and two-coordinate polygons anchored at a.

One defensible measured property is therefore

**the number of protected bits in a validated certificate from a declared family, for this model and record.**

Use “certified achievable repair budget,” not “the intrinsic minimum number of bits the network needs.” A sufficient continuous outer bound on the anchored polygons is acceptable for an upper certificate; it may be conservative because the real problem is on a grid. A handful of sampled invisible alternatives is not enough to certify emptiness.

If p is chosen using the original or its margins and cannot be reproduced from public information, its identifier must be protected and counted. For a public menu of J primes, an explicit construction stores a `ceil(log_2 J)`-bit index plus the appropriate checksum payload. Include a data-free prime larger than `2Q` as a guaranteed fallback. The complete encoded length, including the selector, is the quoted price. Different payload lengths do not provide a free side channel.

This gives two related outputs:

| Output | Meaning | Current status |
|---|---|---|
| Statistical budget from Theorems 1–2 | A pre-tamper budget with failure probability at most delta over recording | Proved under the stated conditions |
| Validated budget from (20) | This realized model/record passes an all-tampers check for the chosen decoder | Criterion proved; scalable implementation not established |
| Exact optimal price for this model | Minimum over all admissible protocols with carefully specified description costs | Not claimed |

This framing fits the original sample-versus-bits question. It does not justify launching layer, scale, and training-regime sweeps before the validation computation is reliable and affordable. Start with one final linear head and one N-versus-budget curve; gamma can be one explanatory baseline.

## 8. Focused novelty review

The search covered the closest primary work on model repair, syndrome sketches/document exchange, and quantized consistency. Relevant theorem/construction sections, not just abstracts, were inspected. The assessment below is a bounded review, not proof that no related result exists.

| Closest work | What it already supplies | Consequence for this paper |
|---|---|---|
| [Dodis, Ostrovsky, Reyzin and Smith, *Fuzzy Extractors*, SIAM J. Comput. 2008](https://arxiv.org/abs/cs/0602007), Section 5 Construction 3; Section 8 | Syndrome helper data for exact recovery from a nearby corrupted string; discussion of probabilistic correctness | Pre-tamper helper bits, syndrome decoding, and changing error quantifiers are not new by themselves. Our objective is protected storage, not their secrecy/entropy objective. |
| [Cheng and Li, *Efficient Document Exchange and Error Correcting Codes with Asymmetric Information*](https://arxiv.org/abs/2007.00870), introduction and model | Communication/redundancy bounds when additional information about error locations is available | “Side information reduces checksums” is too broad a novelty claim. Our side information is numerical input/output observations, with a sample-size law rather than supplied location subsets. |
| [Gao and Lafferty, *Model Repair: Robust Recovery of Over-Parameterized Statistical Models*](https://arxiv.org/abs/2005.09912), Section 4 and neural-model extensions | Exact model repair from training design and training-induced redundancy; linear and nonlinear feature settings | Neither model repair nor repair of trained/random-feature models is new. Their Section 4 uses a row-space condition and an independent-coordinate contamination model. The full-grid, quantized-record, protected-budget tradeoff here is different. |
| [Jacques, *Error Decay of (almost) Consistent Signal Estimations from Quantized Gaussian Random Projections*](https://arxiv.org/abs/1406.0022), Theorem 2 and Section 4.2 | Fast consistency-error decay for Gaussian quantized observations and sparse original classes | The inverse-N boundary mechanism and sparse quantized geometry are established. Our task adds exact grid recovery, a sparse damaged checkpoint, and an optimized protected certificate. Simply converting a known radius to bits would be a weak contribution. |
| [Jacques, Hammond and Fadili, *Dequantizing Compressed Sensing*](https://arxiv.org/abs/0902.2367), Section IV Proposition 1, including the infinity-norm case | Sparse Gaussian geometry measured in non-Euclidean output norms | Gamma should not be advertised as a newly discovered network quantity. Its role in a validated repair-budget estimator would need its own evidence. |

The source-derived summaries are intentionally short; the mathematical arguments earlier in this document are self-contained project derivations.

**Assessment.** I did not identify a direct duplicate of the exact finite-grid, fixed-original protected-bit theorem (with an adaptive one-coordinate tamper and the checksum-collision summation above) among these closest works. That is narrower than claiming priority for the full direction. The proof uses classical ingredients, and the Gaussian benchmark alone is not enough to establish the user's desired top-percentile conference novelty.

The defensible candidate contribution is:

> Retained numerical observations reduce the protected information needed for exact model repair; a fixed-original sample–bits theorem explains the reduction, and a pre-tamper audit turns it into a validated budget for an actual trained row.

The theorem and the empirical pricing procedure must be connected. A theorem for one encoder paired with budget plots from an unrelated heuristic would leave the main claim unsupported.

## 9. Bounded next step and stopping rule

The proof phase now has a usable core: Theorems 1–3, the explicit encoder/decoder, and the pre-tamper validation criterion. Do not add a general multi-layer theory, a new metric paper, or more uniform results.

The one remaining development gate is an affordable implementation of a **sound upper certificate** for (20), including selector bits and exact numerical conventions, on a representative frozen final-head slice. It may use conservative bounds and a fallback; it need not solve the optimal lattice problem. Measure one central N-versus-bits curve, with the theorem's regime, data-free baseline, validated achieved lengths, and runtime separated clearly. Do not plot the minimax converse as a per-checkpoint lower bound.

If the audit is useful and affordable, this is a contained theory-plus-method paper worth preparing for submission, with a focused final related-work/priority check. If it is only a loose gamma bound, or the implementation cannot validate the claimed budget, additional layer/training sweeps will not fix the novelty gap. I would not currently describe the project as already meeting the top-percentile ICML target.

No large code repository is needed for the next gate. The minimal data are one intact cached feature matrix, one original stored row, its retained output codes and any offsets, plus the exact small recording routine. The attached reports/scripts do not by themselves supply that numerical cache. For experiments without fresh dither, (20) remains applicable, but the new statistical theorems do not automatically describe their observed slopes.

## Appendix: checks completed in this review

The proof was checked for the timing of training, dither and modulus selection; quantized-cell conventions; the all-tampers quantifier; checksum aliases; infinite-sum convergence and constants; and the converse's fixed-budget/minimax scope.

Small computational checks passed:

* 120 numerical checks of the Gaussian one-observation inequality (11), across scales from `10^-5` to `10^3`.
* Checks of the shell counts and the bracket bound in (14) across small dimensions and multiple N/A settings.
* 60 exact integer difference checks of the dither collision-product identity, exhaustively enumerating the offsets for a small feature matrix and an anchor obtained by fitting that matrix.
* Exhaustive enumeration of the same-gamma example, obtaining four versus one record-compatible originals as stated.

The numerical checks support the algebra and the examples; they are not proofs of inequalities over continuous parameter ranges. Those proofs are written above. No large attached script was rerun and no new neural-network experiment is claimed.
