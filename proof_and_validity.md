**Protected model repair: focused proof and validity phase**

This document answers the agreed four tasks as far as the supplied evidence permits. It gives a targeted audit, a proved value-ambiguity theorem, a pre-tamper coding theorem with matching cases, executable reference constructions, and a clearly separated comparison figure. It does **not** certify the reported 1,287-bit Llama converse or claim an optimal Llama repair scheme.

The main finding is that the timing of the certificate changes the information requirement. The logarithm of a candidate list for one damaged checkpoint can be strictly smaller than the protected budget required before the checkpoint is known. This gap can be information-theoretic, even if the decoder knows every error location. Treating it as purely computational would give the paper the wrong target.

The positive result is a complete theorem and construction for **product transcript fibers**: records that independently restrict each stored coordinate to a finite set. A concrete diagonal linear model with iid inputs supplies such fibers and exhibits value uncertainty shrinking with N. For large residual alphabets, the construction matches the necessary budget exactly for suitable field sizes and within additive O(s) otherwise. For arbitrary residual alphabets there is a constructive factor-two information bound for unknown-location errors; its general coloring implementation is not efficient. An efficient binary special case has an exact sample–certificate law. These are deliberately bounded results, not claims about arbitrary neural heads.

The code is exact in its finite-field operations and has passed exhaustive small-instance verification. Proofs are provided below; the experiments validate implementations, not theorems. No formal proof assistant was used.

**1. Targeted audit of the supplied files**

Only `interval_audit.py`, the relevant configuration and forward/repair cells of `llama_full_certificate_to_repair.ipynb`, and `audit_v3_report.json` were inspected. The source module depends on Torch, which is not installed in this execution environment, and no Llama weights/features were supplied. The new implementations use pure Python and NumPy. The source notebooks and GPU experiments were not rerun.

The notebook is the previous 3,534-bit experiment: N=256, 16 erasures, 114 checks modulo 2^31−1. It is not an implementation of the new 1,088-bit construction. The JSON gives aggregate results, rather than per-row witnesses or the code for `C_uniform_converse`. Both converse entries have `fraction_exact = 0.0`. It also confirms one lost true value among 4,096 row instances at eta=10^−5.

Five details in the candidate code matter to the claimed converse.

1. `_codes_match` computes `cur = h @ z` and then `cur + (candidate - z[j])*h[:,j]` in floating point. This is an affine **surrogate based on the damaged row**. It is not a direct evaluation of the candidate with the original recording function. Floating-point evaluation of an identical mathematical model can depend on this decomposition.
2. Nominal membership in `[-R+code*width, -R+(code+1)*width)` is not exactly the inverse of the notebook's actual FP32 computation `floor((logit+R)/width)`. The addition itself rounds.
3. Positive eta accepts an expanded cell. This is useful for a proposal stage, but it does not certify equality to an actual stored code under a fixed exact-code protocol.
4. `verify_row` and `estimate_row` count `(column,label)` descriptions. An unchanged row can appear once for every column. Signed-zero labels also require representation-level bookkeeping. The core does not deduplicate full candidate rows; whether a missing caller does so is unknown.
5. The estimator samples with replacement over those descriptions and uses ordinary approximate Wilson endpoints. These are not automatically simultaneous confidence bounds for a product of many selected row candidate sets.

Three elementary counterexamples were reproduced using explicit FP32 arithmetic:

| Issue | Example | Result |
|---|---|---|
| Actual quantizer versus nominal cells | y=−2^−20, R=64, width=0.5 | FP32 quantizer returns 128, while y is outside nominal cell [0,0.5). |
| Affine update versus candidate evaluation | h=(1,1), z=(2^24,1), replace first weight by 0 | FP32 surrogate returns 0; direct candidate dot product returns 1. All listed weights/features are BF16-representable. |
| Upper-open cell versus the screen | retained nominal cell [0,0.5), current output 0.5, eta=0 | `excursion > 0` does not flag; the quantizer's next code is 129. |

The second example establishes that the two functions are not universally interchangeable; its large perturbation is not claimed to occur in the reported small-amplitude Llama experiments. The first and third examples directly disprove the stated eta=0 equivalences. These are protocol issues, not evidence that the qualitative experimental trend is false.

`count_candidates` uses closed endpoints, slack, and float32-rounded search endpoints. Its intended output is an upper proposal count. Such a count is not a packing lower bound. Even exact enumeration of `_codes_match` would certify only that function's acceptance rule, not the original cached FP32 computation.

**Audit conclusion.** The qualitative localization/value-ambiguity findings remain useful diagnostics. The 1,287- and 1,480-bit numbers are not presently certified lower bounds under the original exact-record protocol. Raising eta to 3×10^−5 does not, by itself, repair this logical issue. A proof can use a fixed numerical recording function or an explicitly specified bounded-error observation relation, but must consistently use that same model for the original, the alternatives, and the encoder.

`canonical_witness.py` supplies an exact rational reference for a declared ideal-linear recording convention. It preserves raw label identities, verifies explicit witnesses, deduplicates unchanged rows, and counts Cartesian products of verified row families. It does not retroactively validate an FP32 cache. A new canonical record must be labeled as such if this convention is used experimentally.

**2. What iid examples do to one unresolved value**

Fix an original scalar parameter v0 and a public scalar family f_v(x)=g(x)+h(x)v. The observation is a fixed quantizer applied to f_v(x). For every sample define R_i^+ as the largest allowed positive displacement from v0 in the connected quantizer-cell preimage containing v0, and R_i^− analogously. Either can be infinite. For a linear scalar response, this preimage is an interval. The N-sample interval has one-sided radii

\[
R_N^+=\min_{i\leq N}R_i^+,\qquad R_N^-=\min_{i\leq N}R_i^-.
\tag{1}
\]

Open endpoints do not affect the probability statements below; for discrete counting they must be preserved. Assume iid inputs and, for both signs, constants 0<c≤C and t0>0 such that

\[
c t\leq F_\pm(t):=\Pr(R_i^\pm\leq t)\leq C t,
\qquad 0<t\leq t_0.
\tag{2}
\]

This is a boundary-density assumption: a small parameter displacement has a probability proportional to its size of crossing a relevant observation boundary. It is an assumption to prove for a chosen model, not something inferred from an observed regression slope.

**Theorem A (finite-sample value ambiguity).** Let 0<delta<1 and

\[
a_N=\frac{\ln(4/\delta)}{cN},\qquad
b_N=\frac{\delta}{4CN}.
\]

When both are at most t0, with probability at least 1−delta,

\[
b_N<R_N^\pm\leq a_N\quad\text{for both signs}.
\tag{3}
\]

For a grid of spacing Delta containing v0, if the public parameter-domain endpoints are at least a_N away from v0, its compatible-label count K_N satisfies on that event

\[
2\lfloor b_N/\Delta\rfloor+1
\leq K_N\leq 2a_N/\Delta+1.
\tag{4}
\]

Consequently the typical count has order 1/(N Delta) in the regime where this quantity is large, for fixed confidence and fixed boundary-density constants. The count eventually saturates at one; a power law for raw cardinality cannot persist below that point.

**Proof.** Independence gives Pr(R_N^±>t)=(1−F_±(t))^N≤exp(−cNt), while Pr(R_N^±≤t)≤NF_±(t)≤NCt. Apply the first bound at a_N and the second at b_N and union-bound four events. If both radii exceed b_N, all grid points whose distance from v0 is at most b_N are in the interval, including the endpoints at that distance. If both radii are at most a_N, the interval contains at most 2a_N/Delta+1 grid points. The domain assumption prevents clipping from invalidating the lower count. ∎

If F_±(t)=lambda_± t+o(t), then separately for each sign,

\[
\Pr(NR_N^\pm>u)\longrightarrow e^{-\lambda_\pm u}.
\tag{5}
\]

This follows by substituting u/N in the exact survival probability. No independence between the two limiting radii is asserted or needed.

For an unsaturated uniform quantizer of width w, suppose additionally that the relevant phase is uniform within its cell conditional on |h| and its sign. Then

\[
F_\pm(t)=\mathbb E\min\{t|h|/w,1\}.
\tag{6}
\]

If 0<E|h|<infinity, dominated convergence gives F_±(t)/t→E|h|/w. Thus the scale is w/(N E|h|). A heavy-tail index above one does not change the N exponent under these assumptions. In particular, the proposed tail-index-4.5 explanation for N^−1.22 does not follow. Real undithered logits need not have the conditional phase property; equation (2) is the more general condition.

For a fixed finite grid, if Delta≤t0, the probability of retaining either adjacent grid alternative is at most 2exp(−cNDelta). To include closed endpoints, use F(Delta−)≥cDelta, obtained by taking a limit from below in (2). This gives an eventual singleton guarantee for this scalar family. It does not rule out alternatives at other columns of a neural row.

There is a separate dependence issue when these are the actual training examples. Iid training inputs do not imply iid boundary radii conditional on a model fitted to those same inputs. Theorem A applies to a fixed scalar family evaluated on independent inputs, or requires an additional argument for training-dependent families. Theorem D below avoids conditioning on a fitted model by proving one event uniformly over its entire predetermined model class. It does not characterize the outputs of a particular training algorithm. If the original is uniquely reproducible from the retained dataset and a known deterministic training procedure, then unrestricted exact retraining already gives L=0; a nontrivial original-model class must account for unknown training randomness or other unrecoverable state.

Theorem A formalizes a familiar quantized-consistency mechanism. Related 1/N-type results already appear in the quantized reconstruction literature under their own assumptions. It is not claimed as a new coding principle or a complete neural-head theorem. [Jacques, 2016](https://arxiv.org/abs/1406.0022).

**3. From prediction sets to a pre-tamper certificate**

Fix a retained transcript ell. Assume its full original-model fiber is exactly

\[
\Theta_\ell=A_1\times\cdots\times A_d,
\qquad m_j=|A_j|\geq1.
\tag{7}
\]

The sets are public functions of the retained record and the declared original-model class. They are known to the encoder before damage and do not depend on the future checkpoint. This is the critical structural assumption. Lists obtained only after locating a change in a damaged Llama row do not automatically satisfy it.

First allow arbitrary substitutions in at most s coordinates, within a public alphabet containing each A_j. No per-change magnitude restriction is imposed unless all substitutions used below satisfy it. The encoder must work for every original in (7) and every such tamper, including one chosen after seeing the certificate. This original-class promise is mathematical; knowing a training dataset cannot silently impose additional restrictions on it.

Define

\[
B_r(m)=\sum_{\substack{S\subseteq[d]\\|S|\leq r}}
\prod_{j\in S}(m_j-1).
\tag{8}
\]

Sort the sizes in nonincreasing order and let

\[
P_k(m)=\prod_{j=1}^{\min(k,d)}m_{(j)}.
\tag{9}
\]

**Theorem B (same-guarantee information bounds).** The minimum fixed protected length for this transcript obeys

\[
\max\{\lceil\log_2 B_s(m)\rceil,
       \lceil\log_2 P_{2s}(m)\rceil\}
\leq L_\ell^*
\leq\lceil\log_2 B_{2s}(m)\rceil
\leq2\lceil\log_2 B_s(m)\rceil.
\tag{10}
\]

The upper bound is attained by a deterministic finite pre-tamper coloring construction; it is not in general computationally efficient.

**Proof.** Two originals can share a damaged checkpoint exactly when their Hamming distance is at most 2s. Necessity follows by the triangle inequality. For sufficiency, partition their differing coordinates into two sets of size at most s and form a checkpoint agreeing with each original on one set and with their common values elsewhere. Thus valid certificate classes are precisely codes with minimum distance at least 2s+1, or equivalently colors in the graph joining pairs at distances 1 through 2s.

Fix any original z in the product. Its Hamming ball of radius s contains exactly B_s(m) possible originals, all sharing checkpoint z. They require different certificates. Independently, vary only the 2s coordinates with largest alphabets. Every pair in this family has distance at most 2s, so the whole family is a clique and needs P_2s(m) different certificates. Its pairs can share different checkpoints; a single common checkpoint is unnecessary for this clique argument.

Every vertex has exactly B_2s(m)−1 neighbors. Greedily color vertices in a public order with at most B_2s(m) colors. Encode the original's color before the tamper. Decode by finding a candidate within s changes of the received checkpoint having that color; uniqueness follows from the coloring, and the original exists.

Finally, every vector in a radius-2s ball can be injected into an ordered pair of radius-s ball vectors: in the public coordinate order put its first s nonzero differences in the first vector and the rest in the second. Their union reconstructs the original vector. Hence B_2s≤B_s^2, proving the last inequality. ∎

This establishes a factor-two information characterization for this structural class under a common uniform guarantee. It does not prove an exact optimum or a polynomial-time factor-two method for arbitrary alphabets. The graph and code ingredients are classical. [Dodis et al., secure sketches](https://arxiv.org/abs/cs/0602007); [Kokkala and Östergård, Hamming-power coloring](https://arxiv.org/abs/1607.01605).

**Theorem C (an efficient construction, with exact matching cases).** Remove singleton coordinates, which can be restored from the transcript; if none remain, zero bits suffice. Let n be the number remaining and M=max m_j. Set k=min(2s,n). Choose a prime p≥max(n+1,M), rank each original symbol inside its public set A_j, and store k Vandermonde checks of the rank vector modulo p. This uses

\[
L_{\rm rank}=\lceil k\log_2 p\rceil
\tag{11}
\]

meaningful bits with joint tuple packing. Alternatively storing the entire rank tuple uses ceil(log2 product m_j) bits. Both choices and their lengths are fixed from the transcript before damage.

If n≥2s and every m_j equals a prime p>n, then

\[
L_\ell^*=\lceil2s\log_2p\rceil,
\tag{12}
\]

and the rank construction is optimal in protected bits. More generally, still assuming n≥2s, for equal size M≥n+1, choosing a prime M≤p<2M gives

\[
\lceil2s\log_2M\rceil\leq L_\ell^*
\leq\lceil2s\log_2M\rceil+2s.
\tag{13}
\]

The prime interval used here is the standard Bertrand bound; exact optimality (12) does not require it.

**Proof.** Map a damaged symbol that lies outside A_j to an arbitrary rank, say zero. An unchanged coordinate maps to its true rank, so there remain at most s wrong ranks. Any k columns of a k-row Vandermonde matrix with distinct locators are independent. For n≥2s, two rank vectors within s errors of the mapped checkpoint and having the saved syndrome would differ on at most 2s coordinates and must coincide. For n<2s, all n ranks are determined by n independent checks. The literal tuple alternative is immediate.

A decoder uses the residual power moments to recover the error-locator polynomial and the error amplitudes. For u actual errors the u×u moment Hankel matrix factors as a nonsingular Vandermonde matrix, a nonzero diagonal amplitude matrix, and its transpose. The recurrence is therefore determined by 2u moments. The reference implementation tries u≤s, solves these small systems, finds roots among the n public locators, and checks the full syndrome and rank alphabet. This is polynomial, although not optimized for large n and s. Once ranks are recovered, public lookup in A_j returns the exact stored labels.

The lower bound in (12) is the clique term in Theorem B. When M is not a suitable prime, the stated p yields less than 2s additional bits before ceiling; this gives (13). ∎

The rank reduction is essential: a field must accommodate remaining ranks, not necessarily all original q-bit labels. It is valid only when those rank sets are defined from the retained record before the tamper. This condition is not established for the per-damaged-row Llama lists.

**4. Why knowing error locations afterwards is not enough**

Consider instead at most s erasures: the damaged object marks the erased coordinates, so the decoder knows their positions perfectly. The encoder still does not know them beforehand. In the product fiber, two originals are confusable exactly when their distance is at most s. Repeating the preceding argument gives

\[
\max\{\lceil\log_2 B_{\lfloor s/2\rfloor}(m)\rceil,
       \lceil\log_2 P_s(m)\rceil\}
\leq L_{\ell,\rm erase}^*
\leq\lceil\log_2 B_s(m)\rceil.
\tag{14}
\]

Use min(s,n) rank checks rather than min(2s,n) for an efficient erasure construction. Equal prime alphabets p>n with n≥s yield the exact optimum ceil(s log2 p). This is a different guarantee from unknown-location substitutions; the two curves must not be interchanged.

**Exact counterexample to the claimed purely computational gap.** Let n coordinates each have two possible original values, and allow any two coordinates to be erased. Every fully erased two-coordinate candidate list has exactly four originals, so its log count is two bits. Nevertheless,

\[
L^*=\lceil\log_2(n+1)\rceil.
\tag{15}
\]

For n=128256, this is 17 protected bits, not two. This can be realized with valid finite numerical symbols: originals 0 or 1, erasure replacement 2, and a quantizer that places 0 and 1 in the same cell while exposing 2. The screen is then perfect. No oracle support is required.

**Proof.** The all-zero vector and the n unit vectors form a clique: every pair differs on at most two coordinates and can produce a common marked checkpoint. Therefore n+1 certificates are required. For achievability set r=ceil(log2(n+1)), assign each coordinate a distinct nonzero r-bit column, and save the XOR of columns whose original bit is one. After erasure, subtract the intact coordinates. Any two distinct nonzero binary columns are independent, so the one or two missing bits are uniquely recovered. ∎

Exactly the same bit optimum holds for one unknown binary error. Its confusability graph is again the distance-two graph. The full graph's chromatic number need not equal n+1, but the ceiling in the bit budget makes the above lower and upper bounds coincide.

This counterexample does not prove that 1,088 bits are necessary for Llama. It proves that a post-tamper candidate count cannot establish that their excess is computational. It also explains why the clique bound can be stronger than a same-checkpoint packing bound. The underlying Hamming-code fact is classical; its role here is to correct the research target.

**5. A complete sample–certificate theorem in an explicit linear model**

Let a_j range over {0,...,q−1}, independently in the original-model class, and set

\[
w_j=1+(a_j+1/2)/q,\qquad
f_w(h)_j=w_jh_j,\quad j=1,\ldots,d.
\tag{16}
\]

This is a diagonal linear head with d mutable weights. For each iid input draw U in [0,1]^d with uniform independent coordinates and set h_j=2/(1+U_j). Record the one-bit prediction

\[
Q(f_w(h)_j)=1\{w_jh_j\geq2\}
=1\{(a_j+1/2)/q\geq U_j\}.
\tag{17}
\]

The threshold and input law are public, fixed independently of the original. The original is fixed before a physical experiment varies N; the theorem is uniform over the whole predetermined class. Allowed damage is any at-most-s weight substitutions in this grid. A public numerical cap of one covers all these changes. No tamper location is given to the decoder. The retained record uses Nd output bits plus input storage, kept separate from L.

Every coordinate transcript restricts its label to a consecutive grid interval. Hence the full transcript fiber is exactly a product and Theorems B–C apply. Let M_j(N) be the largest cardinality of any interval in the grid partition induced by coordinate j's N thresholds. There exists a realizable transcript with coordinate sizes exactly M_j(N), because model coordinates can be selected independently. Every other fiber embeds in it. Therefore the optimal uniform-over-transcripts budget for this fixed input sequence is exactly the coding optimum for that product of sizes M_j(N).

**Theorem D (nonasymptotic iid sample–certificate bounds).** Let 0<delta<1, s≤d/2, and N≥1. Define

\[
A_N=\left\lceil\frac{q}{N+1}\right\rceil,\qquad
U_N=\min\left\{q,\left\lceil\frac{q}{N}\ln\frac{dq}{\delta}\right\rceil\right\}.
\tag{18}
\]

For every input sequence, each M_j≥A_N. With probability at least 1−delta, all M_j≤U_N. Consequently, for the same event and the same uniform unknown-location tamper guarantee,

\[
\max\left\{
\left\lceil\log_2\sum_{k=0}^s\binom dk(A_N-1)^k\right\rceil,
\left\lceil2s\log_2 A_N\right\rceil
\right\}
\leq L_X^{\max}
\leq
\left\lceil\log_2\sum_{k=0}^{2s}\binom dk(U_N-1)^k\right\rceil.
\tag{19}
\]

An efficient sufficient bound on that event is 2s checks over any prime p≥max(d+1,U_N), or the literal rank tuple if shorter. In the regime A_N≥d+1, a suitable p<2U_N yields

\[
\left\lceil2s\log_2 A_N\right\rceil
\leq L_X^{\max}
\leq \left\lceil2s\log_2 U_N\right\rceil+2s.
\tag{20}
\]

Thus, in this specified large-residual-alphabet regime, necessity and efficient sufficiency have the same leading term

\[
2s\log_2\frac{q}{N+1},
\tag{21}
\]

up to an additive O(s log log(dq/delta)+s) gap. This is a logarithmic-in-N protected-bit decrease, not a universally linear-in-N one. Neither the regime nor the constants are asserted for the supplied Llama head.

**Proof.** N thresholds divide q ordered labels into at most N+1 nonempty blocks. The largest therefore has at least A_N labels. If a block contains k consecutive labels, no sample threshold lies between its first and last labels, an interval of length (k−1)/q. Union-bound over at most q possible starting labels and d coordinates:

\[
\Pr(\max_j M_j\geq k)
\leq dq\left(1-\frac{k-1}{q}\right)^N
\leq dq e^{-N(k-1)/q}.
\tag{22}
\]

If U_N<q, apply (22) with k=U_N+1; otherwise the upper cardinality bound is automatic. Apply Theorem B to the worst realizable fiber and use monotonicity in each alphabet size. For (20), A_N≥d+1 implies the field can be chosen between U_N and 2U_N. The rank code works for every original and every allowed tamper; the probability is only over the input sequence. The quotient U_N/A_N is at most a constant times ln(dq/delta) for N≥1, yielding the stated additive gap. ∎

The encoder can calculate actual M_j or actual fiber sizes from the records. It does not have to trust the probabilistic upper bound when building its code. Outside the sample event it can use a larger valid certificate; the theorem bounds the high-probability required budget, not an undetectable failure of the encoding protocol.

The zero-bit threshold is also explicit: by (22) with k=2,

\[
N\geq q\ln(dq/\delta)
\quad\Longrightarrow\quad
\Pr(L_X^{\max}=0)\geq1-\delta.
\tag{23}
\]

When residual alphabets become small, equation (20)'s large-alphabet premise no longer applies. The general bound (19) remains valid but can become loose. In particular, residual uncertainty at many possible coordinates can retain a location-dependent coding cost. One should not extrapolate the earlier slope through this regime.

For known erasures, replace the confusability radius 2s by s throughout the coding step. In the corresponding large-alphabet regime the leading value term is s log2(q/(N+1)). This difference is why the coefficient inferred from local lists cannot be assigned automatically to the original uniform substitution problem.

**Exact binary endpoint.** In the same linear model with q=2, a coordinate remains unresolved precisely if none of its N thresholds lies between the two grid points. This has probability 2^−N, independently across coordinates. Therefore

\[
K_N\sim\operatorname{Binomial}(d,2^{-N}).
\]

For s=1 unknown error, the optimal budget for the observed input sequence is exactly

\[
L_X^{\max}=\lceil\log_2(K_N+1)\rceil,
\quad
\Pr(L_X^{\max}\leq r)
=\Pr(K_N\leq2^r-1).
\tag{24}
\]

The Hamming syndrome described above achieves it. This is a fully specified sample–certificate law and efficient matching construction, although its ingredients are elementary and classical. It is a control theorem, not sufficient ICML novelty on its own.

**6. Why the product assumption is a real restriction**

A generic neural output row depends on a sum of weight-feature products. Its transcript fiber couples the weights. It is not generally a product over coordinates. The supplied single-change lists depend on the damaged checkpoint and therefore cannot simply be inserted as the A_j of Theorem B.

There is a small linear counterexample showing why this matters. Let the model output a+b, with original weights a,b in {0,...,q−1}, and retain that sum exactly as a finite code. Consider the transcript a+b=q−1, so its q originals are

\[
\theta_i=(i,q-1-i),\quad i=0,\ldots,q-1.
\]

Allow at most one changed weight. For every two originals theta_i, theta_j, the checkpoint (i,q−1−j) is within one change of each. Thus the q originals form a clique and any pre-tamper certificate requires ceil(log2 q) bits. Saving a suffices, because the retained sum determines b, so this bound is tight.

Yet for any fixed damaged checkpoint there are at most two compatible originals: one obtained by restoring its first coordinate, the other by restoring its second. Its local log candidate count is at most one bit. The optimum can therefore be log2 q even when every post-tamper list has only two entries. Additional identical sum observations do not remove this ambiguity.

This is already a one-change linear-row repair problem. It demonstrates that replacing the full transcript fiber by a collection of local candidate lists is not a harmless simplification. It does not assert that the Llama columns exhibit this exact dependence. A paper claiming a sharp Llama-head rate needs an argument that handles its coupled fiber, or an explicit narrower model whose assumptions are actually satisfied.

**7. Implementations and validation**

`rank_repair.py` implements Theorem C on public integer-label intervals, including joint tuple packing and mapping damaged values outside the interval to a provisional rank. The encoder receives no tamper, support, or test residual. The decoder receives no original. It chooses the shorter public strategy among rank checks and a literal rank tuple. For binary residual alphabets it also uses classical Hamming/BCH syndromes to reduce the small-alphabet cost.

For a binary vector of length n, let m=ceil(log2(n+1)), choose a verified primitive representation of GF(2^m), and use distinct nonzero locators alpha_j. Saving the odd syndromes with powers 1,3,...,2t−1 uses tm bits. Binary amplitudes imply S_2k=S_k^2, so these determine all 2t consecutive moments needed to correct t unknown errors. Their first shifted Hankel matrix has nonzero amplitudes alpha_j, and the same locator proof applies. For s erasures, take t=ceil(s/2) and solve the erased columns as a binary linear system. These are classical constructions; no new BCH theorem is claimed. The reference field tables validate a full nonzero cycle before use.

`row_value_certificate.py` implements the report's separate conditional row-value code directly on uint16 labels. It stores s row-moment checks of row sums modulo 131071 for up to 131070 rows. At s=64 this is 1,088 meaningful bits and 136 bytes. The decoder must receive the correct row and column of every changed weight. This location list is not provided to the encoder. Wrong supplied columns can lead to an incorrect repair with matching row checks; this is not an authentication scheme.

The row code was executed with 128,256 rows, eight columns, and 64 changed labels, using a certificate written first. It recovered the entire array exactly. This exercises the full row-locator range and actual serialization but is **not** a Llama-head rerun or a runtime benchmark. The full 4,096-column head and its caches were not available.

The reproducible tests include 5,706 exhaustive unknown-error cases, 52,874 exhaustive erasure cases, 400 mixed-alphabet randomized cases, and 4,124 exhaustive-support checks of the binary BCH path for selected original vectors. They check actual pre-tamper encodings against the enumerated legal damages. Six high-alphabet configurations attain the proved optimal bit count. Exact witness verification is compared with brute-force enumeration, including distinct signed-zero labels and deduplication of unchanged vectors. The complete counts and field validations are in `validation.json`.

`run_analysis.py` generates the figure and its source data. The controlled panel uses d=24 diagonal weights, q=4096 labels, s=2 unknown changes, and nested sample prefixes. For each prefix it selects a worst-size realizable transcript fiber, computes it again from the original prediction record, writes a certificate, and only then generates 30 damages. There are 330 successful controlled reconstruction checks in total. Selecting this fiber is legitimate for evaluating the fixed-input uniform optimum; it is not a claim that the selected original was independently trained before these samples.

The controlled plot places proved lower bounds and constructed sufficient budgets under the same guarantee. For example, at N=256 the lower bound is 30 bits and the implemented certificate uses 31; at N=1024 both are 21 bits. At small alphabets the generic field code is less sharp; the binary BCH option reduces this gap at the binary endpoint. This is one controlled input sequence, not a Monte Carlo estimate of the probability in Theorem D.

The Llama panel is explicitly different: it displays the supplied average log candidate counts and trial standard deviations at N=256 and N=2048, alongside the conditional 1,088-bit code. It contains no certified Llama information lower bound and no new Llama achieved-recovery curve. The 3,968-bit data-free baseline is stated in its caption. Connecting these panels as if they shared a model or guarantee would be incorrect.

To reproduce the new work with Python, NumPy, and Matplotlib:

```bash
python verify_repair.py
python run_analysis.py --audit-json /absolute/path/audit_v3_report.json
```

**8. Research decision and remaining evidence**

This phase supplies a correct value-decay argument and a rigorous sample–certificate theorem with genuinely matching constructions in specified regimes. It also identifies a genuine obstruction to treating the reported 121-bit count as the target optimum. These results are mathematically useful, but their coding and scalar-quantization ingredients are classical. I would not describe them alone as establishing exceptional ICML novelty.

The remaining central problem is precise: handle the coupled transcript fiber of the intended one-change-per-row head, without assuming its damaged-checkpoint-specific lists were available to the encoder. More model scales, an internal layer, or a new fitted tail exponent will not resolve this. Nor should one promise that 121 bits are achievable before an upper bound for the same guarantee exists.

For the empirical converse, the smallest missing item is the `C_uniform_converse` source cell and its per-row candidate/count data, including verification convention, hits, draws, and any deduplication. An explicit family of accepted raw-label alternatives is more useful than another aggregate median. A canonical recording function and the corresponding features/row values/codes are needed to validate actual witnesses. The supplied JSON alone cannot support that reconstruction.

For a full 1,088-bit Llama demonstration, the original/damaged label arrays and retained-record locator outputs are needed on the user's GPU machine. The new row-code implementation is ready to be connected to them, but its test here does not establish the locator's correctness. No broad code repository is needed.

The bounded next step is to validate one actual common-transcript witness family and decide whether a non-product upper bound can be related to it. If that connection yields only the existing generic graph bounds, stop expanding the ICML claim rather than accumulating auxiliary results. If it yields a new sharp rate or a substantive construction for the intended head class, the present proof and code provide a clean foundation for the paper.
