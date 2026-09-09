# Protected model repair from retained predictions

## A consolidated proof document and review of the Llama evidence

**Question.** How many protected bits must we write before a sparse change to restore a model, when its finite-precision predictions on N inputs are already available?

**Main assessment.** We have a sound information-theoretic formulation, a conditional sample–certificate bound exploiting sparse differences, and an implementable decoder with a precise error-and-erasure guarantee. A complete Llama-head run now supports the construction's feasibility. We do not yet have a matching lower bound for the Llama setting, a sample-complexity theorem for its actual localization rule, or an implementation achieving the behavior-sensitive information bound. These are distinctions between established claims, not implementation details to hide.

The strongest current experimental statement is:

> In a controlled last-layer experiment, 256 retained 8-bit full-logit predictions supply location hints that allow a development-selected 3,534-bit certificate to restore a 64-weight BF16 tamper. The complete certificate-to-repair path runs in about 9.4 seconds once the head and features are available.

That is a sufficient repair budget, not a measurement of the minimum necessary information.

### Reading guide

Sections 1–3 specify the problem and its information limits. Sections 4–5 prove what iid examples and sparse geometry can resolve. Sections 6–8 construct and prove the practical decoder and explain its relationship to behavioral relevance. Sections 9–12 audit the evidence, novelty, and remaining claims. All mathematical results have proofs below; references identify the classical ingredients rather than substituting for proofs.

## 1. The problem and the resources

### 1.1 Original, tamper, observations, and certificate

Let \(\Theta\) be a nonempty finite class of possible original model representations. Architecture, finite weight alphabet, and public restrictions specify the class before identifying the particular original. The class must not consist of an uncharged copy of that original disguised as public information.

Write \(f_\theta(x)\) for the recorded prediction. Raw logits, probabilities, and generated token identities are different observation models. Our Llama experiments retain **raw logits for every vocabulary entry**, quantized to eight bits; they do not retain generated answers or ground-truth training labels.

Each original \(\theta\) has a nonempty set \(\mathcal T(\theta)\) of allowed damaged checkpoints. A principal example is at most \(s\) changed scalar weights, optionally with a numerical magnitude limit \(\rho\) on each change. The information results permit a different damaged-checkpoint alphabet. The implemented algebraic decoder specializes to damaged coordinates that remain valid finite weight labels.

For fixed inputs \(X=(x_1,\ldots,x_N)\), define the retained transcript

\[
\ell_X(\theta)=\bigl(Q_b(f_\theta(x_1)),\ldots,Q_b(f_\theta(x_N))\bigr).
\tag{1}
\]

The quantizer includes its range, clipping convention, cell boundaries, and any metadata necessary to interpret the codes. Input identities and ordering are also available. The transcript is intact after tampering. If an original-dependent scale, output selection, or codebook is transmitted, it belongs to the information accounting.

For a realizable transcript \(\ell\), let

\[
\Theta_\ell=\{\theta:\ell_X(\theta)=\ell\},\qquad
F_\ell(z)=\{\theta\in\Theta_\ell:z\in\mathcal T(\theta)\}.
\tag{2}
\]

Thus \(F_\ell(z)\) is precisely the set of originals still possible after seeing both the examples and the damaged checkpoint.

The certificate \(C_\ell(\theta)\in\{0,1\}^L\) is written **after the examples are available and before the tamper is chosen**. The decoder receives \((X,\ell,z,C_\ell(\theta))\) and public settings. It does not receive the original checkpoint, original error locations, original error values, or an oracle test of candidate correctness.

A certificate length can be fixed from public settings or from development data before a test tamper. Selecting the length from the realized test residual is a different, unauthorized information protocol.

### 1.2 Behavioral target and exact recovery

Let \(d_{\mathrm{beh}}\) be a fixed metric or pseudometric on \(\Theta\). Require

\[
d_{\mathrm{beh}}\bigl(\theta,\operatorname{Dec}_\ell(z,C_\ell(\theta))\bigr)\leq\varepsilon.
\tag{3}
\]

The decoder returns a member of \(\Theta\); changing the reconstruction class changes some bounds. The tolerance is in distance units, not squared-distance units. Population mean total variation is one suitable pseudometric. KL divergence is not a metric and cannot be inserted into the triangle-inequality proofs below without a separate argument.

A pseudometric deliberately ignores parameter differences that do not matter on the domain. Exact representation recovery is stronger: it asks for the original stored symbols themselves. Setting \(\varepsilon=0\) in a behavioral pseudometric need not imply identical symbols. The Llama certificate decoder targets the stronger, exact-symbol goal.

### 1.3 Four separate costs

| Resource | Role | Accounting |
|---|---|---|
| Original/damaged weight representation | Defines the possible symbols and numerical changes | Stored weight bits, plus scales/codebooks |
| Retained input–output record | Side information already available to repair | For N inputs, m output scalars, b bits each: Nmb output bits, plus inputs and metadata |
| Protected certificate | Additional information written before tampering | L meaningful bits; separately report byte padding |
| Working computation | Produces proposals and computes checks | Arithmetic format, runtime, working memory |

There is no theorem equating one ordinary record bit with one protected bit. The distinction is an operational assumption: examples are already retained and intact, whereas additional protected model-dependent storage is budgeted. If retaining and protecting the examples has the same cost as protecting arbitrary bits, a total-storage comparison must count them too.

FP32 temporary arithmetic does not supply extra original-model information when computed from permitted decoder inputs. It does incur compute and memory costs. BF16 weights and an INT8 checkpoint are different original models; repairing the latter restores its own quantized original, not the BF16 reference.

### 1.4 Quantifiers

For fixed \(X,\ell\), define \(L^*_{X,\ell}(\varepsilon)\) as the minimum fixed certificate length satisfying (3) for **every** \(\theta\in\Theta_\ell\) and every \(z\in\mathcal T(\theta)\). We suppress \(X\) when fixed.

For iid inputs, an operational uniform budget is

\[
L^{\mathrm{unif}}(N,\varepsilon,\delta)
=\min\left\{L:\Pr_X\left[\max_{\ell\text{ realizable}}L^*_{X,\ell}(\varepsilon)\leq L\right]\geq1-\delta\right\}.
\tag{4}
\]

These uniform guarantees differ from success probability over a chosen distribution of tampers against one fixed Llama head. Our empirical budgets estimate the latter for a particular decoder. They do not evaluate (4).

The attacker may know the transcript and certificate. A uniform successful event below already covers all legal tampers on that event; no security through hidden code parameters is assumed.

## 2. Basic information bounds

### Theorem 1. A separated candidate family forces protected bits

If some \(F_\ell(z)\) contains \(M\) originals with pairwise behavioral distance strictly greater than \(2\varepsilon\), then

\[
L^*_\ell(\varepsilon)\geq\lceil\log_2 M\rceil.
\tag{5}
\]

**Proof.** Two such originals assigned the same certificate would give the decoder identical inputs. Its output could not lie within \(\varepsilon\) of both, by the triangle inequality. Thus all \(M\) originals require distinct certificate strings, and \(2^L\geq M\). ∎

Equivalently, with \(\operatorname{Pack}_{>r}(A)\) the maximum size of a strictly r-separated subset,

\[
L^*_\ell(\varepsilon)\geq
\left\lceil\log_2\max_{z:F_\ell(z)\ne\varnothing}
\operatorname{Pack}_{>2\varepsilon}(F_\ell(z))\right\rceil.
\tag{6}
\]

The candidate family must share the **same checkpoint and the same actual transcript**. Counting arbitrary sparse vectors, or observing that an optimizer failed, does not produce this lower bound.

### Theorem 2. Exact partition characterization

For nonempty \(A\subseteq\Theta\), define

\[
\operatorname{rad}(A)=\min_{a\in\Theta}\max_{\theta\in A}d_{\mathrm{beh}}(\theta,a),
\qquad \operatorname{rad}(\varnothing)=0.
\]

Let \(K_\ell(\varepsilon)\) be the fewest groups in a partition of \(\Theta_\ell\) such that every group \(A\) satisfies

\[
\operatorname{rad}(A\cap F_\ell(z))\leq\varepsilon
\quad\text{for every }z.
\]

Then

\[
L^*_\ell(\varepsilon)=\lceil\log_2K_\ell(\varepsilon)\rceil.
\tag{7}
\]

**Proof.** The equal-certificate classes of any valid encoder form a partition into at most \(2^L\) groups. For each group and checkpoint, the decoder output is a valid center of the remaining candidates. Conversely, encode the group index of a qualifying partition. For a received group and checkpoint choose a center of their candidate intersection. Such a center exists by finiteness. For impossible inputs return an arbitrary model. ∎

A separate encoding chosen after observing the damaged checkpoint would miss the pre-tamper requirement. Equation (7) uses one partition across every possible checkpoint.

### Corollary 2.1. Zero bits, covers, and monotonicity

\[
L^*_\ell(\varepsilon)=0
\iff \operatorname{rad}(F_\ell(z))\leq\varepsilon\text{ for all }z.
\tag{8}
\]

If \(\operatorname{Cov}_\varepsilon(\Theta_\ell)\) is the minimum number of radius-\(\varepsilon\) balls centered in \(\Theta\) covering the transcript fiber, then

\[
L^*_\ell(\varepsilon)\leq
\lceil\log_2\operatorname{Cov}_\varepsilon(\Theta_\ell)\rceil.
\tag{9}
\]

A refined record with \(\Theta_{\ell'}\subseteq\Theta_\ell\) cannot increase the optimum.

**Proof.** Equation (8) is the one-group case. For (9), store an index of a covering center that contains the original, ignoring the damaged checkpoint. For refinement, restrict the old partition to the smaller fiber; previously valid centers remain valid. ∎

Nested prefixes of one example sequence yield refinement. Higher output precision yields refinement only for nested quantizers. Changing weight alphabets changes the model class, so this does not establish a general ordering between BF16 and INT8 repair budgets. Nor must a development-selected heuristic budget decrease with every added sample.

## 3. A graph view and finite-sample bounds

### Proposition 3. Graph bounds

For \(r\geq0\), make a graph \(G_r(\ell)\) on \(\Theta_\ell\). Join two originals when they can produce a common damaged checkpoint and their distance exceeds r. Then

\[
\lceil\log_2\chi(G_{2\varepsilon}(\ell))\rceil
\leq L^*_\ell(\varepsilon)
\leq\lceil\log_2\chi(G_\varepsilon(\ell))\rceil.
\tag{10}
\]

**Proof.** A valid encoder must distinguish each edge of the first graph, by Theorem 1's two-point argument; it is therefore a proper coloring. For the upper bound, transmit a color in a proper coloring of the second graph. Any two candidates with that color and a shared checkpoint have distance at most \(\varepsilon\). Returning any one candidate meets the target. ∎

At \(\varepsilon=0\) the bounds coincide. For positive tolerance, pairwise constraints need not characterize a common radius-\(\varepsilon\) center; Theorem 2 remains the exact statement. The connection between zero-error side information and coloring is classical, as in [Witsenhausen, 1976](https://dl.acm.org/doi/10.1109/TIT.1976.1055607). We do not claim graph coloring as a new coding principle.

### 3.1 Independent samples

Assume \(X_1,\ldots,X_N\) are iid from a fixed distribution P. The model class, tamper relation, and per-input observation function are fixed independently of this sample. Fresh prediction inputs are the clean experimental interpretation. Merely knowing the training examples does not justify these assumptions if the model class or analysis was selected using those same examples; reuse needs its own conditioning or generalization argument.

For a pair of originals define

\[
\alpha_b(\theta,\theta')
=\Pr_{X\sim P}\{Q_b(f_\theta(X))\ne Q_b(f_{\theta'}(X))\}.
\tag{11}
\]

Let \(\mathcal E_r\) be the unordered pairs that share some legal damaged checkpoint and have distance greater than r. Write \(H_r=|\mathcal E_r|\), and

\[
S_N(r,b)=\sum_{\{\theta,\theta'\}\in\mathcal E_r}
(1-\alpha_b(\theta,\theta'))^N.
\tag{12}
\]

At N=0 each factor is one, including when \(\alpha_b=1\).

### Theorem 4. Pair survival and simultaneous resolution

A fixed pair has identical N-example transcripts with probability exactly \((1-\alpha_b)^N\). Consequently,

\[
\Pr_X\{\exists\ell,z:\operatorname{diam}(F_\ell(z))>r\}\leq S_N(r,b).
\tag{13}
\]

If \(\alpha_b\geq\kappa>0\) for every pair in \(\mathcal E_r\), then

\[
S_N(r,b)\leq H_r e^{-\kappa N}.
\]

For \(H_r\geq1\), the condition

\[
N\geq\frac{\ln H_r+\ln(1/\delta)}{\kappa}
\tag{14}
\]

ensures all candidate sets have diameter at most r with probability at least \(1-\delta\). Setting \(r=\varepsilon\) proves zero-certificate recovery on that event. If \(H_r=0\), the diameter conclusion holds without samples.

**Proof.** Independent input draws give the pair-survival probability. A candidate set of diameter greater than r contains a pair in \(\mathcal E_r\) with equal transcripts. Union-bound these events and use \(1-u\leq e^{-u}\). A nonempty set of diameter at most \(\varepsilon\) has radius at most \(\varepsilon\) by choosing any member. ∎

Replacing r by \(2\varepsilon\) does not generally prove zero-bit recovery at tolerance \(\varepsilon\).

### Theorem 5. An intermediate certificate upper bound

Let \(E_N(r)\) count surviving pairs in \(\mathcal E_r\). For an integer \(j\geq0\),

\[
\Pr\{E_N(r)\geq j+1\}\leq S_N(r,b)/(j+1).
\tag{15}
\]

Define \(c(j)=\lfloor(1+\sqrt{1+8j})/2\rfloor\). On \(E_N(\varepsilon)\leq j\), simultaneously for every transcript,

\[
L^*_\ell(\varepsilon)\leq\lceil\log_2 c(j)\rceil.
\tag{16}
\]

**Proof.** The expected surviving-pair count is \(S_N\), so (15) is Markov's inequality. A graph of chromatic number k contains a vertex-critical k-chromatic subgraph with at least k vertices, each of degree at least k−1; it therefore has at least \(k(k-1)/2\) edges. Each transcript graph has at most j edges. Solve the inequality for k and apply Proposition 3. ∎

For \(S_N>0\), choosing \(j=\lfloor S_N/\delta\rfloor\) gives a failure probability less than \(\delta\). If \(S_N=0\), choose j=0. This is an existence bound; coloring a neural-model graph is not the practical decoder.

### Proposition 6. Why coverage and precision assumptions are necessary

Suppose two confusable originals have distance greater than \(2\varepsilon\) and separation probability \(\alpha\). With probability \((1-\alpha)^N\), their common transcript requires at least one protected bit. Thus a simultaneous zero-bit guarantee with confidence \(1-\delta\) requires

\[
(1-\alpha)^N\leq\delta.
\tag{17}
\]

**Proof.** The pair-survival event supplies the two-element family in Theorem 1. ∎

For \(0<\alpha<1\), this implies \(N\geq\ln(1/\delta)/[-\ln(1-\alpha)]\). If \(\alpha=0\), no finite N removes that pair.

Two examples explain the obstruction. For scalar predictions \(w h(X)\), let \(h=1/\sqrt p\) with probability p and zero otherwise. Models w=0 and w=u have RMS prediction distance \(|u|\), but when the rare outputs are distinguished their separation probability is only p. The normalized fourth moment is \(1/p\). This example presumes the output range admits the rare values. Alternatively, let h=1 always and choose two weights in the same quantizer cell, farther apart than \(2\varepsilon\), with a common allowed tamper. Their separation probability is zero. Repeated iid observations do not reveal additional precision.

## 4. Sparse geometry and a sample–certificate theorem

### 4.1 Finite symbols and numerical values are different objects

There are d protected coordinates. Each original coordinate has a public label in a finite alphabet \(\mathcal A\subseteq\{0,\ldots,q-1\}\), mapped to a numerical value by \(\nu\). Write a for the label vector and \(w=\nu(a)\) for its numerical vector. Nonfinite original values are excluded. If distinct labels have the same numerical value, as with signed zero, the behavioral metric may identify them although exact-symbol repair distinguishes them.

This notation accommodates actual BF16 bit patterns. It avoids treating subtraction of floating-point bit patterns as numerical subtraction. For exact-symbol coding, sparsity counts changed representations, including a change between signed-zero patterns if allowed; a numerical-value-only change count would not bound such symbol errors.

Freeze the feature map and write ideal retained outputs as

\[
f_a(x)=A(x)\nu(a)\in\mathbb R^m.
\tag{18}
\]

For a last-layer matrix W, this is \(Wh(x)\). For this section specialize the behavioral metric to a public PSD matrix M:

\[
d_M(a,a')=\|\nu(a)-\nu(a')\|_M,
\qquad \|u\|_M=\sqrt{u^TMu}.
\tag{19}
\]

M may be singular. Examples include \(M=\mathbb E[A(X)^TA(X)]\), giving RMS logit distance, or a declared reference-weighted quadratic form. They are not automatically KL distance. No dense M is constructed in the Llama experiment.

### 4.2 Sparse signatures

Two originals sharing an s-sparse damaged checkpoint differ in at most \(k_{\max}=\min(2s,d)\) coordinates. If every tamper changes numerical values by at most \(\rho\), their numerical differences are bounded by \(2\rho\).

A signature \(\sigma\) records a nonempty support T of size at most \(k_{\max}\), and distinct ordered labels \((a_j,a'_j)\) on that support. It defines

\[
u_{\sigma,j}=\nu(a_j)-\nu(a'_j),\qquad
v_{\sigma,j}=a_j-a'_j,
\tag{20}
\]

with zeros elsewhere. The numerical vector u controls behavior and predictions, while the integer vector v controls the certificate. We include all allowed label pairs satisfying the numerical magnitude cap, whether or not the signature can occur in the restricted original class. This makes an outer set, never a lower-bound family.

Let \(\mathcal S_\varepsilon\) contain the signatures with \(\|u_\sigma\|_M>\varepsilon\), and let \(K_\varepsilon=|\mathcal S_\varepsilon|\). If \(r_{\mathcal A}\) is the number of permitted ordered distinct label pairs, then

\[
K_\varepsilon\leq\sum_{k=1}^{k_{\max}}\binom dk r_{\mathcal A}^k,
\qquad r_{\mathcal A}\leq q(q-1).
\tag{21}
\]

For \(1\leq k_{\max}\leq d/2\) and \(r_{\mathcal A}\geq1\),

\[
K_\varepsilon\leq k_{\max}
\left(\frac{\mathrm e d r_{\mathcal A}}{k_{\max}}\right)^{k_{\max}}.
\tag{22}
\]

**Proof.** Count choices of support and ordered pairs. Use \(\binom dk\leq(\mathrm e d/k)^k\); the function \(k\ln(\mathrm e d r_{\mathcal A}/k)\) is nondecreasing on this range, so each summand is at most the final bound. If no pairs are allowed, the set is empty. ∎

The logarithm scales with \(s\log(d/s)+s\log q\), rather than describing every full original separately. The count can still be huge.

For a uniform grid \(\nu(j)=a_0+\Delta j\), numerical and label differences satisfy \(u=\Delta v\), so signatures with identical v can be merged. With \(r=\min(q-1,\lfloor2\rho/\Delta\rfloor)\), the count becomes

\[
K_\varepsilon\leq\sum_{k=1}^{k_{\max}}\binom dk(2r)^k.
\tag{23}
\]

Without a magnitude limit set r=q−1. This simplification does not apply directly to nonuniform BF16 values.

### 4.3 Quantizer widths and unresolved differences

Suppose every reachable scalar quantizer cell has width at most \(w_b\). A uniform unsaturated b-bit quantizer over \([-R,R)\) gives

\[
w_b=2R\,2^{-b}.
\]

Equal codes imply coordinatewise output differences at most \(w_b\). Define

\[
\mathcal S_{X,b,\varepsilon}=
\{\sigma\in\mathcal S_\varepsilon:
\|A(X_i)u_\sigma\|_\infty\leq w_b\text{ for all }i\},
\quad U_{X,b,\varepsilon}=|\mathcal S_{X,b,\varepsilon}|.
\tag{24}
\]

This is only an outer approximation to actual equal-code ambiguity. Small differences can cross cell boundaries; satisfying (24) does not prove equal codes. Unbounded clipping cells invalidate the width assumption.

### Theorem 7. A pre-tamper random certificate

Choose a prime \(p\geq q\) and a public matrix \(B\in\mathbb F_p^{t\times d}\) with independent uniform entries, independent of the original and conditionally independent of everything else after X is fixed. Before tampering, store

\[
C_B(a)=Ba\pmod p.
\tag{25}
\]

Decode by returning any original in \(\Theta\) compatible with the received checkpoint, the **actual** transcript, and (25). For fixed X, with probability at least \(1-U_{X,b,\varepsilon}p^{-t}\) over B, this decoder achieves distance at most \(\varepsilon\) for every original and every allowed tamper.

**Proof.** For any signature, v is nonzero modulo p: a nonzero label difference has absolute value below p. A uniform row has zero dot product with v with probability \(1/p\), and t independent rows give \(p^{-t}\). Union-bound all signatures in (24). On the event that none has zero syndrome, any returned candidate farther than \(\varepsilon\) from the true original would supply a harmful surviving signature with zero syndrome, a contradiction. The true original is always feasible. This event excludes all bad pairs at once, including those an attacker chooses after seeing B. ∎

For \(U\geq1\),

\[
t=\left\lceil\frac{\ln U+\ln(1/\delta_B)}{\ln p}\right\rceil
\tag{26}
\]

suffices for setup failure probability at most \(\delta_B\); if U=0 take t=0. Tuple packing uses \(\lceil t\log_2p\rceil\) bits, while separate fixed-width checks use \(t\lceil\log_2p\rceil\).

This construction may require a large public random matrix, expensive counting, and exhaustive decoding. A short pseudorandom seed is not automatically equivalent to an independent matrix. Those costs are not part of L but must not be described as negligible.

### Theorem 8. Joint sparse sample–certificate bound

For each harmful signature let

\[
\beta_b(\sigma)=\Pr_{X\sim P}\{\|A(X)u_\sigma\|_\infty>w_b\}.
\]

For iid inputs and independent B, the probability that the simultaneous recovery guarantee fails is at most

\[
p^{-t}\sum_{\sigma\in\mathcal S_\varepsilon}(1-\beta_b(\sigma))^N.
\tag{27}
\]

If \(\beta_b(\sigma)\geq\kappa>0\) uniformly, this is at most

\[
K_\varepsilon e^{-\kappa N}p^{-t}.
\tag{28}
\]

For \(K_\varepsilon\geq1\), the sufficient certificate allocation is

\[
t=\max\left\{0,
\left\lceil\frac{\ln K_\varepsilon+\ln(1/\delta)-\kappa N}{\ln p}\right\rceil\right\}.
\tag{29}
\]

For an empty harmful set take t=0.

**Proof.** A fixed signature survives all sample-width constraints with probability \((1-\beta_b)^N\), and independently collides under B with probability \(p^{-t}\). Every harmful ambiguity requires both events for some signature. Union-bound the joint events, then apply \(1-u\leq e^{-u}\) and solve (28) for t. ∎

The heterogeneity in (27) is the useful interpretation: examples remove changes that they frequently expose; rarely visible changes survive longer. Equation (29) is an achievable staircase bound. It is not a universal linear law for the optimum L(N), a bit-for-bit exchange between records and certificates, or a guarantee for the structured decoder in Section 6.

A data-free upper cap is exact recovery with \(\min(2s,d)\) Vandermonde checks over a sufficiently large field. The random construction need only be used when its bound is better and its computation is acceptable.

## 5. Assumptions that make samples informative

### Proposition 9. Visibility from a moment condition

For a numerical difference u let \(g_u(X)=\|A(X)u\|_\infty\). Suppose on the relevant harmful sparse differences that

\[
\mathbb E g_u^2\geq c^2\|u\|_M^2,\qquad
\mathbb E g_u^4\leq C(\mathbb E g_u^2)^2,
\tag{30}
\]

where c>0 and C≥1. For any \(\tau\in(0,1)\),

\[
\Pr\{g_u\geq \tau c\|u\|_M\}
\geq\frac{(1-\tau^2)^2}{C}.
\tag{31}
\]

Consequently, if \(\tau c\varepsilon>w_b\), Theorem 8 applies with \(\kappa=(1-\tau^2)^2/C\).

**Proof.** Put Y=\(g_u^2\), \(\mu=\mathbb E Y>0\), and \(A=\{Y\geq\tau^2\mu\}\). Then

\[
(1-\tau^2)\mu\leq\mathbb E[Y1_A]
\leq\sqrt{\mathbb E[Y^2]\Pr(A)}.
\]

Squaring gives the probability bound. On A, \(g_u\geq\tau\sqrt\mu\geq\tau c\|u\|_M\). For harmful signatures this exceeds the quantizer width under the stated condition. ∎

The same proof yields Theorem 4's pairwise visibility assumption if \(\|u\|_M\) is replaced by the chosen behavioral distance and the moment domination is proved for that distance.

For a scalar linear output and RMS output distance, c=1. A fixed-reference softmax quadratic form is

\[
\|u\|_{M_{\mathrm{ref}}}^2
=\mathbb E_X\operatorname{Var}_{v\sim p_{\mathrm{ref}}(\cdot|X)}[(A(X)u)_v].
\tag{32}
\]

Since variance is at most the largest squared coordinate, c=1 is also valid in (30). This metric ignores common logit shifts and downweights unlikely outputs. The reference distribution must be declared, and any model-dependent information supplied to the decoder must be accounted for. A fourth-moment bound is still needed. This form is a surrogate, not an asserted global equality to KL or TV.

A large fourth-moment ratio indicates that an effect is concentrated on rare inputs. This explains a possible mechanism for poor sample coverage. It does **not** certify that the actual Llama feature distribution satisfies a useful uniform C, and the completed experiments did not estimate that constant over all relevant sparse directions.

### 5.1 Numerical execution and the theorem's observation model

Theorems 1–3 hold for any declared finite transcript, including a computational one. The exact pair-survival identity in Theorem 4 additionally requires a fixed per-input observation function. Batch-dependent numerical outputs do not automatically satisfy that specification.

For the geometric bound, a rigorous extension is possible if computed scalar outputs differ from their ideal linear values by at most \(\eta\), uniformly over the relevant originals, inputs, and permitted execution shapes. Equal quantized computed outputs then imply

\[
\|A(X_i)u\|_\infty\leq w_b+2\eta.
\tag{33}
\]

**Proof.** Insert and subtract the two computed outputs. Their difference is at most one cell width, and each computation contributes at most \(\eta\). ∎

Replace \(w_b\) by \(w_b+2\eta\) in (24)–(31). The independent-input width events still justify that sufficient geometric bound. A measured constraint slack is not a proof of a uniform \(\eta\).

For a row correction fitted against computed current logits, two output-rounding errors can enter the interval bounds; solver arithmetic contributes further error. A slack must cover the relevant discrepancies if feasibility is claimed. The previous cache mismatch was consistent with a shape-dependent rounding issue, but the assertion alone did not prove its cause. Model version, cache provenance, execution settings, and batch shape are all relevant. A successful small cache check is useful validation, not a universal numerical-error theorem.

This distinction does not affect the exact finite-field recovery theorem below: that theorem is conditional on the actual erasure/error counts, however the proposal was obtained.

## 6. A structured certificate and exact decoder

### 6.1 Encoder

Let \(a\in\{0,\ldots,q-1\}^d\) be the original symbol vector. Choose a prime

\[
p>\max(d,q-1),\qquad \alpha_j=j+1\quad(0\leq j<d).
\]

Before tampering write the t residues

\[
c_k=\sum_{j=0}^{d-1}a_j\alpha_j^k\pmod p,
\qquad k=0,\ldots,t-1.
\tag{34}
\]

No future tamper locations are needed. Locator generation is public and does not require storing a dense random matrix. This is a classical Vandermonde error-correction construction, not a new coding theorem; the polynomial-code foundation is [Reed and Solomon, 1960](https://epubs.siam.org/doi/10.1137/0108018).

For Llama's head,

\[
d=128256\times4096=525336576,
\quad q=2^{16},\quad p=2^{31}-1.
\]

Labels are unsigned BF16 bit patterns. Finite original values use a subset of the 65,536 possible patterns. Separately packing each residue uses 31 bits. FP32 or INT64 temporary buffers do not change that stored payload. A full unrestricted 32-bit symbol alphabet would need a different field or representation.

### 6.2 Data stage and residual

Using only permitted inputs, a data rule returns a provisional symbol vector \(\bar a\) and an erasure set E. Let

\[
e=|E|,\qquad u=|\{j\notin E:\bar a_j\ne a_j\}|.
\]

An erasure is a flagged coordinate whose value remains unknown; a false flag consumes the same check as a true flag. The decoder knows E but does not know u.

Compute the residual syndrome directly from \(\bar a\):

\[
S_k=c_k-\sum_j\bar a_j\alpha_j^k\pmod p.
\tag{35}
\]

### Theorem 10. Exact recovery from examples and a syndrome

Suppose \(e\leq t\leq d\) and

\[
e+2u\leq t.
\tag{36}
\]

Then the original vector is the unique field vector with syndrome c and at most \(\lfloor(t-e)/2\rfloor\) disagreements with \(\bar a\) outside E. It can be recovered algebraically without enumerating the \(\binom ds\) possible supports.

**Uniqueness proof.** Two candidates in that radius differ only on E and at most \(2\lfloor(t-e)/2\rfloor\) other positions, hence at most t positions total. Equal syndromes would put their difference in the kernel of the t-row Vandermonde matrix. Any k≤t columns are independent: their first k rows have nonzero Vandermonde determinant because the locators are distinct. The difference must vanish. The original lies in the stated radius. ∎

**Constructive proof.** Define

\[
F(x)=\prod_{j\in E}(x-\alpha_j)=\sum_{l=0}^e f_lx^l,
\quad
T_k=\sum_{l=0}^e f_l S_{k+l},\quad 0\leq k<t-e.
\tag{37}
\]

Writing \(d_j=a_j-\bar a_j\pmod p\), expansion gives

\[
T_k=\sum_{j\notin E}d_jF(\alpha_j)\alpha_j^k.
\tag{38}
\]

There are u nonzero terms with distinct locators and nonzero amplitudes. Let their locator polynomial be \(G(x)=\prod_{j\notin E:d_j\ne0}(x-\alpha_j)\), monic of degree u. Its coefficients give a recurrence for T. If u>0, the u-by-u Hankel matrix \((T_{i+j})_{i,j=0}^{u-1}\) factors as

\[
V\,\operatorname{diag}(d_jF(\alpha_j))\,V^T,
\]

where V is a nonsingular u-by-u Vandermonde matrix. Thus the recurrence has degree exactly u and is uniquely determined by at least 2u available moments. Berlekamp–Massey recovers it without knowing u in advance. Factor G over the field to recover its roots and hence the outside-error positions; all roots must be valid non-erased locators. If u=0 the outside list is empty.

On the union of those positions and E, solve for d from the first e+u original syndrome equations. The square Vandermonde system is nonsingular, and e+u≤t. Check all t equations and return \(a=\bar a+d\pmod p\). The recovered original is in its declared alphabet. This is a finite constructive decoder. ∎

The recurrence algorithm is classical: [Massey, 1969](https://ieeexplore.ieee.org/document/1054260/). The implementation uses polynomial factorization with bounded randomized splitting; exhaustion raises an algorithmic failure. Uniqueness does not guarantee a bounded randomized implementation never exhausts its attempts. A public-seed retry does not require extra certificate bits, and any retries should be included in runtime and failure reporting.

### 6.3 Cost and limitations

Computing a full certificate or a full damaged-head syndrome costs O(td) field operations in the implementation. The small algebraic work is polynomial in t: the reference uses quadratic polynomial operations and a cubic small linear solve. Polynomial arithmetic also depends on the field bit length. It does not scan all d locators during root finding.

The GPU field arithmetic uses signed INT64. Each product of residues is below \(p^2<2^{62}\); Mersenne reduction is exact. Each chunk has at most \(2^{20}\) terms, reduced before summing, so the chunk sum is below \(2^{51}\). These bounds prevent signed overflow. Numerical head fitting and modular label arithmetic remain separate.

When (36) fails, the decoder can reject **or return a wrong vector with the same syndrome**. Matching a short certificate does not authenticate correctness against every adaptive tamper. The evaluator can compare to the original; the deployed decoder cannot silently receive that comparison or a free original hash.

With no data hints, take \(\bar a\) equal to the damaged checkpoint and E empty. For at most s wrong symbols, 2s checks suffice (assuming 2s≤d). This is a deterministic sufficient baseline. It is not a converse for the BF16 alphabet or for a more restricted tamper class.

## 7. What the examples buy for this decoder

### Proposition 11. Localization accounting

In localization-only mode, leave the damaged labels unchanged in \(\bar a\). Suppose there are \(s_0\) true wrong symbols; E contains c true locations and v false flags. Then

\[
e=c+v,\quad u=s_0-c,\quad
T_{\mathrm{burden}}=e+2u=2s_0-c+v.
\tag{39}
\]

**Proof.** A true flag removes one error from the outside-error count and adds one erasure, reducing the burden by one. A false flag adds an erasure without removing an error. Substitute the counts. ∎

Therefore each correct location hint saves one field check and each false flag costs one relative to 2s₀. If all s₀ locations are known, s₀ checks suffice for their values. In localization-only mode the burden cannot fall below s₀; reaching fewer checks requires additional value information or a different code.

For a fixed e with 64 actual errors,

\[
T_{\mathrm{burden}}=e+2(64-c).
\]

This is the informative measured quantity. The number of changed output cells, a solver loss, or a mean support recall by itself is not a certificate budget.

### 7.1 Numerical proposal used by the experiments

For one head row, let H contain retained hidden vectors and let \([l_i,u_i)\) be the original output cells. Given the damaged numerical row z, fit a correction v:

\[
\min_v\|v\|_1\quad\text{subject to}\quad
l-Hz\leq Hv\leq u-Hz,\quad |v_j|\leq\rho.
\tag{40}
\]

Closed interval endpoints are an outer relaxation; finite arithmetic uses the declared slack. The true correction is feasible for ideal linear outputs under the magnitude rule. It need not be the minimum-ℓ1 solution. A finite iteration budget need not find a feasible solution.

Using \(a=l-Hz\), \(b=u-Hz\), and \(\tau\sigma\|H\|_2^2<1\), the primal–dual updates are

\[
\begin{aligned}
y^+&=y+\sigma H\bar v-\sigma\operatorname{clip}(y/\sigma+H\bar v,a,b),\\
v^+&=\operatorname{clip}\bigl(\operatorname{soft}(v-\tau H^Ty^+,\tau),-\rho,\rho\bigr),\\
\bar v^+&=2v^+-v.
\end{aligned}
\tag{41}
\]

**Derivation.** The indicator of the interval constraint is applied to Hv. Moreau's identity gives the first line from interval projection. The proximal map of \(\|v\|_1+1_{[-\rho,\rho]^D}(v)\) is soft-thresholding followed by clipping, giving the second. The third is extrapolation. These are the updates of [Chambolle and Pock](https://optimization-online.org/2010/06/2646/). ∎

The experiment compares all vocabulary output codes, fits only mismatching rows, and ranks coordinates by proposal magnitude. Invisible errors remain for the global certificate decoder. In exact per-row evaluation, an unchanged row cannot create a mismatch; finite arithmetic can violate that implication. The current solver uses 350 iterations and returns its maximum interval violation.

The chosen proposal magnitudes are only scores. The exact original values are supplied through syndrome decoding, not by trusting the continuous fit. The mathematical guarantee does not require the fit to converge if its actual flags meet (36).

### Corollary 11.1. From a data-rule guarantee to a pre-tamper budget

Suppose a fixed data rule, on a sample event of probability at least \(1-\delta\), produces \(e+2u\leq t_N\) simultaneously for all originals and legal tampers. A certificate of \(t_N\) checks chosen before tampering guarantees exact recovery on that event, subject to successful execution of the algebraic algorithm.

**Proof.** Apply Theorem 10 to each original and tamper on the common sample event. ∎

This is the precise missing link between N and the implemented code. Theorems 4 and 8 do not prove the hypothesis for the ℓ1 scoring rule. A small candidate set or a good random-matrix existence bound does not imply that this particular optimizer locates errors.

### Proposition 12. A conditional bridge from value accuracy to symbol accuracy

Suppose a uniform weight grid has spacing \(\Delta\), the true numerical vector is w, and a continuous estimate \(\widetilde w\) obeys \(\|\widetilde w-w\|_2\leq\eta\). Nearest-grid rounding makes at most

\[
r\leq\left\lfloor4\eta^2/\Delta^2\right\rfloor
\tag{42}
\]

wrong symbols, also bounded by d. With no erasures, 2r checks suffice when within the dimension regime.

**Proof.** A wrongly rounded coordinate must differ numerically from the original by at least \(\Delta/2\). Sum its squared error over wrong coordinates. ∎

This is conditional on an estimation bound not proved for (40). A global minimum BF16 spacing makes such a bound extremely weak. The current experiment does not use value proposals, so (42) is a clarification of the earlier theory rather than a reported practical achievement.

## 8. Behavioral relevance and what exact repair does not optimize

The information optimum ignores changes within the declared behavioral tolerance. The implemented full-head symbol code protects every changed coordinate equally. Selecting calibration-important weights for simulated attacks does not make the certificate behavior-aware.

### Proposition 13. A sufficient public relevance reduction

Let J be a public set of coordinates. Suppose

\[
\sup\{\|u\|_M:\operatorname{supp}(u)\subseteq J^c,
\|u\|_0\leq s,\|u\|_\infty\leq\rho\}\leq\varepsilon.
\tag{43}
\]

If the decoder restores the original exactly on J and leaves the damaged checkpoint unchanged outside J, its final error is at most \(\varepsilon\) in \(d_M\), provided that resulting reconstruction belongs to the declared reconstruction class.

A conservative sufficient condition for (43) is

\[
\rho\sum_{\text{largest }\min(s,|J^c|)}\sqrt{M_{jj}}\leq\varepsilon,
\quad j\in J^c.
\tag{44}
\]

**Proof.** The final error is confined to at most s untouched damaged coordinates outside J and obeys the magnitude cap. Apply (43). For (44), the PSD seminorm obeys the triangle inequality, so \(\|u\|_M\leq\sum_j|u_j|\|e_j\|_M=\sum_j|u_j|\sqrt{M_{jj}}\). Maximize over allowed supports. ∎

For a product alphabet the reconstruction membership condition is automatic when all remaining labels are valid. For an additional constrained model class it must be checked. The description of an original-dependent private J is not free. Joint directions and cancellations matter, so small individual weights or diagonal scores alone are not a behavioral theorem.

This optional reduction is rigorous but **not implemented or evaluated** in the completed Llama study. The current exact repair implies restoration of any behavior determined by the unchanged backbone and identical repaired head under the same execution conditions. It does not measure how many fewer bits would suffice at a nonzero behavioral tolerance.

## 9. Focused review of the completed experiments

### 9.1 Evidence available for this review

I inspected the three theory files and the current pilot and full-certificate notebook source. Numerical results below are those supplied by the user in this conversation. The per-trial CSVs, GPU logs, original checkpoint bytes, retained caches, and serialized certificate from the user's machine are not present in this workspace. Thus this is a source-and-results review, not an independent rerun or a verification of unprovided trial artifacts.

The final notebook was corrected from an incorrectly assumed erasure count of 64 to the actual frozen value of **16**, while retaining 114 checks. Its cache check was also corrected to reproduce the pilot's batch size. Neither change establishes a new mathematical result.

### 9.2 What the implemented instance actually is

| Item | Completed setting |
|---|---|
| Model used as experimental original | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Protected part | Entire untied output head, 525,336,576 scalar entries |
| Feature extractor | Fixed BF16 backbone; cached final-token hidden vectors |
| Original/head representation | BF16; raw bit-pattern labels for the code |
| Numerical head arithmetic | FP32, TF32 disabled |
| Ordinary outputs | Full vocabulary raw logits; uniform 8-bit cells over [−64,64) |
| Cell width | 0.5 logit units |
| Input distribution | Sampling with replacement from a fixed WikiText-2-derived empirical pool |
| N for the complete demonstration | 256 |
| Simulated damage | 64 numerical weight changes, one per affected row |
| Selected erasures | 16, from the fixed proposal rule |
| Certificate | 114 residues modulo 2³¹−1 |
| Evaluation-only information | True tamper and original-head SHA-256 |

Hidden vectors are reusable because the backbone is assumed intact. They are computational caches of the available inputs and unchanged feature extractor, not additional freely supplied information about the unknown head. Their storage and the cost of preparing them must nevertheless be included in a complete system inventory.

The tamper generator samples from calibration-selected row and column pools and uses legal INT8-derived numerical changes rounded to BF16. The result is about this damage distribution. It does not cover all s-sparse changes, physical memory bit flips, nonfinite faults, backbone corruption, or adversarially chosen supports.

The pretrained checkpoint is publicly downloadable. In the operational problem an unavailable original cannot be recovered by downloading or embedding a private copy for free. Here the public checkpoint is an experimental fixture kept outside the decoder's allowed access, representing an otherwise unavailable original. This access restriction must be stated; the experiment is not evidence that certificates beat redownloading that publicly available fixture.

### 9.3 Matched-precision results already supplied

The relevant development-selected budgets from the matched comparison are:

| Layout | N | BF16 certificate bits | INT8 certificate bits |
|---|---:|---:|---:|
| Concentrated | 64 | 3,968 | 3,968 |
| Concentrated | 128 | 3,968 | 3,968 |
| Concentrated | 256 | 3,968 | 3,968 |
| Dispersed | 64 | 3,968 | 3,968 |
| Dispersed | 128 | 3,658 | 3,782 |
| Dispersed | 256 | 3,534 | 3,534 |

The supplied summaries report actual algebraic decode success 1.0 in each cell. The source config specifies ten development and thirty test trials, but the actual executed counts and confidence intervals must be checked against the saved CSVs before publication. The full-scan demonstration is a replay of the predetermined first locked test instance, not an additional independent trial to append to that count.

The matched perturbations have closely aligned numerical magnitudes and behavioral damage. At N=256, equal budgets do not establish equivalence of BF16 and INT8 repair difficulty. The measured code uses the same 31-bit field checks for both, mainly changing its budget through localization quality. It does not directly use INT8's smaller value alphabet to shorten an individual check. This is a meaningful control against the previous unmatched-amplitude explanation, not a proof that weight precision has no information cost.

The earlier quantization controls were:

| Head | KL from BF16 reference | TV from BF16 reference |
|---|---:|---:|
| BF16 | 0 | 0 |
| INT8 | 0.000182 | 0.007303 |
| INT4 | 0.072091 | 0.140776 |

INT4's larger baseline behavioral shift supports treating it as a secondary utility control. These values are pre-tamper shifts, not restoration errors. They do not justify any claim that four-bit restoration reaches the original BF16 model.

### 9.4 Full-certificate demonstration

The user-reported complete result is:

| Quantity | Value |
|---|---:|
| Actual changed weights | 64 |
| True locations among 16 flags | 16 |
| False flags | 0 |
| Errors outside flags | 48 |
| Sufficient burden for this instance | 112 checks |
| Frozen certificate supplied | 114 checks |
| Meaningful certificate bits | 3,534 |
| Serialized certificate bytes | 442 |
| Ordinary output storage | 31.3125 MiB |
| Changed retained output cells | 5,934 |
| Proposal interval violation | 0.219346 |
| Proposal iterations | 350 |
| Decoded nonzero patch entries | 64 |
| Decoded support matches evaluator | True |
| Recovered values on support match evaluator | True |
| Complete repaired-head SHA-256 matches original | True |

The arithmetic is

\[
e+2u=16+2\cdot48=112\leq114.
\]

The flags have 100% precision and 25% recall on this instance. The decoder then finds and repairs the remaining 48 locations using the certificate. The development-selected 114-check budget corresponds, at e=16 and s₀=64, to allowing c≥15. The reported development mean recall 0.2484375, multiplied by 64, corresponds to a mean of 15.9 caught locations, consistent with that rule. The current instance catches 16, leaving two checks of slack.

The phrase “required checks” in the output should be read as **sufficient error-and-erasure burden for this instance**. The theorem does not say 112 checks are necessary among all possible encoders or that fewer checks could never decode this particular vector. Reporting 112 as the new allocated length would also use test information after the fact; the budget remains 114.

Exact support and label comparisons, together with the controlled fact that only the simulated tamper and returned patch change the head, give a direct equality argument. The whole-head hash is additional strong verification. Hash equality alone is not a mathematical collision-free proof and the hash is evaluator-only, not an extra 256-bit decoder certificate.

The violation 0.219346 prevents claiming that the numerical solver found a feasible optimum. It does not contradict Theorem 10, which uses the returned flags rather than the proposal's values. Nor does it mean the true correction was infeasible; the earlier numerical check reported zero true-correction violation. These are different statements.

### 9.5 Timing and storage

| Stage | Reported seconds |
|---|---:|
| Original full-head certificate scan | 8.655664 |
| Certificate serialization | 0.004403 |
| Reported encoding total | 8.660067 |
| Localization | 0.245581 |
| Damaged full-head syndrome scan | 8.843010 |
| Algebraic decode | 0.292639 |
| Applying patch | 0.000629 |
| Total repair | 9.385136 |
| Evaluator verification | 0.876572 |

| GPU memory measurement | GiB |
|---|---:|
| Encoder peak allocated | 4.971189 |
| Encoder additional allocated over entry baseline | 3.960938 |
| Decoder peak allocated | 6.928223 |
| Decoder additional allocated over entry baseline | 5.917970 |

The full certificate and damaged syndrome were computed from the full heads. The earlier multi-trial notebook instead built the mathematically equivalent residual from the known simulated sparse patch to avoid repeated full scans. The complete run closes that computational shortcut for one instance.

The damaged scan accounts for about 94% of reported repair time. More localization optimization is therefore not the main current runtime opportunity. These are one-run measurements on the user's setup, not repeat-averaged performance. The supplied report does not itself contain a GPU identifier, software versions, or model commit; attach the actual run metadata before using an H200-specific paper claim.

Timings exclude model download/loading and preparation of original output records and hidden features. The encoding subtotal also omits the separately placed small GPU-to-CPU transfer of the certificate array, and the benchmark/hash passes are outside it. Describe the reported 8.66 seconds with those boundaries or use a single encompassing timer if an exact encoding-latency claim is needed. The total repair timer is broader than the sum of its listed sub-stages, which is expected from conversions and transfer overhead.

Allocated GPU peaks are not whole-system memory: they exclude CPU RAM, CUDA allocator reservations, and some external allocations. The body has already been released. No hidden claim of repairing an 8B-model deployment in only 6.93 GiB total system memory follows.

Separately packed payloads compare as follows:

\[
L_{\mathrm{baseline}}=2\cdot64\cdot31=3968\text{ bits}=496\text{ bytes},
\]

\[
L_{\mathrm{demo}}=114\cdot31=3534\text{ bits},
\quad \lceil3534/8\rceil=442\text{ bytes}.
\]

The meaningful-bit saving is 434 bits, or 10.9375%; actual byte-file saving is 54 bytes. There are two padding bits in the 442-byte file. The ordinary output record is \(256\cdot128256\) bytes, or 31.3125 MiB. This supports a protected-storage reduction when those predictions already exist. It is not a reduction in total storage compared with retaining just the data-free certificate.

Public settings and their serialization are outside the 442-byte file. The code records an original-derived numerical rho as public experimental metadata. For a deployed uniform scheme that bound must be fixed as part of the public tamper specification; transmitting an additional original-specific bound or codebook is model-dependent side information that must be inventoried. Calibration pools and INT8 reference scales are used to generate attacks, not supplied to the repair algorithm as hidden support hints.

### 9.6 What the concentrated result does and does not show

The independent LP attained approximately 3.8e−14 interval violation with support recall 0.171875 on its diagnostic instance. This makes it less plausible that the concentrated failure is solely a consequence of the fast solver's iteration cap. It shows that the particular convex objective/returned solution is poor at localization there.

It does **not** show that the original sparse correction is unidentifiable. A feasible low-ℓ1 correction can differ from the true sparse correction even when the latter is uniquely identifiable within the actual sparse alphabet class. Also, one LP optimizer does not characterize every optimizer. A lower bound requires multiple actual originals with common transcript and common checkpoint, as in Theorem 1.

The current safe statement is: location hints were useful for the sampled dispersed tampers, while the same method did not reduce the frozen budget for concentrated tampers. Explanations involving support geometry are plausible but not established as information-theoretic impossibility.

### 9.7 Baseline scope

The 3,968-bit baseline handles arbitrary 64-symbol damage anywhere in the full head. It is valid and transparent. It is not known optimal, even without examples. A post-tamper list of support and original values is not a comparable pre-tamper baseline.

For the paper, state whether “dispersed” is merely the test distribution inside the full sparse tamper class or a promise given to the decoder. If it is a decoder promise of one error per row, every competing method may exploit it, including row-level information from the output mismatch screen. The current baseline does not exploit that promise. Therefore describe the 10.9% saving as a reduction **relative to the chosen full-head syndrome baseline**, not relative to the optimum for dispersed faults.

The experiments use full vocabulary logits, which are more informative than the scalar losses or accuracy summaries commonly kept in evaluation reports. The paper should say “retained full-logit predictions”; generic “evaluation logs” could suggest a broader observation model than was tested.

## 10. Statistical claims and reproducibility boundaries

Development selects e and t to cover all ten development cases for each setting. This is an empirical choice, not a uniform certificate that every future tamper will be decodable. Test support is used to score the frozen rule, not to allocate its certificate.

If the final per-setting test count is indeed 30 and all succeed, report **30/30**, together with an interval and the exact trial distribution. Do not call it 100% guaranteed recovery. For a fixed pipeline and independent Bernoulli trials, an elementary one-sided 95% lower confidence bound for 30 successes is

\[
p_{\mathrm{success}}\geq0.05^{1/30}\approx0.905.
\tag{45}
\]

**Justification.** At any success probability below this endpoint, the probability of observing 30 successes is less than 0.05. Inverting that test yields the endpoint. This bound is conditional on the sampling and fixed-pipeline assumptions; it is not a simultaneous bound over all table cells.

The notebook also computes Wilson intervals. Either appropriate stated convention is acceptable; do not mix interval types without labels. Small samples support a pilot-scale reliability claim, not a high-reliability security guarantee.

The same trial uses nested N prefixes and matched precision supports, so comparisons are paired. Do not multiply N values, precisions, or layouts into an artificial count of independent trials. Repeated draws from a fixed pool are iid from that empirical distribution, conditional on the pool, but are not new independent documents from the full text population.

The source uses separate development and test pools and nominally separate seed ranges. However, the conversation records repeated debugging and method choices after inspecting results, including the matched-precision control. A final rerun with the same historical test seeds is reproducible, but it is not untouched confirmatory evidence for a pipeline changed using their outcomes. Label these results as development/pilot evidence unless run provenance shows otherwise. If independent confirmation is necessary, it should be one bounded run after freezing the final implementation, not another exploratory grid.

The current full-demo and pilot sources also have two minor replay differences: the demo requests top-16 directly while the pilot takes the first 16 of a top-64 ranking, which can differ at score ties; and the INT8-derived change is expressed through an algebraically equivalent but differently ordered FP32 calculation. Neither invalidates the demonstrated repair, but exact replay should compare the realized support, BF16 changed labels, input order, and erasure set. Do not call the two programs byte-identical without that comparison.

For an archival result preserve model/data revisions, cache provenance, input indices, public settings, environment versions, GPU identity, frozen choices, per-trial outcomes, and the full-demo report. No new large experiment is required merely to recover metadata that already exists.

## 11. Claim-by-claim review

| Claim | Mathematical/evidential basis | Current status |
|---|---|---|
| Harmful ambiguity requires protected information | Theorem 1; exact characterization in Theorem 2 | Proved for the stated finite-class problem |
| More nested examples cannot increase the optimal budget | Corollary 2.1 | Proved; not automatically a property of the heuristic's chosen budget |
| iid finite-precision predictions remove distinguishable pairs | Theorem 4 | Proved under independent per-input observations |
| Sparse geometry gives a sample–certificate bound | Theorems 7–8 | Proved for the random matrix and exact feasibility decoder |
| A useful uniform visibility constant holds for Llama | Would require verifying assumptions of Proposition 9 or another argument | Not established |
| The efficient decoder attains the sparse information bound | Would need a bridge from examples to a provable data-rule burden | Not established |
| e+2u checks suffice for actual residual labels | Theorem 10 | Proved; classical coding mechanism |
| Samples reduce that sufficient budget in the tested setting | Matched dispersed results | Supported empirically, with the statistical scope above |
| Full-head encoding and repair are computationally feasible | Direct scans, actual patch, evaluator equality checks | Supported by one complete run |
| 3,534 bits are necessary or near-optimal | Would require a matching converse in the same instance/class | Not established |
| Concentrated changes intrinsically need the full budget | Would require indistinguishable candidate families or another converse | Not established |
| The implementation spends bits according to behavioral relevance | Would require an implemented behavior-sensitive scheme | Not established; it repairs exact symbols |
| BF16 and INT8 have the same minimum repair information | Would require matched information limits, not equal achieved checks | Not established |
| The final pipeline has a uniform adversarial guarantee at 114 checks | Would require uniform control of data-stage errors/flags | Not established |

The main logical gap is explicit:

\[
\underbrace{\text{few harmful ambiguities after N examples}}_{\text{information/geometry theorem}}
\quad\not\Rightarrow\quad
\underbrace{\text{our proposal finds enough correct locations}}_{\text{efficient decoder hypothesis}}.
\]

Neither the successful full scan nor a low optimizer residual fills that gap. Conversely, the exact recovery theorem does not depend on resolving it: it already supplies a conditional guarantee for whatever hints the examples produce.

## 12. Research assessment and a contained next step

### 12.1 What is classical and what would need to be new

The graph formulation has a direct classical side-information precedent. Syndrome-based recovery from a nearby damaged object also predates this project: secure sketches explicitly study recovering an object from a close version and a short sketch, alongside privacy requirements that are not the objective here. See [Dodis, Ostrovsky, Reyzin, and Smith, *Fuzzy Extractors*, 2008](https://arxiv.org/abs/cs/0602007). Error-and-erasure algebra and the optimization updates are classical, as cited above. Matrix checksum methods also have a long history in [Huang and Abraham, *Algorithm-Based Fault Tolerance for Matrix Operations*, 1984](https://graal.ens-lyon.fr/~abenoit/CR02/papers/abft2.pdf).

Consequently, “a few checks repair sparse model damage” is not itself a sufficient novelty claim. The graph and union-bound derivations are useful foundations but should not be sold as major new information theory merely because the object being repaired is a neural head.

The candidate research contribution is more specific: **quantify the additional protected information needed after actual finite-precision model predictions are available, accounting for sparse geometry and domain relevance, and give a practical way to extract that benefit.** The current work gives a coherent beginning and a functioning instance. It has not yet established a sharp characterization or a strong algorithmic advance over those ingredients. This review checks key precedents; it is not an exhaustive priority search, and it makes no claim of being the first such framework.

### 12.2 What the paper can responsibly say now

A defensible current story is:

> Retained predictions can supply useful error-location information for sparse model repair. We formalize the residual-information problem, prove conditional sample–certificate bounds, and implement a prediction-assisted syndrome decoder. On a full Llama output head, the demonstrated protocol reduces its chosen certificate-only budget by 10.9% while performing exact recovery with a 442-byte serialized certificate.

The theoretical promise “protect only behaviorally important uncertainty left by the examples” is broader than the exact-recovery experiment. It must be presented as the formulation and a theoretical direction, not as a completed property of the practical encoder.

**Assessment:** the engineering feasibility concern is now substantially addressed. The combination of broad classical bounds, a conditional classical decoder, and a modest protected-bit saving does not yet substantiate a top-percentile ICML novelty claim. More test repetitions alone cannot change that assessment. This is not evidence that the original question is uninteresting; it identifies the missing scientific connection.

### 12.3 Next action, without expanding the project

First consolidate the already-produced evidence into one certificate-versus-N figure and this exact-repair runtime table. Verify actual trial counts and frozen settings from the saved CSVs; display uncertainty and name the full-head syndrome baseline accurately. This is reporting and provenance work, not a new research direction.

Then settle one theoretical decision within the existing formulation: **can we make a useful quantitative connection between sample geometry and the residual burden of a decoder we can actually run?** Corollary 11.1 identifies the needed statement; the same-class lower bound in Theorem 1 determines whether a near-optimality claim is even plausible. We should not promise both a sharp converse and a new optimal decoder without evidence that the class permits them.

A bounded analysis should use the current last-layer, finite-alphabet, sparse-change setting. If it yields a useful condition explaining the dispersed/concentrated distinction or a meaningful gap to necessity, it strengthens the one existing story. If it does not, the honest endpoint is a narrower proof-of-concept paper or a reassessment of the venue ambition. Do not add languages, model sizes, precision grids, or unrelated applications to conceal that gap.

The current completed experiments are sufficient to stop implementation expansion. They are not sufficient to stop evaluating the central novelty claim.

## Appendix A. Decoder interface and certificate timing

A minimal operational protocol is:

```text
PUBLIC: architecture, intact feature extractor, allowed original alphabet,
        tamper class, input records, output quantizer, locator formula,
        field prime, fixed localization rule, erasure count, check count.

ENCODE(original head, retained inputs):
    retain original quantized outputs
    scan original symbol vector to compute t syndrome checks
    pack checks; write certificate before damage

REPAIR(damaged head, retained input-output record, certificate):
    compute numerical proposal using only those inputs
    select erasure locations using the fixed rule
    scan damaged symbol vector to obtain its syndrome
    subtract from unpacked original checks
    decode outside errors and erased values
    validate returned labels; apply patch

EVALUATE(original kept separately, attack record, repaired head):
    compare recovered support and labels
    verify complete head equality/hash
    record failures, incorrect accepted outputs, timings, and resources
```

The encoder need not know the future support; a sparse patch stored after discovering the damage is not this protocol. The actual full demo selects a predetermined input sequence from a precomputed original-output pool after the tamper step; moving that deterministic selection before tampering leaves the information unchanged and states the timing more cleanly. It does not query new original outputs after the damage.

The notebooks enforce the interface by function arguments and code discipline, not by process-level isolation. Evaluator variables still exist in the same notebook process. The source inspection supports the claim that the decoder does not read them; it is not an operating-system security boundary.

Focused checks executed during this review used independent full-vector CPU encodings on 1,024-coordinate, 16-bit-symbol examples with 64 changed positions and 114 checks. Both the 16-correct-flag case (burden 112) and the 15-correct-plus-one-false-flag case (burden 114) recovered exactly. The notebook's actual packing/unpacking functions also round-tripped 114 field residues in exactly 442 bytes. These checks validate the boundary arithmetic and implementation mechanics; they are not additional Llama experiments or evidence of statistical reliability. All 45 equation labels and Markdown/math delimiter pairs were checked for consistency.

## Appendix B. Necessary distinctions to preserve in the manuscript

1. **Necessary bits versus sufficient bits.** Packing lower bounds constrain every valid decoder. The measured syndrome burden describes one construction.
2. **Actual cells versus outer strips.** Width constraints can overcount ambiguous differences. Their failure cannot certify ambiguity.
3. **Uniform recovery versus sampled faults.** All-original/all-tamper theorems require their stated uniform events. A successful finite test set does not supply those events.
4. **Stored symbols versus numerical effects.** BF16 label arithmetic is exact field arithmetic; fitting logits uses numerical values. They cannot be interchanged.
5. **Behavioral recovery versus exact symbols.** Exact symbols suffice for behavior but may use unnecessarily many bits.
6. **Available outputs versus ordinary evaluation summaries.** The experiment keeps every logit code, not merely accuracy, loss, or a chosen answer token.
7. **Conditional iid pool sampling versus population coverage.** Sampling with replacement from a fixed cache gives the former, not an automatic guarantee for all language inputs.
8. **Classical components versus project novelty.** The scientific contribution must be the relationship established between examples, precision, sparse changes, and repair information, not a renaming of parity checks.

