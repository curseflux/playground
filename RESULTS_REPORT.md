# Repair budget: consolidated results report

**Companion to `repair_theory_and_evidence_consolidated.md`.** That document states the problem,
the theory, and the evidence available as of the full-head demonstration. This one reports
everything measured since, on the same Llama-3.1-8B output head, and says which file produced
each number.

Hardware and software for every measurement below: NVIDIA A100-SXM4-40GB, torch 2.4.0+cu118,
`meta-llama/Meta-Llama-3.1-8B-Instruct`, 2026-09-09.

---

## 0. What changed, in one paragraph

The consolidated document reports a **10.9% certificate saving** (3,534 bits against a 3,968-bit
data-free baseline) using an l1 localiser to pick 16 erasure flags. That number was capped by
design: at 16 flags against 64 errors the error-and-erasure burden `e + 2u` is at best 112 against
128, so **no value of N could have produced more than 12.5%**. Replacing the l1 localiser with
exact interval arithmetic changes the picture completely. The retained record identifies the
changed row and column in **every** case, and leaves a median of 3 candidate weight values per row
at N=2048. The residual information is **121 bits** where the data-free baseline is 3,968 — a
**97.0% reduction** — and an efficient construction reaches **1,088 bits (72.6%)**. Along the way
the decay law was measured: after the column is pinned, surviving candidates fall as `N^-1`.

---

## 1. Provenance

| # | Code | Output | Establishes | Report §|
|---|---|---|---|---|
| P1 | `llama_repair_certificate_pilot.ipynb` | (in consolidated doc §9.3) | matched-precision budgets, 3,968 / 3,534 bits | 2 |
| P2 | `llama_full_certificate_to_repair.ipynb` | (consolidated doc §9.4-9.5) | full-head exact repair, SHA-256 match, 9.4 s | 2 |
| P3 | `repair_decoder.py` | — | RS error-and-erasure decoder; Chambolle-Pock l1 localiser | 2 |
| A1 | `llama_interval_audit.ipynb` | `interval_audit_report.json` | first interval audit: candidate counts, slope, ladder | 3 |
| A2 | `interval_audit.py` | — | reusable core: intervals, exact counting, subsample estimation, screen. Unit-tested against exhaustive brute force | 3, 8 |
| A3 | `llama_interval_audit_v2.ipynb` | `audit_v2_report.json` | screen calibration, pool controls, N-sweep to 2048 | 4, 5, 6 |
| A4 | `llama_audit_v3.ipynb` | `audit_v3_report.json` | screen at N=2048, 8 trials per pool mode, uniform converse | 4, 5, 6, 7 |
| A5 | `finalize_audit.py` | `finalize_report.json` | per-row slope fit, residual-vs-N curve, ladder at N=2048 | 5, 6 |

Configuration is embedded in each `*_report.json` under `config`, and provenance (GPU, torch
version, timestamp) under `provenance`.

---

## 2. Prior evidence (from the consolidated document)

Reproduced here for continuity; source P1-P3.

| Quantity | Value |
|---|---|
| Protected part | full untied output head, 128,256 x 4,096 = 525,336,576 BF16 entries |
| Retained record | full-vocabulary raw logits, uniform 8-bit cells over [-64, 64), width 0.5 |
| Damage model | 64 changed weights, one per affected row ("dispersed"), INT8-derived, rho-bounded |
| Data-free baseline | `2s` RS checks over `p = 2^31-1` = **3,968 bits** (496 B) |
| Reported result | 114 checks = **3,534 bits** (442 B), 10.9% saving |
| Verification | recovered support and labels exact; whole-head SHA-256 match |
| Repair time | 9.385 s total, of which 8.84 s is the damaged full-head syndrome scan |
| Record size | 256 x 128,256 bytes = 31.3125 MiB |

**Design ceiling.** With 16 erasures frozen against 64 errors, `e + 2u = 16 + 2(64-c) >= 112`
against a 128-check baseline. The maximum achievable saving in that configuration is **12.5%**,
independent of N. The reported 10.9% is therefore a measurement of the l1 localiser's ~25% recall,
not of the information in the record.

---

## 3. Method: exact intervals in place of the l1 localiser

Source: A2, used by A1/A3/A4.

Under the one-change-per-row model, hypothesise that column `j` of row `v` changed by `delta`.
With `c_i` the **damaged** logit on sample `i` and `[l_i, u_i)` the cell named by the retained
code,

```
l_i <= c_i + delta * h_j(x_i) < u_i        for every i = 1..N
```

Each sample is one linear inequality in `delta`, so with `a_i = l_i - c_i`, `b_i = u_i - c_i`:
`delta` lies in `[a_i/h_ji, b_i/h_ji)` when `h_ji > 0`, the reversed interval when `h_ji < 0`, and
`h_ji = 0` gives a feasibility test. Intersecting over `i`, clipping to the public `rho` bound,
and counting BF16 values in the result gives the exact candidate set for that column. Nothing in
this reads the original checkpoint or the tamper.

The interval is computed in float64 while codes were produced in float32, so a declared tolerance
makes it an **outer** bound; candidates inside it are then re-verified exactly (or, when the set is
too large to enumerate, sampled and verified with a Wilson interval). Correctness evidence is in
§8.

---

## 4. The record identifies location completely

### 4.1 Screen calibration (source A3, `audit_v2_report.json: item1_*`)

Comparing recomputed codes to cached codes **exactly** flags 236 rows for 64 real changes. The
false flags are FP32 replay noise: recomputing logits at a different batch shape moves them by a
mean of `1.2e-6` (max `4.2e-5`), enough to cross a cell boundary occasionally. Flagging a row only
when its logit leaves the retained cell by more than `eta`:

| eta | flagged | recall | false positives |
|---|--:|--:|--:|
| 0 | 236 | 1.00 | 172 |
| 1e-6 | 168 | 1.00 | 104 |
| 3e-6 | 77 | 1.00 | 13 |
| **1e-5** | **64** | **1.00** | **0** |
| 1e-4 … 1e-3 | 64 | 1.00 | 0 |

### 4.2 The screen holds at 8x the samples (source A4, `audit_v3_report.json: A_screen_by_N`)

At `eta = 1e-5`, across all four coordinate pools and both `N = 256` and `N = 2048`:
**flagged = 64, recall = 1.00, false positives = 0**, in all 8 cells. The tolerance does not have
to be re-tuned as N grows.

### 4.3 Columns are pinned too (source A4, `B_trials`)

Across 8 tampers x 4 pools x 2 values of N, the mean number of rows whose changed **column** is
uniquely determined is **64.0 of 64** everywhere except one cell (salient at N=256: 63.875).
Against the l1 localiser's ~25% coordinate recall, this is not an improvement of degree.

**Location costs zero protected bits.** Both the row and the column come from the record.

---

## 5. The value term: a two-regime law

### 5.1 Regimes (source A3, `audit_v2_report.json: item4_law`)

| N | 1 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| median feasible columns | 3790 | 1403 | 373 | 20 | **1** | 1 | 1 | 1 | 1 | 1 | 1 |
| median candidates | 1.8e7 | 3.7e5 | 5.6e4 | 3.0e4 | 220 | 134 | 54 | 30 | 13 | 7 | 4 |

- **Location regime (`N < 32`).** Steep and irregular while columns are eliminated.
- **Value regime (`N >= 32`).** The column is unique; only the value is unresolved.

The pool provides 2,048 **distinct** inputs (sampled without replacement from 3,530 held-out
texts), so this is not an artifact of resampling.

### 5.2 The exponent (source A5, `finalize_report.json: per_row_slope`; A3)

Fitting `log2(candidates)` against `log2(N)` **per row**, each row entering the value regime when
its own column count reaches 1 (median `N = 32`; 63 of 64 rows fitted):

| coordinates | per-row slope (mean ± SE) | per-row median | aggregate fit |
|---|--:|--:|--:|
| salient | **-1.025 ± 0.064** | -0.897 | -1.062 |
| uniform | **-1.221 ± 0.071** | -1.082 | -1.217 |

Predicted: **-1**, from `width(I) ~ WIDTH / (N |h_j|)`.

**Salient coordinates are consistent with -1** (0.4 SE). **Uniform coordinates are significantly
steeper** (3.1 SE from -1). A plausible mechanism, *not tested here*: the binding constraint is the
sample with the largest `|h_j(x_i)|`, and for a heavy-tailed feature `max_i |h_j|` grows with N,
giving `N^-(1+1/alpha)` for tail index `alpha`. The observed -1.22 corresponds to
`alpha ~ 4.5`. **Test:** regress `log(max_i |h_j(x_i)|)` on `log N` from the cached features — a
few lines, no new GPU work.

*Note.* `audit_v2_report.json` reports a slope of -1.566 for the uniform pool. That is a fitting
artifact: its window opens where the **median** column count first reaches 1, while a tail of rows
is still unpinned, producing a spurious -6.88 step from N=32 to 64. The per-row fit in A5
supersedes it.

### 5.3 Residual information against N (source A5, `residual_bits_by_N`)

Sum of `log2(candidates)` over the 64 changed rows — the information the record leaves:

| N | 1 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| salient (bits) | 1495 | 837 | 606 | 487 | 406 | 343 | 272 | 207 | **145** |
| uniform (bits) | 1633 | 974 | 767 | 563 | 457 | 377 | 315 | 227 | **168** |

In the value regime this falls by **68 bits per doubling** (salient) and **79** (uniform), against
64 predicted for an exact `N^-1` law over 64 rows.

---

## 6. The budget ladder

Source: A4 (`D_final_table`, 8 trials, mean ± sd) and A5 (`ladder`).

| Scheme | Bits | Bytes | vs data-free |
|---|--:|--:|--:|
| Data-free full-head RS (`2s` checks, `p = 2^31-1`) | 3,968 | 496 | — |
| Reported in the consolidated document (l1 flags + RS) | 3,534 | 442 | 10.9% |
| Row-summary code + screen (`2\|E\|` checks, `p = 2^17-1`) | 2,176 | 272 | 45.2% |
| **Row locators + interval column ID (`\|E\|` checks, `p = 2^17-1`)** | **1,088** | **136** | **72.6%** |
| Information left by the record, N=256 (counting bound) | 312 ± 20 | 39 | 92.1% |
| **Information left by the record, N=2048 (counting bound)** | **121 ± 14** | **16** | **97.0%** |

`|E| = 64` is measured, not assumed (§4.2). The two information rows are salient-pool means over 8
trials; uniform-pool values are 397 ± 21 and 178 ± 14 bits (90.0% and 95.5%).

**The last two rows are counting bounds, not constructions.** They say how much information
remains; they do not exhibit an encoder that spends only that much.

### 6.1 The gap between 1,088 and 121 is computational

The construction spends **17 bits per flagged row** where the information is **1.9 bits per row**
at N=2048. The waste is one thing: a Reed-Solomon-style code must carry field elements large
enough to give every one of the 128,256 rows a distinct locator, which forces at least 17 bits per
check — so it pays again for the location the screen already supplied for free. Spending less
requires an encoder written *before* the tamper that nonetheless addresses only the rows that will
break, i.e. efficient decoding against per-coordinate candidate sets. **This is stated as an open
computational-versus-informational gap, not closed.**

---

## 7. What makes the record informative

### 7.1 A 2x2 over coordinate pools (source A4, `B_trials`; 8 trials per cell)

Rows and columns drawn either from the calibration salience pools (`topk(mean_p, 256)` rows,
`topk(feature_energy, 512)` columns) or uniformly at random. Residual bits, mean ± sd:

| | salient columns | uniform columns |
|---|--:|--:|
| **salient rows**, N=256 | 312 ± 20 | 390 ± 22 |
| **uniform rows**, N=256 | 302 ± 14 | 397 ± 21 |
| **salient rows**, N=2048 | 121 ± 14 | 182 ± 15 |
| **uniform rows**, N=2048 | 112 ± 13 | 178 ± 14 |

Main effects (SE from the eight trials):

| effect | N=256 | N=2048 |
|---|--:|--:|
| column salience | **+86.8 bits (12.5 SE)** = +1.36 bits/row | **+63.9 bits (12.9 SE)** = +1.00 bits/row |
| row salience | -1.9 bits (0.3 SE) = -0.03 bits/row | -6.6 bits (1.3 SE) = -0.10 bits/row |

**Columns matter; rows do not.** That is what the geometry requires: interval width goes like
`WIDTH / (N |h_j|)`, a property of the column. The row enters only through where its logit happens
to sit inside its cell.

**Quantitatively.** The median feature energy is 10.52 in the salient columns and 2.82 in the
uniform ones, a ratio of 3.73, predicting a shift of `0.5 * log2(3.73) = 0.95` bits per row. The
measured effect at N=2048 is **1.00 bits per row**. At N=256 it is 1.36, higher — consistent with
some rows not yet being fully in the value regime there.

### 7.2 The headline generalises

An earlier concern was that the saving was an artifact of measuring on the top 0.025% of the head
by salience. It is not: at N=2048 the residual is **121 bits** for salient coordinates and
**178 bits** for uniform ones, against 3,968 data-free. Both are above 95%.

---

## 8. The uniform converse

Source: A4 (`C_uniform_converse`), 512 sampled untouched rows.

| N | median feasible columns | hidden size D | median bits/row | one-sided lower bound | converse over s=64 |
|---|--:|--:|--:|--:|--:|
| 256 | 4096 | 4096 | 23.14 | 23.13 | **1,480 bits** |
| 2048 | 4096 | 4096 | 20.15 | 20.11 | **1,287 bits** |

**These candidates form a packing family with a common transcript and a common damaged
checkpoint**, so by Theorem 1 of the consolidated document, `s` times the per-row `log2`
lower-bounds every encoder that must handle every legal `s`-sparse tamper.

`median_feasible_columns` equals `D` exactly. On an untouched row nothing needs explaining, so
`delta = 0` is feasible for every column and none is eliminated. **This is why the number is far
larger than the per-tamper residual and why it answers a different question**: it counts changes
that are invisible *anywhere* in the head, not alternatives to a tamper that happened.

The converse decays at **-1.00 bits per doubling** (23.14 -> 20.15 over three doublings) — the same
exponent as the per-tamper residual.

### 8.1 The two numbers are not in conflict

The 1,088-bit construction (§6) is **below** the 1,287-bit uniform converse. That is not a
contradiction: they cover different tamper classes. The construction repairs every tamper the
record can **see** — the screen must flag the row. Tampers the record cannot see are what the
converse covers, and no construction avoids paying for them. Which number applies is decided by
whether a uniform guarantee or an average-case one is demanded, and the invisible tampers are
precisely those whose effect on the model's own outputs is below the resolution at which those
outputs were retained.

---

## 9. Numerical tolerance is part of the record convention

Three separate failures during this work had the same cause: comparing quantized outputs
**exactly** when the two sides came from different arithmetic paths.

1. **The original screen** flagged 172 false rows at `eta = 0` (§4.1).
2. **A cache-validation check** using `torch.equal` had a **93% chance of failing on a valid
   cache**: the per-code flip rate is 5.2e-6, so probing 4 samples x 128,256 rows expects ~2.7
   differing codes.
3. **Candidate verification** rejected genuine candidates, the true original among them, at
   `N * 5.2e-6` per row — 0.13% at N=256 but **1.07% at N=2048**. A synthetic replay at N=2048
   loses the true original in 4 of 300 rows at `eta = 0` and 0 of 300 at `eta = 1e-5`.

All three are fixed by applying the same declared `eta` at every stage. After the fix
(source A4, `B_truth_check`): the true original is lost in **1 of 4,096 row-instances (0.024%)** at
`eta = 1e-5`, against ~22 expected at `eta = 0`. The single remaining loss suggests `eta = 3e-5`
would be a safer operating point; the sweep in §4.1 shows recall stays 1.0 and false positives stay
0 up to `eta = 1e-3`.

`eta` cannot be arbitrarily small: one float32 ULP at logit magnitude 64 is **7.6e-6**, and
verification only becomes a reliable superset of the exact-code convention for `eta >= 3e-6`.

**For the manuscript:** a retained-prediction record is not a bit-exact object. Any consumer that
compares quantized outputs exactly will reject genuine reconstructions at a rate linear in N, and
the failure presents as data corruption rather than as arithmetic.

---

## 10. Correctness evidence for the measurement code

Source A2, `interval_audit.py`, tested against exhaustive enumeration over the full BF16 alphabet
on synthetic rows:

| Property | Result |
|---|--:|
| Interval is an outer bound and retains the true column | 40/40 |
| `verify_row` equals exhaustive brute force (own convention) | 40/40 |
| `verify_row` vs the code-equality convention | 37/40 (boundary ULPs; see §9) |
| Chunked sample reduction equals unchunked | 40/40 |
| Subsample estimator's 95% interval covers the exact count | 29/30, median relative error 1e-4 |
| Screen recall 1.0 at small `eta`; flag count monotone in `eta` | pass |

Two bugs were found by these tests and fixed before any reported run: the estimator divided hits by
the requested draw count while truncating draws for narrow columns (coverage 20/25 -> 25/25), and
the tamper seed used `hash()` on a string, which Python salts per process, so tampers would not
have reproduced across runs.

`finalize_audit.py` (A5) was validated on a synthetic sweep with a known per-row slope of -1: it
recovers **-0.999 ± 0.031** and correctly skips rows that never pin or saturate immediately.

---

## 11. Claim status

| Claim | Basis | Status |
|---|---|---|
| The record determines the changed row | A3, A4; recall 1.0, 0 false positives, 8/8 cells | Measured |
| The record determines the changed column | A4; 64.0/64 rows in 7 of 8 cells | Measured |
| Candidates decay as `N^-1` after the column is pinned | A5; -1.025 ± 0.064 (salient) | Measured, salient |
| The same exponent holds for uniform coordinates | A5; -1.221 ± 0.071 | **Rejected at 3.1 SE**; steeper |
| Column feature energy is the mechanism | A4; +1.00 bits/row measured vs +0.95 predicted, 12.9 SE | Measured |
| Row salience is irrelevant | A4; ≤1.3 SE, sign changes | Consistent with no effect |
| 121 bits remain at N=2048 (salient), 178 (uniform) | A4, A5 | Measured, counting bound |
| 1,088 bits suffice as a construction | §6; `\|E\|` measured, coding cost arithmetic | Sufficient, not shown necessary |
| Any encoder needs >= 1,287 bits for all legal tampers | A4; packing family, common transcript and checkpoint | Measured lower bound |
| The 1,088-vs-121 gap is computational | Field-size argument, §6.1 | Argued, not proved |
| The result holds beyond one tampered linear layer | — | **Not tested** |
| The result holds against adaptive adversarial tampers | — | **Not tested** |
| A uniform guarantee at 1,088 bits | Requires all tampers visible to the screen | **Not established** |

---

## 12. Open items

1. **Tail-index test** for the -1.22 uniform slope (§5.2). Cached features, a few lines.
2. **`eta = 3e-5`** rather than 1e-5, given the single residual truth-check loss (§9).
3. **One mid-stack MLP layer**, to show the analysis applies to any tampered linear layer given
   cached inputs to it. Bounds the "only the last layer" objection.
4. **Whether "dispersed" is a promise to the decoder or only the test distribution** (consolidated
   document §9.7). If it is a promise, every competing baseline may use it, and the honest
   data-free location cost is `log2 C(131072, 64) = 792` bits rather than the full-head 1,558.

Items 1 and 2 are analysis. Item 3 is one run. Item 4 is a definition to fix in the manuscript.
