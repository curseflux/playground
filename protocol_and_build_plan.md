# Protocol freeze, audit build, and whole-head accounting

Three things that are not experiments and must exist before any experiment means
anything. Written against `fixed_original_proof_and_novelty_review.md` (Prop. 4 audit,
Lemma 2 construction) and `sharp_sample_theorem.md` (Lemma 1 gain bound).

---

## 1. Protocol freeze — the decisions still open

"Freeze the protocol" is not paperwork. Across the five documents these choices are
still undecided or **mutually contradictory**, and several change the headline number.

| # | Decision | Status | What it moves |
|--:|---|---|---|
| 1 | **Dither during recording** | Existing runs have **none**; Thms 1–2 of the fixed-original review **require** fresh independent dither | Without it the new theorems do not describe the measured slopes at all. Re-recording needed. |
| 2 | **Guarantee** | Now per-original; older docs written for uniform (`consolidated` §1.4 defines `L^unif`) | 6 bits vs 0.46 bits per 8× data |
| 3 | **Is "dispersed" a promise to the decoder or just the test distribution?** | Open — `consolidated` §9.7 and §12 item 4 | Honest data-free baseline: 1,558 bits vs 792 bits |
| 4 | **Changes per row `m`** | Theorems assume `m=1`; the planned sweep goes to `m=64` | Decoder feasibility (see §3 and Addendum 2) |
| 5 | **Record convention: exact codes or bounded-error?** | Open — `proof_and_validity.md` §1 showed the FP32 quantiser is *not* the nominal cell | This is what voided the 1,287-bit converse |
| 6 | **Tolerance `η`** | `RESULTS_REPORT` §12 item 2 proposes 3e-5 over 1e-5, undecided | Truth-check loss rate |
| 7 | **Weight lattice** | Theorems use uniform `Δ`; BF16 is log-spaced; `sharp_sample_theorem` explicitly disclaims a uniform BF16 lattice | Every bit count that divides by `Δ` |
| 8 | **Saturation** | Theorems assume unsaturated; the real record clips to [−64, 64) | Correctness at the tails |
| 9 | **Does the screen's row list count as free?** | Undecided | 790 bits |
| 10 | **Magnitude cap `ρ`: public and enforced?** | Stated but not consistently applied | Whether long thin invisible slivers are legal |
| 11 | **What is stored vs. recomputed** (features, inputs, offsets/dither) | Undecided | Dither must be reproducible; if stored, it is storage |

Deliverable: one page fixing all eleven, dated, referenced by every later run. Anything
later that violates it is a new protocol, not a new result.

---

## 2. The audit — what it is, how to build it, how to test it

### What it is

Proposition 4. Before any damage, while the original `a` is still in hand, form

```
A_p(a,H,ℓ) = { b ≠ a : |supp(b−a)| ≤ 2, Record(b) = ℓ, C_p(b) = C_p(a) }
```

If this is **empty**, `a` is provably recoverable from every legal one-coordinate
tamper. Prop. 4 also proves emptiness is *necessary* for the unique-candidate decoder,
so the test is exact, not conservative.

### The simplification that makes it computable

From §2 of the fixed-original review: a `≤2`-sparse difference has zero checksum **iff
every entry is a multiple of `p`**. So the audit is not "compare all originals". It is:

> For each column pair `(j,k)`: does the invisible polygon contain a nonzero point of
> the **coarse** lattice `pΔ·Z²`?

That is exactly the geometry in `coupling_note.md` §1, intersected with a sublattice of
spacing `pΔ` instead of `Δ`.

### A cheap sufficient test

The polygon contains no nonzero coarse point whenever its **coordinate projections**
both lie inside `(−pΔ, pΔ)`. So compute, per pair, the projection of the polygon onto
each axis and compare to `pΔ`. Only pairs that fail this cheap screen need exact
lattice enumeration.

Cheaper still, and free: Lemma 1 gives `‖δ‖∞ < W/(Δγ₂)` for **every** confusable pair
at once. If `W/(Δγ₂) ≤ p`, the audit passes by fiat and needs no per-pair work.

### Why the audit exists at all

Because Lemma 2 picks `p` from the **worst case**. The audit lets you pick a smaller `p`
and prove it is still safe **for this model**. That is the entire mechanism behind
"price the insurance for this specific network" — without it there is no per-model
number, only a worst-case one.

### Test plan: proving it never falsely certifies

Toy scale only: `d ∈ {4,5,6}`, `q ∈ {5,7,9}`, `N ∈ {3,…,8}`, so the full grid `q^d` and
all legal tampers can be enumerated by brute force.

| Property | Test |
|---|---|
| **No false certification** (the one that matters) | audit says certified ⟹ brute force confirms exact repair for *every* legal tamper |
| **Necessity** (Prop. 4's converse) | audit says not certified ⟹ brute force exhibits two admissible candidates |
| **Modular aliases** | plant a difference that is an exact multiple of `p` and confirm it is caught |
| **Exact cancellation** | duplicate two feature columns so a `(δ, −δ)` pair is invisible |
| **Cell boundaries** | place logits exactly on cell edges; check half-open conventions both ways |
| **Saturation** | values at ±64 |
| **`m = 2,3`** | same tests with `≤2m`-sparse differences |

The gate is the first row. A single false certification is a correctness bug, and it
must be found on a 6-weight toy rather than on a Llama head.

---

## 3. Whole-head accounting — and a floor that is easy to miss

### The problem

The theorems certify **one row** (`d = 4096`, one change). The head has 128,256 rows and
64 total changes. One certificate per row is `128,256 × ~13 bits ≈ 1.7 Mbit` — hopeless.
So the certificate must be written before knowing which rows break, yet cost as if only
64 rows mattered.

### The shape of the answer

Two levels, which the project already has half of:

* **Inner (within a row):** which column, which value. This is Lemma 2's checksum.
* **Outer (across rows):** which rows. A syndrome over 128,256 positions with ≤64
  errors. This is the existing 2,176-bit "row-summary code + screen" line in
  `RESULTS_REPORT` §6.

If the screen reliably names the damaged rows, the outer level is free and only the
inner level is paid. Whether that is legitimate is protocol decision #9. Under the
**per-original** guarantee it is defensible: with probability `1−δ` no change is
invisible, so the screen misses nothing.

### The floor nobody has stated

Inside a row, the checksum family must give 4,096 columns distinct one-dimensional
subspaces of `F_p^r`, costing `r(p,d)·log₂p` bits. Minimising over `p`:

| p | r | bits/row |
|--:|--:|--:|
| 2 | 13 | 13 |
| 3 | 9 | 15 |
| 17 | 4 | 17 |
| 4099 | 2 | 25 |

**The minimum is 13 bits, and `log₂ 4096 = 12`.** The locator cost is *conserved* — no
choice of `p` escapes it, exactly as block-splitting failed to in Addendum 1.

```
floor for this construction family : 13 × 64 = 832 bits
their reported construction        :            1,088 bits
value-only information floor       :             ~165 bits
```

So roughly **12 of every 13 bits per row buys the column index** — which the interval
method already recovers for free from the record. That gap is not slack in their
implementation; it is intrinsic to any code that must address 4,096 columns.

### What this means for the build

1. The honest whole-head number for this construction family is **~832–1,088 bits**, and
   the existing 1,088 is already within 30% of its family's floor. Do not expect tuning
   to find 165 bits.
2. Reaching the value-only floor requires a construction that **does not pay for column
   location** — one that uses the record's own column identification inside the code
   rather than re-encoding it. That is a genuine open design problem, not an
   implementation detail.
3. Therefore the honest report is a **two-line result**: the achieved certified budget,
   and the information floor, with the gap named and attributed to column addressing.
   Quoting only the second would repeat the 121-bit mistake.

---

## 4. Order of work

1. Freeze the eleven decisions in §1. Hours, and everything downstream is undefined
   without it.
2. Build the audit and pass the toy gate in §2, **no-false-certification first**.
3. Decide #9, then build the two-level head certificate and report both lines from §3.
4. Only then run the `N` ladder and the concentration sweep.

Steps 2 and 3 carry the risk. Step 4 is the cheap part.
