# SDPA perf findings — open threads to investigate deeper

## Read batching is a +39% REGRESSION once the op is bandwidth-bound (per-tile trickle wins)

**Status:** parked at per-tile default; measured, not shipped. Investigate deeper later.

**Box / op:** Blackhole p150b, 110 cores @ 1.35 GHz. `scaled_dot_product_attention` (FlashAttention-2).

**Exact case:** flagged `LOOSE_CASES` shape **1×10×9472×128**, bf16, MHA, self-attn, non-causal,
tile-aligned, `fp32_dest_acc_en=False` (throughput regime), HiFi2. Sq_t = Skv_t = 296 tiles,
Dt = 4, chunk 8, n_kv_chunks = 37, ~370 work units / 110 cores → ~4 wu/core.

**Why we were looking:** after the compute-side cuts (P10 → PV packer-L1-accumulate fusion,
eltwise reconfig-drop, bf16 accumulators, super-fast exp `InputClamping::None` + packer ZERO_RELU),
the op crossed from compute-bound to **DM/read-bound**:
- compute floor (reader+writer NoC stub) = **4.887 ms**
- e2e (per-tile reader, parked) = **5.275 ms**, reads-visible = **7.18%** (was ~1.8% at session start)

**The finding:** re-enabling the parked R3 reader batching (straddle-safe divisor predicates
`batch_q = (Sq_t % Sq_chunk_t == 0)`, `batch_kv = (Skv_t % Skv_chunk_t == 0)` — both TRUE here) —
i.e. batch a whole KV chunk's tile reads behind ONE `noc_async_read_barrier` instead of
read+barrier per tile — was a **REGRESSION**:

| variant | e2e | reads-visible |
|---|---|---|
| per-tile (parked) | **5.275 ms** | 7.18% |
| batched | **7.352 ms (+39%)** | **33.4%** |
| compute floor (reader stub) | 4.897 ms | — (control, unchanged in both) |

The reader-stub compute floor was identical (4.897 ms) in both runs — the clean control proving
the +39% is entirely the reads path, not AICLK drift.

**Burst size when batched (what changed):** per KV chunk, behind ONE barrier the reader issues
- K read: `Dt × skv_valid` = 4 × 8 = **32 tiles** (64 KB bf16)
- V read: `skv_valid × Dt` = 8 × 4 = **32 tiles** (64 KB bf16)

so two 32-tile / 64 KB bursts per KV chunk, × ~148 chunk-iterations on the critical-path core,
× 110 cores bursting concurrently. Per-tile (parked, wins) = 1 read + 1 barrier per tile
(steady trickle) under the `KV_DEPTH=2` double-buffer.

**Interpretation:** bandwidth / NoC-congestion-bound, **not** latency-bound. The batched burst
front-loads reads that congest the NoC across 110 cores and overlap compute *worse* than the
steady per-tile trickle. The `examples/double_buffer` caveat "skip batching if bandwidth-bound"
undersells it — here it's a large loss, not merely flat. So the whole latency-lever family
(read batching, deeper CB) is dead for this shape. The real lever is fewer **bytes**
(bf8 K/V halves the ~1.8 GB) or fewer redundant re-reads (each (b,h)'s 4.8 MB of K/V is re-read
**37×**, once per q-chunk → mcast / `examples/shared_input_reuse`, T3).

**Total DRAM read volume:** ~1.8 GB (37 q-chunks/(b,h) × 2 × 2.4 MB × 10 (b,h)), ~370 GB/s
effective (1.8 GB / ~4.9 ms).

**Open questions for deeper investigation:**
- DRAM-bandwidth saturation vs NoC-link congestion? Measure achieved GB/s vs Blackhole DRAM peak
  (`/perf-measure` NoC calibration). ~370 GB/s observed — is that at peak?
- Sweep intermediate batch sizes (2 / 4 / 8 tiles per barrier) — is there a congestion sweet spot,
  or is per-tile strictly best when bandwidth-bound?
- `KV_DEPTH` interaction (2 → 3): Perf 1 measured KV_DEPTH=3 as −0.37% when compute-bound; unknown
  when read-bound.
- Core placement / DRAM-facing topology: can the 110 cores' reads be spread to cut contention?

**Where the code is:** `kernels/scaled_dot_product_attention_reader.cpp` — `read_tiles<cb, batch,
ablate>` + the `batch_q` / `batch_kv` predicates (kept `false`, with this summary in the comment).
Measurement instruments still present (default-off, byte-identical): `SDPA_ABLATE_READER/WRITER`
(reader/writer NoC stubs), `SDPA_ZONE_PROFILE` (per-phase zones).

---

## NoC-multicast (KV read-once + broadcast) — TWO variants BUILT + MEASURED

Two multicast variants were built and measured on device (both opt-in, default OFF — the fast Q-outer
path stays default for every shape). They differ in ONE decision — whether the loop is restructured —
and that decision is the whole story:

| variant | flag | loop | compute floor | e2e (flagged shape) | vs baseline 5.28 ms |
|---|---|---|---|---|---|
| **A. Q-outer LOCKSTEP** | `SDPA_LOCKSTEP=1` | Q-outer **preserved** (reuses shipped fused compute) | **4.888 ms** (unchanged) | **4.921 ms** | **−6.7% (1.072× WIN)** |
| B. KV-outer restructure | `SDPA_MCAST=1` | KV-outer (4 resident q-chunk states) | 7.620 ms (**+56%**) | 7.626 ms | +44.3% (loss) |

The delta between them is entirely the compute floor: preserving the Q-outer loop (variant A) keeps the
4.888 ms floor, so the broadcast's read-contention reduction shows up as a real win; restructuring to
KV-outer (variant B) inflates the floor +56%, swamping any read benefit. **Lesson: for a multicast to
help SDPA, it must NOT restructure the loop — put the broadcast on top of the unchanged Q-outer compute.**

---

### VARIANT A: Q-outer LOCKSTEP multicast — MEASURED +6.7% WIN (compute floor PRESERVED)

**Status:** BUILT (opt-in `SDPA_LOCKSTEP=1`, default OFF) and MEASURED. This is the variant that WORKS.
The thesis — keep the compute floor at 4.888 ms by NOT restructuring, put broadcast on top, shave the
exposed reads — is confirmed with a structural proof (the floor number) and a real e2e win.

**What was built:** the loop structure is UNCHANGED from baseline (Q-outer — one q-chunk + one `(m,l,O)`
state per core at a time), so the **shipped optimized compute kernel is reused VERBATIM** (`fuse_rowsum` /
`fuse_oaccum` / packer-L1-acc, `Skv_chunk_t=8` → 37 KV chunks). The ONLY changes from baseline: (1) work
is head-grouped — each head's q-chunks live on one grid row (10 heads × 11 = 110); (2) the KV stream is
delivered by ONE column-0 injector per row (`Mcast1D` PerRow) that reads each KV chunk once from DRAM and
`noc_async_write_multicast`es it, while the row marches its KV index in LOCKSTEP over "waves" (wave w =
every core folds its w-th owned q-chunk over the full KV loop, one state live; `num_waves = ceil(37/11) =
4`; the mcast semaphore handshake is the barrier; KV CB double-buffered so the injector prefetches j+1).
The compute cannot tell whether K/V arrived via DRAM or mcast — it just consumes `cb_k_in`/`cb_v_in`. Only
the reader + writer + host wiring are new: `scaled_dot_product_attention_lockstep_{reader,writer}.cpp`,
`_lockstep_eligible` / `_create_lockstep_program_descriptor` (reuses the baseline compute kernel + CB
layout). Last wave is clamp-padded (short-wave cores recompute the last q-chunk, writing it again with the
same value) so every core stays active every wave — no idle receivers, critical path stays 4 waves = the
baseline max-per-core.

**Measured (same session, warm median ×5 device-kernel ns, flagged 1×10×9472×128, bf16 HiFi2 fp32_dest_acc_en=False):**

| variant | flag | e2e | vs baseline |
|---|---|---|---|
| baseline Q-outer | default | 5.275 ms | — |
| **Q-outer lockstep, broadcast ON** | `SDPA_LOCKSTEP=1` | **4.921 ms** | **−6.7% (1.072×)** |
| Q-outer lockstep compute FLOOR (reads+writes stubbed) | `+ SDPA_MCAST_ABLATE_READER=1 SDPA_MCAST_ABLATE_WRITER=1` | **4.888 ms** | — |
| Q-outer lockstep, broadcast OFF (per-wave per-core DRAM re-reads) | `+ SDPA_MCAST_NO_BCAST=1` | 8.183 ms | +55% |

- **CRITICAL THESIS CHECK — the compute floor is 4.888 ms, IDENTICAL to the baseline Q-outer floor.** The
  grouping + lockstep did NOT raise the floor (structural proof: the compute kernel is byte-identical to
  the shipped one). This is the single number that distinguishes this variant from the KV-outer one.
- **Broadcast SHAVES the exposed reads.** e2e 4.921 ms sits only 0.033 ms (0.67 %) above the floor —
  reads-visible dropped from the baseline's **0.387 ms (7.3 %)** to **0.033 ms (0.67 %)**. So the broadcast
  captured ~91 % of the baseline's read headroom (Amdahl ceiling was 4.896 ms). Mechanism: the baseline
  has 110 cores independently hammering DRAM (contention → 0.39 ms exposed); the lockstep funnels reads
  through **10 injectors** (10× less DRAM contention) and delivers over the NoC, which hides behind
  compute under the double-buffer. Read volume 1.8 GB → ~192 MB (10 injectors × 4 waves × 4.8 MB ≈ 9× less).
- **The lockstep barrier does NOT stall.** e2e within 0.7 % of the floor ⇒ the injector's read+mcast+
  handshake is fully hidden behind one step of QK^T+softmax+PV compute (double-buffered); for this uniform
  non-causal shape the 11 cores march in tight lockstep. The 10:1 injector serial-read path did NOT expose
  more than the baseline's independent reads — it exposed LESS (contention win).
- **Broadcast is load-bearing here** (unlike variant B): broadcast ON 4.921 vs OFF 8.183 ms. Without it,
  the lockstep's per-wave re-read structure has all 110 cores re-reading their head's KV 4× → worse DRAM
  contention than baseline. The broadcast is what turns the per-wave structure from a pessimization into a
  win.

**Conclusion:** a multicast CAN help SDPA — but only by staying Q-outer (floor untouched) and letting the
broadcast cut DRAM read contention. Measured +6.7 %, correct (PCC ≥ 0.999 across the eligible class incl.
golden long-context cells, no hang under `--dev`). Kept opt-in pending broader validation; recommended to
ship default-on for the eligible class (a real, low-risk improvement).

---

### VARIANT B: KV-outer restructure — MEASURED +44% REGRESSION (roofline confirmed)

**Status:** BUILT (guarded, **opt-in `SDPA_MCAST=1`** — default OFF) and MEASURED on device. Verdict, now
with evidence instead of a model: **net loss, not shipped on.** The roofline's design-first prediction
(below the fold) was correct; this section replaces the prediction with the measured numbers. The fast
Q-outer path stays the default for every shape (zero regression); the mcast path is kept in-tree gated
behind `SDPA_MCAST=1` because a measured negative + a working KV-outer reference is more useful than a
deleted branch.

**What was built:** `Mcast1D` PerRow — one row-family per `(b,h)`, the column-0 injector reads each KV
chunk **once** from DRAM and `noc_async_write_multicast`es it across its 11-core row (10 groups × 11 =
110 on Blackhole). Loop restructured Q-outer → **KV-outer**: each core holds the online-softmax `(m,l,O)`
state of the several q-chunks it owns RESIDENT in rotating CBs and folds each broadcast KV chunk into all
of them. Kernels: `scaled_dot_product_attention_mcast_{reader,compute,writer}.cpp`; guard +
`Mcast1D` wiring in the program descriptor (`_mcast_eligible` / `_create_mcast_program_descriptor`).

**Measured (same session, warm median ×5 device-kernel ns, flagged shape 1×10×9472×128, bf16, HiFi2,
fp32_dest_acc_en=False):**

| variant | flag (all vs the flagged shape) | e2e | vs baseline |
|---|---|---|---|
| baseline Q-outer | default (mcast off) | **5.284 ms** | — |
| **mcast, broadcast ON** | `SDPA_MCAST=1` | **7.626 ms** | **+44.3%** |
| mcast, broadcast OFF (KV-outer, redundant per-core DRAM reads) | `SDPA_MCAST=1 SDPA_MCAST_NO_BCAST=1` | 7.654 ms | +44.9% |
| mcast compute floor (reads+writes stubbed) | `SDPA_MCAST=1 SDPA_MCAST_ABLATE_READER=1 SDPA_MCAST_ABLATE_WRITER=1` | **7.620 ms** | — |

**The regression is ENTIRELY the KV-outer restructure raising the compute floor — NOT the broadcast:**

- **The mcast compute floor is 7.620 ms vs the baseline Q-outer floor 4.888 ms — the restructure raises
  the floor +56%.** This is the killer, exactly the roofline's cost (iii). Two compounding causes, both
  consequences of the restructure: (a) holding **4 q-chunks' `(m,l,O,q_scaled)` state resident** (~320
  tiles) does not fit L1 at `Skv_chunk_t=8`, forcing `Skv_chunk_t=4` → **74 KV chunks instead of 37**, so
  2× the per-chunk phase-boundary / init / reduce overhead; (b) this variant's compute is **unfused** (no
  `fuse_rowsum`/`fuse_oaccum`/packer-L1-acc that the shipped Q-outer kernel uses) plus the rotation
  bookkeeping (an extra pop-to-scratch for the running max, and `mul→temp`+`add` for the l/O updates).
- **The broadcast itself is neutral-to-slightly-POSITIVE, and negligible.** Broadcast ON (7.626) beats
  broadcast OFF (7.654) by only **0.028 ms** — the broadcast does cut the exposed reads, but reads are
  **~99.9% hidden**: the compute floor (7.620, reads+writes+handshake all removed) sits **0.006 ms** below
  the broadcast-ON e2e (7.626). So delivering KV once and broadcasting saves essentially nothing, because
  the reads it removes were already behind compute.
- **Roofline's two "extra cost" predictions, measured:** (i) the **10:1 consumer→injector serial
  dependency** and its semaphore handshake — REAL (the code has it), but its EXPOSED cost is ~0 (the
  handshake + reads + writes together add only 0.006 ms over the no-handshake compute floor; the dominant
  compute hides it). (ii) **NoC broadcast congestion** — did NOT materialize as exposed cost (broadcast ON
  even edged out broadcast OFF; delivery is hidden). Both predicted costs are real but invisible next to
  the inflated compute floor.
- **Read volume did drop ~37× as designed** (1.8 GB baseline → ~48 MB: 10 injectors each read their head's
  K/V once). Achieved baseline read BW ≈ 341 GB/s (1.8 GB / 5.284 ms). But the reduction is **worthless**:
  reads were ~93% hidden to begin with (baseline reads-visible 7.1% / 0.375 ms), so removing them can save
  at most 0.375 ms — a ceiling the +56% floor inflation swamps many times over.

**Conclusion (measured, not modeled):** same lesson as read-batching — cutting read **bytes/volume/count**
does nothing when reads are hidden behind compute; and here the restructure REQUIRED to enable the mcast
(KV-outer + resident multi-q-chunk state) more than doubles the per-chunk overhead and forces a finer KV
chunk, raising the compute floor +56% and turning a 7.1%-ceiling read optimization into a +44% net loss.
Even a maximally-optimized KV-outer compute (matching the 4.888 ms floor) could recover only the ~0.375 ms
of exposed reads and, per the ON-vs-OFF delta, the broadcast contributes only ~0.028 ms of that — so the
scheme cannot beat the Q-outer baseline. The only levers with headroom remain on **compute** (the 4.888 ms
floor), not data movement. The prior design-first roofline verdict is CONFIRMED.

---

### (original design-first roofline prediction, retained for the record — CONFIRMED by the measurement above)

**Why it can't win — the reads are already ~93% HIDDEN behind compute (ablation):**

| variant | e2e (median warm ×5) |
|---|---|
| baseline (all DRAM) | **5.272 ms** |
| reader NoC stub (0 read bytes) | 4.896 ms |
| writer NoC stub (0 write bytes) | 5.245 ms |
| reader+writer stub (0 DRAM bytes) = **compute floor** | **4.888 ms** |

- **reads-visible = 7.12 % (0.375 ms)**; op is **compute-bound** at a 4.888 ms floor. Amdahl ceiling of any
  read optimization = +7.1 %. mcast does not reduce the compute floor (identical FLOPs) and the KV-outer
  restructure forces each core to hold ~4× the `(m,l,O)` state resident — predicted to RAISE the floor.
  **Predicted:** realistic ~0% or negative. → Measured: **+44%** (floor rose +56%).
