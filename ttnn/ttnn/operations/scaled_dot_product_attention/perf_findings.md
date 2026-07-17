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

## NoC-multicast (KV read-once + broadcast) — roofline says NO-WIN, NOT built (design-first verdict)

**Status:** rooflined before writing any kernel (perf-roofline-dm Mode A). Verdict: **do not build.**
No code written; working tree unchanged.

**Idea considered:** eliminate the 37× redundant K/V DRAM re-read by grouping each (b,h)'s 37 q-chunks
onto an ~11-core rectangle (10 groups × 110 cores), restructuring Q-outer→KV-outer, and having one
injector per group read each KV chunk **once** and `noc_async_write_multicast` it to the rectangle
(`Mcast2D` + `mcast_pipe`, per `examples/shared_input_reuse`). Cuts DRAM read volume 37× (1.8 GB → ~48 MB).

**Why it can't win — the reads are already ~93% HIDDEN behind compute (fresh ablation, same session):**

| variant | e2e (median warm ×5) |
|---|---|
| baseline (all DRAM) | **5.272 ms** |
| reader NoC stub (0 read bytes) | 4.896 ms |
| writer NoC stub (0 write bytes) | 5.245 ms |
| reader+writer stub (0 DRAM bytes) = **compute floor** | **4.888 ms** |

- **reads-visible = 7.12 % (0.375 ms)**, writes-visible = 0.51 %. The op is **compute-bound** at a
  4.888 ms floor; DRAM reads overlap ~93 % with compute and expose only 0.375 ms.
- **Amdahl ceiling:** the ABSOLUTE best any read optimization can do is drive exposed reads to 0 →
  e2e 4.896 ms = **1.077× / +7.1 %**. mcast is a read optimization, so 7.1 % is its hard ceiling.
- **mcast does NOT reduce the 4.888 ms compute floor** — identical online-softmax FLOPs (loop-order +
  data-source change, not a math change). The floor *is* the bottleneck, and mcast leaves it untouched.
- **mcast can't even reach the 7.1 % ceiling.** Modeled transfer: per group the injector reads a
  128 KB KV chunk from DRAM (10 injectors, low contention) and mcasts it to ~10 cores; per-chunk compute
  (QK^T+softmax+PV over each core's ~4 q-chunks) dwarfs a 128 KB delivery, so mcast reads hide behind
  compute **just like the per-core reads do today** — same ~0 net on exposed latency. Worse, mcast adds
  two costs absent today: (i) a **10:1 consumer→injector serial dependency** (today all 110 cores read
  independently in parallel — no core waits on another; mcast makes 10 cores block on 1 injector +
  semaphore handshake per KV chunk), and (ii) **NoC broadcast congestion** on a grid the batching
  dead-end above proved is congestion-bound (front-loading NoC traffic = +39 %). The KV-outer restructure
  also forces each core to hold ~4× the (m,l,O) online-softmax state resident and interleave q-chunks —
  adding reconfig/state-switch overhead that RAISES the compute floor.

**Predicted:** ceiling +7.1 %, realistic ~0 % or negative (restructure raises floor + broadcast re-triggers
congestion). Not worth the complexity/hang-risk of the KV-outer restructure + 10-group injector pipeline.
Same lesson as batching: reducing read **bytes/volume/count** does nothing when reads are hidden and the
bottleneck is compute (+ NoC congestion). The only levers with headroom are on **compute** (the 4.888 ms
floor), not data movement.
