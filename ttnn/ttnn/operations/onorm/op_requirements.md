# Operation Requirements: onorm

## Definition

- **Formula**:
  `n[b,t,h,:] = o[b,t,h,:] * rsqrt(mean(o[b,t,h,:]^2) + eps) * weight[:]`
  `out[b,t,h*V+c] = n[b,t,h,c] * sigmoid(gate[b,t,h*V+c])`
  i.e. `out = flatten_heads( RMSNorm_over_V(o) * weight ) * sigmoid(gate)` — the
  Kimi-Linear KDA **s6** tail, fused on-chip.

- **PyTorch Reference** (standalone):

  ```python
  def pytorch_onorm(o, gate, weight, epsilon=1e-5):
      """o: [B,T,HV,V] head-major; gate: [B,T,HV*V] pre-sigmoid; weight: [...,V]."""
      B, T, HV, V = o.shape
      of = o.to(torch.float32)
      rms = torch.sqrt(torch.mean(of ** 2, dim=-1, keepdim=True) + epsilon)   # [B,T,HV,1]
      normed = of / rms * weight.to(torch.float32).reshape(-1)                # [B,T,HV,V]
      normed_flat = normed.reshape(B, T, HV * V)                              # head-major flatten
      out = normed_flat * torch.sigmoid(gate.to(torch.float32))               # [B,T,HV*V]
      return out.to(o.dtype)
  ```

- **Import Path**: `from ttnn.operations.onorm import onorm`

- **Function Signature**:

  ```python
  onorm(
      o: ttnn.Tensor,                                        # [B, T, HV, V]  head-major, TILE, bf16
      gate: ttnn.Tensor,                                     # [B, T, HV*V]   flat token-major, TILE, bf16 (pre-sigmoid)
      weight: ttnn.Tensor,                                   # [1, 1, 1, V]   RMSNorm scale, TILE, bf16
      epsilon: float = 1e-5,
      compute_kernel_config: ttnn.DeviceComputeKernelConfig = None,
  ) -> ttnn.Tensor                                           # [B, T, HV*V]   flat token-major, TILE, bf16
  ```

  Fixed KDA s6 geometry: `HV = 32`, `V = 128`, `FLAT = HV*V = 4096`, `T % 32 == 0`.

---

## Why this queue is all-perf

`TARGET − SUPPORTED = ∅`. Both axes are already fully covered:

| Axis | TARGET | SUPPORTED | Gap |
|---|---|---|---|
| `dtype` | `[ttnn.bfloat16]` | `[ttnn.bfloat16]` | — |
| `layout` | `[ttnn.TILE_LAYOUT]` | `[ttnn.TILE_LAYOUT]` | — |

`INVALID = []`, `EXCLUSIONS = []`, `xfail_expected = 0`. This is deliberate — `feature_spec.py`
pins the single KDA bringup configuration and states there is no generality backlog. So every
phase below is a **measured perf refinement**: it adds nothing to `SUPPORTED`, and its acceptance
is gated on device-ns, not on a category moving.

`feature_spec.py` declares no `LOOSE_CASES`, so **no shape is perf-flagged**; the target regions
below are verifier-selected from the Phase-0 measurements recorded in `verification_report.md`.

### The measurement that orders this queue

The design assumed the op is DRAM-bandwidth-bound. **It is not.** Measured on Blackhole p150:

- per-core achieved bandwidth is **3.1 GB/s** against the ~17.9 GB/s single-core NoC ceiling;
- **P7b — `sigmoid(gate)` on the SFPU — is 152.7 µs of a 239 µs kernel (63.9 %)**;
- one 32-token block costs **~240 µs on one core, essentially independent of core count**
  (T=64 → 2 cores → 239.6 µs; T=640 → 20 cores → 244.1 µs);
- every compute-only config knob moves the number (`fp32_dest_acc_en=False` 1.095×, LoFi 1.047×,
  `math_approx` 1.030×, all three 1.112×) — which a DRAM-bound op would not do.

So: the op is **SFPU-bound on the MATH thread**, and the design's "that lever will not pay
because we are DRAM-bound" dismissals (`op_design.md` §1.5 `RECONFIG_MODE`, §6.2 block-size
amortization) are void and are re-opened by Refinement 3.

---

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Measurement discipline (applies to every phase below — this op punishes sloppy timing).**
> Single-shot profiler numbers for onorm are not reproducible across processes (a 248 µs vs
> 102 µs swing on identical config is on record). Every number you report must be either a
> median of ≥ 5 **trial-major interleaved** repetitions inside one process
> (`test_onorm_trials.py` is the harness) **or** a wall-clock mean of ≥ 10 back-to-back
> dispatches after warm-up with explicit `synchronize_device` — and the two methods must agree.
> Also: the per-phase `MaybeDeviceZoneScope` wraps each helper *including* its `cb_wait_front`,
> so a starved phase reads as a slow phase. Never attribute cost to a phase on zone time alone;
> corroborate with a compute-only knob that cannot touch the NoC.

> **Config-spanning guard set** (referenced by every `Done when` below). onorm has one kernel
> path, one layout and one placement, so the guard set is the shape/occupancy span plus the
> config span: `(B=1,T=32)` 1 core, `(B=1,T=128)` 4 cores, `(B=1,T=640)` 20 cores,
> `(B=8,T=640)` 110 cores — each at the default compute config, plus `(B=1,T=640)` at
> `math_approx_mode=True` and at `MathFidelity::LoFi`. No regression means no shape in that set
> gets slower beyond measurement spread (< 2 %).

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: `[ttnn.bfloat16]` (== TARGET)
- **SUPPORTED layout**: `[ttnn.TILE_LAYOUT]` (== TARGET)
- **SUPPORTED shape-derived axes**: none — `INPUT_TAGGERS = {}` (fixed `HV=32`/`V=128`, tile-aligned `T`)
- **SUPPORTED op-specific axes**: none as registry axes; `epsilon` and `compute_kernel_config` are free parameters
- **EXCLUSIONS**: `[]`
- **Cores**: multi-core from day one — `split_work_to_cores(grid, B*ceil(T/32), row_wise=True)`; measured `CORE COUNT` = 2 / 4 / 20 / 110 on the four profiled shapes
- **Compute config**: caller-overridable; default = HiFi4 + `math_approx_mode=False` + `fp32_dest_acc_en=True` + `dst_full_sync_en=False`, resolved through the single exported `default_compute_kernel_config()`
- **Golden baseline**: **5 / 5 registry cells passing**; `supported_fail = xpass_drift = xfail_wrong_mode = 0` (`verifier_report.json`)
- **Accuracy baseline**: PCC 0.999993, rel-RMS 0.0037, got/true ratio median 0.9997 (no scale bug) across 4 shapes
- **Device-ns baseline** (median of 5 interleaved trials): 239,587 ns @ B=1/T=64 · 239,690 ns @ B=1/T=128 · 244,110 ns @ B=1/T=640 · 538,989 ns @ B=8/T=640

---

### [~] Refinement 1 — Get `sigmoid(gate)` off the saturated MATH thread

**Type**: perf

**Goal**: P7b (`unary<Sigmoid<>, cb_gate_tiles, cb_gate_sig>`) is **63.9 % of the kernel** —
152.7 µs of 239.1 µs per core at B=1/T=640, and the same share on every other supported shape,
because it is a fixed 128 SFPU tile-ops per token-block. It is genuine SFPU work, not a
`cb_wait_front` stall: it responds to compute-only knobs that cannot touch the NoC, and it
costs 1.19 µs/tile ≈ 37 ns per SFPU vector op under a 32-bit DEST. Reduce its contribution to
the critical path on **all** supported cells. No SUPPORTED change.

Two concrete leads, in order of expected payoff — plus the catalog:

1. **Run the sigmoid on the PACK thread instead of MATH.** `sfpu_activation_helpers.hpp` already
   exposes `SigmoidActivation<VecMode, Fast>` and `apply_activation_from_pack()`, which route to
   `sigmoid_tile_init_pack<Fast>()` / `sigmoid_tile_pack<vec_mode, Fast>()`. Today `onorm`'s MATH
   thread carries 100 % of the SFPU volume while PACK is comparatively idle, so this is a
   thread-rebalance, not a cost reduction — potentially most of the 64 % rather than a few
   percent of it. `op_design.md` §9 dismissed this header as "only fills the `Activation` slot of
   `matmul_block` / `add_bias_bcast_rows`"; that is true of its current *call sites*, not of the
   mechanism. Establish first whether the pack-side path can be driven from an `eltwise_chain` /
   `unary` shape, or whether P7b/P7c must become a single chain whose pack stage applies the
   activation.
2. **Per-call `fast_and_approx`, decoupled from the global `math_approx_mode`.**
   `sigmoid_tile<VectorMode, fast_and_approx>` and `sigmoid_tile_init<fast_and_approx>` take the
   approximation as a *template* argument, but `kernel_lib`'s `Sigmoid<Slot>` hardcodes the
   accurate variant. A `SigmoidFast` activation would buy the sigmoid's share of the measured
   1.030× global `math_approx_mode` win **without** relaxing the `rsqrt` in P2. Cheap, and a
   useful A/B even if lead 1 lands.
3. **Catalog**: `ttnn/ttnn/operations/examples/master.md` → `compute_fusion` (engine choice and
   what SFPU-vs-FPU actually costs) and `sfpu_tile_scope` (SFPU work-scoping — note it does *not*
   apply to the sigmoid itself, whose result is a full tile, but its ns-per-vector-op model is
   the right cost model to reason with).

**Verifier notes**: first because it is the largest single lever and it is **topology-independent**
— it is a per-tile SFPU cost that survives whatever Refinement 2 does to the work split, so doing
it first means it is never re-done. Two hard constraints:
(a) **Do not reach for `fp32_dest_acc_en=False`.** It is measured at 1.095× and the PCC margin
would absorb it, but `eval/prompts/onorm.txt` directs fp32 sum-of-squares accumulation in DST. If
you want that 9.5 %, preserve fp32 accumulation for P1 by another mechanism, or get an explicit
documented deviation — never flip it silently.
(b) **Do not collapse P7b and P7c into one DEST-resident SFPU chain.** That is the measured 0.58×
mistake (`compute_fusion`), on the op's largest compute volume. If lead 1 forces the two phases
into one chain, the multiply must still land on the **FPU**.
The `cb_gate_sig` L1 hop exists precisely to feed the FPU unpacker; if a pack-side sigmoid removes
the need for it, that CB's 64 pages (128 KB) come back to the L1 budget — hand that headroom to
Refinement 3, do not spend it here.

**Done when**: measured device-ns improves on B=1/T=640 and B=8/T=640 (the two occupancy regimes
where P7b's 64 % is on the critical path), with the per-phase zone breakdown re-recorded to show
P7b's share actually fell; no regression across the config-spanning guard set; the golden suite is
green; and `test_onorm_precision_baseline.py` still passes with its numbers recorded in
`changelog.md` (a `fast_and_approx` sigmoid *will* move them — it must stay inside PCC ≥ 0.9995
and rel-RMS < 0.02).

**OUTCOME (partial)** — both named leads were implemented, measured and closed; the
premise behind them turned out to be unavailable in hardware. Evidence in `changelog.md`.

- **Ablation first.** A third `SIGMOID_ENGINE` value, `"ablate"`, removes the sigmoid
  payload while keeping every CB wait/push, DEST window and NoC transfer. It confirms the
  phase is genuine SFPU work and not a `cb_wait_front` stall: B=1/T=640 drops 244,495 →
  92,212 ns. The sigmoid really is **152 µs (62 %)**, and 37 ns/vector-op × 32 vector-ops
  × 128 tiles matches the catalog's measured Blackhole SFPU rate (`sfpu_tile_scope`:
  ~24 ns rsqrt, ~28 ns recip per vector op). There is no misconfiguration to fix.
- **Lead 1 (pack thread) — built, correct, measured 0.991–0.994× on all three shapes.**
  It is kept as a live non-default `SIGMOID_ENGINE = "pack"` knob. It does not win, and
  the reason is structural rather than tunable: a Tensix has **one SFPU**, shared by MATH
  and PACK, and DEST half-sync pins the two threads to within one window — so relocating
  the SFPU volume relocates the bottleneck instead of overlapping it, and the pack path
  additionally pays `apply_activation_from_pack`'s SEMWAIT + dest-offset-flip + WAIT_SFPU.
  The premise "PACK is comparatively idle" is true of the *thread* and false of the
  *engine*. This closes lead 1 — no configuration of it can pay.
- **Lead 2 (`fast_and_approx`) — provably a no-op, no measurement possible.** The LLK's
  `_calculate_sigmoid_` / `_init_sigmoid_` **ignore `APPROXIMATION_MODE`** on both
  Blackhole and Wormhole B0 (`tt_llk_*/common/inc/sfpu/ckernel_sfpu_sigmoid.h`): accurate
  and fast are the same 6-entry LUT. There is no `SigmoidFast` to build. Closed.
- **Third lever, found and adopted: `GATE_DEST_TILES`.** Phase 0 opened one DEST window
  *per tile* in both gate phases (`InputLifecycle::Streaming` clamps the chain's
  `block_size` to 1). Moving both P7b **and** its 1:1 twin P7c to `InputLifecycle::Chunked`
  makes the tiles-per-DEST-window a knob, adopted at **4** (`DEST_AUTO_LIMIT` under
  `fp32_dest_acc_en` + half sync). Monotonic 1 → 2 → 4 and **1.003–1.006× on every cell of
  the config-spanning guard set**, no cell regressing, free in L1. Small because the phase
  is dominated by the SFPU payload, not per-window overhead — with the sigmoid ablated the
  same knob is worth 1.016×.
- **Not done here, and why:** the one large remaining lever is the DEST width, which
  constraint (a) explicitly fences off. It is now *priced* rather than guessed — see
  Refinement 1b. `cb_gate_sig` was **not** eliminated (the pack engine still materialises
  it), so its 64 pages are not new headroom for Refinement 3.

---

### [ ] Refinement 1b — Cash the 16-bit DEST: 1.208×, with P1's fp32 sum-of-squares preserved

**Type**: perf

**Goal**: Refinement 1 established that `sigmoid(gate)`'s 152 µs is irreducible SFPU
*throughput* — one SFPU per core, no approximation variant, per-window overhead already
amortised. Exactly one lever moves it on a fixed core count, and R1 measured it instead of
guessing at it:

| B=1/T=640, median of 5 interleaved trials | whole kernel | sigmoid payload (vs its own ablation) |
|---|---|---|
| `fp32_dest_acc_en=True` (today's default) | 244,312 ns | **152,167 ns** |
| `fp32_dest_acc_en=False` | **202,256 ns (1.208×)** | **109,354 ns (1.39×)** |

**28 % of the sigmoid's cost is the 32-bit DEST**, and the non-sigmoid remainder is
unchanged (92.1 → 92.9 µs), so the win is specifically the SFPU's. Note this is much
larger than the 1.095× Phase 0 recorded for the same flag.

Measured precision cost of the flip (4 shapes, `probes/probe_005.py`):

| | PCC | rel-RMS | got/true median |
|---|---|---|---|
| fp32 DEST on | 0.999993 | 0.0037 | 0.9997 |
| fp32 DEST off | **0.999988** | **0.0056** | 1.0026 |

Both sit comfortably inside Refinement 1's own stated bar (PCC ≥ 0.9995, rel-RMS < 0.02) —
rel-RMS is 3.5× under the limit — and the got/true spread stays centred on 1.0, so this is
rounding, not a scale bug.

**The work is not "flip the flag".** `eval/prompts/onorm.txt` directs fp32 sum-of-squares
accumulation in DST, and R1's constraint (a) forbids flipping it silently. Two routes, in
order of preference:

1. **Preserve P1's fp32 accumulation by another mechanism** and take the 16-bit DEST for
   everything else. P1 is the only fp32-sensitive step: it DEST-accumulates `o²` over
   `V_TILES = 4` tiles (`DestAccumulation::Enabled` → `OutputLifecycle::DestAccumulation`).
   Candidates: the packer's L1 accumulator (`PackTileL1Accumulation` — but the catalog
   records it as fp32-DEST-only, so check that first), a `Float32` `cb_sumsq` with the
   accumulation moved out of DEST, or the `row_reduce_accumulate` catalog entry's
   `dest_accum_pairs` shape. Note `fp32_dest_acc_en` is a whole-kernel compile-time
   config, so "per-phase fp32" means changing *where* the accumulation lives, not the flag.
2. **Explicit documented deviation** with the numbers above, if route 1 costs more than it
   saves. This needs sign-off — it is a contract change, not a knob turn.

Free rider: `fp32_dest_acc_en=False` raises `DEST_AUTO_LIMIT` from 4 to 8, so
`GATE_DEST_TILES` could then go to 8. Re-sweep it (`_DEST_TILE_LIMIT` in
`onorm_program_descriptor.py` is the one place that constant lives).

**Verifier notes**: ordered immediately after its parent and **before Refinement 2**, per
the partial-tick protocol. Like R1 it is per-tile and topology-independent, so it survives
R2's work-split change unchanged. Everything R1 built is reusable: the ablation engine
prices the sigmoid payload on its own, and
`test_onorm_sigmoid_engine.py::test_dest_acc_trial` already measures both DEST widths
trial-major interleaved.

**Done when**: measured device-ns improves on B=1/T=640 and B=8/T=640 with the chosen
mechanism landed as the default; `test_onorm_precision_baseline.py` passes with the new
numbers recorded in `changelog.md` and inside PCC ≥ 0.9995 / rel-RMS < 0.02; the
`fp32_dest_acc_en=True` path still works for a caller who asks for it (it is a public
`compute_kernel_config` field, so it cannot become a lie); no regression across the
config-spanning guard set; and the golden suite plus `test_onorm_knobs.py` are green. If
route 2 is taken, the deviation must be written into `changelog.md` and
`default_compute_kernel_config()`'s docstring, naming the contract line it departs from.

---

### [ ] Refinement 2 — Cross-core re-tile: stop leaving 108 of 110 cores idle at small `T`

**Type**: perf

**Goal**: the work unit is one token-block = 32 tokens, because the head-major → flat re-tile
fuses exactly 32 tokens into one output tile-row. So `B*ceil(T/32)` is the total unit count, and
at the shapes a decode/short-prefill caller actually issues that number is tiny:

| Shape | token-blocks | cores used | measured device ns |
|---|---|---|---|
| B=1, T=32  | 1  | **1**  | ~250,000 (wall clock) |
| B=1, T=64  | 2  | **2**  | 239,587 |
| B=1, T=128 | 4  | **4**  | 239,690 |

**These take the same ~240 µs as the 20-core T=640 case** — the machine is 96–99 % idle and the
latency is identical. Nothing else in this queue can touch them: Refinement 1 shrinks the
per-core constant, Refinement 3 tunes it, but only a finer work unit adds cores.

Implement the design's own lamp #1 (`op_design.md` §1.5, "cross-core re-tile"): split a token-block
across cores so each core normalizes its own subset of the 32 tokens, then NoC-writes its
4096-byte row-major slices into the owning core's `cb_rm_flat_rows` at byte offset `t*FLAT*2`,
so the 32 rows of one output tile-row can come from up to 32 different cores. The design
deliberately left this reachable: `cb_rm_flat_rows` is *already* a plain row-major L1 stripe
addressed by token row, and a remote writer filling row `t` honours exactly the contract the local
untilize already honours. Use `mcast_pipe.hpp`'s `SenderPipe` / `ReceiverPipe`
(`ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`) plus one semaphore per block — **do not** hand-roll
`noc_async_write_multicast` + `noc_semaphore_set/wait/inc`. Catalog: `master.md` →
`shared_input_reuse` (the `mcast_pipe` + `ttnn.Mcast2D` host wiring worked end-to-end) and
`tensix_all_reduce_ring_transport` (direction-sensitive NoC contention — relevant to how you lay
the exchange group out). No SUPPORTED change.

**Verifier notes**: **scheme-change, stands alone** — the new cross-core topology *is* the work,
and it is the only entry in this queue that is not a knob-turn. Ordered after Refinement 1
because R1's lever is per-tile and topology-independent (it carries over unchanged), and before
Refinement 3 because R3 re-tunes block factors against the *final* structure — tuning them now
would be thrown away.

Three things the design already guarantees you, and one it warns about:
- `cb_rm_flat_rows` must stay **exactly** `FLAT_TILES` pages per owning core, filled from and
  drained to the buffer base every block. Larger lets the ring wrap mid-block and the tilize
  address generator — which assumes one contiguous `[32, FLAT]` stripe of stride `FLAT*2` bytes —
  reads garbage. Smaller deadlocks the untilize (`op_design.md` risk 1).
- Keep reads on the reader/NoC0 and writes on the writer/NoC1; reads issued on NoC1 measured
  4.8× slower (`master.md` → `noc_placement`).
- Grid layout: keep `row_wise=True` ordering discipline — a column line measured 2.91× slower.
- The win is bounded by the exchange: you are trading DRAM-read parallelism for NoC traffic that
  did not exist before. Measure the crossover and, if a small block count is genuinely better
  served single-core, dispatch on it rather than always paying the exchange.

**Done when**: measured device-ns improves substantially on B=1/T=32, B=1/T=64 and B=1/T=128
(the under-filled shapes — these are the target region, and a several-fold win is the bar, not a
few percent), `CORE COUNT` in the profiler CSV rises accordingly for those shapes; no regression
on B=1/T=640 or B=8/T=640, which were already core-saturated; no regression across the
config-spanning guard set; the golden suite and `test_onorm_knobs.py` are both green (the knob
suite is what proves the block factors survived the restructure); and
`test_onorm_precision_baseline.py` still passes — a cross-core exchange must not perturb the
numerics at all, so the recorded PCC/rel-RMS should be bit-comparable, and any drift is a bug.

---

### [ ] Refinement 3 — Re-tune the compute block surface against the final structure

**Type**: perf

**Goal**: with the SFPU no longer dominating (R1) and the work split reshaped (R2), the remaining
~86 µs of per-core compute — P4 normalize 29.3 µs, P7c 19.2, P1 11.9, P5 9.4, P2 8.6, P6 4.3,
P7a 3.6 — becomes the critical path, and it is spread over **25 helper invocations per token-block**
(`NCH*5 + 1 + 2*(FLAT_TILES/GC)`). Two catalog levers apply directly and are measured to compound:

1. **Turn data-format reconfig off.** All twelve CBs are `Float16_b`, so the dtype never changes
   anywhere in the kernel and every helper's `BinaryDataFormatReconfig::Input` /
   `PackTileReconfig::Output` / `ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT` /
   `ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure` is wasted MMIO at each of those 25
   boundaries. `master.md` → `compute_block_size` measures **up to 1.19×** for exactly this, "largest
   where there are the most transitions". The uniform `Float16_b` format was chosen by the design
   *specifically* to make this a legal one-line-per-helper knob-turn (`op_design.md` §1.5, risk 11).
2. **Co-tune the block factors.** `NORM_CHUNK_TOKENS` (8), `GATE_CHUNK_TILES` (64),
   `TOKENS_PER_BLOCK` (32), `O_DEPTH` (2), `DM_BLOCK_TILES` (8), `DM_DEPTH` (4), and
   `GATE_DEST_TILES` (4, added and perf-selected by R1 — its ceiling is `_DEST_TILE_LIMIT`,
   which rises from 4 to 8 if R1b lands the 16-bit DEST, so re-sweep it then). Only the two DM
   knobs and `GATE_DEST_TILES` have ever been *perf*-selected; the compute-side factors were chosen from an L1-budget
   argument that assumed the op was DRAM-bound. Same `compute_block_size` mechanism: coarser blocks
   amortize the per-invocation fixed cost, whole tiles are the granularity floor, and the curve has
   diminishing returns — so name the direction, sweep, and take the measured optimum.
3. **Rider — `weight` mcast.** Each core re-reads 4 weight tiles (8 KB). It is ~0.7 % of traffic, so
   it is *not* worth its own phase, but `cb_weight` is already a standalone reader-produced,
   never-popped CB, and if R2 has already stood up `mcast_pipe` wiring, swapping this producer for
   `SenderPipe::send()` / `ReceiverPipe::receive()` is a reader-only change. Take it if it is nearly
   free at that point; drop it if it is not.

No SUPPORTED change.

**Verifier notes**: **last, and it must be last** — every lever here is a knob-turn whose optimum
depends on the structure R1 and R2 leave behind, so running it earlier means re-running it. Both
levers are knob-turns on the block surface the planner already exposed, which is why they share one
phase rather than getting three (the catalog's "several cheap levers in one phase" sizing).

Four constraints:
- `test_onorm_knobs.py` already proves every one of these knobs is live and that the host budget
  assert rejects out-of-L1 settings while naming what to lower — **use it as the correctness net for
  the sweep**, and add any new setting you adopt to `KNOB_SETTINGS` / `COMBOS`.
- The L1 budget is the binding constraint on coarsening: at 533 pages (1,091,584 B ≈ 74 % of the
  CB-available L1) the two re-tile buffers are already 256 pages, and they scale with
  `TOKENS_PER_BLOCK`. `TOKENS_PER_BLOCK = 64` only fits once `NORM_CHUNK_TOKENS` or
  `GATE_CHUNK_TILES` comes down — the assert enforces the order (`GATE_CHUNK_TILES` first, then
  `NORM_CHUNK_TOKENS`). **R1 did NOT eliminate `cb_gate_sig`** — its `pack` engine still
  materialises it, and the design's FPU-fed-from-L1 argument for P7c is unchanged — so its 64
  pages are *not* new headroom. Do not budget for them.
- Raising `TOKENS_PER_BLOCK` **reduces the core count** at small `T` (it coarsens the work unit).
  After R2 the two interact directly — do not tune `TOKENS_PER_BLOCK` on B=8/T=640 alone and then
  regress B=1/T=64.
- Reconfig-off is only correct while the dtype is genuinely constant across every boundary. It is
  today (all CBs are `o.dtype`), and `SUPPORTED["dtype"]` is single-valued so it cannot change
  under you — but if a future refinement ever widens `dtype`, this knob has to become conditional.
  Leave a comment saying so at the point you flip it.

**Done when**: measured device-ns improves on B=1/T=640 and B=8/T=640 with the chosen settings
landed as the new defaults in `onorm_program_descriptor.py`; the sweep evidence (median of ≥ 5
interleaved trials per candidate, with spread) is recorded in `changelog.md`; no regression across
the config-spanning guard set; `test_onorm_knobs.py` and the golden suite are green; and
`test_onorm_precision_baseline.py` still passes — reconfig-off must be bit-neutral (PCC unchanged),
so any PCC movement from lever 1 is a bug, not a trade.
