# Operation Requirements: rms_norm

## Definition
- **Formula**: `y = x / sqrt(mean(x², dim=-1) + ε) · γ` (γ ≡ 1 when absent), reduced over the last dimension
- **PyTorch Reference**:
  ```python
  def rms_norm_ref(x, gamma=None, epsilon=1e-6):
      xf = x.float()
      y = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
      if gamma is not None:
          y = y * gamma.float().reshape(-1)
      return y.to(x.dtype)
  ```
- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:
  ```python
  rms_norm(
      input_tensor: ttnn.Tensor,
      *,
      gamma: Optional[ttnn.Tensor] = None,           # shape (1, 1, 1, W)
      epsilon: float = 1e-6,
      compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,  # None -> default_compute_kernel_config()
      memory_config: Optional[ttnn.MemoryConfig] = None,                     # output placement; None -> DRAM interleaved
  ) -> ttnn.Tensor
  ```
  One `ttnn.generic_op` dispatch per call (LOCAL or W_SPLIT program variant, selected by `select_blocking()`).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (EXCLUSIONS: `{dtype: float32, fp32_dest_acc_en: False}` — permanent, precision convention)
- **SUPPORTED layout**: [TILE, ROW_MAJOR] (native: tilize/untilize inside the kernel, no host-side layout ops)
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, "none"}; gamma_layout ∈ {TILE, ROW_MAJOR, "none"}; memory_layout ∈ {INTERLEAVED}
- **Cores**: full grid. Two built regimes: **LOCAL** (tile-rows split across the grid, whole row resident per core) and **W_SPLIT** (lines of `w_split` cores share an R-range; Σx² partials gathered to the line root, rinv multicast back via `ttnn.Mcast`). `select_blocking()` is the single source of `w_split` / `core_w_tiles` / `block_rows`.
- **Compute config**: caller's `ComputeConfigDescriptor` passed as-is (fidelity / approx honoured; `math_approx_mode` → `Rsqrt<Approx>`); default = HiFi4 + fp32_dest_acc_en=True
- **Golden baseline**: 659 supported_pass / 5566 xfail_expected / 33900 invalid_skipped; supported_fail = xpass_drift = xfail_wrong_mode = 0 (per `verifier_report.json`)

### [x] Refinement 1 — Numerical configurability: fp32_dest_acc_en=False + bfloat8_b (input and γ)

**Goal**: add `False` to `SUPPORTED["fp32_dest_acc_en"]` (bfloat16 input; `{float32, False}` stays in EXCLUSIONS), add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and to `SUPPORTED["gamma_dtype"]`, and derive every intermediate-CB page format from the resolved `fp32_dest_acc_en` (Float32 when on, Float16_b when off). Native, in-kernel: no host-side typecasts. Cells that fail out of the box go to `EXCLUSIONS`, not to a separate refinement. Moves the 1221 interleaved `xfail_expected` cells (`fp32_dest_acc_en=False` ×753, `gamma_dtype=bfloat8_b` ×259, `dtype=bfloat8_b` ×209) to `supported_pass`, including all 12 interleaved `_perf_case` loose cases (bf16 / HiFi2 / fp32_dest_acc_en=False / TILE bf16 γ).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **Anchor**: this refinement lands the full config contract of the interleaved perf loose cases (`fp32_dest_acc_en=False` + HiFi2). HiFi2 is already accepted. After this lands, those 12 cells must be `supported_pass`, not xfail and not EXCLUDED.
- Ledger audit 3 (page format vs DEST width) is folded in here. Today `cb_sumsq_raw`, `cb_partial`, `cb_gathered`, `cb_rinv_col`, `cb_rinv_bcast` and `cb_rinv` are hard-coded `ttnn.float32`, and `_per_row_bytes()` / the writer's gather and multicast byte counts use `t_f = tile_size(float32)`. With fp32 DEST off these must become Float16_b. Derive them from **one** `t_int` / `int_fmt` value in `create_program_descriptor` that feeds the CB descriptors, `_per_row_bytes()` (the L1 budget, so `block_rows` grows when it halves) and the writer's `f32_tile_bytes` CT arg (rename it). Do not restate it.
- bfloat8_b γ in TILE layout: the writer's `load_gamma` reads only face0-row0 and face1-row0 (16 elements × `es_g` bytes). That addressing is meaningless for a block-float tile (shared-exponent section plus mantissa section). For bf8b γ, read whole tiles (`core_w_valid_tiles` pages, one barrier). `bf8b × ROW_MAJOR` γ is INVALID.
- bfloat8_b input: only TILE and tile-aligned cells are valid (INVALID removes bf8b×RM and bf8b×non-aligned), so the reader's `zero_w_pad_lanes<es>` is never instantiated for bf8b. Add a `static_assert`/host guard rather than relying on it silently. `t_in` = 1088 B feeds `cb_x` / `cb_y` sizing through `ttnn.tile_size(dtype)` already.
- `default_compute_kernel_config()` is the single place the default lives. Whether to move it to `fp32_dest_acc_en=False` (matching `ttnn.rms_norm`) is a contract decision the skill covers. If it moves, the golden tagger follows it automatically.
- Build it performantly. Refinement 3 optimizes the sharded version of this exact config, and Refinement 5 the interleaved one, so the Float16_b intermediate path must keep the same fused-chain structure (no extra L1 round-trips).

**Done when**: every interleaved cell with `fp32_dest_acc_en=False`, `dtype=bfloat8_b` or `gamma_dtype=bfloat8_b` (outside INVALID and EXCLUSIONS) is `supported_pass`; the 12 interleaved `_perf_case` loose cases pass their 0.9995 PCC gate; the acceptance and precision-baseline tests stay green; verifier loud categories are 0.

### [x] Refinement 2 — Sharded placements: HEIGHT / WIDTH / BLOCK_SHARDED, consumed in place

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED`, `WIDTH_SHARDED` and `BLOCK_SHARDED` to `SUPPORTED["memory_layout"]`, for both TILE and ROW_MAJOR input, with a sharded output that inherits the input's shard spec (the harness passes `memory_config=input.memory_config()`). Per `op_design.md` → "Sharded placements", these are **placements of the two already-built regimes**:
- HEIGHT_SHARDED → LOCAL. One shard core is one line, `w_split = 1`.
- WIDTH_SHARDED / BLOCK_SHARDED → W_SPLIT. One line per shard-grid row, `w_split` = shard-grid width, `core_w_tiles` = shard width in tiles. The existing gather + `ttnn.Mcast` combine is reused.

The input shard is the per-core block: `cb_x` (TILE) is backed on it zero-copy via `ttnn.cb_descriptor_from_sharded_tensor`, and the output CB is backed on the output shard the same way. Block = the whole resident shard (sharded default), sub-chunked only if the footprint check fails. Moves the 1550 fp32_dest_acc_en=True sharded `xfail_expected` cells to passing (HEIGHT ×490, WIDTH ×531, BLOCK ×529), plus every sharded cell that also needs Refinement 1's axes (another ~2795).

**Verifier notes**:
- **No skill covers `memory_layout` / sharded placement.** Do not use `/memory-layouts` (that skill covers RM/TILE layout). The required topology is fully specified in `op_design.md` (Regimes rows "HEIGHT_SHARDED placement of LOCAL" and "WIDTH / BLOCK_SHARDED placement of W_SPLIT"; Dataflow Strategy → Sharded placements).
- **Native means zero-copy.** A core must never read its own local shard back through a `TensorAccessor` as if it were interleaved. That only *tolerates* the layout and does not implement it; it would be sent back as incomplete even with green golden cells. TensorAccessor stays for interleaved I/O and for γ.
- **The shard grid replaces `select_blocking()`'s core choice** for sharded inputs. Lines come from the shard spec (grid, orientation, shard shape), not from the first `num_lines·w_split` row-major device cores. The Mcast receiver groups must be the shard-grid rows; groups that are not a single rectangle are already handled by the helper. Keep `select_blocking()` the single source of every knob: give it a sharded branch that *derives* `w_split` / `core_w_tiles` / `block_rows` from the shard spec. Do not add a parallel knob table.
- **Ragged shards.** The perf and resilience loose cases use ragged geometries (e.g. 9×10 grid with a padded last column, `[736, 128]` on 8×10). The last shard row or column holds padding: valid tile counts per core become runtime args, and nominal CT extents stay uniform (same uniform-address rule as W_SPLIT today).
- **Padding inside a backed shard.** For non-aligned W in TILE, the pad lanes of the last W tile live in the caller's input shard, and the pad-poison loose cases fill them with 1000. Zeroing them in place as the interleaved reader does would **mutate the caller's input tensor**. Mask them in compute instead (e.g. a mask applied to the last W tile in `sumsq_block`), or make a local copy of only the partial-W tile. Whichever is chosen, record it in the ledger.
- **ROW_MAJOR shards.** `tilize` reads `cb_x_rm` at stride `core_w_tiles·32·es`. A RM shard can back `cb_x_rm` directly only when its stick stride matches; otherwise a core-local L1→L1 re-stride is the minimal step (still no DRAM and no accessor read of the own shard). `RM × sharded × TILE γ` is INVALID; RM γ pairs with RM sharded.
- The 8192-row wide-prefill shapes are interleaved-only by construction (a resident shard does not fit). Shapes whose shard does not fit L1 are skipped by the harness as infeasible (`oom_guard`), not charged to the op.
- Order: this is the hardest remaining generality item (block-sharded, ragged grids), so it comes right after the numeric bundle and before any perf phase. It also lands the contract of the sharded `_perf_case` loose cases that Refinement 3 targets (WIDTH_SHARDED, bf16, fp32_dest_acc_en=False, HiFi2), so Refinement 1 must already be in. If the implementer can only land part of it, ship `[~]` and file `Refinement 2b` for the remaining schemes. Never use a `ttnn.to_memory_config` / `ttnn.sharded_to_interleaved` wrapper: it is a second dispatch and violates the one-dispatch rule.
- L1 ledger: add a sharded column (cb_x backed on the shard takes no allocator budget, so the `x_block_depth·br·cw·T_in` term drops out). Re-derive `block_rows` = whole shard and state the mechanism caps that clamp it.

**Done when**: every HEIGHT/WIDTH/BLOCK_SHARDED cell outside INVALID and EXCLUSIONS is `supported_pass` or infeasible-skipped by the harness (never op-failed); that includes the sharded `_perf_case`, resilience and pad-poison loose cases; a grep of the kernels shows no accessor read of the core's own shard; interleaved cells and acceptance tests show no regression; verifier loud categories are 0.



### [x] Refinement 2b — Sharded placements: HEIGHT / WIDTH / BLOCK_SHARDED, consumed in place (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 2 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 3 FAIL: REGRESSION — prior-passing golden cells no longer pass (responsible cells 4491/4904). A prior-passing cell that failed, hung, or never ran (suite hung before reaching it) is a regression.
```

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression.

**Outcome**: diagnosed as a harness-side timeout, not an op failure. Every one of the 3000 cells the gate charged (`error/infra`, "not reported by its device job") sat in one of four golden chunks (002 / 004 / 006 / 012) that the broker killed for producing no output for 300 s.
- Why the chunks went silent: the eval runner prints nothing while pytest runs. Its precompile warm pass fails (`ImportError: bench_devfilter`, infra), so every distinct program was JIT-compiled inline and serially. At about 1.2 s per test, a cold chunk of ~250–500 real tests runs past 300 s.
- The op had no hang: the triage `TT_THROW: ... build failed` lines are the compiler being interrupted by the kill.
- Re-run of the FULL golden suite on the R2 kernels, unchanged, through the same chunked runner: 4993 passed, 0 hangs, 0 errors.
  - The only 2 failures are the two INVALID `test_translated` bf8b × w_non_aligned cells. They hit the documented host `ValueError`; before R2 they also failed, as xfail.
  - Verifier: supported_pass 4980, supported_fail 0, xfail_wrong_mode 0. Every R1-passing cell passes (0 regressions).
  - Slowest chunk: 254 s.
- No code changed. The JIT cache under `built/` now holds every golden program.
### [x] Refinement 3 — Speed up the width-sharded decode profile (cross-core combine latency)

**Type**: perf

**Goal**: the `feature_spec.LOOSE_CASES` perf cases at `WIDTH_SHARDED` decode, `(1,1,32,W)` for W ∈ {1024, 2304, 5120, 7168} with shard grids 8×1, 9×1, 8×4, 7×4, carry `achievable_ns` = 4110 / 4617 / 5267 / 5481 ns (Blackhole p150b at 1350 MHz; clock-scale per the feature_spec header). Config: bf16, HiFi2, fp32_dest_acc_en=False, TILE bf16 γ, soft PCC 0.9995. There is one tile-row per core and the x data is already in L1, so the kernel time is the W_SPLIT rendezvous: partial gather → root combine + finalize → rinv multicast → normalize. Speed these cells up toward the clock-scaled `achievable_ns`, using the relevant patterns in `ttnn/ttnn/operations/examples/master.md` (`mcast_topology`, `tensix_all_reduce`, `tensix_all_reduce_compute`) and the design's perf lamps:
- L4: gather only the column-0 data of the partials, not whole tiles.
- L5: `McastConfig(handshake=False)`, safe by the causal chain.
- L6: issue the sumsq of the next work before waiting on rinv.
- TREE_COMBINE / COMBINE_SCATTER: the deferred regime rows.

The implementer owns the lever choice. No SUPPORTED change.

**Verifier notes**: measure the exact config above, never a fp32_dest_acc_en=True stand-in. This phase is one ⭐⭐⭐ restructure at most (combine topology) plus cheap knobs (L4/L5). Do not also take on the block-sharded throughput work; that is Refinement 4. L4 also shrinks `cb_gathered` (`k·br` tiles on every core, about 412 KB at `w_split = 103`), which is the over-capacity ledger finding folded in here.

**Done when**: measured device-kernel ns improves on the four WIDTH_SHARDED decode perf cases (record before/after against the clock-scaled `achievable_ns`); their 0.9995 PCC gate still holds; the golden suite is green; and there is no regression across the config-spanning guard set (one representative per kernel path × layout × placement: LOCAL/W_SPLIT × TILE/RM × interleaved/height/width/block).

**Outcome**:
- **Measured.** Device-kernel ns on BH p150b at 1350 MHz, so `achievable_ns` applies unscaled. Decode `(1,1,32,W)`, W = 1024 / 2304 / 5120 / 7168:
  - before: 4323 / 5453 / 5647 / 6508
  - after: 2978 / 3347 / 3750 / 3792
  - speedup 1.45 / 1.63 / 1.51 / 1.72×; now 0.69–0.72× of `achievable_ns` (4110 / 4617 / 5267 / 5481)
  - PCC 0.99998
- **Levers, all kept and all knobs.** Approximate gain on 7×4, in landing order:
  - γ off the gather path, then tail-only zero fill + issue at kernel start: −0.5 µs
  - column-scoped fused rinv finalize (`RinvColumn`): −0.6 µs
  - L4 faces-0/2 gather: −0.5 µs
  - L5 `handshake=False`: −0.1 µs
  - x·γ prescale into `cb_xg` during the rendezvous wait: −0.84 µs
- **No regression.** The guard probe ran one representative per path × layout × placement against the R2 commit: every shape is equal or faster, e.g. HEIGHT 1024×4096 38.7 → 31.4 µs, BLOCK 7168×1024 55.4 → 31.1 µs, interleaved 32×7168 8.5 → 6.8 µs.
- **What binds now** (7×4 zones):
  - sumsq + collapse ≈ 0.95 µs before the partial exists
  - partial landing ≈ 0.7 µs: write-barrier ack round trip + atomic inc. This is latency, not fan-in; 8-core lines show the same 0.7 µs.
  - root combine + finalize ≈ 0.9 µs
  - multicast ≈ 0.6 µs: the loopback send fences on every receiver's ack
  - xg·rinv + store ≈ 0.4 µs
- **Next steps.** Replace the barrier + atomic with flushed writes + a per-sender flag write on the same NoC/VC (the chain-pipe ordering pattern; kernel_lib documents no ordering between departed writes and an atomic). Let the root normalize from its own `cb_rinv_bcast` instead of waiting for the loopback fence. Scope the collapse's within-tile SFPU reduce to column 0.
- **Why not done here.** These are new synchronization protocols with hang risk, the targets are already beaten by ~30%, and a tree / scatter combine would only add serial rendezvous at one tile-row per core.

### [ ] Refinement 4 — Speed up the block-sharded prefill profile (per-core compute throughput)

**Type**: perf

**Goal**: the BLOCK_SHARDED perf loose cases carry `achievable_ns` targets:

| Shape | Shard grid | `achievable_ns` |
|---|---|---|
| (1,1,640,6144) | 9×10, ragged | 19039 |
| (1,1,640,7168) | 7×10 | 21669 |
| (1,1,7168,1024) | 8×10, ragged | 18871 |
| (1,1,7168,2048) | 11×10, ragged | 25947 |
| (1,1,8192,1024) | 8×8 | 25640 |

Config: bf16 / HiFi2 / fp32_dest_acc_en=False. Each core holds a multi-tile-row resident block (e.g. 32 tile-rows × 4 tiles), so these cells are bound by the compute pipeline (sumsq pass + normalize pass, per-phase reconfigs), not by DRAM. Speed them up toward the clock-scaled targets using the compute-side patterns in `ttnn/ttnn/operations/examples/master.md`:
- `compute_block_size`: one helper call per whole shard.
- `compute_fusion`: lamp L7 in the design, i.e. FPU DEST-reuse versus packing `x·γ` to L1 and multiplying from the CB, measured at 0.82–0.94× in the catalog.
- `sfpu_tile_scope`.

No SUPPORTED change.

**Verifier notes**: these are compute-side T2 levers, so several fit in one phase. Gate each one on a measured win. Remove a lever that does not move device ns.

**Done when**: measured device ns improves on the five BLOCK_SHARDED perf cases; their 0.9995 PCC gate holds; the golden suite is green; and there is no regression across the same config-spanning guard set as Refinement 3.

### [ ] Refinement 5 — Speed up the interleaved perf profile (DRAM-bound prefill, latency-bound decode)

**Type**: perf

**Goal**: the 12 interleaved perf loose cases (`(1,1,rows,W)`, rows ∈ {32, 640, 7168, 8192}, bf16 / HiFi2 / fp32_dest_acc_en=False). Verifier measurement at the fp32_dest_acc_en=True stand-in: all 12 already run at or under their `achievable_ns`. Prefill reaches about 380–420 GB/s of x+y traffic, e.g. 8192×7168 takes 556 µs on 110 cores against a 1032 µs reference. Decode 32×1024 uses only 4 cores (5.8 µs; `MIN_CORE_W_TILES = 8` caps `w_split`). Re-measure at the real config once Refinement 1 lands. Then run `/perf-ceiling-dm` on the prefill shapes and file levers only where there is headroom. Candidate levers from the design's lamp table and `ttnn/ttnn/operations/examples/master.md`:
- L9 / `bank_stagger`: every core starts on the same DRAM bank.
- L1 / L3: `block_rows` so that each core has ≥ 2 blocks to overlap read with normalize+store; `x_block_depth`.
- L8: the `L1_CB_BUDGET_BYTES` constant, which decides LOCAL vs W_SPLIT on 8192×7168.
- L10 / `double_buffer`: read chunk size.
- L2: `min_core_w_tiles` / `w_split` for the decode shapes.

No SUPPORTED change.

**Verifier notes**: if `/perf-ceiling-dm` shows the prefill shapes within a few percent of the DRAM ceiling, restrict this phase to the decode shapes (latency-bound, small grid occupancy). If those don't move either, close it with the measurement recorded rather than inventing a lever. All knobs here are knob-turns on parameters `select_blocking()` already owns. Change the constant there, not in the kernels.

**Done when**: measured device ns improves on the most-impacted interleaved perf shapes, or the phase is closed with a recorded ceiling measurement showing no headroom. The 0.9995 PCC gate holds, the golden suite is green, and there is no regression across the config-spanning guard set.
