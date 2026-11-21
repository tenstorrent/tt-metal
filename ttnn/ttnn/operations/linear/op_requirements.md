# Operation Requirements: linear

## Definition
- **Formula**: `out[..., i, j] = sum_k a[..., i, k] * b[k, j] (+ bias[j])`, over the last two dimensions
- **PyTorch Reference**: `torch.matmul(a, b) + bias` (the weight is `[K, N]`: `torch.nn.functional.linear(a, b.T, bias)`); the graded reference is `reference()` in `eval/specs/linear/spec.py` (float64 on the stored inputs)
- **Import Path**: `from ttnn.operations.linear import linear`
- **Function Signature**:
  ```python
  def linear(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, bias: ttnn.Tensor = None,
             memory_config: ttnn.MemoryConfig = None, dtype: ttnn.DataType = None, program_config=None,
             compute_kernel_config: ttnn.ComputeConfigDescriptor = None, core_grid=None) -> ttnn.Tensor
  ```
- **Specification**: `eval/specs/linear/spec.py` (governs over this file). Target: Quasar on ttsim. Perf gates here are **structural counts** (the backend has no clock; any gain in time stays unconfirmed until a timed backend or hardware measures it).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment=[tile_aligned], rank=[4], weight_form=[matrix, same_rank], batch=[single]
- **SUPPORTED placement axes**: memory_config=[DRAM, L1], sharding=[interleaved, width], output_memory=[DRAM, L1/width, L1/width/auto]
- **SUPPORTED op-specific axes**: bias ∈ {None, row, vector}, dtype_argument ∈ {same}, fp32_dest_acc_en ∈ {True, False}
- **Regime**: R-2D-stream (op_design.md): `[block_h × block_w]` output blocks, K streamed whole per NEO into DEST, bias preloaded into DEST, rounds of C blocks dealt to clusters
- **Cores / thread plan**: whole rounds over the 8×4 grid (`split_work_to_cores`, row-wise); `THREAD_PLAN` (act reader 1, weight reader 1, C = 4, W = 2)
- **Compute config**: `compute_kernel_config` exposed (descriptor or device config, R4); default HiFi4 + `fp32_dest_acc_en=True` from `default_compute_kernel_config()`
- **Golden baseline**: 331 supported_pass (324 cells + 1 loose + 6 regression), 2916 xfail_expected, 1080 invalid_skipped, 7 rejections, 0 loud (per `verifier_report.json`)

### [ ] Refinement 1 — Shape generality: ranks 2 and 3, a batch of activations, ragged M / N / K

**Goal**:
- add `2` and `3` to `SUPPORTED["rank"]`;
- add `broadcast` to `SUPPORTED["batch"]`;
- add `mn_non_aligned` and `k_non_aligned` to `SUPPORTED["alignment"]`.

Two pieces of real kernel work come with it:
1. **K padding must not reach the sum (`k_non_aligned`).** Overwrite the padding of the last K tile in-kernel: the activation's pad columns and the weight's pad rows. Do it in the reader (explicit sync on that ring) or in compute. Overwrite, never multiply by a mask: `0 × NaN` is NaN, and a tensor produced by another op may carry any padding.
2. **Build the deferred regime R-rows-to-neos (op_design.md, Regimes).** The new tall-M, narrow-N shapes (`2048x64 @ 64x64`, `1x512x128 @ 128x64`) have `col_blocks < C`. Today they pad half the NEOs of every round. In R-rows-to-neos the NEOs of a round split rows: the activation ring goes BLOCKED, the weight ring ALL, chosen by a host predicate.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- The verifier probed every out-of-phase bf16 INPUTS entry with only SUPPORTED widened (probe_002). All 48 cells passed: two placements per shape, both DEST widths, both bias forms, ranks 2 and 3, batched, ragged M / N / K. `plan_blocking` already folds images into rows from the padded shape.
- `k_non_aligned` passes there only because the golden runner's tile padding is zero. With the padding filled non-zero (`ttnn.fill_implicit_tile_padding`, 7.0 on `a` and 3.0 on `b`), the `1x1x32x50 @ 1x1x50x64` product is off by 295 against a reference max of 28. So item 1 is a correctness requirement for real inputs, not polish (spec GUIDANCE, *Padding*).
- R-rows-to-neos is a knob-turn on the built structure: same kernels, the two rings' access patterns swapped. It is here because this refinement opens the shapes it serves. Ship the new path filling the grid, not correct-only.
- `32x4096 @ 4096x32` (wide K, `M_t = N_t = 1`) runs on one NEO of one cluster. Its only lever is split-K (Refinement 5); do not hold this refinement on it.
- Comes before Refinement 2 so float32 extends the final shape path once.

**Done when**:
- every bf16 cell of the new rank / batch / alignment values passes in the golden suite (0 loud);
- a test with non-zero, and with NaN / inf, tile padding in both operands passes for a `k_non_aligned` shape;
- `2048x64 @ 64x64` runs with no padded NEO slot (a `select_blocking` / regime pin, as `test_linear_plan.py` pins Phase 0);
- the Phase 0 cells and `test_linear_plan.py` stay green.

### [ ] Refinement 2 — float32 operands

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]` for every shape and placement value in SUPPORTED at that point, both bias forms included. float32 with `fp32_dest_acc_en=False` stays refused (rule R1; the golden suite marks those cells strict-xfail whatever SUPPORTED says). The compute config is already exposed (Phase 0); this refinement makes the rings and the bias path dtype-aware.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- The verifier probed float32 with SUPPORTED widened. Every call fails at program build: `hardware_config.cpp:193 ... consumes FP32 DFB 'cb_act_tiles' with enable_32_bit_dest=true, but provides no unpack_modes entry for this DFB`. Every FP32 ring the compute kernel consumes needs an explicit `UnpackMode`: UnpackToSrc for the two matmul operands, and for the bias whatever the new bias path needs.
- The bias preload `unary_bcast<ROW>` asserts on 32-bit tiles (`bcast.h:71,109`; op_design.md Key Risks). The float32 bias needs another route into DEST, for example the raw SFPU row-broadcast add. Keep the fold-into-DEST structure: no product ring.
- The output ring's page becomes Float32 (output ABI, R9). `tile_bytes` doubles, so `select_blocking` halves `block_k` where the ring cap binds (ResNet-sized K). It is derived from `a.buffer_page_size()`, so no knob changes by hand.
- Quasar has no `bfloat8_b` / `bfloat4_b` (target page, family facts): do not add them and queue nothing for them.
- The numeric list grades float32 at `c = 45` with a tf32 accumulator (`NUMERIC_RESOLVED`), so the matmul's 19-bit SrcA/SrcB is within the requirement. Record PCC and relative RMS in the precision baseline (extend `test_linear_precision_baseline.py` with float32).

**Done when**:
- every float32 + `fp32_dest_acc_en=True` cell passes in the golden suite;
- float32 + `fp32_dest_acc_en=False` is refused (strict-xfail);
- 0 loud categories, and the precision baseline has float32 rows.

### [ ] Refinement 3 — Read each shared operand once: weight and activation multicast

**Type**: perf

**Goal**: build the deferred regimes R-mcast-weight and R-mcast-act (op_design.md, Regimes). One reader per row block (weight) and per column round (activation) reads each chunk once and multicasts it to the clusters that share it. Non-injector clusters' readers become receivers into the same ring layout; compute, the rings and the split stay unchanged. Free-selected region (no loose case of the specification is perf-flagged): the models' supported bf16 shapes that re-read a shared operand.
- Qwen3-VL `1x1x128x256 @ 256x512`: 16 clusters, `num_row_blocks = 4`. The weight is read 4× from DRAM: 512 tile reads for 128 unique tiles. The activation is read 4×: 128 tile reads for 32 unique.
- ResNet-50 fc `1x1x32x2048 @ 1x1x2048x1024`: 8 clusters, `rounds_per_row_block = 8`. The activation is read 8× from its L1 shards: 512 tile reads for 64 unique.

Patterns: `mcast_topology` (two 1-D channels on a 2-D split; the naming inversion is the trap) and `shared_input_reuse` in `ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.

**Verifier notes**:
- This is the cheapest-traffic split of `l1_ledger.md` and the design's Traffic ranking. The ledger defers it as "unexercised on this part", which is not a positive reason (L1 ledger audit, verification_report.md). That finding is folded here.
- Multicast between clusters is listed as *not covered* by `.claude/references/quasar_op_examples/README.md`. Prove one minimal cluster-to-cluster multicast into a ring on ttsim before restructuring the readers. If it cannot be made to work at this base, ship `[~]` with the evidence and file the blocker as Refinement 3b.
- Comes before Refinement 4 because 4's reader knobs land on the readers this refinement rewrites.

**Done when** (structural, read off the op by a test or probe; the time gain is unconfirmed on ttsim):
- Qwen weight DRAM tile reads per call go 512 → 128 and activation reads 128 → 32;
- ResNet activation tile reads go 512 → 64, or at most one per K chunk per injector;
- the golden suite is green, results are bit-identical to Phase 0 (same K order);
- no regression across the guard set: one representative per kernel path × placement (activation DRAM / L1 / width-sharded, result DRAM / width-sharded) × bias (none / row / vector) × DEST width, plus the one-device-session test.

### [ ] Refinement 4 — Data-movement knobs: a second weight-reader hart and K-chunk overlap

**Type**: perf

**Goal**: co-tune the cheap knobs the planner exposed (op_design.md, Perf lamps *Weight reader threads* and *Overlap*):
1. a second hart on the dominant byte stream, `cb_weight_tiles`. The BLOCKED consumer needs `P | block_k * block_w` (`DataflowBuffer.md:307`). R + W = 5 ≤ 6, and the plan stays one host constant.
2. at least two K chunks per round where the coarsest `block_k` swallows all of K. Qwen: `block_k = K_t = 8`, one chunk, so no reader/compute overlap inside a round. Or raise `WEIGHT_DEPTH` to 3, whichever the structural count favours.

Pattern pointers: `split_reader`, `double_buffer` in `ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.

**Verifier notes**: `select_blocking` stays the single source of `block_k`. Change its rule, never patch a value downstream. Run `/perf-ceiling-dm` with `--aiclk-mhz` given explicitly (ttsim has no clock) first: keep the hart change only where the ceiling shows headroom.

**Done when** (structural; the time gain is unconfirmed on ttsim):
- weight-stream reader harts go 1 → 2 on ResNet and Qwen;
- Qwen K chunks per round go 1 → ≥ 2;
- the golden suite is green and results are bit-identical;
- no regression across the same guard set as Refinement 3.

### [ ] Refinement 5 — Split K across clusters for the shapes with few output tiles

**Type**: perf

**Goal**: build the deferred regime R-splitK-cores (op_design.md). Each split cluster accumulates a K sub-range into DEST and packs a Float32 partial into its own `cb_partial` ring. A gather cluster adds the `splits − 1` partials into DEST before the bias and the final pack (semaphore-gated unicast or a multicast gather). Region: the supported shapes whose independent axes under-fill the grid.
- ResNet fc (`M_t = 1, N_t = 32`) engages 8 of 32 clusters.
- Llama `1x1x32x64 @ 1x1x64x128` engages 1.
- After Refinement 1, `32x4096 @ 4096x32` runs on one NEO.

The K range is already a runtime argument (`k_tile_start`, `k_tiles`). Patterns: `tensix_all_reduce`, `eltwise_l1_vs_dest_accumulate` in `ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.

**Verifier notes**: a scheme-change: stands alone. Splitting K changes the floating-point order, so check the result against the `dot` bound (c = 41 / 45) and PCC 0.99 at both DEST widths, not bit-identity. Gate it on headroom first (`/perf-ceiling-dm`): ResNet's 4 MiB weight already crosses DRAM once over 2 channels, so more clusters pay only if the per-cluster read issue, not DRAM, is the limit. If ResNet shows no headroom, keep the regime for the wide-K / tiny-output shapes only, where one NEO is the bound.

**Done when** (structural; the time gain is unconfirmed on ttsim):
- clusters engaged on ResNet go 8 → ≥ 16, unless the ceiling shows no headroom, which is recorded;
- NEOs at work on `32x4096 @ 4096x32` go 1 → ≥ 8;
- the golden suite is green within the numeric bounds;
- no regression across the guard set of Refinement 3.
