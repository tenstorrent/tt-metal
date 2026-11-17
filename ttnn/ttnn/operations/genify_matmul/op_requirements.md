# Operation Requirements: genify_matmul

## Definition
- **Formula**: `Y = A @ B` (B shared `(K, N)`, or batched `(..., K, N)` with A's leading dims). Then:
  - none → `Y`
  - bias → `Y + bias[0, n]`
  - bias_relu → `relu(Y + bias[0, n])`
  - swiglu → `out[..., m, 32j+c] = silu(Y[..., m, 64j+c]) · Y[..., m, 64j+32+c]` (width N/2)
  - addcmul → `residual + gate[0, n] · Y`
- **PyTorch Reference**:
  ```python
  def pytorch_genify_matmul(A, B, *, bias=None, activation=None, residual=None, gate=None):
      y = torch.matmul(A.float(), B.float())
      if bias is not None:
          y = y + bias.float()
      if activation == "relu":
          y = torch.relu(y)
      elif activation == "swiglu":
          *lead, n = y.shape
          t = y.reshape(*lead, n // 64, 2, 32)
          y = (torch.nn.functional.silu(t[..., 0, :]) * t[..., 1, :]).reshape(*lead, n // 2)
      if residual is not None:
          y = residual.float() + gate.float() * y
      return y
  ```
- **Import Path**: `from ttnn.operations.genify_matmul import genify_matmul`
- **Function Signature**:
  ```python
  genify_matmul(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, *, bias: ttnn.Tensor = None,
                activation: str = None, residual: ttnn.Tensor = None, gate: ttnn.Tensor = None,
                dtype: ttnn.DataType = None, memory_config: ttnn.MemoryConfig = None,
                compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Perf guard set**: `GENIFY_MATMUL_PERF=1 eval/eval_test_runner.sh tests/ttnn/unit_tests/operations/genify_matmul/perf <out>` records device ns for 14 interleaved shapes × {HiFi4, HiFi2}. Every refinement extends it with one representative per new kernel path / placement / dtype it adds, so later perf phases can measure regressions.

### [x] Phase 0 — Core Implementation (regime R1 `interleaved_out_split`)

- **SUPPORTED dtype**: [bfloat16]; **weight_dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, m_non_aligned}; weight_batch = single
- **SUPPORTED op-specific axes**: placement = interleaved (A, B and the output in DRAM or L1); epilogue = none; fp32_dest_acc_en = True
- **Cores**: full grid. The independent M×N output split comes from `choose_split()`, which is blocking-aware and traffic-ranked within `GRID_W_SLACK` of the best W. Each core walks its region in L1-solved blocks, with the K-walk accumulated in L1 (packer L1 acc).
- **Compute config**: honours the caller's `math_fidelity` / `math_approx_mode`. `fp32_dest_acc_en=True` only. The default comes from `default_compute_kernel_config()` (HiFi4, fp32 DEST).
- **Golden baseline**: 94 supported_pass, 2088 xfail_expected, 0 xpass_drift, 0 xfail_wrong_mode. There is 1 supported_fail, a harness L1-capture false-negative on an expected-`ValueError` test (see verification_report.md).

### [x] Refinement 1 — Numerical configurability (activation, weight and DEST formats)

**Goal**: Add these values to SUPPORTED, for every placement and epilogue the op supports at the time:
- `ttnn.bfloat8_b` and `ttnn.float32` to `SUPPORTED["dtype"]`;
- `ttnn.bfloat8_b` and `ttnn.bfloat4_b` to `SUPPORTED["weight_dtype"]`;
- `False` to `SUPPORTED["fp32_dest_acc_en"]`.

Then:
- Derive every CB page format from its tensor: A, B, the output, and the partials (Float16_b when `fp32_dest_acc_en=False`).
- Keep the subblock limit keyed on DEST width (`dest_limit` 8 with 16-bit DEST).
- Keep `{"dtype": float32, "fp32_dest_acc_en": False}` as the permanent EXCLUSION.

Cells that fail out of the box go to `EXCLUSIONS` with a reason, not into their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **Anchor for Refinement 3.** The PERF FOCUS 1 contract is bf16 A × **bfloat4_b** B, LoFi, `fp32_dest_acc_en=True`, interleaved DRAM, swiglu. This refinement lands its weight dtype; Refinement 2 lands the epilogue. LoFi / HiFi2 are already honoured (not gated).
- **Mixed formats are the norm here.** The model cases pair:
  - bf16 × bf4b / bf8b;
  - fp32 A → bf16 out (FLUX time_mod);
  - bf16 A → bf8b out (PDL);
  - bf8b × bf8b with 16-bit DEST (PDL W pass).

  The output `dtype` override already works: bf16 → fp32/bf8b is tested in `test_genify_matmul_extended.py`. The L1 solve already takes tile bytes from the tensors, so bf4b weights (576 B tiles) automatically get coarser blocks.
- **fp32 output precision.** The helper reloads the Float32 partials through SrcA on the last K-block, which truncates them to tf32. The precision baseline shows it as a −2.6e-4 median got/true bias on K-walk shapes. fp32-out still measured rel-RMS 0.00047 against the 0.005 gate. If the `(float32, True)` cells miss, `UnpackToDestFp32` tagging on `cb_partials` is the lever, not a tolerance change.
- **Expected unblock.** 24 loose cases are blocked only by `weight_dtype`, plus the cartesian dtype / acc cells.

**Done when**: every golden cell (cartesian and loose) whose only unsupported axes are among dtype / weight_dtype / fp32_dest_acc_en passes or is an EXCLUSION with a stated reason. The phase-0 suites stay green, and the guard set shows no device-ns regression on bf16.

**Outcome**: SUPPORTED widened to dtype {bf16, bf8b, fp32}, weight_dtype {bf16, bf8b, bf4b} and fp32_dest_acc_en {True, False}. The only EXCLUSION is the permanent fp32 × acc=False. Golden: 60/60 cartesian and 27/27 newly unlocked loose cells pass. No kernel change was needed. New precision knob `DEST16_BLOCK_K_MAX = 8`: with 16-bit DEST, a single long DEST chain gave rel-RMS 0.18 at K = 16k; capped, it is 0.019, and it is also 0.76–0.96× device time. The bf16 / fp32-DEST blocking is unchanged.

### [ ] Refinement 2 — Fused epilogues: bias, bias_relu, swiglu, addcmul

**Goal**: Add `bias`, `bias_relu`, `swiglu` and `addcmul` to `SUPPORTED["epilogue"]`. Each runs inside the same device program, on the accumulated block, before the output is packed. Follow op_design.md "Block Operation Realization" rows 2', 3a, 3b and 3c:

- **Compute.**
  - `matmul_block<…, LastBlockTarget::Interm, …>` into `cb_partials`. It is now allocated whenever epilogue ≠ none, still sized to exactly one block.
  - Then one epilogue block op:
    - `add_bias_bcast_rows` (+ PACK_RELU for bias_relu);
    - `eltwise_chain` swiglu: Strided gate/up tiles, Sigmoid·Mul·Mul;
    - `eltwise_chain` addcmul: FPU mul with the row-broadcast gate, then SFPU add of the residual.
- **Reader.**
  - `cb_bias_row` / `cb_gate_row`: one N-block row, read **once per block**.
  - `cb_residual_tiles`: row groups, depth 2.
  - For swiglu: the gate/up column permutation within each N-block.
- **Host.**
  - `q_n ≥ 2` for swiglu, so whole gate/up pairs land per core and per block.
  - `out_block_n = block_n / 2`.
  - Bias, gate and residual join the `generic_op` tensor list.
  - The epilogue CB terms (`E_bias`, `E_addcmul` in the L1 ledger) go **into `_footprint`** before the solve.

**Verifier notes**:
- No skill in the inventory covers fused epilogues; the design doc's realization table and the `compute_fusion` catalog entry are the references. Use no DEST reuse ahead of an FPU op.
- **Anchor for Refinement 3.** The flagged prefill gate_up runs swiglu on a **bf4b** weight, so this refinement must pass that exact loose case: (1,2048,5120) @ (5120,8704), LoFi, fp32 DEST. Order it after Refinement 1.
- **Bias dtype.** The bias is bf8b when the weight is bf4b (golden helper rule), so the bias CB format is its own, not the weight's.
- **Build it performantly.** Refinement 3 optimizes this exact path:
  - no second pass over the output;
  - no per-row-group re-read of the bias row;
  - the residual stream double-buffered;
  - the R1 split still filling the grid.
- **Batch interaction.** Epilogues must also hold under `block_batches > 1` once Refinement 7 lands batched weights; keep the bias / gate lifecycle per block, not per kernel.
- **Never a host-side or ttnn-op epilogue.** That is a prompt MUST.

**Done when**: every golden cell whose only unsupported axis value is one of the four epilogues passes (cartesian INPUTS plus the interleaved loose cases, among them PERF FOCUS 1 and 4 and the FLUX addcmul cases), within `helpers` tolerances, with phase-0 and Refinement 1 cells still green.

### [ ] Refinement 3 — Speed up PERF FOCUS 1 (Qwen3.8 prefill gate_up + SwiGLU, 2048 tokens)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags `_qwen_prefill(2048, 5120, 8704, epilogue="swiglu")` as the mandatory perf target. Its contract:
- A (1, 2048, 5120) bf16 × B (5120, 8704) **bfloat4_b**;
- LoFi, `fp32_dest_acc_en=True`;
- interleaved DRAM; swiglu; bf16 out.

The goal is to match or beat `achievable_ns` = 564.5 µs; the roofline is 428.7 µs. The soft precision gate is `LOW_FIDELITY_TOLERANCES[(bf16, True)]`.

Under R1 this shape re-reads A ~10× and B ~11× from DRAM. The bf16×bf16 / epilogue-none twin measures 4.3 ms, ~8× off, entirely traffic-bound. Primary lever: the deferred regime **R1a `interleaved_2d_mcast`** from `ttnn/ttnn/operations/examples/master.md` (`mcast_topology`, `shared_input_reuse`):
- a row injector multicasts each A K-block along its grid row;
- a column injector multicasts each B K-block down its column, via the multicast helper (`ttnn.Mcast` + `McastArgs`);
- the compute kernel and CB inventory stay unchanged (op_design.md "Inter-Tensix contracts").

After the topology lands, co-tune the R1 knobs it exposes:
- `block_k_tiles` cap vs `depth_act` / `depth_weight` (`double_buffer`, `trid_double_issue`);
- `GRID_W_SLACK` / the factorization priority;
- `q_n` / subblock (`matmul_output_subblock`);
- `InitMode::None` after the first block;
- the split reader (`split_reader`).

No SUPPORTED change.

**Verifier notes**:
- This is a ⭐⭐⭐ T3 restructure: one topology per phase. Do not bundle the dram_sharded or batched work.
- **Folded L1-ledger finding.** When a core's whole walk is one K-block (S1, `nb = 1`), the second slot of `cb_act_tiles` / `cb_weight_tiles` is dead L1. It also means compute only starts after the whole K is read (zero read/compute overlap). Resolve it here as part of the `block_k` / depth co-tune, and update `l1_ledger.md`.

**Done when**: measured device ns on the flagged shape improves toward 564.5 µs, its soft precision gate holds, and the golden suite is green. There must be no regression across the config-spanning guard set: the perf guard dir plus one representative per kernel path × dtype × epilogue × placement added so far, including a 1-D (M ≤ 32) shape and an L1-resident A/output shape.

### [ ] Refinement 4 — dram_sharded placement (regime R4: weight streaming from the DRAM banks)

**Goal**: Add `"dram_sharded"` to `SUPPORTED["placement"]`:
- A is L1 width-sharded over **K**;
- B is DRAM width-sharded, one column shard per bank (`device.dram_grid_size().x`; the last shard may be partly padding; swiglu shards are 64-column granules);
- `memory_config` is `ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG` with **no shard spec**. The op builds the output's L1 width-shard spec itself and returns exactly that placement.

Required topology (op_design.md R4):
- **Workers.** One per bank, from `get_optimal_dram_bank_to_logical_worker_assignment` (no hardcoded coordinates). Each reads its bank's B shard with coalesced reads (`read_coalescing`).
- **A broadcast.** A's K-slices live on A's own cores, so each A shard crosses the NoC once, broadcast from its owning core (rotating / explicit senders) to the irregular set of bank workers. Use `McastArgs` / `ttnn.Mcast`, `irregular_receiver_set_mode`.
- **K-blocks.** Divide the sender's shard width. B's next K-block is issued before waiting on the A round (stall-shadow).
- **Output.** Written to its L1 shard: local when the worker owns it, otherwise a NoC write.
- The existing EXCLUSION `{"placement": "dram_sharded", "weight_batch": "batched"}` stays.

**Verifier notes**:
- **Scheme-change, stands alone.** A is cut along K (dependent), so the A broadcast topology *is* the work. No skill covers it, so the topology above is the spec.
- **Read A where it lives.** No reshard or interleaved copy of A; that is a prompt MUST.
- **Ordering.** Hardest-first, and a true dependency of Refinement 6 (PERF FOCUS 2 runs dram_sharded).
- **Mixed formats.** Qwen decode is bf16 × bf4b, LoFi, so Refinement 1's formats must carry onto this path.
- **swiglu.** The decode gate_up cases need it, from Refinement 2.

**Done when**: every dram_sharded golden cell (cartesian INPUTS and the Qwen decode / LM-head loose cases) passes, except the permanent EXCLUSION. The requested placement and the op-chosen shard spec are honoured, and all earlier cells stay green.

### [ ] Refinement 5 — block_sharded placement (regime R3: cross-core A K-slice exchange)

**Goal**: Add `"block_sharded"` to `SUPPORTED["placement"]`:
- A is L1 block-sharded: rows over grid rows, **K over grid columns**;
- B is DRAM interleaved;
- the output is L1 block-sharded on A's grid with exactly the requested shard spec.

Required topology (op_design.md R3/R3a):
- **Per core.** The block is `shard_m_tiles × out_shard_w`, walking K one peer shard per K-block (`num_k_blocks = grid cols`).
- **Own K-slice.** Zero-copy from `cb_act_shard` (`ttnn.cb_descriptor_from_sharded_tensor`), selected by `In0SourceFn`.
- **Peer K-slices.** Arrive in `cb_act_tiles`. Use unicast reads as the stepping stone, then a rotating-sender row multicast (each core injects its own shard on its K turn). B column slices are read from DRAM.
- **Output.** In place via `cb_output_shard` when `block_n` equals the shard width; otherwise a streaming CB plus a local write.
- **L1 budget.** The CB budget must account for A's and the output's resident shards; the existing `l1_cb_budget` fragmentation-aware view does.

**Verifier notes**:
- **Scheme-change, stands alone.** The reduction dim is split across cores, so A moves between peers. No skill covers sharding with a cross-core dependency, so the topology above is the spec.
- **The own shard must not be re-read through a `TensorAccessor`.** That would only tolerate the layout, not implement it, and does not count as done.
- **Model inputs.** PDL's block-sharded convs are bf8b with bias / bias_relu (Refinements 1 and 2). The ASPP-pool cell (M = 1 on an 8×1 grid) under-fills the grid; correctness only here.

**Done when**: every block_sharded golden cell (cartesian INPUTS and the PDL res4 / res5 / ASPP loose cases) passes with the requested shard spec, and earlier cells stay green.

### [ ] Refinement 6 — Speed up PERF FOCUS 2 (Qwen3.8 decode MLP down)

**Type**: perf

**Goal**: The second flagged case, `_qwen_decode(1, 4352, 5120, cores=8)`:
- A (1,1,1,4352) bf16 width-sharded on 8 cores × B (4352, 5120) **bfloat4_b**, DRAM width-sharded over the banks;
- LoFi, fp32 DEST, dram_sharded.

The goal is to match or beat `achievable_ns` = 36.9 µs; the roofline is 27.2 µs (12.5 MB of bfp4 at 460.8 GB/s). The soft precision gate is `LOW_FIDELITY_TOLERANCES[(bf16, True)]`.

This is a weight-streaming, DRAM-bound regime. Relevant levers in `ttnn/ttnn/operations/examples/master.md`:
- B stream depth 3–4 with transaction-id barriers (`trid_double_issue`);
- coalesced per-bank reads (`read_coalescing`);
- start-bank stagger (`bank_stagger`);
- B reads on a second DM RISC (`split_reader`);
- amortizing the A broadcast round over a larger `block_k`.

No SUPPORTED change.

**Done when**: measured device ns on the flagged shape improves toward 36.9 µs, its soft precision gate holds, and the golden suite is green. There must be no regression across the guard set (one representative per kernel path × placement × dtype, including the interleaved PERF FOCUS 1 shape).

### [ ] Refinement 7 — height_sharded placement + batched weights (local shards, independent batch split)

**Goal**: two independent axes, bundled into one refinement.

1. **height_sharded.** Add `"height_sharded"` to `SUPPORTED["placement"]` (regime R2):
   - A is L1 height-sharded; each core's shard holds all of K for its rows, so the work stays local.
   - A is consumed **in place**: `cb_act_shard` via `ttnn.cb_descriptor_from_sharded_tensor`, one `cb_push_back`, no NoC read of the own shard.
   - The block is `shard_m_tiles × out_n` with `block_k = tensor_k_tiles`.
   - B comes from DRAM.
   - The output goes in place into the requested L1 height-sharded spec on A's cores (`cb_output_shard` when the block spans the full shard width; otherwise a streaming CB plus a local write).
2. **Batched weights.** Add `"batched"` to `SUPPORTED["weight_batch"]` (regime R1d, a knob-turn):
   - `grid_b > 1` in the factorization;
   - `weight_batch_stride` in the reader, which already carries it;
   - `MatmulBlockShape.batch = block_batches`;
   - validate the batched B layout on interleaved placements.

**Implementation skill**: /interleaved-parallel

**Verifier notes**:
- **Knob-turns, so bundled.** Both are knob-turns of R1 with no new cross-core topology. `/interleaved-parallel` applies to the batched part (the `grid_b` independent-axis split on interleaved tensors). No skill covers sharded placement (do not attach `/memory-layouts`); the CB-placement pattern above is the spec.
- **Honest SUPPORTED.** An own-shard read through a `TensorAccessor` passes the golden cells but does not implement height_sharded. Send it back, do not tick it.
- **Model inputs.**
  - PDL's height-sharded convs need bf8b + bias / bias_relu (Refinements 1 and 2).
  - The batched-weight PERF FOCUS 3 case needs bf8b, `fp32_dest_acc_en=False` and B in L1. Refinement 1 lands the formats; L1-interleaved B already works.
- **Follow-up candidate (not a refinement).** The B multicast across A's cores (R2a) is the perf successor for PDL height-sharded shapes. No PDL height-sharded case is perf-flagged, so it stays a report note.

**Done when**: every height_sharded and every batched-weight golden cell passes (cartesian INPUTS and the PDL loose cases, including PERF FOCUS 3), except the permanent `dram_sharded × batched` EXCLUSION. The requested output shard spec is honoured, and all earlier cells stay green.

### [ ] Refinement 8 — Speed up PERF FOCUS 3 (PDL bilinear upsample, H pass: batched weight)

**Type**: perf

**Goal**: The third flagged case:
- A (1,32,512,128) bf8b × B (1,32,128,1024) bf8b, B in **L1** interleaved;
- LoFi, **`fp32_dest_acc_en=False`**; interleaved; epilogue none.

The goal is to match or beat `achievable_ns` = 123.0 µs; the roofline is 43.5 µs. The soft precision gate is `LOW_FIDELITY_TOLERANCES[(bf8b, False)]`.

32 independent batches of a small-K matmul, compute- and write-heavy. Relevant levers in `ttnn/ttnn/operations/examples/master.md`:
- 16-bit DEST allows 8-tile subblocks (`matmul_output_subblock`, 1.46×);
- the `block_batches` / grid_b co-tune, so one `matmul_block` call spans several batches (`compute_block_size`);
- output write batching and NoC placement (`noc_placement`).

No SUPPORTED change.

**Done when**: measured device ns on the flagged shape improves toward 123.0 µs, its soft precision gate holds, and the golden suite is green. There must be no regression across the guard set (one representative per kernel path × placement × dtype, including the PERF FOCUS 1 and 2 shapes).

### [ ] Refinement 9 — Speed up PERF FOCUS 4 (FLUX.2 prompt-stream FF1 + SwiGLU, 256 tokens)

**Type**: perf

**Goal**: The fourth flagged case, `_flux(256, 6144, 18432, epilogue="swiglu")`:
- A (1,256,6144) bf16 × B (6144,18432) bf16;
- HiFi2, fp32 DEST, interleaved DRAM, swiglu.

The goal is to match or beat `achievable_ns` = 865.9 µs; the roofline is 508.6 µs, bound by the 226 MB bf16 weight. With M = 8 tiles the traffic-optimal split is `grid_m = 1`, so B crosses DRAM exactly once. A (8 tiles × K) is then reuse-shared by every core, so the lever is **R1b `interleaved_stream_mcast_a`**: one injector broadcasts each A K-block (`shared_input_reuse`, 1.7×). Co-tune B stream depth / trids (`trid_double_issue`) and the bank stagger (`bank_stagger`).

Its epilogue-none twin (256×6144×9216) is 989 µs vs 453 µs stock today. The guidance / timestep linear_2 (M = 1, 1.6× stock) share the regime.

No SUPPORTED change.

**Verifier notes**: Reuse the multicast infrastructure Refinement 3 builds. This is a different topology (one-to-all A broadcast instead of row/column), so it gets its own phase.

**Done when**: measured device ns on the flagged shape improves toward 865.9 µs, its soft precision gate holds, and the golden suite is green. There must be no regression across the guard set (one representative per kernel path × placement × dtype, including the PERF FOCUS 1–3 shapes).
