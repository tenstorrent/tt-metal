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

### [x] Refinement 2 — Fused epilogues: bias, bias_relu, swiglu, addcmul

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

**Outcome**: SUPPORTED epilogue = {none, bias, bias_relu, swiglu, addcmul}, all running in the one program on the L1 block (`matmul_block<Interm>` → `add_bias_bcast_rows` / `eltwise_chain`).
- Golden: 240/240 cartesian interleaved epilogue cells and 13/13 interleaved loose epilogue cases pass, among them PERF FOCUS 1 (rms 0.0085 against the 0.03 limit) and 4. The unit dir is green.
- Fused cost on PERF FOCUS 1: 1655 µs against 1572 µs for the epilogue-none twin. An ablation shows the plumbing is free and the remaining ~5% is the serialized SFPU silu, already cut from 124 to 83 µs by the bf16-precision sigmoid knob.
- Next lever (Refinement 3): fold silu·up into the last K-block's pack (subblock-local gate/up permutation), or overlap the epilogue with the next block's K-walk.

### [x] Refinement 3 — Speed up PERF FOCUS 1 (Qwen3.8 prefill gate_up + SwiGLU, 2048 tokens)

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

**Outcome**: PERF FOCUS 1 measured **1673 → 463.5 µs** device time (one fresh run each; target `achievable_ns` 564.5 µs beaten, 8% above the 428.7 µs roofline). Precision is unchanged: PCC 0.99992, rel-RMS 0.0085 against the 0.03 gate. Levers: R1a row/column operand multicast (1673 → 494 µs, 110 cores, DRAM 504 → 64 MB); staggered line injectors so no core injects both operands (494 → 464 µs); single-walk K split (resolves the folded L1 finding, 0.79× on 384×1536×1024); an R1a gate `MCAST_MIN_KBLOCK_BYTES = 32 KB` (64×16384×256 16-bit DEST was +11% under R1a with 16 KB K-blocks, now R1 again). Guard set: every 2-D shape is 0.22–0.56×, nothing slower beyond ±1% noise. Measured null, kept as a knob: the 2×2 subblock, 477.8 µs vs 463.7 µs for 1×4 (`SUBBLOCK_MIN_UNPACK = False`). **Bottleneck now**, by ablation: the matmul + partials floor is 413 µs (reader and epilogue stubbed); the serialized swiglu epilogue costs ~35 µs and exposed data movement ~16 µs. **Next**: (1) fold silu·up into the last K-block's pack, or overlap the epilogue; this needs a matmul_block helper change to pack half a subblock. (2) The design's injector stall-shadow (trid-tagged read of K-block kb+1 issued before multicasting kb), which would also let R1a win on the tiny-K-block shapes now gated to R1. Neither was done here: the target is met, and each is a structural change of its own.

### [x] Refinement 4 — dram_sharded placement (regime R4: weight streaming from the DRAM banks)

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

**Outcome**: `dram_sharded` is in SUPPORTED as regime R4. Topology:
- one worker per bank (optimal NOC_0 assignment), each streaming its bank shard with one coalesced read per K-block, issued before the K-block's A round;
- every A shard multicast once from its own core in its K turn (rotating senders via `ttnn.Mcast` + `McastArgs`). The receiver group is the workers' bounding rectangle with the handshake restricted to the workers, because the helper's rotating mode needs a rectangle and the 8 scattered bank workers exceed its 3-rectangle irregular multicast;
- the output packed in place into an op-built L1 width shard (shard w on bank w's worker).

Golden: 310/310 single-weight cartesian dram_sharded cells plus the 10 Qwen decode / LM-head loose cases pass; 60 + 90 xfail on the two permanent EXCLUSIONs. The 57 translated DRAM-sharded tests pass, among them program-cache reuse. Earlier suites are green.

Baseline device time, for Refinement 6:
- PERF FOCUS 2: 42.7 µs (target 36.9, roofline 27.2), B-stream-bound at ~64% of DRAM peak. Each of the 8 K-blocks waits its own read barrier before the next bank read is issued.
- LM head 1×5120×16384: 147.8 µs (127.0 stock).
- gate_up 32×5120×9216 swiglu: 102.6 µs (64.0 stock).

Next levers are Refinement 6's: trid-pipelined B stream depth 3–4, and a larger A round per handshake.

### [x] Refinement 5 — block_sharded placement (regime R3: cross-core A K-slice exchange)

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

**Outcome**: `block_sharded` is in SUPPORTED as regime R3. The own shard is zero-copy via `cb_act_shard` + `In0SourceFn`; peers come over a rotating-sender row multicast (`BS_ROW_MCAST`, with unicast as the other setting); B is read from DRAM per core; the output is packed in place in the requested shard. Golden `-k block_sharded`: 311 passed (300 cartesian + 11 PDL), 150 xfail (fp32 × 16-bit-DEST EXCLUSION, batched weight), 0 failed. PDL device time with the multicast: res5 shortcut 79.8 µs (stock 72.4), res4 reduce 18.7 (16.6), res5 expand 45.6 (24.2), ASPP 1×1 19.6 (13.8). Remaining headroom: wide-N shapes are B-bound (B crosses DRAM once per grid row). Next lever: R3a's B column multicast.

### [x] Refinement 6 — Speed up PERF FOCUS 2 (Qwen3.8 decode MLP down)

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

**Outcome**: PERF FOCUS 2 measured **42.8 → 29.2 µs** device time (one fresh run each; target `achievable_ns` 36.9 µs beaten, 7% above the 27.2 µs roofline); precision unchanged (PCC 0.999993, rel-RMS 0.0050 vs the 0.9995 / 0.03 gate). The ablation at the start showed the one-worker-per-bank R4 was **compute-bound**, not B-bound: matmul alone 38.7 µs, B stream alone ~28 µs. Levers: (1) a `cores_per_bank` split (`DS_CORES_PER_BANK`, auto: 5 workers × 4 columns per bank, 40 compute cores) — 42.8 → 34.8 µs, compute down to 10.6 µs, DM now binding; (2) the A rounds moved to the writer RISC (BRISC / NoC1, `DS_ACT_ON_WRITER`, `split_reader`) so they stop sitting between the reader's B issues and barriers — 35.0 → 28.9 µs, A now fully hidden (A-stubbed run 29.7 µs ≈ full); (3) a trid-tagged B ring (`DS_B_AHEAD` / `DS_DEPTH_WEIGHT`, `trid_double_issue`) — correct but measured flat to +3% at depth 2–3 once 5 slice readers share each bank, so parked at 1 / 2 as a live knob. Same levers on the other R4 shapes: LM head 1×5120×16384 148.5 → 105.8 µs (stock 127.0), gate_up 32×5120×9216 swiglu 103.4 → 63.1 µs (stock 64.0); the 1-tile-shard cartesian shape (C = 1) is unchanged; the rest of the guard set is within ±1% (block_sharded res5 shortcut re-measured 3×: +0.8%). **Bottleneck now**: the B stream (compute-stubbed 28.8 µs ≈ full; B alone 27.9 µs vs the 27.2 µs roofline, ~97% of DRAM peak). **Next**: what is left is ~1.5 µs of fixed cost — the first K-block's exposed B latency before compute starts and the output tail — e.g. a smaller first K-block (split A shard 0's round) to start compute earlier. Not done: it is within ~6% of the DRAM roofline and each remaining µs needs a structural change of its own.

### [x] Refinement 7 — height_sharded placement + batched weights (local shards, independent batch split)

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

**Outcome**: SUPPORTED placement += `height_sharded` (R2, built as R3's single-K-shard case on A's shard cores: the resident shard is compute's A operand via `cb_act_shard`, no `cb_act_tiles`, no NoC read of the own shard; output in place in its L1 height shard, N-split streamed fallback), weight_batch += `batched` (R1d on interleaved; R2 / R3 shards mapped onto whole images or one image's rows by `ShardRows`). Golden: 450/450 cartesian HS, 150/150 batched interleaved + block_sharded, 12/12 PDL HS loose, PERF FOCUS 3; unit net green. Perf finding: R2 is B-DRAM-bound (each shard core reads all of B: PDL res3 reduce 20.2 µs vs stock 6.9 µs); the lever is the deferred R2a B multicast. PERF FOCUS 3 baseline 70.8 µs (stock 123.0 µs).

### [x] Refinement 8 — Speed up PERF FOCUS 3 (PDL bilinear upsample, H pass: batched weight)

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

**Outcome**: PERF FOCUS 3 measured **70.3 → 65.2 µs** device time (median of 5 fresh runs each; stock 123.0 µs, roofline 43.5 µs); precision PCC 0.9989, rel-RMS 0.017 (gate 0.997 / 0.15). The ablation at the start showed the output **DRAM write stream binds**: write-only 59.9 µs, while the reads (A re-read 3× from DRAM by the 32×1×3 split, 6.6 MB) sat before compute as a ~10 µs prologue and competed with the writes for DRAM. Levers: (1) **operand residency** (`RESIDENT_SPLIT_BLOCKS = 4`): a one-K-block S1 walk is split along one axis with the smaller per-core operand held fronted (matmul_block `WaitAndRetainOnLastBlock`; `act_resident` = A held across N-blocks, `weight_resident` = B held across M-blocks), so compute and the first writes start after the resident operand + one block; (2) **`L1_READ_COST = 0.25`** in the split ranking (L1-sourced operand bytes do not load DRAM): PERF FOCUS 3 moves to 32×3×1 with A resident, so A crosses DRAM once (2.2 MB) and B is re-read from L1 instead; (3) **write batching** (`OUT_WRITE_BATCH_ROW_GROUPS`): correct but flat to +1.6% at 4 / 8, parked at 1 as a live knob. Also measured: the 16-bit-DEST 8-tile subblock already holds (1×8 on the new block); `block_batches = 2` (16×{3,2,1}×{1,3} splits) 71.6–81.6 µs, so one batch per core stays; writer on NoC0 107 µs, so NoC1 stays; write bank-stagger, N-block rotation and posted writes are all within noise. Same levers elsewhere: the PDL upsample W pass 15.6 → 12.7 µs; 2048×256×512 bf16 −11% / −14% (HiFi4 / HiFi2); 2048×128×3072 −6% / −4%. The rest of the guard set is within ±2% (plans unchanged). A residency split that would lose the R1a multicast is refused: 2048×256×512 bf8b had measured +11% before that gate. **Bottleneck now**: the interleaved DRAM write stream. Write-only time is 59–61 µs at 32, 64 or 96 cores (≈ 300 GB/s for 1088 B tile writes). Full 65 µs = that floor plus ~5 µs of read prologue, because every core reads its resident A and first B block before its first write. **Next**: shorten that prologue. Options: B multicast within each batch's 3-core group (it re-reads the same L1 B 3×, 13 MB on NoC0), or a staggered core start. Not done: the multicast needs a batch-line topology, a scheme change, and the whole remaining gap is ~8% above a measured write floor that no read-side lever moves.

### [x] Refinement 9 — Speed up PERF FOCUS 4 (FLUX.2 prompt-stream FF1 + SwiGLU, 256 tokens)

**Type**: perf

**Goal**: The fourth flagged case, `_flux(256, 6144, 18432, epilogue="swiglu")`:
- A (1,256,6144) bf16 × B (6144,18432) bf16;
- HiFi2, fp32 DEST, interleaved DRAM, swiglu.

The goal is to match or beat `achievable_ns` = 865.9 µs; the roofline is 508.6 µs, bound by the 226 MB bf16 weight. With M = 8 tiles the traffic-optimal split is `grid_m = 1`, so B crosses DRAM exactly once. A (8 tiles × K) is then reuse-shared by every core, so the lever is **R1b `interleaved_stream_mcast_a`**: one injector broadcasts each A K-block (`shared_input_reuse`, 1.7×). Co-tune B stream depth / trids (`trid_double_issue`) and the bank stagger (`bank_stagger`).

Its epilogue-none twin (256×6144×9216) is 989 µs vs 453 µs stock today. The guidance / timestep linear_2 (M = 1, 1.6× stock) share the regime.

No SUPPORTED change.

**Verifier notes**: Reuse the multicast infrastructure Refinement 3 builds. This is a different topology (one-to-all A broadcast instead of row/column), so it gets its own phase.

**Done when**: measured device ns on the flagged shape improves toward 865.9 µs, its soft precision gate holds, and the golden suite is green. There must be no regression across the guard set (one representative per kernel path × placement × dtype, including the PERF FOCUS 1–3 shapes).

**Outcome**: PERF FOCUS 4 measured **1125 → 645 µs** device time (median of 3 fresh runs, range 630–646). The `achievable_ns` target of 865.9 µs is beaten, and the result is 1.27× the 508.6 µs roofline. Precision: PCC 0.99999, rel-RMS 0.0084.
- **What landed.** R1b `interleaved_stream_mcast_a` on a 1-D N split of 96 cores: every core streams its own B columns, and one injector broadcasts each A K-block from the writer RISC (BRISC / NoC1). Co-tuned levers:
  - the row-aware N quantum (`Q_N_ROW_AWARE`: 72 → 96 cores, 2×2 subblock);
  - `ACT_STREAM_DEPTH = 4` (683 → 633 µs);
  - the injector stall-shadow `ACT_STREAM_AHEAD = 2`: −3 to −4% on A-heavy shapes, noise on FF1.
- **Gates.** The split ranking caps R1b by the single injector's throughput (`ACT_STREAM_INJECTOR_COST = 18`), and R1b runs only on weight-streaming shapes. Together these keep text-encoder down and x_embedder on their old plans.
- **Same regime, other shapes:**
  - the twin, 559 → 321 µs;
  - guidance / timestep linear_2, 308 / 407 → 206 / 209 µs;
  - context_embedder, 2226 → 351 µs;
  - attn-out txt, 888 → 144 µs;
  - FF2 txt, 570 → 323 µs;
  - text-encoder qkv, 1314 → 232 µs.
- **Guard set.** PERF FOCUS 1–3, block_sharded, height_sharded, 4096³, the W pass and the L1 shapes are all within ±1% (PERF FOCUS 3: median of 5).
- **Bottleneck now (ablation).** The B DRAM stream: B alone takes 550 µs, ≈ 411 GB/s for interleaved 2 KB tile reads from every bank. The A broadcast adds 28 µs, but its lockstep keeps the cores on the same DRAM rows: with A stubbed and compute on, the run is slower (744 µs vs 688 µs). Compute alone is 274 µs, and ~55 µs of compute stays outside the stream.
- **Next.** A split B reader (half of each B slot filled by BRISC on NoC1 under a local slot handshake), or a rotating injector for A-heavy shapes (a rectangular landing zone as in R4). Neither was done: each is a structural change, and the gain depends on NoC0 link saturation, which was not measured. Same-bank coalesced reads measured slower (627 vs 578 µs).

### [x] Refinement 10 — Weight multicast for sharded A (R2a height_sharded, R3a block_sharded): PDL 1×1 convs

*Filed by the run operator after Perf 1, from the shape-set review of every model case against stock.*

**Goal**: no SUPPORTED change. Speed up the `height_sharded` and `block_sharded` regimes, now the op's largest gap to stock. Every PDL 1×1-conv model case in `LOOSE_CASES` (bfp8, LoFi, the model's shard layouts in `extras.act_shard`) is 1.1–3.1× slower than its `achievable_ns`. Times are from the golden run after Perf 1:
- R2 (`height_sharded`) re-reads all of B from DRAM on every shard core (op_design.md R2 → R2a):
  - res3 shortcut 8192×256×512 (bf16 A): 41.0 µs vs 13.1 stock
  - res3 reduce 8192×512×128: 20.1 vs 6.9
  - res3 expand 8192×128×512: 23.5 vs 8.2
  - res2 shortcut 32768×128×256: 19.0 vs 9.8
- R3 (`block_sharded`) reads its B column slice from DRAM once per grid row (R3 → R3a):
  - res5 reduce 2048×2048×512: 40.3 vs 16.1
  - res5 expand 2048×512×2048: 43.0 vs 24.2
  - res5 reduce 2048×1024×512: 19.8 vs 10.6

Build the designed successors:
- R2a: one injector reads each B chunk once and multicasts it to all of A's shard cores.
- R3a: B column multicast down each grid column, beside the existing A row multicast.

Make them the path for every height- and block-sharded call where they are correct. Carve out only shapes where they measure slower.

**Implementation skill**: /perf-measure for the breakdown first; the multicast helpers in `ttnn/cpp/ttnn/kernel_lib/mcast/`.

**Done when**:
- Every PDL sharded model case is at or under its clock-scaled `achievable_ns`, or its remaining gap is explained by a measured floor.
- Golden stays fully green: 1,512 supported pass, 0 fail.
- The guard set shows no regression beyond ±3% elsewhere, PERF FOCUS 1–4 included.

**Outcome**:
- **What was built.** R2a and R3a: one fixed injector per B group multicasts B rounds, and the first bias / gate row rides one row round on the same channel. R3 is gated at ≥ 3 MiB of saved B re-reads.
- **HS results.** All 12 PDL cases ran 0.38–0.81× of before. Examples: res3 shortcut 41.3 → 15.7 µs, res3 reduce 20.5 → 8.4, res3 expand 23.5 → 9.9, res2 shortcut 18.9 → 10.7. They are now 1.02–1.26× achievable.
- **BS results.**
  - res5 shortcut 78.0 → 30.4 (0.42× achievable), res5 expand 44.5 → 24.0 (0.99×), res5 reduce 19.3 → 14.9, res5 reduce 2 40.5 → 22.9, ASPP 1×1 19.8 → 18.4.
  - The gated shapes (res4 reduce / expand / reduce 5, ASPP projection) are unchanged, as is ASPP pool (one grid row).
- **Guard set.** PERF FOCUS 1–4 and the guard shapes are within noise.
- **Remaining gaps are measured floors, not B traffic.**
  - HS: two ~1.4 µs handshake rounds on 86–103-core groups, plus the injector's serial tile-read issue.
  - R3 res5 reduce 2: the A row exchange is ~5.7 µs of bytes plus handshakes, and the 4 B rounds add 3.3 µs serialized behind them on the reader.
  - res4 reduce: 16.0 µs remains with the matmul and the A bytes both stubbed. The cause is thin 2-tile K-blocks that each pay a full partials pack.
- **Next.** Move the B channel onto the writer RISC so B rounds overlap the A rounds (this also lifts the R3 gate), then pipeline the A exchange two rounds deep. Not done here: each is a separate structural change to the writer / A exchange, outside this heading's B-multicast scope.

### [x] Refinement 11 — Interleaved split for narrow N and mid M (FLUX.2 proj_out, text-encoder projections, single-block projections, mid-M prefill)

*Filed by the run operator after Perf 1, from the shape-set review of every model case against stock.*

**Goal**: no SUPPORTED change. Fix the interleaved model cases still slower than their `achievable_ns`. Times are from the golden run after Perf 1:
- FLUX.2 `proj_out` 2048×3072×128 (N = 4 tiles): 164.7 µs vs 61.0 stock.
- Text encoder (bf16 × bfp8, HiFi2):
  - `qkv_proj` 512×5120×3072: 232 vs 139
  - `o_proj` 512×4096×2560: 96.7 vs 79.4
  - `down` 512×16384×5120: 551 vs 509
- FLUX.2 single blocks (M = 2,304, HiFi2):
  - QKV 2304×6144×9216: 1390 vs 1245
  - out 2304×12288×6144: 1983 vs 1689
  - FF1 + SwiGLU 2304×6144×18432: 2614 vs 2398
- Qwen3.8 prefill (bfp4, LoFi):
  - qkvg 4096×5120×3584: 440 vs 372
  - in_proj 1024×5120×4160: 165 vs 139
  - the M = 512 rows: 0.89–0.94× of stock

Start from a measured breakdown of each case: the regime and split `choose_split` picks, per-core DRAM traffic, and stage times.

Likely levers:
- Split-K (R1c) or a 2-D split for narrow N.
- Choosing between R1, R1a and R1b by measured cost instead of the current ranking.
- Block and subblock sizes for M = 2,304.

**Implementation skill**: /perf-measure

**Done when**:
- Each listed case is at or under its clock-scaled `achievable_ns`, or its remaining gap is explained by a measured floor.
- Golden stays fully green.
- The guard set shows no regression beyond ±3%, PERF FOCUS 1–4 included.

**Outcome**: device ns (one fresh run each, 1350 MHz, stock `achievable_ns` in brackets):
- **At or under stock**: proj_out 160.5 → **57.2** µs (61.0); text-encoder qkv 232.7 → **111.4** (139.2); FLUX single QKV 1382 → **1117** (1245), out 1978 → **1426** (1689), FF1 + SwiGLU 2588 → **2155** (2398); Qwen in_proj M = 1024 147.1 → **137.2** (139.0); Qwen M = 512 rows in 100.2 (108.8), out 52.1 (52.1), gate_up 194.3 (214.4), down 101.2 (111.7), qkvg 94.8 (101.3); qkvg M = 4096 409.4 → **372.5** (371.9, +0.2%: noise).
- **Above stock, gap = a measured floor**: text-encoder o_proj 96.1 → 85.2 (79.4) and down 541.6 → 524.7 (508.6). Ablation: DM-only (compute skipped) 82.3 / 509.5 µs, compute-only 60.3 / 405.2 µs, sync floor 14.7 / 37.9 µs. Both are bound by the R1a line injectors' serial read → multicast chain (B lines of 1.1 MB / 8.4 MB per injector on an 8-row grid; M = 16 tiles caps the split at 80–88 cores).
- **Levers** (all host knobs, one source of truth each): (1) `SPLIT_COST_MODEL` — `choose_split` blocks every candidate and ranks by `split_cost` = max(padded K-walk × fidelity tile-K cost, DRAM stream, busiest core's reads, busiest core's writes), with `q_n_min` candidates and a padding-aware S3 block search; (2) `MIN_K_BLOCKS = 8` — R1a helper walks re-split into ≥ 8 K-blocks (shorter pipeline fill; Qwen M = 512 −4..−12%); (3) `R1A_INJECTOR_AHEAD = 2` — R1a injector reads K-block kb+1 under its own trid while multicasting kb, depth-3 rings, gated to injector-bound walks (proj_out 66.6 → 58.5; it loses 4–8% where the K-walk is as long as the injector stream, so it is path-gated there); (4) `NONE_RAW_WALK` — the Perf 1 raw K-walk + an identity copy for epilogue-none multi-K-block walks, gated to compute-bound splits (−4..−5% on FLUX single QKV / out and Qwen M = 4096; +2–4% on DM-bound ones, so gated).
- **Guard set** (140 cells, every perf file incl. PERF FOCUS 1–4, sharded): geomean 0.93×, no cell slower beyond +2.9% (R1b / M = 1 cells with identical plans, noise band); PERF FOCUS 1 391.1 → 391.4, 2 29.8 → 29.9, 3 66.6 → 66.0, 4 617.5 → 616.8. Golden interleaved slice 429 passed / 75 xfailed; translated + regression unchanged (212 passed, the 14 pre-existing cache-count failures).
- **Next** (not done): move the R1a B-line injection onto the writer RISC (BRISC / NoC1, as R1b does for A) so B reads overlap the A-line rounds — the only lever that moves the te_o / te_down DM floor; the in-RISC prefetch cannot, because its read issue blocks the same RISC that multicasts. Not done here: it is a reader / writer topology restructure of its own, and both remaining gaps are within 9% of stock with a measured floor.

### [x] Refinement 12 — Hang: bf16 batched-weight matmul (PDL upsample H pass in bf16)

*Filed by the run operator for the #65150 follow-up (the model integration on tt-metal branch `avelickov/65150-matmul-integration`). Refinements 12–18 aim the op at the roofline: `extras.roofline_ns` is now set on every golden model case. Device times below are from run 1137's last golden run, with the roofline in brackets.*

**Goal**: no SUPPORTED change. A supported cell hangs. Fix it first: until it is fixed, every golden run stops at it.
- `genify_matmul(A (1, 32, 512, 128) bf16, B (1, 32, 128, 1024) bf16 batched, out bf16)`, HiFi2, `fp32_dest_acc_en=False`, every tensor in DRAM: dispatch timeout. It hangs with B in L1 too. In bfp8 (the model's dtype, PERF FOCUS 4) the same call runs.
- Golden cases:
  - `test_op_loose[A1x32x512x128_B1x32x128x1024-...-dtype=BFLOAT16-...-fp32_dest_acc_en=False-...]`
  - `test_op_feature[A1x32x512x128_B1x32x128x1024_weight_l1-...]`
- `--dev` triage in #65150 had the writer's tail split (`tail_split.hpp`) waiting on the reader's K-walk, in regime R1d (batched weight).

Find the root cause and fix it for every shape and dtype it can reach. A guard on this one shape does not count. A bf16 tile is 2048 B against bfp8's 1088 B, so CB page counts and K-block sizes differ between the two dtypes. Check the K-walk / tail-split handshake for every block-size combination the split chooser can produce.

**Implementation skill**: /debug-ttnn-op

**Done when**:
- Both golden cases pass, with and without `--dev`.
- A focused test in `tests/ttnn/unit_tests/operations/genify_matmul/` pins it: bf16 and bfp8, B in DRAM and in L1.
- Golden is otherwise green, and PERF FOCUS 4 (the bfp8 H pass) is no slower.

**Outcome**: root cause = the tail write split (`tail_split.hpp`) held the WHOLE last block's output for the reader, but on a multi-batch block (block_batches > 1; bf16 plans R1a with block_batches = 32, bfp8 plans block_batches = 1) the reader only serves after its walk over the block's later batches -> output CB full -> compute stalls -> reader stuck in `cb_reserve_back` (triage: 64 writers in `wait_at_least`, 64 readers in `cb_reserve_back`). Fix: the split covers only the tail block's LAST batch, whose output exists only after the reader's final K-block push, so the handshake has no cycle for any block_batches / K-block / depth / dtype. Both golden cases pass with and without `--dev`; bf16 H pass now runs in 136.0 us (it hung before). bfp8 H pass: 66.6 -> 67.6 us (medians of 4 A/B runs on the same box, +1.5%: inside the noise band; bfp8's block_batches = 1 goes through the same handshake). Not done here: the bf16 H pass is 2x the bfp8 one. Its plan (R1a, 32-batch blocks of 2x4 tiles, 8x8 grid) has not been profiled; that is perf work for a later round.

### [x] Refinement 13 — Split output: `chunks` (FLUX.2 QKV ×3, time_mod ×17)

*Filed by the run operator for the #65150 follow-up.*

**Goal**: implement `chunks > 1`. The entry point already takes it and raises UnsupportedAxisValue. Today FLUX.2 runs genify at full N and splits the result with `ttnn.chunk`. Together with the concat of Refinement 14, those extra ops cost 25.7 ms per denoising step, 8% of the step.

Contract: `genify_matmul(A, B, ..., chunks=c)` returns a list of `c` tensors.
- Output i is columns [i·w, (i+1)·w) of epilogue(A @ B), where w = N_out / c (N_out = N, or N / 2 for swiglu).
- N_out % c == 0 and w % 32 == 0; otherwise ValueError.
- Every chunk gets `memory_config` and `dtype`.
- The program writes each chunk directly: no full-width intermediate, no extra op. The writer picks the destination tensor by output tile column (one TensorAccessor per chunk, or an address table in the runtime args). The tiles each core computes do not change, so the split choice and the compute kernel stay as they are.

Scope: interleaved placement, every epilogue (FLUX.2 needs none and bias). Sharded placements with `chunks` may raise UnsupportedAxisValue.

Golden: the six `test_op_feature[..._chunksN-...]` cases:
- FLUX.2 double QKV, img and txt;
- FLUX.2 single QKV;
- time_mod ×17 (fp32 A + bias, bf16 out);
- two small ones.

**Done when**:
- The six cases pass.
- Each FLUX.2 `chunks` case runs within 2% of the same call without `chunks`; LOOSE_CASES has each one. The split is free.
- Golden is green, and nothing else is slower.

**Outcome**: the writer (and the reader's tail-write share) route each output tile column to its chunk through an address table in the runtime args (`kernels/out_chunks.hpp`); split, blocking and compute unchanged. All six golden cases pass (PCC >= 0.99999 per chunk). Device ns, `chunks` vs the same LOOSE call: double QKV img 962.8 vs 963.9 us; single QKV 1121.1 vs 1122.1 us; time_mod 1605.5 vs 1602.7 us (+0.2%); double QKV txt, 3 runs each, median 328.8 vs 325.4 us (+1.0%, both spread 320-336 us: noise). The split is free. The bottleneck is the matmul itself; nothing left to do for chunks.

### [x] Refinement 14 — A as a list of K segments (FLUX.2 single-block out = [attention | MLP])

*Filed by the run operator for the #65150 follow-up.*

**Goal**: implement a list `input_tensor`. The entry point raises UnsupportedAxisValue for one today. FLUX.2's single-block out projection multiplies concat([attn (M, 3072), mlp (M, 9216)], dim=-1) by a (12288, 6144) weight. Today the model builds that concatenation with `ttnn.concat`, the other part of the 25.7 ms per step.

Contract: `genify_matmul([A_0, ..., A_{s-1}], B, ...)`.
- Every A_i is (..., M, K_i), with the same leading dims, dtype, layout and placement.
- K_i % 32 == 0 and Σ K_i == B.shape[-2]; otherwise ValueError.
- Result = epilogue(concat(A_i, dim=-1) @ B), without building the concatenation. The reader maps K tile kt to its segment and local tile (one TensorAccessor per segment, or a segment table in the runtime args).

Scope: interleaved A (DRAM or L1), every epilogue, any number of segments from 1 up (FLUX.2 uses 2). Sharded segments may raise UnsupportedAxisValue.

Golden: the three `test_op_feature[..._concatK...]` cases.

**Done when**:
- The three cases pass.
- Both FLUX.2 cases run within 2% of the single-tensor call of the same shape; LOOSE_CASES has the single out.
- Golden is green, and nothing else is slower.

**Outcome**: the reader's `read_act` (R1 / R1a, every residency / head-ramp variant) and R1b's A-streaming writer map global K tile kt to its segment and local tile through a segment table in their RT args (`kernels/a_segments.hpp`); the split, blocking, CBs and compute kernel are those of the single-tensor call (the host plans on the logical concat shape). All three golden `concatK` cases pass. Device ns, segments vs the single-tensor LOOSE call: single out M = 2304 1425.9 vs 1426.4 us; final block M = 2048 1202.3 vs 1201.7 us (both within 0.05%). The concat is free. The bottleneck is the matmul itself; nothing left to do for the segmented A.

### [x] Refinement 15 — Small-M weight streaming at the DRAM roofline (Qwen3.8 M = 128, FLUX.2 prompt stream, M = 1)

**Type**: perf

*Filed by the run operator for the #65150 follow-up.*

**Goal**: no SUPPORTED change. These interleaved calls are DRAM-bound on the weight: `roofline_ns` is their DRAM bytes at 460.8 GB/s, and they sit 1.2–2.9× off it. Genify µs, roofline in brackets:
- Qwen3.8 prefill, M = 128 (bf16 × bfp4, LoFi; A in L1 in the GDN layers), 16–64 calls per prefill each:
  - MLP gate_up + SwiGLU 128×5120×8704: 109.9 (56.8). This is PERF FOCUS 2.
  - GDN in_proj 128×5120×4160: 64.2 (28.3)
  - MLP down 128×4352×5120: 56.1 (30.0)
  - out_proj 128×1536×5120: 23.2 (12.4)
  - full-attention qkvg 128×5120×3584, A in DRAM: 80.1 (27.2). Stock is 52.9 µs, so this one is slower than stock.
- FLUX.2 prompt stream, M = 256 (bf16 × bf16, HiFi2), 8 calls per step each:
  - QKV 256×6144×9216: 326.7 (262.8)
  - attention out + addcmul 256×6144×3072: 145.0 (96.0)
  - FF1 + SwiGLU 256×6144×18432: 643.2 (508.6)
  - FF2 256×9216×6144: 327.5 (262.8)
  - context_embedder 256×15360×3072, once per image: 349.3 (225.3)
- FLUX.2 at M = 1 (fp32 or bf16 A):
  - timestep and guidance linear_2 1×6144×6144: 200.1 / 201.6 (167.3 / 165.5)
  - time_mod 1×6144×52224 + bias: 1603 (1409)
  - guidance linear_1 1×256×6144: 14.1 (7.7)
- Narrow N or short K at M = 2048:
  - x_embedder 2048×128×3072: 47.9 (30.2)
  - proj_out 2048×3072×128: 57.0 (30.2)

At these M the whole job is streaming B from the 8 DRAM banks once, at full bandwidth, with compute and the A broadcast hidden underneath. Start with a measured breakdown of the M = 128 gate_up case (/perf-measure: DM-only and compute-only ablations, bytes read per core, which banks each core reads). Then apply levers such as:
- Every core streams a disjoint slice of B: no B multicast, no B re-read. Each core keeps enough reads in flight to cover DRAM latency, and the slices spread evenly over the banks.
- A (at most 1.3 MB here) is multicast once, or read in place where it already lives in L1.
- Split-K with a cross-core reduction where N is too narrow to give every core a column slice (qkvg, proj_out, M = 1).
- The bank-local streaming of the dram_sharded regime (R4), applied to an interleaved B: each core reads the pages of its nearest bank.

**Implementation skill**: /perf-measure, /perf-ceiling-dm

**Done when**:
- Each listed case is within 1.15× of its `roofline_ns`, or its remaining gap is explained by a measured floor.
- Golden is green.
- No other model case in LOOSE_CASES or FEATURE_CASES is slower beyond ±3%.

**Outcome**: device µs before → after (roofline), one fresh run each (`perf/test_genify_matmul_perf_r15.py`).
- **Within 1.15× (FLUX.2 non-swiglu):** QKV 331 → 283 (263), attn-out + addcmul 148 → 107 (96), FF2 332 → 278 (263), context_embedder 352 → 251 (225), linear_2 fp32 / bf16 205 / 209 → 168 / 173 (167 / 166), time_mod 1600 → 1352 (1409).
- **Faster, still off roofline (Qwen3.8 M = 128):** in_proj 63.6 → 48.2 (28.3), down 55.9 → 41.5 (30.0), qkvg 81.0 → 49.8 (27.2; stock 52.9, now faster), out_proj 22.9 → 22.8 (12.4).
- **Unchanged:** gate_up + SwiGLU 109 → 107 (56.8), FF1 + SwiGLU 631 → 614 (509, noise), linear_1 10.5 (7.7), x_embedder 47.7 (30.2), proj_out 58.3 (30.2).
- **What was built: the R1b bank-local split.**
  - Each core owns units 8 B columns apart, so each K row is one contiguous bank read.
  - Each bank's units go to the cores of the grid row of its NoC0 home (its optimal reader); spare cores fill the most loaded classes.
  - The output follows the same stride. The split is one full-grid rectangle, so A is multicast once.
- **Also built: rotating A injectors** (3 on the bank split, K-block r from injector r % 3). Each issues its next turn's read before sending when the slots are free.
- **Measured findings.**
  - The first ablation of the M = 128 gate_up showed B, not A, binding. B alone was 93 µs: 576 B tile reads, and a core's NoC0 B rate fell 3–4× with the grid rows its responses crossed.
  - Coalescing alone gave 78 µs. Bank-local home-row placement gave down's B stream 34 µs (~430 GB/s). NoC1 reads were 1.8× slower.
- **Bottleneck now.**
  - Qwen M = 128: the 1.3 MB A multicast (~40 µs at 5 rounds, ~5 µs per 256 KB round plus the turn hand-over), then a ~5 µs compute / write tail.
  - swiglu: its (gate, up) pair spans banks 2s and 2s + 1, whose home rows differ. Under the bank split half its bytes cross rows (gate_up 101–111 vs 107 µs flat, FF1 667 vs 631), so swiglu keeps the flat split (`BANK_LOCAL_SWIGLU = False`, a live knob).
  - x_embedder / proj_out are R1a shapes this lever does not touch.
- **Next, not done (each is its own structural change).**
  - A multicast on both NoCs, or receivers acking a round ahead (needs a counter-signal handshake in the helper), for the Qwen A floor.
  - Paired gate / up cores exchanging Y_up for swiglu.
  - Split-K for proj_out.
- **Guards.** Golden interleaved cartesian 375 passed / 75 xfailed, R15 loose slice 19/19, feature 10/10. Perf guard, R1b guard and R11 sets: no cell slower beyond ±3% (swiglu ff1 / gate_up re-measured 3×: medians 637 / 632 and 107.0 / 107.9); text-encoder gate 504 → 479.

### [x] Refinement 16 — Mid-M compute efficiency (text encoder at 512 tokens, Qwen3.8 prefill M = 512 / 1024)

**Type**: perf

*Filed by the run operator for the #65150 follow-up.*

**Goal**: no SUPPORTED change. At M = 512–1024 the roofline is compute: 70% of the 110-core FPU peak at the call's fidelity. These calls are 1.25–2.2× off it. Genify µs, roofline in brackets:
- Text encoder, M = 512 (bf16 × bfp8, HiFi2), 40 calls per image each (gate and up are separate calls, 80 together):
  - gate / up 512×5120×16384: 502.6 (403.5). This is PERF FOCUS 1.
  - down 512×16384×5120: 522.6 (403.5)
  - qkv_proj 512×5120×3072: 111.9 (75.7)
  - o_proj 512×4096×2560: 85.4 (50.4)
- Qwen3.8 prefill, M = 512 (bf16 × bfp4, LoFi):
  - GDN in_proj: 100.7 (51.2)
  - MLP gate_up + SwiGLU: 194.2 (107.2)
  - MLP down: 100.5 (53.6)
  - full-attention qkvg: 95.1 (44.1)
  - out_proj: 52.2 (24.4)
- Qwen3.8 prefill, M = 1024:
  - out_proj 1024×1536×5120: 75.6 (39.2)
  - qkvg: 119.9 (88.3)
  - in_proj: 137.4 (102.4)
  - down: 133.2 (107.2)
- out_proj (K = 1536) at M = 2048 / 4096: 135.6 / 225.0 (75.7 / 151.3)

Refinement 11's outcome measured the text-encoder o_proj and down as bound by the R1a line injectors' serial read → multicast chain: B lines of 1.1 MB / 8.4 MB per injector on an 8-row grid, with M = 16 tiles capping the split at 80–88 cores. The lever it named but did not build is moving the R1a B-line injection onto the writer RISC (BRISC / NoC1, as R1b does for A), so B reads overlap the A-line rounds. Levers:
- That writer-RISC B injection.
- A split that keeps all 110 cores busy at M = 16 tiles: 2-D, or split-K.
- Subblock and K-block sizes against DEST at HiFi2 and LoFi: bigger blocks, fewer packs.
- Reads overlapped with compute along the K-walk: deeper CBs where L1 allows.

**Implementation skill**: /perf-measure

**Done when**:
- Each listed case is within 1.15× of its `roofline_ns`, or its remaining gap is explained by a measured floor.
- Golden is green.
- No other model case is slower beyond ±3%.

**Outcome**: Three levers landed. (1) R1a split lines: the A row lines on the writer RISC, rotating line injectors with staged, paced reads, and the head ramp. (2) The R1b ring fallback (depth 3 when depth 4 would split M). (3) Operand-aware subblocks.
Device µs, before → after (roofline):
- Text encoder: gate 478.6 → 403.1 (403.5, PERF FOCUS 1 at the roofline), down 525.5 → 430.1 (1.07×), qkv 113.3 → 93.7, o_proj 86.6 → 68.9.
- Qwen3.8 M = 512: in 101.6 → 75.1, gate_up 195.7 → 161.0, down 102.6 → 81.8, qkvg 96.3 → 72.9, out 53.0 → 45.0.
- Qwen3.8 M = 1024: in 139.6 → 125.0 (1.22×), qkvg 120.9 → 107.2 (1.21×), down 134.4 → 125.2 (1.17×), out 76.2 → 75.7.
- out_proj M = 2048 / 4096: 136.4 / 226.4 → 129.0 / 226.2.

Every case is now within 1.05–1.2× of its larger measured floor (compute-only = DM stubbed; DM-only = compute stubbed; table in the changelog). What keeps those floors above the roofline:
- M = 512 is 16 tile rows, which caps the 2-D split at 80–88 cores (the roofline assumes 110); the K-walk alone is ~1.25× the roofline there.
- out_proj (K = 48) writes its whole output after the last K-block. M-blocking the region to overlap the writes measured worse (B re-reads).

Next I would try a non-rectangular split that fills 110 cores at M = 16 tiles, and an out_proj walk that keeps B resident across M-blocks, delivered in K-blocks during the first one. I did not build them because both are new topologies, beyond the levers named here. Golden interleaved slices are green and the perf guards are within noise.

### [x] Refinement 17 — Decode weight streaming (dram_sharded) at the DRAM roofline

**Type**: perf

*Filed by the run operator for the #65150 follow-up.*

**Goal**: no SUPPORTED change. Qwen3.8 decode (bf16 × bfp4, LoFi, M = 1 and batch 16 / 32) streams each bank's weight shard, so the roofline is the weight bytes at 460.8 GB/s. Per token these matmuls take 9.3 ms in the model against an 8.2 ms roofline. Genify µs, roofline in brackets:
- full-attention qkvg 1×5120×3584, 16 calls per token: 32.1 (22.4). This is PERF FOCUS 3, and stock is faster at 29.2 µs.
- LM head, last vocab chunk 1×5120×12928: 99.8 (80.8)
- GDN in_proj 1×5120×4608, 48 calls: 32.2 (28.8)
- out_proj 1×1536×5120, 64 calls: 11.0 (9.6)
- MLP down 1×4352×5120, 64 calls: 29.6 (27.2)
- MLP gate_up + SwiGLU 1×5120×9216, 64 calls: 62.3 (57.6)
- LM head 1×5120×16384, 3 calls: 106.3 (102.4)

Start with qkvg: N = 3584 is 112 tiles, 14 per bank, so the banks are balanced, yet it is the furthest off. Measure the per-bank stream time against the A broadcast and the output tail. Levers:
- More than one worker per bank: split a bank's shard by K or by N between two cores.
- Larger DRAM reads, more of them in flight.
- The A broadcast overlapped with the first B reads.
- The ragged last bank of the LM head's 12928-wide chunk (404 tiles).

**Implementation skill**: /perf-measure, /perf-ceiling-dm

**Done when**:
- Every decode case is within 1.10× of its `roofline_ns`, or its remaining gap is explained by a measured floor.
- Golden is green.
- Nothing else is slower beyond ±3%.

**Outcome**: device µs before → after (roofline): qkvg 31.9 → 24.2 (22.4, 1.08×; stock 29.2), LM head last chunk 100.9 → 83.9 (80.8, 1.04×), in_proj 33.1 → 30.3 (28.8, 1.05×), down 29.5 → 29.3 (27.2, 1.08×), gate_up + SwiGLU 62.4 → 60.1 (57.6, 1.04×), LM head 104.1 → 100.7 (102.4, 0.98×), batch 32 gate_up 62.0 → 60.3, batch 32 down 30.1 → 29.6 (1.09×); out_proj 11.3 → 11.6 / batch 16 11.3 → 11.2 (9.6, 1.17–1.21×) is at its measured floor (compute + A payload stubbed 10.8 µs; every payload stubbed 9.3 µs, 3.5 µs without the 8 A rounds). Levers: ragged bank slices + the smallest compute-hiding cores_per_bank (qkvg was compute-bound at C = 2: 1120 tile products per core), home-row segment placement (the shared row 9 slowed both of its banks ~20 % when a slice reader sat left of the col-6 bank), the LM head's ragged last bank now splits like the others (C = 3, 17-column slices). The B ring at 2 K-blocks in flight measured flat to +6 % (lm_last) and stays parked at 1. Bottleneck now: the bank stream itself (compute stubbed: qkvg 23.1, lm_head ~98). Next lever, not tried: fewer A rounds for short-K shapes (out_proj's 8 handshaked rounds cost ~6 µs alone, overlapped) — e.g. workers reading A's shards directly — because out_proj's B stream (10.8 µs stubbed-compute) would still bind.

### [x] Refinement 18 — PDL: sharded 1×1 convs and the bilinear upsample

**Type**: perf

*Filed by the run operator for the #65150 follow-up.*

**Goal**: no SUPPORTED change. PDL's 55 matmuls take 0.97 ms per inference against a 0.33 ms roofline. They are small, so fixed costs dominate, and most of the sharded ones are slower than stock. Genify µs (roofline, stock):
- Height-sharded (R2):
  - res2 shortcut 32768×128×256: 11.0 (5.0, 9.8)
  - res2 reduce 128→64: 4.9 (1.3, 4.0)
  - res2 expand 64→256, ×3: 9.1 (2.5, 8.7)
  - res3 expand 128→512, ×4: 10.0 (2.5, 8.2)
  - res3 reduce 512→128, ×3: 8.7 (2.5, 6.9)
  - res3 shortcut (bf16 A): 15.8 (5.0, 13.1)
  - decoder and head projections: 4.4–5.9 (0.2–1.3, 3.5–4.7)
- Block-sharded (R3):
  - res4 reduce 1024→256, ×5: 12.0 (2.5, 9.4)
  - res4 reduce 512→256: 18.8 (5.0, 16.6)
  - res5 reduce 2048→512, ×2: 22.3 (10.1, 16.1)
  - res5 reduce 1024→512: 14.6 (5.0, 10.6)
  - ASPP 1×1, ×2: 17.6 (5.0, 13.8)
  - ASPP projection, ×2: 18.4 (3.2, 16.8)
  - ASPP image pool (M = 1), ×2: 9.1 (1.2, 6.9)
- Upsample (interleaved, batched weight):
  - H pass, ×3: 66.6 (43.5, 123.0). This is PERF FOCUS 4.
  - W pass, ×3: 12.4 (5.0, 15.4)

Measure the fixed cost first: ablate the smallest case, the center head 32768×32×32 (4.4 µs against stock 3.5). Refinement 10's outcome holds the costs it measured for the sharded regimes. Then remove what is not the matmul:
- fewer synchronization rounds;
- fewer CB and semaphore setup steps;
- B read once per core group;
- less per-core runtime-arg traffic.

**Implementation skill**: /perf-measure

**Done when**:
- Every PDL case is at or under its stock `achievable_ns`, and the H pass is within 1.2× of its `roofline_ns`; otherwise the remaining gap is explained by a measured floor.
- Golden is green.
- Nothing else is slower beyond ±3%.

**Outcome**: Measured the fixed cost first: on heads it was two serial injector deliveries (B, then the bias row), not the acks. Landed: R2 one-shot handshake-free B channel with the row riding in B's delivery, injector early push, one B group per rectangle of the HS core set, a B-read helper on the injector's BRISC, and R3 split roles (B half on BRISC/NoC1 with one-K-block rounds and no traffic gate). PDL sharded set, device µs before → after (stock):
- **Fixed cost:** heads 4.4 → 2.8 (3.5).
- **HS:** res3 shortcut 15.7 → 12.6 (13.05), res3 expand 9.9 → 8.3 (8.2), res3 reduce 8.4 → 6.6 (6.9), res2 reduce 4.9 → 3.8 (4.0).
- **R3:** res5 reduce 2 23.0 → 16.6 (16.1), ASPP 1×1 18.4 → 14.0 (13.8), res5 reduce 14.9 → 11.6 (10.6), res4 reduce 1024 12.1 → 9.8 (9.4), res4 shortcut 17.2 → 14.2 (16.2), ASPP projection 18.6 → 16.7 (16.8).

Now at or under stock: 13 of 23 sharded cases plus the W pass. Five more are within 1–3%: res2 shortcut, res2 reduce 256→64, res3 expand, decoder skip 8192, ASPP 1×1.

Measured floors for the rest (ablation, handshakes kept):
- **H pass:** 66.8 µs against a 62.6 µs write-only floor (17.8 MB of 1088 B tile writes at ~285 GB/s). The 1.2× roofline target (52 µs) is below that floor. Splitting the writes over both NoCs measured 65–82 µs.
- **ASPP pool:** 8.8–9.4 against a 6.5 µs A-round floor. Its A rounds slow from ~0.6 to ~1.0 µs each under B traffic.
- **res4 reduce 512→256:** 18.4 against a 15.9 µs A-exchange floor that overlaps imperfectly with its 512-product matmul.
- **res5 reduce / reduce 2:** A floors 9.0 / 12.3, with B and compute adding on top.

Next lever, not done: an R3a A exchange without the per-round loopback write barrier (each sender's include-source multicast waits for all write acks before it can ack the next round). It needs a send-without-loopback mode in the shared Mcast helper, outside this op. The one-shot concurrent exchange was tried and measured worse (parked: `BS_A_ONE_SHOT`).

### [x] Refinement 19 — Interleaved tall, narrow matmuls on all cores (PDL 1×1 convs read in place)

**Type**: perf

*Filed by the run operator for the #65150 follow-up, after the model re-measurement on the ported op.*

**Goal**: no SUPPORTED change. In the PDL model, `genify_conv.py` reads these convs' activations in place (interleaved in DRAM or L1) and writes an L1 interleaved output. Stock instead shards the activation first and runs a sharded conv. On these shapes the interleaved split leaves most cores idle: M = 2,048–32,768 rows, K = 256–2,048, N = 32–2,048, and B (at most 2 MB) is small next to A (up to 8.9 MB). Together they take 330 µs of PDL's 907 µs of matmul time per inference. They are the ten new LOOSE_CASES `placement=interleaved` PDL entries.

In-model genify µs. In brackets: stock's sharded matmul (`achievable_ns`) + the sharding step stock runs before it; then `roofline_ns`.
- A in DRAM:
  - res3.0 conv1 32768×256×128: 41.3 (9.4 + 24.7; 19.4)
  - res4.0 conv1 8192×512×256: 25.4 (16.6 + 11.9; 10.0)
  - res5.0 conv1 2048×1024×512: 17.6 (10.6 + 7.4; 6.1)
  - res5.0 shortcut 2048×1024×2048: 33.7 (72.3 + 7.2; 20.2)
- A in L1:
  - decoder res2 projection 32768×256×32, ×2: 21.6 (3.6 + 12.3; 1.3)
  - decoder res3 projection 8192×512×64, ×2: 12.4 (4.6 + 6.8; 1.3)
  - ASPP 1×1 2048×2048×256, ×2: 28.6 (13.8 + 6.6; 5.0)
  - ASPP projection 2048×1280×256, ×2: 18.9 (16.8 + 6.5; 3.2)
  - res3.0 shortcut after the stride-2 subsample 8192×256×512: 19.1 (13.1; 5.0)
  - res4.0 shortcut after the stride-2 subsample 2048×512×1024: 17.2 (16.2; 5.0)

The first integration logged 27–49 cores for the worst of these (`choose_split` appeared to over-weight B re-reads against each core's own A read).

Start from a measured breakdown of decoder res2 (32768×256×32) and res3.0 conv1 (32768×256×128): cores used, bytes each core reads, stage times. Levers:
- An M-split over all 110 cores when B is small: each core reads its own A rows and all of B (or B multicast once).
- Enough A reads in flight per core to cover L1 / DRAM latency; the output written by the core that computed it.
- The split cost model re-ranked for these shapes against measured times.

**Implementation skill**: /perf-measure

**Done when**:
- Each listed case meets one of these:
  - at or under its `achievable_ns` (stock's sharded matmul alone, without the sharding step);
  - within 1.5× of its `roofline_ns`;
  - its remaining gap is explained by a measured floor.
- Golden is green.
- No other model case is slower beyond ±3% (the Qwen3.8 PERF FOCUS cases and the sharded PDL cases are guards).

**Outcome**: the tall cases now take an M-only split over 64–86 cores (was 27–49). It adds: B and the bias row multicast once per rectangle (`R1_B_MCAST`), the row kept across M-blocks, at most 8 A tiles in flight per core (`R1_A_INFLIGHT_TILES = 4`, against a NoC collapse when the whole grid bursts its K-block reads), a DRAM bank stagger, and L1-output and per-group-B terms in `split_cost`. Device µs (bench), before → after:
- res3.0 conv1: 41.6 → 27.6, within 1.5× of its roofline.
- res4.0 conv1: 27.4 → 20.6.
- decoder res2: 22.6 → 12.1.
- decoder res3: 12.3 → 9.8.
- res3 shortcut: 20.0 → 16.8.
- res5.0 shortcut: 33.5, under its achievable.

In-model, these ten calls drop from ≈322 to ≈271 µs per inference (bench numbers × call counts). Upsample W pass: 12.8 → 11.3 µs. No guard cell is slower outside noise.

**What binds now (measured floors):**
- With every operand payload and the compute stubbed, the L1-interleaved output write plus sync alone takes 4.3–10.6 µs. That is above decoder res2's 3.6 µs achievable; stock reads A from its own shard and writes in place.
- decoder res2 / res3: the all-to-all L1 A read, ~6 µs over the local-read floor.
- res4.0 conv1: the A DRAM stream at ~300 GB/s, plus the single injector's 139 KB B read (+3.4 µs).
- res3 shortcut: the output-write floor plus the B delivery.
- The M = 2048 R1a cases (ASPP 1×1 28.6, ASPP projection 18.9, res5.0 conv1 17.3, res4 shortcut 16.6) did not move: 4–8 serialized line-multicast rounds of ~3 µs.

**Next, and why not here:** a B delivery not bound by one core's DRAM read rate. The distributed slices are built and parked (`R1_B_DISTRIBUTED`): concurrent multicasts into one rectangle serialize (20.6 → 57 µs), and ordered 2 / 4 slices lose (23.9 / 25.0). Also a deeper R1a round pipeline for the M = 2048 cases. Neither had a winning setting within this refinement.
