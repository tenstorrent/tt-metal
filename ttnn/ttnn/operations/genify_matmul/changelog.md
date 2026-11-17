# Changelog: genify_matmul

## Phase 0 — Core Implementation
- **Date**: 2026-11-17
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer →
  verifier). Regime R1 `interleaved_out_split`:
  - an independent M×N output split over the full device grid;
  - per-core blocks solved coarsest-first against the L1 budget;
  - the K-walk accumulated in L1 by the `matmul_block` helper (packer L1 acc, ragged last K-block);
  - one `ttnn.generic_op` dispatch per call.
- **SUPPORTED at Phase 0**:

  | Axis | Values |
  |---|---|
  | dtype | bfloat16 |
  | weight_dtype | bfloat16 |
  | layout | TILE |
  | fp32_dest_acc_en | True |
  | placement | interleaved (DRAM or L1) |
  | epilogue | none |
  | alignment | tile_aligned, m_non_aligned |
  | weight_batch | single |

- **Accuracy achieved**: PCC ≥ 0.9999985, max_abs_err ≤ 0.018, relative RMS 0.0017, ULP p99 3–4.
  Measured on 5 shapes (single tile through 2048×4096×2048 and K = 16384) with
  `test_genify_matmul_precision_baseline.py`. Bit-identical across repeated calls.
- **Golden suite at Phase 0**:
  - 94 supported_pass, 2088 xfail_expected, 0 xpass_drift, 0 xfail_wrong_mode
    (`verifier_report.json`);
  - 1 supported_fail: a harness L1-capture false-negative on a correctly raised `ValueError`.
- **Issues encountered (fixed by the verifier)**:
  - **Grid factorization.** Ranking W strictly first produced DRAM-traffic-pathological splits:
    9472×5120×3456 took 28 ms with every core reading all of A. It is now `choose_split()`: within
    `GRID_W_SLACK` (0.25) of the best W, take the min of aggregate read traffic *including* the
    block re-reads. Measured 0.21–0.46× device time on the mis-split shapes and no regression beyond
    noise on the guard set; golden geomean 0.95×.
  - **L1 CB budget.** It ignored fragmentation (a freed L1 tensor above a live one), which gave a
    CB/L1-buffer clash TT_THROW. It now ends at the lowest allocated L1 block.
  - **DRY.** The duplicate `TILE` constant was removed.
- **Tests added**:
  - `test_genify_matmul_precision_baseline.py`;
  - `test_genify_matmul_extended.py`: output-dtype override, fragmented L1, L1 weight;
  - `perf/test_genify_matmul_perf_guard.py`: opt-in device-ns guard set (`GENIFY_MATMUL_PERF=1`).
  - `test_genify_matmul.py` (acceptance, 19 tests) already existed and is unchanged.

## Refinement 1 — Numerical configurability (activation, weight and DEST formats)
- Date: 2026-11-17
- What was done:
  - **SUPPORTED widened.** `dtype` += bfloat8_b, float32. `weight_dtype` += bfloat8_b, bfloat4_b.
    `fp32_dest_acc_en` += False. `{"dtype": float32, "fp32_dest_acc_en": False}` stays the permanent
    EXCLUSION. No new EXCLUSION was needed: every cell this refinement unlocks passed out of the box.
  - **Formats needed no change.** Phase 0 already derived every CB page format and tile size from
    its tensor (A, B, output) and the partials from the DEST width (Float32 / Float16_b), plus
    `dest_limit` 4 / 8. The `matmul_block` helper owns the unpack/pack reconfig, so there is no
    kernel change.
  - **Precision fix: new knob `DEST16_BLOCK_K_MAX = 8`** (program descriptor;
    `Geometry.block_k_cap`).
    - *Problem.* With 16-bit DEST the solve put all of K into one DEST chain. The bf16 running sum
      swamps small addends, so the error grows with K: rel-RMS 0.04 / 0.07 / 0.18 at
      K = 2k / 4k / 16k. Stock ttnn.matmul measures 0.011 / 0.014 / 0.027 on the same shapes.
    - *Fix.* With 16-bit DEST, `block_k` is now capped at 8 tiles. The cross-K-block sum moves to
      the packer: L1 accumulation in `cb_partials`, kept Float16_b as specified.
    - *Effect.* K = 16k drops to rel-RMS 0.019. Cap 2: 0.020. Float32 partials only reached 0.0145,
      a marginal gain, so Float16_b is kept.
    - *Scope.* fp32 DEST is uncapped (`block_k_cap = tensor_k_tiles`), so the bf16 / fp32-DEST
      path's blocking is unchanged.
    - *Perf.* The cap also wins on device time from better read/compute overlap:
      64×16384×256 108 µs vs 143 µs uncapped; 384×1536×1024 89 vs 101 µs; 1×2048×1536 35 vs 41 µs;
      2048×2048×1536 LoFi 363 vs 377 µs. Cap 4 is within noise of 8 on time and slightly worse on
      LoFi, so 8 stays.
- Accuracy achieved (`test_genify_matmul_precision_matrix`, HiFi4, randn; full table in
  `tests/.../genify_matmul/precision_matrix_results.md`):
  - fp32 DEST:
    - bf16 out: PCC 0.999999, rel-RMS ≤ 0.0017;
    - bf8b out: PCC 0.99997, rel-RMS ≤ 0.0085;
    - fp32 out: PCC 1.000000, rel-RMS ≤ 0.00073. The tf32 partials reload (median got/true
      0.9994) stays well inside 0.005, so no `UnpackToDestFp32` was needed.
  - 16-bit DEST: bf16 out rel-RMS ≤ 0.019; bf8b out ≤ 0.025 (gate 0.15).
  - Shapes: 32×32×32, 1×2048×1536, 384×1536×1024, 2·64×4096×512, 64×16384×256.
  - The weight dtype barely moves these numbers, because the reference holds the quantized
    operands.
- Golden test progress, on the cells this refinement targets (interleaved, epilogue none, single
  weight):
  - cartesian: 60/60 supported pass, 12 xfail (the fp32 × acc=False EXCLUSION);
  - loose cases: 27/27 newly unlocked pass. These are the 20 Qwen prefill bf4b, 2 FLUX fp32
    timestep, 4 text-encoder bf8b and the PDL upsample W pass (bf8b, 16-bit DEST, L1 A/out).
  - After the cap change, the 37 cells with 16-bit DEST were re-run and still pass.
- Issues encountered:
  - **Uniform-input metric artifact.** An all-positive `torch.rand` input gives outputs whose
    mean dwarfs their std, so std-relative metrics only measure bf16 output rounding at the mean.
    The matrix uses zero-centred uniform instead.
  - **fp32 out at LoFi.** It measures rel-RMS ~0.025, which is the hardware's LoFi floor. The
    LOW_FIDELITY fp32 row was calibrated at HiFi2, which is what FLUX uses. Per the skill, the matrix
    gates below-HiFi4 cells on PCC only.
  - **Finding (Phase-0 split, not a format regression).** 512×5120×3072 picks grid (1,1,96), so
    every core reads all of A. That costs 1.38 ms in bf16 and 1.33 ms with a bf8b weight, against
    139 µs stock. The 2-D factorizations that would cut A traffic fail `_placement_for`
    (gn = 12 > gx = 11, or gm = 12 > gy = 10). This is Refinement 3's mcast topology / factorization
    co-tune.
  - **Format vs bf16 twin (device ns).**
    - 2048×5120×4160 LoFi: bf4b weight 1.02 ms vs bf16 1.55 ms. The bf4b tiles are 576 B, so the
      solve picks coarser K-blocks.
    - 512×5120×3072 HiFi2: bf8b 1.33 ms vs bf16 1.38 ms.
    - 1×6144×6144 HiFi2: fp32 A 401 µs vs bf16 305 µs. A is read whole by each of the 96 cores,
      so 2× A bytes cost 1.3×.
- Tests added:
  - `test_genify_matmul_precision_matrix.py`: 540 cases, 5 shapes × A {bf16, bf8b, fp32} ×
    B {bf16, bf8b, bf4b} × acc × {HiFi4, HiFi2, LoFi} × {randn, uniform}. The fp32 × acc=False
    cells assert `ExcludedCell`.
  - `perf/test_genify_matmul_perf_guard.py::test_perf_formats`: one representative per new format
    path (bf4b LoFi, bf8b HiFi2, fp32 A, bf8b × bf8b 16-bit DEST ×2, capped 16-bit-DEST K-walk).
    Baseline device ns: 1016036, 1343873, 404437, 20310, 44777, 108567.

## Refinement 2 — Fused epilogues: bias, bias_relu, swiglu, addcmul
- Date: 2026-11-17
- What was done: `SUPPORTED["epilogue"]` is now {none, bias, bias_relu, swiglu, addcmul}. Every
  epilogue runs inside the same single `generic_op` program, on the block accumulated in L1, before
  the output is packed. There is no second pass over the output and no host-side or ttnn-op epilogue.
  - **Reused**: the R1 split and blocking, `cb_partials`, the matmul_block call (now templated on
    `LastBlockTarget`), the writer, and the reader's K-walk.
  - **Compute**: with an epilogue, `matmul_block<…, Interm>` lands the block in `cb_partials` (once
    per batch slice), then `pack_reconfig_l1_acc(0)`, then one epilogue block op:
    - bias / bias_relu: `add_bias_bcast_rows<RowBroadcast, TileRowMajor>`, with PACK_RELU set and
      reset around it for bias_relu;
    - swiglu: `eltwise_chain` over (block_m, block_n/2), Strided gate/up `CopyTile` + a
      kernel-local `Silu` chain element + `MulBinary`;
    - addcmul: `eltwise_chain` with an FPU `Mul` against the row-broadcast gate, then a `CopyTile`
      of the residual and an SFPU `AddBinary` (no DEST reuse ahead of an FPU op).
    - The epilogue-none path is unchanged (`LastBlockTarget::Out`).
  - **Reader**:
    - swiglu gate/up column permutation inside each N-block;
    - the bias / gate N-block row, read once per block and pushed after the first batch's K-walk so
      it never stalls the K-block prefetch;
    - residual row groups (sb_h × block_n, depth 2) after each batch's K-walk.
  - **Host**:
    - `q_n ≥ 2` for swiglu, `out_block_n = block_n/2`, and the writer region in output columns;
    - the `E_bias` / `E_addcmul` / `cb_partials` terms are inside `_footprint`, so the solve
      prices them; `_block_k_max` derives its fixed part from `_footprint`;
    - bias / gate / residual join the tensor list, and the `validate()` gate requires
      interleaved TILE epilogue tensors.
  - **Knobs** (single source in the program descriptor): `DEPTH_RESIDUAL = 2`,
    `EPILOGUE_BLOCK_TILES = 4` (measured flat, 1708 → 1696 µs, kept as a live knob),
    `SWIGLU_FAST_SILU = True` (see below).
- Accuracy achieved (PCC / relative RMS, limit in brackets):
  - PERF FOCUS 1, Qwen prefill 2048×5120×8704 bf16×bf4b LoFi swiglu: 0.999988 / 0.0085 [0.9995 / 0.03].
    The other Qwen prefill and FLUX swiglu cases are the same to 3 digits.
  - FLUX addcmul (128 / 256 / 2048 × 6144×3072, HiFi2): 0.999994 / 0.0037–0.0038.
  - FLUX time_mod 1×6144×52224, fp32 A, bias → bf16: 0.999991 / 0.0060.
  - 384×1536×1024, fp32 out: bias 0.00071, bias_relu 0.00086, swiglu 0.00159, addcmul 0.00077
    [0.99999 / 0.005].
  - 384×1536×1024, bf8b with 16-bit DEST: rms 0.010–0.024 [0.15].
  - 384×1536×1024, bf16×bf4b: rms 0.0017–0.0025 [0.01].
- Golden test progress (slices run):
  - cartesian interleaved epilogue cells: 240/240 pass, 48 xfail (the permanent fp32 × acc=False
    EXCLUSION);
  - interleaved loose epilogue cases: 13/13 pass, among them PERF FOCUS 1 (prefill 2048 swiglu,
    bf4b), PERF FOCUS 4 (FLUX 256 swiglu), the 3 FLUX addcmul cases and the fp32 time_mod bias.
  - The epilogue cells on height / block / dram-sharded placements stay xfail until Refinements 4, 5
    and 7.
  - Unit dir: 611 passed, 42 skipped (env-gated perf guard).
- Perf (device ns, fused vs the epilogue-none twin of the same shape and formats, one fresh run each):
  - swiglu 2048×5120×8704 bf4b LoFi: 1655 µs vs 1572 µs;
  - addcmul 2048×6144×3072 HiFi2: 1773 µs vs 1716 µs;
  - bias_relu 2048×256×512 bf8b: 20.4 µs vs 20.8 µs (noise);
  - bias 384×1536×1024: 102.9 µs vs 101.6 µs.
  - The split and the block are unchanged by the epilogue (11×9, 6×32 on PERF FOCUS 1). The output
    CB is half the size, which gives K-block 9 instead of 8.
  - Ablation with the chain compute stubbed (`CKL_ELTWISE_CHAIN_SKIP_COMPUTE`) gives 1564 µs, the
    same as the twin. The Interm / `cb_partials` plumbing is free, and the remaining overhead is the
    epilogue's own compute, which runs serialized after the matmul on each core.
  - That compute was SFPU-bound: the fp32-precision sigmoid inside silu took ~1.7k cycles per output
    tile. `SWIGLU_FAST_SILU` (bf16-precision sigmoid, applied only to non-fp32 outputs) cut it from
    124 to 83 µs, with swiglu accuracy unchanged at 3 digits.
- Issues encountered: none blocking.
  - The precompile farm's best-effort up-front compile fails on this box with an sfpi toolchain
    mismatch (`__xtt_vector8`). The tests compile locally and are unaffected.
  - **Finding for Refinement 3.** The swiglu epilogue still costs ~5% of PERF FOCUS 1. It sits on
    the critical path because each core runs it after its last K-block. Two levers: overlap it with
    the next block's K-walk (only matters when there are several blocks per core), or fold
    silu·up into the last K-block's PostComputeFn with a subblock-local gate/up permutation. The
    second needs a helper change: the packer would have to pack only half the subblock.
- Tests added:
  - `test_genify_matmul_epilogue.py`, 40 cases:
    - 4 epilogues × 5 shapes (single K-block, 2-D grid, folded batch, M = 1, deep ragged K);
    - × 3 formats (bf8b with 16-bit DEST, fp32 out, bf16×bf4b with a bf8b bias);
    - × L1 A / out;
    - × a pinned multi-block-per-core walk.
  - `perf/test_genify_matmul_perf_guard.py::test_perf_epilogues`: one representative per epilogue
    path at the model config, each with its epilogue-none twin.

## Refinement 3 — Speed up PERF FOCUS 1 (Qwen3.8 prefill gate_up + SwiGLU, 2048 tokens)
- Date: 2026-11-17
- What was done:
  - **Regime R1a `interleaved_2d_mcast`** (scheme-change). R1's region, blocking, CBs and compute
    kernel are unchanged; only the reader's operand source changes.
    - **Reader.** On a 2-D rectangle split, one injector per grid row-line reads each A K-block from
      DRAM and multicasts it into the line's `cb_act_tiles`. One injector per column-line does the
      same for B into `cb_weight_tiles` (`McastArgs` sender/receiver pipes; the handshake is "slot
      reserved"). Every core delivers A then B per K-block, so the two channels cannot deadlock.
    - **Kernel groups.** The injector roles are CT flags (`a_sends`, `b_sends`), one reader kernel
      group per role; R1 attaches both channels as absent.
    - **Uniform walk.** Every core walks the nominal `nb_m × nb_n` blocks, so the CB pointers stay
      in lockstep and the multicast lands at the sender's own slot address. Padded-tail blocks of
      ragged edge regions carry 0 valid tiles (the `block_extent` clamp in the reader and writer).
    - **Host.** `uses_operand_mcast()` predicate (knob `OPERAND_MCAST`); mcast-aware `read_traffic`
      (A once per row-line, B once per column-line), so the split now goes to the W-min 11×10
      rectangle on PERF FOCUS 1. Two `ttnn.Mcast` channels at disjoint semaphore ids; `rect` vs
      `rect_t` picks RowMajor or ColumnMajor lines.
  - **Staggered injectors** (`STAGGER_INJECTORS`, `injector_offsets()`): `Staggered` fixed senders
    with offsets chosen so no core injects both operands. Before this, core (0,0) read and
    multicast both operands every K-block. 494 → 464 µs.
  - **R1a gate** `MCAST_MIN_KBLOCK_BYTES = 32 KB`. Each round serializes injector read → handshake
    → multicast, so tiny K-blocks are latency-bound. 64×16384×256 (16-bit DEST, 64 K-blocks of
    16 KB) measured 119.8 µs under R1a against 107.7 µs under R1, so it stays on R1.
  - **Single-walk K split** (`SINGLE_WALK_K_BLOCKS = 2`, resolves the folded L1-ledger finding). A
    per-core walk of one K-block is split in two: the read overlaps compute, and the depth-2 second
    slot is no longer dead.
  - **Subblock tie-break knob** `SUBBLOCK_MIN_UNPACK = False`. Measured null: 2×2 is 477.8 µs vs
    463.7 µs for 1×4.
  - **Reused**: R1 split / blocking / CBs / compute kernel / writer. **Added**: the reader's mcast
    delivery path, role kernel groups, channel attachment, `block_extent` clamp, and the knobs above.
- Accuracy achieved: PERF FOCUS 1 (2048×5120×8704, bf16×bf4b, LoFi, swiglu): PCC 0.99992 /
  rel-RMS 0.0085 [0.9995 / 0.03]. Unchanged on every unit and golden case run.
- Golden test progress:
  - interleaved single-weight loose cases: 53/53 pass (among them PERF FOCUS 1 and 4, and the
    FLUX addcmul and time_mod cases);
  - cartesian interleaved TILE single-weight cells: 300 pass, 60 xfail (the permanent fp32 ×
    acc=False EXCLUSION), 0 fail, 0 XPASS;
  - unit dir: 618 passed, 42 skipped (env-gated perf).
- Perf (device ns, one fresh run per variant; before = this refinement's starting commit):
  - PERF FOCUS 1 fused: 1673 → 463.5 µs (target 564.5, roofline 428.7). Epilogue-none twin:
    1583 → 481 µs.
  - Guard set, 2-D shapes: 0.22–0.56× (e.g. 2048×9216×6144 HiFi2 4446 → 987 µs; 4096³ HiFi4
    2998 → 1294 µs; bf4b 2048×5120×4160 LoFi 1020 → 266 µs; FLUX addcmul 1804 → 438 µs).
  - 1-D and other R1 shapes: unchanged within ±1%.
  - L1-resident A/out (new guard cases): 128×5120×8704 swiglu 386 → 359 µs; 384×1536×1024
    98.5 → 30.7 µs.
- Bottleneck now (ablation on PERF FOCUS 1):
  - full run: 464 µs;
  - reader stubbed (CB sync kept): 449 µs;
  - epilogue chain also stubbed: 413 µs (the matmul + packer-L1-acc floor);
  - so the serialized swiglu epilogue is ~35 µs and exposed data movement is ~16 µs.
- Next levers (not done here; the target is met and each is its own structural change):
  - fold silu·up into the last K-block's pack (needs a matmul_block change to pack half a subblock);
  - the injector stall-shadow: a trid-tagged read of K-block kb+1 issued before multicasting kb.
    It also unlocks R1a for the tiny-K-block shapes now gated to R1.
  - Finding: 128×5120×8704 (Qwen prefill-128, 4 M-tiles) runs on a flat 68-core R1 split at 359 µs
    against `achievable` 105 µs. It is B-stream bound with B re-read per M-region; R1b / stream
    work (Refinement 9 territory).
- Issues encountered:
  - The precompile farm still fails on this box (sfpi `__xtt_vector8`), best-effort only.
  - `scripts/run_safe_pytest.sh -k "a and b"` splits the expression, so pass explicit test ids.
- Tests added:
  - `test_genify_matmul_mcast.py` (7 cases):
    - R1a on `rect` and `rect_t`, with empty padded-tail blocks on both axes (bias, swiglu,
      addcmul, none);
    - staggered vs first-core injectors are bit-identical;
    - the single-walk K split.
  - `perf/test_genify_matmul_perf_guard.py::test_perf_l1_resident`: L1-resident A/output on the
    R1a path.

## Refinement 4 — dram_sharded placement (regime R4: weight streaming from the DRAM banks)
- Date: 2026-11-17
- What was done:
  - **Op file.** `SUPPORTED["placement"]` now includes `dram_sharded`. `validate()` checks the R4 layout through `dram_sharded_topology()`: A is L1 WIDTH_SHARDED with all padded rows per shard and shards in core order; B is DRAM WIDTH_SHARDED, one column shard per bank (even width for swiglu); a requested output config is L1 WIDTH_SHARDED with no shard spec, or exactly the op's. The entry point allocates the output on the op-built spec, `dram_sharded_output_memory_config()`. The batched EXCLUSION stays.
  - **Host (`create_dram_sharded_program_descriptor`).**
    - Workers: bank w's optimal worker, `get_optimal_dram_bank_to_logical_worker_assignment(device, NOC_0)`, for every bank with a valid column.
    - Blocking (`choose_dram_sharded_blocking`): `block_n` = the bank shard width; `block_k = kw / s`, where `s` is the smallest divisor of the A shard width that fits L1 and the 16-bit-DEST chain cap; `block_m` is the largest divisor of the rows that fits; `s > 1` only with `block_m == 1`, so a sender's K-block is one contiguous run of its shard.
    - CBs: `cb_output_shard` (17) is backed on the output shard, 0 CB bytes. `cb_act_tiles` is also placed on every non-worker core the broadcast writes.
    - The A broadcast is one `ttnn.Mcast` channel: rotating senders = A's shard cores in shard order (one shard: explicit fixed sender); receivers = the workers' bounding rectangle; `handshake_cores` = the workers.
  - **Reused vs added.**
    - Reused: the reader, compute and writer kernels and their CB protocol; the R1 CT-arg lists, now factored into `_reader_ct` / `_writer_ct` / `_compute_ct`; `_footprint`, which gained an `out_in_place` term; `choose_subblock`; `l1_cb_budget`.
    - Added: the R4 descriptor branch; the reader's `dram_sharded` path (one coalesced bank read per K-block issued before the A round, the round's send/receive, and a sender-only path for non-worker A cores); compute `swiglu_natural_pairs` (a flat Strided walk, row stride 2, over natural (gate, up) bank columns); writer `out_in_place` (retire only).
- Accuracy achieved:
  - PERF FOCUS 2, 1×4352×5120 bf16×bf4b LoFi: PCC 0.999994, rel-RMS 0.0047 (gate 0.9995 / 0.03).
  - gate_up 32×5120×9216 swiglu: PCC 0.999988, rel-RMS 0.0085.
  - LM head 1×5120×16384: PCC 0.999994, rel-RMS 0.0047.
  - cartesian 384×1536×1024 bf16 HiFi4: PCC 0.999999, rel-RMS 0.0017 (gate 0.9999 / 0.01).
- Golden test progress:
  - dram_sharded cartesian (single weight): 310/310 pass, plus 60 xfail on the permanent fp32 × acc=False EXCLUSION.
  - dram_sharded batched: 90/90 xfail on the permanent EXCLUSION.
  - Qwen decode / LM-head loose cases: 10/10 pass.
  - Translated DRAM-sharded tests (`test_served_shapes_blackhole`, `test_cores_per_bank`, `test_program_cache_reuses_the_program`): 57/57 pass.
  - Regression net, unit files `test_genify_matmul`, `_epilogue`, `_extended`, `_mcast`: 73/73 pass.
- Perf, first R4 build (device ns, one fresh run each):
  - PERF FOCUS 2: 42.7 µs (achievable 36.9, roofline 27.2);
  - LM head 1×5120×16384: 147.8 µs (127.0);
  - gate_up 32×5120×9216 swiglu: 102.6 µs (64.0);
  - cartesian 384×1536×1024 (48 one-tile A shards, 48 serialized rounds): 136.5 µs.
  - Finding for Refinement 6: the B stream runs at ~64% of DRAM peak on PERF FOCUS 2, because each K-block's bank read waits its own barrier before the next is issued (the trid double-issue lever). Tiny A shards make the run round-latency-bound (one handshake per shard).
- Issues encountered:
  - The mcast helper's rotating sender needs a rectangular receiver group, and its irregular multicast is capped at 3 rectangles. The Blackhole bank workers are 8 scattered single cores (columns 0 and 6), so the receivers are their bounding rectangle with the handshake restricted to the workers.
  - `run_safe_pytest.sh --profile` splits a `-k` "or" expression; profile through the perf guard file instead.
- Tests added:
  - `test_genify_matmul_dram_sharded.py` (11 cases):
    - decode with 8 / 10 rotating senders;
    - 64 and 32 one-tile shards (M = 1, bias, addcmul);
    - multi-row A (12 and 64 rows; an fp32 M split; swiglu);
    - a single shard (fixed sender) with the 16-bit-DEST K split;
    - a ragged last bank (N = 704) with the output shard order checked;
    - a host plan test pinning the K and M splits.
  - Perf guard `test_perf_dram_sharded` (4 R4 representatives, among them PERF FOCUS 2).
