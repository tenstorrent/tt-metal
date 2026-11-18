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

## Refinement 5 — block_sharded placement (regime R3: cross-core A K-slice exchange)
- Date: 2026-11-17
- What was done:
  - **Op file.** `SUPPORTED["placement"]` now includes `block_sharded`. `validate()` runs `_validate_block_sharded()`: B interleaved; A L1 BLOCK_SHARDED, ROW_MAJOR, a rectangular grid of exactly (K shards) × (M shards), whole-tile shards; the output L1 BLOCK_SHARDED on A's grid with A's shard height, ROW_MAJOR, whole-tile width that owns every output column. The requested spec is honoured exactly; with `memory_config=None` the op builds the default (`block_sharded_output_memory_config`: `[shard_h, ceil(out_n_tiles / cols)·32]`). Layout errors map to `UnsupportedAxisValue`.
  - **Host (`create_block_sharded_program_descriptor`).**
    - Topology (`block_sharded_topology`): core (c, r) owns A rows of grid row r × output shard c; K-block kb = the A shard of grid column kb (`num_k_blocks` = grid columns, `block_k` = shard width, a ragged last K shard via `last_block_k_tiles`).
    - Blocking (`choose_block_sharded_blocking`): `block_n` = the output shard's B columns (×2 swiglu), output packed in place (`cb_output_shard`); `block_m` = the largest divisor of the shard rows that fits. Fallback: whole shard rows with an N split, output streamed through `cb_output_tiles` and the writer's (sharded, all-local) accessor.
    - CBs: `cb_act_shard` (index 5) backed on A's shard (zero-copy); `cb_act_tiles` for peer K-slices; `cb_weight_tiles` / `cb_partials` / epilogue rows as R1.
    - Exchange: one `ttnn.Mcast` channel, one group per grid row, `McastRotatingSenderConfig()` (senders rotate in column = K order). Knob `BS_ROW_MCAST` (default True); False = the unicast stepping stone.
  - **Reader** (`block_sharded` branch): per K-block, B's column slice read from DRAM first (R1's per-tile reads, factored into `read_weight`), then the A round. Own turn: push the resident shard rows into `cb_act_shard` (no NoC). R3a: send them from L1 to the row peers' reserved `cb_act_tiles` slot. Peer turn: `receive(kb)`, or (unicast) read the peer's shard tiles through A's sharded accessor (`read_act`). One barrier per K-block. Under the multicast every core also reserves / pushes its own round's slot so the row's CB write pointers stay in lockstep.
  - **Compute**: `OwnShardSource` (`In0SourceFn`) selects `cb_act_shard` for K-block `own_kb`; `RetireOwnRoundSlot` (`PostKBlockFn`) pops the own round's lockstep slot under the multicast. Other regimes pass `own_kb = NO_OWN_KB`.
  - **Reused**: R1 reader A/B read loops (now lambdas shared by R1 and R3), compute kernel and helpers, the writer (in place or accessor), `_reader_ct` / `_writer_ct` / `_compute_ct`, `_footprint`, `choose_subblock`, `l1_cb_budget`, the `ttnn.Mcast` attachment. **Added**: the R3 descriptor branch, topology / blocking / default output config, reader R3 branch (3 CT args, 1 RT arg), the two compute functors (2 CT args, 1 RT arg).
- Accuracy achieved:
  - PDL res4 reduce 8192×512×256 bf8b LoFi bias_relu: PCC 0.99984, rel-RMS 0.021 (gate 0.999 / 0.06).
  - PDL res5 shortcut 2048×1024×2048 (ragged last grid row: 2048 = 9×224 + 32): PCC 0.99988, rel-RMS 0.017.
  - PDL ASPP projection 2048×1280×256 (5-tile K shards): PCC 0.99984, rel-RMS 0.021.
  - cartesian 384×1536×1024 bf16 HiFi4: PCC 0.9999985, rel-RMS 0.0017 (gate 0.9999 / 0.01).
- Golden test progress:
  - `test_golden.py -k block_sharded`: 311 passed, 150 xfailed, 0 failed, 0 XPASS. That is 300/300 single-weight cartesian cells plus all 11 PDL loose cases (res4 / res5 / ASPP 1×1 / ASPP pool / ASPP projection / res4 shortcut). The xfails are 60 cells on the permanent fp32 × 16-bit-DEST EXCLUSION and 90 batched-weight cells (`weight_batch=batched` is not supported yet; Refinement 7).
  - Regression net: unit files `test_genify_matmul`, `_epilogue`, `_extended`, `_mcast`, `_dram_sharded`: 84/84 pass.
- Perf (device ns, PDL configs bf8b LoFi, one fresh run each; perf guard `test_perf_block_sharded`):

  | PDL case | row mcast (default) | unicast stepping stone | stock |
  |---|---|---|---|
  | res5 shortcut 2048×1024×2048 | 79.8 µs | 91.4 µs | 72.4 µs |
  | res4 reduce 8192×512×256 bias_relu | 18.7 µs | 45.8 µs | 16.6 µs |
  | res5 expand 2048×512×2048 | 45.6 µs | 48.0 µs | 24.2 µs |
  | ASPP 1×1 2048×2048×256 bias_relu | 19.6 µs | 52.5 µs | 13.8 µs |

  - The rotating row multicast is 1.05–2.7× faster than unicast, so it is the default.
  - Finding (not done here; the heading scopes B to per-core DRAM reads): the wide-N shapes are B-bound. B crosses DRAM once per grid row (res5 expand: about 11 MB, 10 rows). The next lever is R3a's B column multicast: B is read once per grid column by a fixed or staggered injector and multicast down the column, as in R1a's `weight` line channel. The per-K-block A/B ordering is the same as R1a's, so it stays deadlock-free.
  - Finding: 1-tile K shards (res4 expand, K = 256 on 8 columns) make every round a 1-tile-wide K-block, which is latency-bound per handshake.
- Issues encountered:
  - The rotating sender sits inside its own receiver group, so its multicast loopback-writes its own `cb_act_tiles` slot. Solution: a lockstep slot on the own round, reserved and pushed by the reader and retired unread by compute.
  - The 16-bit-DEST in-DEST K-chain cap (`DEST16_BLOCK_K_MAX`) is not applied to R3, because `block_k` is the shard width. Every golden cell has `shard_k ≤ 8`.
- Tests added:
  - `test_genify_matmul_block_sharded.py` (12 cases):
    - cartesian-shaped cells across dtypes and epilogues;
    - PDL res5 with a ragged last grid row;
    - ASPP pool (M = 1, 8×1 grid, bf16 → bf8b);
    - a ragged last K shard together with a grid column that owns no output columns;
    - a single K shard (no exchange);
    - unicast vs multicast bit-identical;
    - the default output placement;
    - the M-split and the N-split streamed-output fallbacks under a forced L1 budget.
  - Perf guard `test_perf_block_sharded`: 4 PDL representatives × {mcast, unicast}.

## Refinement 6 — Speed up PERF FOCUS 2 (Qwen3.8 decode MLP down)
- Date: 2026-11-17
- What was done:
  - **Bottleneck first.** Ablation of the Refinement 4 build (42.8 µs) on PERF FOCUS 2 (1×4352 @ 4352×5120 bf4b, LoFi, fp32 DEST): compute stubbed 29.7 µs, B stubbed 40.0, B + A stubbed (compute only) 38.7, all stubbed 10.0 (the A rounds). One worker per bank did 136 × 20 tile-matmuls: **compute-bound**, not B-bound as the R4 finding assumed.
  - **Lever 1 — `cores_per_bank` split** (`DS_CORES_PER_BANK`, auto through `DS_CORE_N_TILES_MIN = 4`). `ds_cores_per_bank()` picks the largest C dividing the bank width into ≥ 4-column slices (whole swiglu pairs) with C · banks ≤ grid; `_ds_bank_cores()` assigns bank w's optimal worker plus its C − 1 nearest free cores. Each worker streams its slice of every K row (`ds_row_reads`: one read per K row; a slice wholly in the last bank's padding reads nothing), computes `block_n = sw / C`, and streams its output through `cb_output_tiles` + the writer's sharded accessor into the bank's output shard on the primary worker. C = 1 keeps the in-place Refinement 4 path. The op-built output placement is unchanged (`primary_ranges`). PF2: C = 5, 40 workers: 42.8 → 34.8 µs (compute-only 10.6 µs; DM-only 32.9 µs now binding).
  - **Lever 2 — A rounds on the writer** (`DS_ACT_ON_WRITER = True`, the `split_reader` idea). Ablation after lever 1: B + compute 30.1 µs, A + compute 14.2 µs, full 35.0 µs — the rounds sat between the reader's B issue and its barrier. The round implementation is now one shared header (`kernels/genify_matmul_ds_act.hpp`: `ActKBlocks`, `worker_round`, `send_only`) used by the reader (knob False) or the writer (knob True, BRISC / NoC1, Mcast `act` channel on `DS_ACT_WRITER_NOC`). The writer produces `cb_act_tiles` per block row, then drains that row's output; non-worker broadcast cores run only the writer. 35.0 → 28.9 µs.
  - **Lever 3 — trid B ring** (`DS_B_AHEAD`, `DS_DEPTH_WEIGHT`, `trid_double_issue`): the R4 reader's B stream became a ring of T K-blocks in flight, each retired by `async_read_barrier_with_trid` (`ds_weight_depth()` sizes `cb_weight_tiles`). Measured T = 1/2/3 at depth 4: 28.9 / 28.9 / 29.0 µs (writer-side A), and +3% on gate_up at T ≥ 2; with reader-side A T ≥ 2 was 3–5% slower. Five slice readers per bank already saturate it (B alone 27.9 µs vs the 27.2 µs roofline), so the knob is parked at T = 1 / depth 2 (no L1 cost) and kept live (unit-tested at T = 2, 3).
  - **Reused**: the R4 descriptor branch, topology, blocking solve, `_footprint` (new `depth_weight` term), the writer's accessor path (R3 fallback) for split slices, the compute kernel unchanged. **Added**: topology fields `primaries` / `cores_per_bank` / `core_n_tiles` / `out_in_place`, `ds_cores_per_bank`, `_ds_bank_cores`, `ds_weight_depth`; reader CT args 27–30 (`ds_row_reads`, `ds_weight_depth`, `ds_b_ahead`, `ds_reader_act`; R3 and accessor CT args shifted); writer CT args 8–15 (the A-round role) and RT arg 12 (A address) under `ds_act`; R1 / R3 writers attach the `act` channel absent.
- Accuracy achieved (LoFi, fp32 DEST, bf16 × bf4b; gate 0.9995 / 0.03):
  - PERF FOCUS 2 1×4352×5120: PCC 0.999993, rel-RMS 0.0050.
  - gate_up 32×5120×9216 swiglu: PCC 0.999984, rel-RMS 0.0088.
  - LM head 1×5120×16384: PCC 0.999993, rel-RMS 0.0049.
- Golden test progress:
  - `test_golden.py -k dram_sharded` (cartesian): 310 passed, 150 xfailed (the permanent EXCLUSIONs).
  - `test_op_loose -k dram_sharded`: 10/10 (Qwen decode, LM head); `-k "block_sharded or PERF FOCUS 1"`: 12/12.
  - `test_translated.py` DRAM-sharded (`served_shapes_blackhole`, `cores_per_bank`, `program_cache_reuses`): 57/57.
  - Unit regression net (`test_genify_matmul`, `_epilogue`, `_extended`, `_mcast`, `_dram_sharded`, `_block_sharded`): 98/98; `_dram_sharded` alone 15/15 after the knob cases were added.
- Perf (device ns, one fresh run each; whole perf guard set new vs the Refinement 5 commit):

  | case | before | after | stock |
  |---|---|---|---|
  | PERF FOCUS 2 1×4352×5120 | 42.9 µs | **29.2 µs** | 36.9 µs |
  | LM head 1×5120×16384 | 148.5 µs | 105.8 µs | 127.0 µs |
  | gate_up 32×5120×9216 swiglu | 103.4 µs | 63.1 µs | 64.0 µs |
  | cartesian 384×1536×1024 (C = 1) | 137.0 µs | 137.0 µs | — |

  Every non-R4 guard row is within ±1% except block_sharded res5 shortcut (+3.0% in the sweep; re-measured 3× each: 79.1 vs 78.5 µs median, +0.8%, noise).
  - Bottleneck now: the B stream (compute-stubbed 28.8 µs ≈ full; B alone 27.9 µs, ~97% of the DRAM roofline). Remaining ~1.5 µs is fixed cost (first K-block's exposed B latency, the output tail); next lever would be a smaller first K-block so compute starts sooner. Not done: within ~6% of the roofline.
- Issues encountered:
  - A non-template `static_assert` inside a discarded `if constexpr` branch still fires (moved the condition into a comment).
  - `run_safe_pytest.sh --profile` with a `-k` containing spaces is split by the Tracy wrapper; select by test id instead.
  - The remote JIT precompile fails on this box (`__xtt_vector8`); it is best-effort and the tests compile locally.
- Tests added:
  - `test_genify_matmul_dram_sharded.py::test_cores_per_bank_split` (4 knob sets on N = 704 with bias: C = 3 with padding-only slices / C = 1, A rounds on writer / reader, B ring T = 1 / 2 / 3).
  - `test_plan_blocking` now pins the PERF FOCUS 2 split (C = 5, 4-column slices, 40 distinct workers, primaries first).

## Refinement 7 — height_sharded placement + batched weights (local shards, independent batch split)
- Date: 2026-11-17
- What was done:
  - **height_sharded = regime R2, built as R3's single-K-shard case.** `height_sharded_topology()` validates A (L1 HEIGHT_SHARDED, whole-tile shards holding all of K) and the requested output (L1 HEIGHT_SHARDED on A's cores, A's shard height and orientation, the whole output width; `height_sharded_output_memory_config()` is the default when `memory_config` is None). It returns a `BlockShardedTopology` with `cols = 1`, `shard_k = K` and an explicit `core_list` (the shard cores in shard order through `ttnn.corerange_to_cores` with the shard orientation), so any core set works, e.g. PDL's 103 cores = 9 full grid rows plus a partial one. The R3 descriptor (`create_block_sharded_program_descriptor`, now `placement`-aware) runs it unchanged otherwise.
    - **A is consumed in place.** With a single K shard (`topo.act_resident`) the compute kernel's A operand is `cb_act_shard` itself (compute CT arg 0), backed on the resident shard by `cb_descriptor_from_sharded_tensor`. The reader only `cb_push_back`s it, with no NoC read and no TensorAccessor on the own shard, and `cb_act_tiles` is not allocated (`_footprint(..., act_cb=False)`).
    - **Block.** `shard_m_tiles × out_n` with `block_k = tensor_k_tiles` (one K-block). The output is packed in place into `cb_output_shard`. Fallback when B's whole-K × whole-N block does not fit L1: the shard rows with N split, the output streamed through `cb_output_tiles` and the writer's local accessor writes.
    - **`cb_weight_tiles` depth** is 1 when the core's whole walk is one K-block (`_weight_depth`): the second slot could never fill. This frees L1 for the coarser in-place block.
  - **Batched weights = regime R1d.** Interleaved A needed no kernel change: `grid_b > 1` in the factorization (flat placement), `weight_batch_stride` in the reader, `MatmulBlockShape.batch = block_batches` were all already plumbed. SUPPORTED and the gate are widened; B must be interleaved TILE (the existing gate).
    - On R2 / R3, `ShardRows` / `shard_rows()` maps each A shard onto (batch, rows) so that no block straddles two batches. Either the shard lies inside one image (`img_m % shard_m == 0`: per-shard b0 / row0), or it holds whole images (`shard_m % img_m == 0`: `block_batches = shard_m / img_m`, `block_m = img_m`, pushed image by image). Any other shard height is refused through validate().
    - The reader skips B reads for a shard's padded batch slots.
  - **Reused**: R3's topology / blocking solve / descriptor / reader `block_sharded` walk / compute `In0SourceFn` path / writer in-place and accessor paths, R1's batch walk. **Added**: `BlockShardedTopology.core_list` / `core_set` / `act_resident`, `height_sharded_topology`, `height_sharded_output_memory_config`, `ShardRows` + `shard_rows`, `_weight_depth`, `_footprint(act_cb=)`, `plan_block_sharded(placement=)`; one `batch_valid` guard in the reader; op-file `_validate_height_sharded`, `_check_shard_batches`.
- Accuracy achieved (PCC / rel-RMS against gate):
  - PDL HS res2 shortcut 32768×128×256 bias (bf8b, LoFi): 0.99982 / 0.0169 (gate 0.999 / 0.06).
  - PDL HS res2 reduce bias_relu: 0.99916 / 0.0175.
  - PERF FOCUS 3 (batched, bf8b, 16-bit DEST): 0.99939 / 0.0172 (gate 0.997 / 0.15).
  - Batched (1,4,128,128)×(1,4,128,256) bf16: 1.000000 / 0.0017 (fp32 DEST), 0.99998 / 0.0064 (16-bit) on HS, interleaved and block_sharded alike.
  - HS M=1 K=2048 with 16-bit DEST: 0.99964 / 0.039. The one-K-block walk keeps a 64-tile chain in DEST; the gate is 0.15.
- Golden test progress:
  - Cartesian `height_sharded`: 450 passed, 90 xfailed (the permanent fp32 × 16-bit-DEST EXCLUSION).
  - Cartesian batched on interleaved + block_sharded: 150 passed. The batched dram_sharded cells xfail as the permanent EXCLUSION.
  - Loose: 12/12 PDL height-sharded cases plus PERF FOCUS 3.
  - Translated `-k batched`: 2/2 (true bmm cell, batched row-vector bias). `-k "sd_matmul or bert"`: 15 passed, as before.
  - Unit regression net (`test_genify_matmul`, `_epilogue`, `_extended`, `_block_sharded`): 78/78.
- Perf (device ns, one fresh run each; new perf-guard rows `test_perf_height_sharded`, `test_perf_batched`):

  | case | genify | stock |
  |---|---|---|
  | PDL res2 shortcut 32768×128×256, 103 cores | 14.6 µs | 9.8 µs |
  | PDL res3 expand 8192×128×512, 86 cores | 18.7 µs | 8.2 µs |
  | PDL res3 reduce 8192×512×128 bias_relu | 20.2 µs | 6.9 µs |
  | PDL res2 reduce 32768×256×64 bias_relu | 9.7 µs | 5.0 µs |
  | PERF FOCUS 3 batched 32×512×128×1024 | 70.8 µs | 123.0 µs |

  - Finding: R2 is bound by B's DRAM traffic. Every shard core reads the whole K×N B, so res3 reduce moves ≈ 6 MB, ≈ 13 µs at 460 GB/s, against 20.2 µs measured. The fix is the deferred R2a `height_sharded_mcast_b` (B once from DRAM, multicast to the shard cores), which the verifier keeps as a report note (no PDL height-sharded case is perf-flagged).
  - PERF FOCUS 3 already beats `achievable_ns`. That is Refinement 8's baseline.
- Issues encountered: None. Every new case passed on its first device run.
- Tests added:
  - `test_genify_matmul_height_sharded.py`, 15 cases:
    - cartesian-shaped HS cells across dtypes and epilogues;
    - PDL HS on 103 and 86 cores (ragged last shard, bf16 → bf8b output);
    - batched B with one image per shard, and with whole images per shard (ragged, plus addcmul);
    - batched B on block_sharded swiglu, and on interleaved none / swiglu / addcmul with bf8b and 16-bit DEST;
    - COL_MAJOR orientation together with the default output placement;
    - the N-split fallback under a forced L1 budget;
    - the shard → batch mapping rules.
  - Perf guard: `test_perf_height_sharded` (4 PDL shapes) and `test_perf_batched` (PERF FOCUS 3).

## Refinement 8 — Speed up PERF FOCUS 3 (PDL bilinear upsample, H pass: batched weight)
- Date: 2026-11-17
- What was done:
  - **Bottleneck first.** Ablation of the Refinement 7 build on PERF FOCUS 3 (32 × 512×128 @ 128×1024 bf8b, B in L1, LoFi, 16-bit DEST; split 32×1×3, one 16×12 block per core, one K-block):

    | run | time |
    |---|---|
    | full | 70.5 µs |
    | compute stubbed | 66.8 µs |
    | reads stubbed | 62.7 µs |
    | writes stubbed | 30.8 µs |
    | write-only | 59.9 µs |
    | read-only | 20.3 µs |
    | compute-only | 10.9 µs |
    | all stubbed | 0.7 µs |

    So the output DRAM write stream (17.8 MB) binds. The reads (A re-read 3× from DRAM, 6.6 MB, plus B 4.4 MB from L1) sit before compute as a whole-region prologue and contend with the writes for DRAM. Write-only time does not move with core count (32 / 64 / 96 cores: 62 / 58 / 59–60 µs): it is a DRAM-side limit of ≈ 300 GB/s for 1088 B interleaved tile writes.
  - **Lever 1 — operand residency** (`RESIDENT_SPLIT_BLOCKS = 4`, `_resident_split`). A one-block, one-K-block S1 walk that K cannot split is split along ONE axis. The smaller per-core operand stays fronted in its CB across the blocks: matmul_block `WaitAndRetainOnLastBlock` on it, and the compute kernel pops it after its last use, the lifecycle that policy hands the caller.
    - `act_resident`: A held per M-block across its N-blocks; B streams.
    - `weight_resident`: B held across the M-blocks; A streams.

    The reader reads and pushes, or line-delivers under R1a, the resident operand only on its first block. Every core of an R1a line walks the same nominal sequence, so the skipped deliveries match. The split is refused when it would drop the R1a multicast (a smaller K-block under `MCAST_MIN_KBLOCK_BYTES`: 2048×256×512 bf8b had measured +11%). N-block width sweep on PERF FOCUS 3 (32×2×1): 16 / 8 / 4 columns = 74.9 / 65.0 / 68.5 µs, so 4 blocks.
  - **Lever 2 — L1-discounted split ranking** (`L1_READ_COST = 0.25`, `Geometry.a_in_l1` / `b_in_l1`). The ranking used to count L1-sourced bytes like DRAM bytes, so PERF FOCUS 3 re-read A from DRAM 3× to save L1 B reads. Discounted, it picks 32×3×1 with A resident: A crosses DRAM once, and B is re-read from L1 over NoC0, overlapped with the write stream.

    | split / residency | time |
    |---|---|
    | 32×1×3 none (old) | 69.7 µs |
    | 32×1×3 weight | 70.2 µs |
    | 32×1×3 act | 72.5 µs |
    | 32×3×1 none | 107 µs |
    | 32×3×1 weight | 103 µs |
    | 32×3×1 act | 65.2–65.9 µs |
    | 32×2×1 act | 65.0 µs |
    | 32×1×1 act | 68.7 µs |

    Residency is also counted in `read_traffic` (a resident operand is read once per core).
  - **Lever 3 — output write batching** (`OUT_WRITE_BATCH_ROW_GROUPS`, `Blocking.write_batch`, writer CT arg 16). The writer drains several row groups per wait / flush / pop, and `cb_output_tiles` holds `DEPTH_OUT` batches. Measured 65.0 / 65.7 / 66.0 µs at 1 / 4 / 8: the writer never stalled on its flushes. It is parked at 1 (byte-identical CB size) as a live knob, unit-tested at 4.
  - **Measured and not kept:**
    - two batches per core (`block_batches` co-tune; 16×3×1 / 16×2×1 / 16×1×3): 81.6 / 77.9 / 71.6 µs;
    - writer on NoC0: 107 µs;
    - output bank-stagger (write-only −3.5%, full within noise), N-block start rotation per core, posted writes: all within noise;
    - deeper A (whole region) and B CBs: within noise.
  - **Reused**: R1's blocking solve (S1 branch), `read_traffic`, the reader's R1 / R1a K-block path, compute `run_matmul` (its input policies are now CT-selected), the writer's row-group loop. **Added**: `Blocking.weight_resident` / `act_resident` / `write_batch`, `_resident_split`, `choose_blocking` (write-batch wrapper around `_choose_blocking`), `Geometry.a_in_l1` / `b_in_l1`; reader CT args 20–21 (later args shifted by 2), compute CT args 22–23 and R1 RT arg 2 (`nb_n`), writer CT arg 16 (accessors shifted by 1).
- Accuracy achieved (PCC / rel-RMS):
  - PERF FOCUS 3: 0.99892 / 0.0172 on 32×3×1 with A resident (gate 0.997 / 0.15).
  - 2048×128×3072 bf16 fp32 DEST: 0.99989 / 0.026.
  - Ragged 300×96×320: 0.99990 / 0.026.
- Golden test progress: `test_golden.py -k "(loose and interleaved) or (not loose and interleaved and (A1x1x2048x256 or batched))"`: 204 passed, 30 xfailed (the expected EXCLUSIONs). Unit net: `test_genify_matmul`, `_mcast`, `_extended`, `_epilogue` 73/73; `_height_sharded`, `_block_sharded`, `_dram_sharded` 42/42; new `_residency` 14/14.
- Perf (device ns; median of 5 for the first two rows, median of 3 for the rest):

  | case | before | after |
  |---|---|---|
  | PERF FOCUS 3 (batched upsample H pass) | 70.3 µs | **65.2 µs** |
  | PDL upsample W pass (A / out in L1) | 15.6 µs | **12.7 µs** |
  | 2048×256×512 bf16 HiFi4 / HiFi2 | 15.3 / 14.7 µs | 13.6 / 12.6 µs |
  | 2048×128×3072 bf16 HiFi4 / HiFi2 | 55.0 / 51.0 µs | 52.0 / 49.0 µs |
  | 2048×256×512 bf8b LoFi 16-bit DEST (multicast gate) | 8.89 µs | 8.82 µs |

  The rest of the guard set is unchanged in plan; one run each:
  - PERF FOCUS 1: 465.5 µs.
  - PERF FOCUS 2: 29.7 µs.
  - Block-sharded res5 expand: 45.8 µs.
  - 1×6144×6144: 316.3 vs 321.0 µs.
  - 4096³: 1293 vs 1294 µs.
  - L1 384×1536×1024: 31.1 vs 30.8 µs.
  - HS res3 expand: 18.7 vs 18.9 µs.
  - 32×1024×2048: 24.4 vs 23.9 µs (+2.1%, noise band).
  - Bottleneck now: the interleaved DRAM write stream. Write-only is 59–61 µs; full is 65 µs, which is that floor plus ~5 µs of read prologue (each core reads its resident A and first B block before its first write). A next lever would be a B multicast inside each batch's 3-core group (B is re-read 3× from L1, 13 MB on NoC0) or a staggered start. Not done: it is a scheme change, and the gap is ~8% above a measured write floor.
- Issues encountered:
  - The R1a multicast gate interacts with any block-shrinking lever. A split that drops the line multicast must be refused; added as an explicit check.
  - Device-time noise on this write-bound shape is ±5% run to run, with occasional +10% outliers, so every decision here used medians of 3–5 runs.
- Tests added:
  - `test_genify_matmul_residency.py`, 14 cases:
    - the PERF FOCUS 3 split under both `L1_READ_COST` settings;
    - weight / act residency on R1a, with every epilogue family and bf8b swiglu;
    - the L1 W pass;
    - ragged M;
    - residency off;
    - the write-batch knob at 4 (with none and swiglu epilogues).
  - Perf guard `test_perf_upsample_w_pass`.

## Refinement 9 — Speed up PERF FOCUS 4 (FLUX.2 prompt-stream FF1 + SwiGLU, 256 tokens)
- Date: 2026-11-17
- What was done: built regime **R1b `interleaved_stream_mcast_a`** and co-tuned it.
  - **Starting point.** PERF FOCUS 4 (A 1×256×6144 bf16 × B 6144×18432 bf16, HiFi2, fp32 DEST, swiglu) ran R1a on an 8×9 split (72 cores): B crossed DRAM once, but only the 9 column injectors read it, ~25 MB each, so those injectors bound the run at 1125 µs.
  - **R1b topology.** A 1-D N split (`grid_m' = 1`, 96 cores, block 8×6, 2×2 subblock). Every core streams its own B columns from DRAM on the reader (NCRISC / NoC0). ONE injector reads each A K-block and multicasts it to every core's `cb_act_tiles` slot. The broadcast runs on the **writer RISC** (BRISC / NoC1), which becomes `cb_act_tiles`' producer on every core (reserve → ack or send → push), so the A handshake never stalls the B stream. The compute kernel is unchanged. Wiring: reader CT 22 is now `operand_mode` (0 R1 / 1 R1a / 2 R1b); writer CT 8 is `act_mode` (0 / 1 R4 rounds / 2 R1b stream), plus CT 17–20 (last K-block, A page size, A residency, injector ahead) and A's accessor after the output's. One `ttnn.Mcast` channel: a fixed in-group sender; a flat split's partial last row makes the receiver set a second rectangle. 1125 → 683 µs.
  - **`Q_N_ROW_AWARE`.** On a 1-D split a core owns all `core_m` rows, so a subblock of height h (largest power of two ≤ 4 dividing `core_m`) needs only `4/h` columns. FF1: `core_n` 8 → 6, 72 → 96 cores, subblock 1×4 → 2×2.
  - **`ACT_STREAM_DEPTH = 4`** (operand rings in K-blocks on an R1b K-walk). Carried in `Blocking.depth_act` / `depth_weight`; the L1 solve and the CB sizes both read it there, and the solve shrinks `block_k` (20 → 10) to pay for it. Every core waits for each K-block's A round, so the slowest core's B read gates the next broadcast; depth gives that coupling slack. Measured FF1 683 / 666 / 633 / 634 µs at depth 2 / 3 / 4 / 5; depth 6 drops the K-block under the R1b gate.
  - **`ACT_STREAM_AHEAD = 2`** (injector stall-shadow). The injector reserves K-block kb+1's slot and issues its reads (read-trid ring) before multicasting kb. Effect: −3 to −4% on the injector-heavy shapes (context_embedder 362.6 → 349.8 µs, text-encoder down 761 → 732 µs); within noise on FF1 (median of 3: 638 at ahead=1, 645 at ahead=2).
  - **Ranking and gates** (`read_traffic`, `uses_act_stream`):
    - R1b counts A once and B per core, capped by the single injector's throughput: `max(A + B, ACT_STREAM_INJECTOR_COST · A)`, cost 18. The B stream runs at ~411 GB/s; one injector's A read + multicast at ~23 GB/s. Without the cap, text-encoder down 512×16384×5120 (A = 19% of B) went R1b at 760 µs against 552 µs under R1a.
    - R1b requires a weight-streaming shape (B bytes ≥ output bytes): the A rounds share the writer RISC with the output drains. Without the gate, x_embedder 2048×128×3072 went R1b at 62.8 µs against 48.5 µs.
    - The existing gates still apply: an A K-block of at least `ACT_STREAM_MIN_KBLOCK_BYTES` = 32 KB, single batch.
  - **Reused:** R1's split, blocking, CBs, compute kernel and block walk; the reader's R1 B path; the writer's block walk; `ttnn.Mcast` + `McastArgs`; the trid helpers. **Added:** `operand_mode()`, `uses_act_stream()`, `_attach_act_stream()`, `_row_subblock_h()`, `_stream_depths()`, the `Blocking` depth fields, the writer's A-stream role, 7 knobs.
- Perf (device ns; one fresh run each unless noted; old = `ACT_STREAM_MCAST = Q_N_ROW_AWARE = False`):

  | case | before | after | stock |
  |---|---|---|---|
  | **PERF FOCUS 4** FF1 256×6144×18432 swiglu | 1125.4 µs | **645 µs** (median of 3; 630–646) | 865.9 µs (roofline 508.6) |
  | its epilogue-none twin 256×6144×9216 | 558.9 µs | **321 µs** | 452.9 µs |
  | guidance linear_2 1×6144×6144 bf16 | 308.3 µs | 206 µs | 192.1 µs |
  | timestep linear_2 1×6144×6144 fp32 | 406.9 µs | 209 µs | 195.4 µs |
  | context_embedder 256×15360×3072 | 2225.8 µs | 351 µs | 405.1 µs |
  | attn out txt 256×6144×3072 addcmul | 887.7 µs | 144 µs | 193.2 µs |
  | attn out 128×6144×3072 addcmul | 526.4 µs | 119 µs | — |
  | FF2 txt 256×9216×6144 | 569.5 µs | 323 µs | 456.2 µs |
  | text-encoder qkv 512×5120×3072 bf8b | 1314.2 µs | 232 µs | 139.2 µs |
  | time_mod 1×6144×52224 fp32 + bias | 1819.9 µs | 1601 µs | 2310.7 µs |
  | timestep linear_1 1×256×6144 fp32 | 19.6 µs | 10.0 µs | 13.1 µs |
  | Qwen prefill-128 gate_up (A / out in L1, bf4b) | 392.9 µs | 114 µs | — |
  | cartesian 32×1024×2048 / 1×2048×1536 | 23.8 / 36.9 µs | 12.9 / 19.7 µs | — |
  | x_embedder 2048×128×3072, text-encoder down 512×16384×5120 | 48.5 / 552 µs | 47.5 / 552 µs (plans unchanged) | 62.9 / 508.6 µs |

  Guard set, plans unchanged (writer / reader CT layout only): PERF FOCUS 1 465.3 µs (465.5), PERF FOCUS 2 29.6 (29.7), PERF FOCUS 3 65.5 (median of 5; 65.2), BS res5 expand 46.1 (45.8), HS res3 expand 18.6 (18.7), 4096³ 1292.5 (1293), W pass 12.8 (12.7), L1 384×1536×1024 31.2 (31.1).
- Bottleneck now (FF1, ablation with hooks in `agent_logs/refinement9_ablation_hooks.patch`):
  - Measured times (depth 4):
    - full: 633 µs
    - compute stubbed (B stream + A broadcast): 578 µs
    - B stream alone, A and compute stubbed: 550 µs (226 MB at ≈ 411 GB/s, interleaved 2 KB tile reads)
    - compute alone: 274 µs
    - everything stubbed: 16 µs
  - So the run is **B-DRAM-stream-bound**; compute overlaps it within ~55 µs.
  - The A lockstep costs ~28 µs on the bare B stream, but helps once compute runs: with compute on and A stubbed, the run takes 744 µs against 688 µs, because the cores drift apart and lose DRAM-row locality.
  - Same-bank coalesced B reads (perf-only ablation) were slower: 627 µs against 578 µs.
- Accuracy achieved (PCC / rel-RMS, HiFi2, fp32 DEST):
  - FF1 swiglu: 0.999990 / 0.0084
  - twin: 0.999983 / 0.0046
  - guidance: 0.999995 / 0.0047
  - timestep fp32: 0.999994 / 0.0067
  - context_embedder: 0.999986 / 0.0046
- Golden test progress: `test_golden.py -k "(loose and interleaved) or (not loose and interleaved and (A1x2048_ or A1x1x32x1024))"`: 204 passed, 30 xfailed (the expected EXCLUSIONs), 0 failed. Unit net (whole `tests/.../genify_matmul/`): 687 passed, 0 failed.
- Issues encountered:
  - The R1b plan change touched every 1-D-split interleaved shape. The plan diff and the per-shape A/B found two regressions, fixed by the injector cost and the weight-streaming gate.
  - `run_safe_pytest.sh --profile` cannot pass a multi-word `-k` through tracy, so the perf file takes a `GENIFY_MATMUL_ONLY` id filter.
  - A perf-only ablation issuing reads on NoC1 from NCRISC hung, as expected: NoC command buffers belong to one RISC. A real split reader needs BRISC to fill half of each B slot under a local handshake.
- Tests added:
  - `test_genify_matmul_stream_mcast.py` (13 cases):
    - R1b selection + correctness on PERF FOCUS 4, M = 1, ragged M / N, a one-row rectangle and every epilogue family
    - fp32 A
    - A in L1
    - a multi-N-block walk (none / swiglu)
    - A residency
    - each knob off
  - `perf/test_genify_matmul_perf_flux.py`: the 4 FLUX cases plus 12 R1b guard cases, with `GENIFY_MATMUL_KNOBS` / `GENIFY_MATMUL_ONLY`.
  - `test_genify_matmul_residency.py` act-resident R1a cases: unchanged and passing again (x_embedder keeps R1a under the weight-streaming gate).
- Next (not done):
  - The B stream itself: 550 µs vs the 508.6 µs roofline. Interleaved 2 KB tile reads from all banks on NoC0 reach ~411 GB/s. A split reader would put half of B on BRISC / NoC1 under a local slot handshake, since the CB still has one producer. It is a structural change, and the gain depends on NoC0 link saturation, which was not measured.
  - The ~55 µs of compute / stream non-overlap at the ends of the walk.
  - A rotating injector for A-heavy shapes, so A reads spread over cores. The helper's rotating mode needs a rectangular receiver set, so it would mean a bounding-rectangle landing zone as in R4.

## Perf 1 — perf tournament round 1 (PERF FOCUS 1: Qwen3.8 prefill gate_up + SwiGLU, 2048 tokens)
- Date: 2026-11-17
- **Focus config (exact golden case):** A 1×2048×5120 bf16 @ B 5120×8704 bfloat4_b, LoFi, fp32_dest_acc_en=True,
  math_approx_mode=False, DRAM interleaved, swiglu, bf16 out. Every knob is in SUPPORTED. Plan: R1a, rect_t 11×10,
  one block per core of 6×28 B-tiles (84 output tiles), subblock 1×4, 14 K-blocks of 12 (last 4).
- **Result: 466.4 → 418.4 µs (−10.3%)**, PCC 0.999917 → 0.999919, rel-RMS 0.00846 → 0.00829. Now below the
  #65148 roofline (428.7 µs) and stock (564.5 µs). All 3 ideas were measured: 2 graduated, 1 null. No regression.
  SUPPORTED is unchanged.

### Instrumentation (permanent)
- `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` is restored as-is from the shared branch
  (`MaybeDeviceZoneScope` = `DeviceZoneScopedN`; compiled out when the profiler is off).
- Zones (one per question; per K-block at most, so the 250-marker budget covers about 25 K-blocks per core):
  - reader: `rd_reserve` / `rd_issue` / `rd_barrier` / `rd_mcast_a` / `rd_mcast_b` (R1 / R1a, per K-block);
    `rd_ds_kblock` (R4) and `rd_bs_kblock` (R3 / R2), per K-block; `rd_row`, `rd_residual`;
  - compute: `cmp_matmul` and `cmp_epilogue`, per block;
  - writer: `wr_wait_out`, `wr_write`, `wr_final_barrier`, `wr_ds_round` (R4), `wr_a_inject` / `wr_a_receive` (R1b).
- The focus run peaks at 144 markers per RISC (under the cap).
- Measurement cost: with the profiler on, the zones add about 0.3 µs on 9–13 µs kernels, which shows up as
  +3 to 5% there. With the zones compiled out those cells are flat (see the guard set below). On the focus shape
  the cost is under 0.2%.
- Tooling:
  - focus bench `tests/.../genify_matmul/perf/test_genify_matmul_perf_focus.py` (`GENIFY_MATMUL_PERF=1`,
    `GENIFY_MATMUL_CHECK=1`);
  - zone parser `agent_logs/perf1_zones.py`;
  - ablation hooks in `agent_logs/perf1_ablation_hooks.patch`, a `GENIFY_ABLATE` env var taking
    reads / mcast / writes / matmul / epi. Perf-only; never commit it applied.

### Measured breakdown (focus, before; device ns, one fresh run each)
- Ablation, peeled cumulatively:

  | run | µs |
  |---|---|
  | full | 466.4 |
  | writes stubbed | 456.8 |
  | swiglu chain stubbed | 428.5 |
  | chain + writes | 370.1 |
  | chain + writes + reads + mcast (matmul only) | 354.8 |
  | everything stubbed (sync / pack-L1-acc floor) | 117.8 |
  | compute only, with the epilogue | 441.5 |
  | data movement only | 284.6 |

- Zones (p50 over 110 cores, sums):
  - compute: `cmp_matmul` 369 µs, then `cmp_epilogue` 85–87 µs;
  - writer: `wr_write` 32 µs (max 58), all at the tail;
  - reader: the kernel ends at 344 µs, after 111 µs blocked in `rd_reserve`.
- **Ranked:**
  1. **Post-matmul tail, ~96 µs.** The swiglu epilogue's own compute is 87 µs, about 1400 cycles per output tile,
     serialized on the math thread. The 17.8 MB output write stream (~58 µs when exposed) is bunched at the end,
     because every core has one block.
  2. **Matmul K-walk.** 354.8 µs against the 318.6 µs FPU floor (168 × 160 tile-K × 16 cycles), plus ~15 µs of
     exposed head read.
  3. **Reader.** Not on the critical path: DM-only is 284.6 µs, and the reader waits on full CBs for 111 µs.
     Roofline gate: no reader-bandwidth idea was floated.

### Portfolio and verdicts (artifacts under `perf_experiments/<slug>/` + `tests/.../genify_matmul/perf_experiments/<slug>/`)
| idea | verdict | before → after (device ns) | domain |
|---|---|---|---|
| `swiglu_epilogue_sfpu`: fused one-pass silu(g)·u SFPU element | **WIN** | isolated focus epilogue 85.3 → 61.8 µs; fp32 out 116.9 → 75.6 µs | every swiglu path (R4 natural pairs 86.6 → 63.7, 16-bit DEST 117.1 → 62.7, bf8b out 85.5 → 61.7); no exceptions; Blackhole only (SFPARECIP) |
| `matmul_kwalk_efficiency`: raw K-walk, DEST advance by address modifier, one partials reserve / push | **WIN** | whole op 466.3 → 441.5 µs; isolated 353.9 → 330.6 µs (16.61 cycles per tile-K, was 17.77); bit-identical | faster on every config measured: HiFi2 −6.3%, HiFi4 −3.3%, 16-bit DEST −7.0%, block_k 8, bfp8 B, 4×8 block |
| `progressive_output_multiblock`: nb > 1 blocks per core so writes overlap the K-walk | NULL | focus 466.6 → 473.4 µs (N2, operand depth 4; flat). Only with the epilogue stubbed: 429.8 → 410.1 | measured regressions: 2048×5120×4160 +20% (N2) / +5.5% (M2), 2048×6144×3072 addcmul +11%, R1b FF1 +5% (N3), focus M2 +5%, M3 +26%. Not graduated |

- **Swiglu menu, also measured** (isolated ns; precision in bf16 ULP):
  - stock-math fusion: 74.3 µs;
  - degree-2 polynomial: 56.2 µs, max ULP 2.72 against 2.39 (baseline) and 2.03 (graduated);
  - `UnpackToDestFp32` on `cb_partials`: 72.0 µs, max ULP 0.53;
  - no-Newton: 68.1 µs, worse precision than baseline.
  - The graduated bf16 grade is the fastest option that is at least as precise as the old chain on every metric.
- **Swiglu, recorded options not graduated:**
  - `UnpackToDestFp32`: a precision gain at +10 µs. The tf32 truncation of the fp32 partials through srcA
    dominates the residual error.
  - The bf16-grade element on fp32 outputs: −14 µs, but +1.5% mean error.
- **Progressive split:** the epilogue already hides all but ~10 µs of the write stream. The split forces a 2×2
  subblock, which costs about as much as it saves. An M-split re-delivers bf4b B and makes the reader the
  bottleneck. It is worth revisiting only if the tail becomes write-bound.

### What graduated (one path; replaced code deleted)
- **`SwiGluMul`** (`kernels/genify_matmul_swiglu_sfpu.hpp`) is the swiglu epilogue's only SFPU step on every
  regime.
  - The precision grade is the existing `silu_fp32_precision && DST_ACCUM_MODE`: fp32 grade for fp32 output on a
    32-bit DEST, bf16 grade otherwise.
  - The local `Silu` element and `MulBinary` in the swiglu chain are deleted. No carve-out.
- **`kw::matmul_accumulate_interm`** (`kernels/genify_matmul_kwalk.hpp`) is the K-walk of every epilogue block
  (LastBlockTarget::Interm).
  - The narrow carve-out, which keeps the helper, covers only what the raw walk cannot express:
    - sb_h > 1 subblocks (e.g. R1b's 2×2 on PERF FOCUS 4);
    - operand residency (`WaitAndRetainOnLastBlock`).
  - The epilogue-none path (LastBlockTarget::Out with the partials reload) also stays on the helper. The raw
    walk has no reload, so it is inexpressible there as written. Extending the walk to that path is the next
    lever: it is the same SETC16 cost on every epilogue-none matmul.
- The zones are extended to the new path (`cmp_matmul` wraps both walks).

### Whole op before / after and guard set
- Focus: **466.4 → 418.4 µs** (zones compiled out: 417.4 µs).
- Guard set: the full `perf/test_genify_matmul_perf_guard.py` (62 cells: every regime × layout × placement
  representative) and `perf/test_genify_matmul_perf_flux.py` (16 cells), new vs the pre-round kernels, one fresh
  run each. Tables: `agent_logs/perf1_guard_cmp.txt`, `agent_logs/perf1_flux_cmp.txt`.
- Wins:
  - FLUX addcmul 2048×6144×3072: 439.2 → 414.1 µs (−5.7%);
  - Qwen prefill-128 gate_up with A in L1: 116.0 → 107.9 (−6.9%; FLUX-file twin 114.4 → 108.4);
  - FLUX attn-out 128×6144×3072 addcmul: 123.0 → 118.0 (−4.1%);
  - PERF FOCUS 4 FF1 (R1b, helper walk + `SwiGluMul`): 636.6 → 624.4 (−1.9%, within noise);
  - W pass: 12.9 → 12.5.
- Every epilogue-none and sharded cell is unchanged within noise.
- Cells over +3% in one run were re-measured 3× (medians):

  | cell | before | after | note |
  |---|---|---|---|
  | 1×6144×6144 HiFi2 | 206.9 | 205.6 | |
  | fp32 1×6144×6144 | 210.9 | 208.2 | |
  | block_sharded res5 expand | 45.7 | 45.3 | |
  | bias_relu 2048×256×512 fused | 9.66 | 9.73 | zones off |
  | bias_relu twin (epilogue none) | 8.81 | 8.79 | zones off |
  | 256×6144×9216 none | 322.4 | 321.9 | zones off |
  | fp32 1×256×6144 | 10.13 | 10.23 | zones off |
  | 32×1024×2048 | 13.11 | 13.42 | zones off; +2.4%, noise band |

- No material regression, so no carve-out was earned by measurement.

### Golden suite
- `test_golden.py`, all 1887 cells in four placement chunks: 1512 passed, 375 xfailed (the permanent
  EXCLUSIONs), 0 failed.
- `test_translated.py` + `test_regression.py`: 212 passed, 20 skipped, 80 xfailed, 14 failed.
  - The 14 failures are all `test_matmul_1d_multiple_output_blocks_per_core`: `num_program_cache_entries` delta 0
    instead of 1. Its parametrizations differ only in reference-op knobs that genify_matmul ignores, so later
    cases hit the program cache.
  - They are pre-existing: the same 14 fail on the pre-round kernels. They are not numerical, and not caused by
    this round.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::matmul_block` (and the LLK `matmul_block` / `matmul_block_no_mop` / `llk_math_matmul` under it) | capability | (1) Every flavour writes the DEST offset (SETC16 via `set_dst_write_addr`) before each output tile's MVMUL sequence. That is ~1 FPU cycle on a 16-cycle LoFi tile-K, and no API offers a multi-tile matmul that advances DEST with an address modifier on the last MVMUL. (2) With `packer_l1_acc` + TileRowMajor + Interm, the helper drains `cb_partials` per K-block on the UNPACK thread (wait / pop per row group), so the unpacker waits for the packer every K-block. Its `caller_owns_pack_target` mode fixes (2) alone (−3.9 µs). | 466300 (whole op); 353910 (isolated K-walk) | 441500; 330641 | `kernels/genify_matmul_compute.cpp:313` → `kernels/genify_matmul_kwalk.hpp:126` |
| no helper covers this (eltwise_chain SFPU binary elements: no `silu(a)*b`) | capability | No fused silu(a)·b element. The chain's binary SFPU path (`SFPU_BINARY_CALL`) passes DEST indices at runtime, but the fully unrolled body with two rows interleaved needs compile-time row offsets from the gate slot. The best pure fusion with the stock sigmoid math was 74307 ns. The chain itself is kept (init / DEST / CB / reconfig). | 85317 (isolated, `silu_tile` + `MulBinary`) | 61810 | `kernels/genify_matmul_compute.cpp:252` → `kernels/genify_matmul_swiglu_sfpu.hpp:202` |

### Next levers (not done)
- Extend the raw K-walk to the epilogue-none `Out` target (add a reload, or a final partials → output pass),
  and to sb_h > 1 subblocks (R1b).
- With the epilogue at 62 µs, the focus tail is the epilogue plus the write stream again (~1:1). A
  progressive split becomes interesting only once the tail is write-bound.

## Perf 2 — perf tournament round 2 (PERF FOCUS 1: Qwen3.8 prefill gate_up + SwiGLU, 2048 tokens)
- Date: 2026-11-17
- **Focus config (exact golden case, all in SUPPORTED):**
  - A 1×2048×5120 bf16 @ B 5120×8704 bfloat4_b → bf16 out;
  - LoFi, fp32_dest_acc_en=True, math_approx_mode=False;
  - DRAM interleaved, swiglu.
  - Plan (unchanged): R1a on rect_t 11×10; one 6×28 block per core; subblock 1×4; block_k 12 × 14 K-blocks (last 4).
- **Result: 418.3 → 391.9 µs (−6.3%).**
  - PCC 0.999919 and rel-RMS 0.00829, both unchanged.
  - Every idea was measured, and all 3 graduated (one of them only as a pair).
  - No regression; SUPPORTED unchanged.

### Measured breakdown (focus, before; device ns, one fresh run each)
- Ablation, peeled cumulatively. Hooks: `agent_logs/perf1_ablation_hooks.patch`, applied only for the runs.

  | run | µs |
  |---|---|
  | full | 418.3 |
  | epilogue stubbed | 405.9 |
  | epilogue + writes stubbed | 347.0 |
  | + reads / mcast stubbed (matmul only) | 331.7 |
  | compute only (no DM) | 393.3 |
  | DM only | 259.8 |
  | everything stubbed (sync / pack-L1-acc floor) | 86.5 |

- Zones (p50 over 110 cores):
  - `cmp_matmul` 345.8 µs, then `cmp_epilogue` 62.2 µs;
  - `wr_write` 31.8 µs (in 3 µs bursts, one per epilogue row);
  - the reader ends at 338 µs;
  - the first K-block of both operands lands 15.7 µs after kernel start on every core.
- **Ranked:**
  1. **Tail, ~72 µs, balanced: the epilogue (62 µs) runs in parallel with a DRAM write wall (~59 µs).**
     - Stubbing the epilogue alone saves only 12 µs. Stubbing the writes after it saves another 59 µs.
     - The whole 17.8 MB output goes out in the last ~60 µs, from all 110 cores on NoC1, at ~300 GB/s. The spec is 460.8 GB/s, a 38.7 µs floor.
     - Both halves have to drop to move the wall.
  2. **K-walk head, 15.7 µs.** It is the K-walk's entire exposed data movement: 345.8 µs with DM, 330.0 µs without.
     - K-block 0 is 3.45 MB chip-wide, read by all 21 injectors at once, against a ~7.5 µs DRAM floor.
     - In steady state the reads are hidden: the reader sits blocked on full CBs for 88 µs.
  3. Raw K-walk overhead, ~12 µs over the 318.6 µs FPU floor. Round 1 graduated this; it was not re-floated.
- **Roofline gate (`/perf-ceiling-dm`):**
  - The write stream is at 65% of DRAM peak, so it has headroom. The skill's Step 4b says a both-NoC split can approach peak.
  - The `noc_estimate` CLI is not built on this box (it needs `--build-tests`). The gate therefore uses the DRAM spec cap instead of the NPE bracket.

### Portfolio and verdicts
Artifacts: `perf_experiments/<slug>/` and `tests/.../genify_matmul/perf_experiments/<slug>/`.

| idea | verdict | before → after (device ns) | domain |
|---|---|---|---|
| `swiglu_epilogue_loadmacro`: SFPLOADMACRO-pipelined SwiGluMul | **WIN** (isolated); REGRESSION on the whole op if graduated alone | isolated epilogue 61,827 → 46,648. Whole op alone 418.4 → 434.3 µs: the epilogue stops pacing the output and the write wall gets congested. With writes stubbed, 408.6 → 394.5 µs | every swiglu path: R4 natural pairs 63.7 → 48.8; 16-bit DEST 62.7 → 45.3; bf8b out 61.8 → 46.9; fp32 grade 75.7 → 54.9; no exceptions. Blackhole only |
| `tail_writes_both_nocs`: the idle reader writes 36% of the last block's rows on NoC0 | **WIN** | write stream 60.1 → ~44 µs (with the epilogue stubbed, 405.7 → 388.8 µs). Whole op alone 418.2 → 415.5 µs (−0.65%) | R1 descriptor (R1 / R1a / R1b, any epilogue, DRAM- or L1-interleaved output). Measured: 2048×5120×4160 −5.3%, FLUX addcmul −1.6%, R1b flat |
| `kwalk_head_ramp`: short first K deliveries [4, 6, 8, 10, 12×11] | **WIN** (focus within noise; material elsewhere) | focus 418.2 → 409.6 µs (−2.05%, 3 stable repeats). 384×1536×1024 bias 35.6 → 33.1 (−7.2%); 8192×4096×64 R1 291.2 → 282.2 (−3.1%) | R1 and the R1a raw walk. Inexpressible elsewhere (see the carve-outs below) |

- **Swiglu menu (isolated focus epilogue).**
  - Every option has identical precision: PCC 0.999999, rel-RMS 0.00163, max 2.03 / mean 0.275 bf16 ULP.

    | option | ns | note |
    |---|---|---|
    | baseline | 61,827 | |
    | tuned | 55,133 | hand-ordered stream; bit-identical |
    | tuned_k2c | 53,637 | bit-identical |
    | lm | 50,330 | |
    | **lm2** | **46,648** | **graduated** |
    | lm3 | 48,107 | replay buffer |

  - Binding thread: the SFPU pass is 56.1 of the 61.8 µs; unpack and pack together are ~6 µs.
  - lm2 runs at 19 cycles per SFPU row, against ~28 for the round-1 element.
  - lm2 changes the product order to (g·r)(1+e)·u. On the bf16 focus block, 2 of 86,016 elements differ from the old order, both at ~2e-38.
- **Tail split menu (NoC0 share; whole op with the epilogue stubbed):**

  | NoC0 share | µs |
  |---|---|
  | 0% (baseline) | 405.7 |
  | 15% | 398.2 |
  | 25% | 396.3 |
  | 30% | 390.8 |
  | **36% (graduated)** | **388.9** |
  | 43% | 392.0 |
  | 50% column cut | 399.7 |
  | 50% alternating tiles | 402.2 |
  | 100% NoC0 | 451.1 (NoC0 writes pile up on the DRAM-column links) |

  Per-core shares were no better than the uniform cut.
- **Head ramp menu (whole focus):**

  | variant | µs | note |
  |---|---|---|
  | ragged block first [4, 12×13] | 418.8 | NULL: the next 12-wide block lands 10 µs late |
  | [4, 8, 12×12, 4] | 414.1 | |
  | [2, 4, 6, 8, 12×11, 8] | 412.7 | |
  | [3, 5, 8, 12×12] | 410.5 | |
  | **auto [4, 6, 8, 10, 12×11]** | **409.6** | **graduated** |
  | auto with B-only compaction | 412.2 | |

  Delivery time grows faster than linearly in its width, so each ramp step lands before compute finishes the previous one.

### What graduated (one path each; replaced code deleted)
- **`SwiGluMul` (lm2)** replaces the round-1 SFPI body in `kernels/genify_matmul_swiglu_sfpu.hpp` outright.
  - It is the only swiglu SFPU step on every regime and both precision grades. No carve-out.
  - The result now lands in the up slot (D1); the chain packs from `SwiGluMul::out_slot`.
- **Tail write split** (`kernels/tail_split.hpp` plus the reader, writer and host).
  - It is the R1 descriptor's only write path, and it is unconditional: the patch's on/off host knob was removed.
  - The output CB keeps a single consumer: the writer posts each last-block batch to the reader through three local semaphores, and pops only after the reader's share has left L1.
  - Under addcmul, the reader serves output batches inside its residual loop, so the depth-2 residual ring cannot deadlock.
  - Domain: R1, R1a, R1b.
    - **Carve-out (inexpressible):** R4 in-place output has nothing to write.
    - **Not extended:** the R2 / R3 / R4 descriptors with a non-in-place output. Their readers have a different tail state (R4 streams B to the end; R3 exchanges A shards), so the split needs separate reader integration there. This is recorded as the next lever, not claimed.
- **Head ramp** (`kernels/genify_matmul_kschedule.hpp`).
  - The reader, the raw K-walk and the helper walk (via `LastKBlockInnerDim` and `MatmulBlockShape`) all follow one delivery schedule.
  - With `GM_K_RAMP` off, the schedule is the identity; a `static_assert` checks this, so the legacy cells keep the same code.
  - Host `k_head_ramp()` is a carve-out predicate (`if (cannot) legacy`). Every carve-out is inexpressible, none is merely unmeasured:
    - R1b: the writer's A stream has its own nominal K loop;
    - residency walks;
    - one-K-block walks: a ramp there would *introduce* packer L1 accumulation where K accumulates wholly in DEST, which changes precision rather than speed;
    - R1a walks on the matmul_block helper (epilogue none, sb_h > 1): the helper strides A rows by block_k, so A cannot be compacted, and a short head still multicasts a full A slot. The ramp's mechanism is gone there; measured flat to +2.3%.
  - `GM_COMPACT_A` (A laid out with the delivery's row stride) is on the raw walk only, enforced by a `static_assert`.
  - B multicasts only its valid `kv·block_n` prefix whenever the ramp is on.
- **Instrumentation** extended (permanent):
  - `rd_tail_write` on the reader. It is occupancy: it includes the wait for the writer's post.
  - `wr_tail_wait_reader` on the writer.
  - The per-K-block reader zones now run once per delivery.
  - The focus run peaks at 168 markers per RISC, under the cap.

### Whole op before / after
- **Focus: 418.3 → 399.9 µs** (SwiGluMul + tail split) **→ 391.9 µs** (+ head ramp).
- Zones after: `cmp_matmul` 337.6 µs (was 345.8: the head), `cmp_epilogue` 46.7 µs (was 62.2). The writer ends ~7 µs after compute.
- Still ahead of stock (564.5 µs) and below the #65148 roofline (428.7 µs).

### Guard set (no regression)
- Coverage: the full `perf/test_genify_matmul_perf_guard.py` (62 cells: every regime × layout × placement representative) and `perf/test_genify_matmul_perf_flux.py` (16 cells).
- Method: new vs the pre-round kernels, one fresh profiled run each. Table: `agent_logs/perf2_guard_cmp.txt`.
- Wins:
  - focus −6.4%; its epilogue-none twin −6.8%;
  - 2048×5120×4160 bf4b −5.2%;
  - 384×1536×1024 bias fused −8.0%;
  - batched (PDL H pass) 75.4 → 69.5 µs (−7.9%);
  - Qwen prefill-128 swiglu with A in L1: −3.5%;
  - 4096³ HiFi2 −2.5%; FLUX 1×6144×6144 −3.3%.
  - Everything else is within ±3%.
- Four cells were over +3% in one run. Each was re-measured 3× for base and new with the zones compiled out; medians, µs:

  | cell | base | new |
  |---|---|---|
  | 256×9216×6144 HiFi4 | 367.7 | 367.1 |
  | 256×6144×9216 HiFi2 | 323.2 | 321.7 |
  | bias_relu 2048×256×512 fused | 9.84 | 9.85 |
  | bias_relu none | 8.80 | 8.93 (+1.5%, noise) |
  | FLUX fp32 1×6144×6144 | 207.8 | 207.1 |
  | FLUX 256×6144×9216 | 332.3 | 324.4 |

- No material regression, so measurement earned no carve-out.

### Golden suite and unit tests
- `test_golden.py`, all 1887 cells in 8 chunks: 1512 passed, 375 xfailed (the permanent EXCLUSIONs), 0 failed. Identical to round 1.
- `test_translated.py` + `test_regression.py`: 212 passed, 20 skipped, 80 xfailed, 14 failed.
  - The 14 failures are the same pre-existing `test_matmul_1d_multiple_output_blocks_per_core` program-cache-count asserts recorded in Perf 1. They are not numerical.
- Unit dir: 687 / 687 passed, including the mcast, residency, stream, sharded, epilogue and precision-matrix tests.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| no helper covers this (eltwise_chain SFPU binary elements: no `silu(a)*b`; SFPI has no load-macro scheduling) | capability | (1) No kernel_lib element computes silu(a)·b. (2) Neither SFPI nor any LLK API exposes SFPLOADMACRO, the facility that lets a load also schedule a MAD, round, store or SWAP on an otherwise idle SFPU sub-unit. (3) SFPI does not software-pipeline across SFPU rows, so dependent SFPMADs and per-face drains leave ~4 bubbles per row. The raw stream issues 18 + 1 cycles per row instead of ~28. A helper that let an element declare "load X also issues op Y on sub-unit Z after N" would cover this. The chain itself is still used (init, DEST / CB sync, reconfig, pack). | 61,827 (isolated focus epilogue, round-1 SFPI element, itself already a round-1 bypass) | 46,648 | `kernels/genify_matmul_compute.cpp:262` → `kernels/genify_matmul_swiglu_sfpu.hpp:272` |

- The round-1 `kw::matmul_accumulate_interm` bypass is unchanged in kind. It now also takes a per-delivery inner width and an A row stride, so `compute_kernel_lib::matmul_block` gains one more missing capability: **a per-K-block A row stride**. The helper strides in0 rows by the nominal `block_k`, which is why the head ramp cannot compact A on the helper walk (the R1a epilogue-none carve-out above).
- The tail split and the head ramp are dataflow-only and bypass no helper.

### Next levers (not done)
- Extend the raw K-walk (or give `matmul_block` a per-K-block A stride) to the epilogue-none and sb_h > 1 walks. That would lift the R1a head-ramp carve-out and round 1's SETC16 cost there.
- Extend the tail split to the R2 / R3 / R4 descriptors with non-in-place output.
- Focus tail now: epilogue 46.7 µs, then ~7 µs of write drain. The K-walk is 337.6 µs against the 318.6 µs FPU floor.

## Refinement 10 — Weight multicast for sharded A (R2a height_sharded, R3a block_sharded): PDL 1×1 convs
- Date: 2026-11-17
- What was done:
  - **Breakdown first (/perf-measure).** Baseline zones: R2 reads all of B on every shard core (res3 reduce: 6 MB of the same 64 tiles from 86 cores); R3 reads its column slice once per grid row.
  - **R2a / R3a weight channel** (scheme-change, the designed successors). B does not vary along the shard rows, so every core of a *B group* needs the same B K-blocks. A B group is a grid column in R3; in R2 it is all of A's shard cores, R2 being R3's single-column case. One code path covers both:
    - **Injector.** A fixed injector per group (`ttnn.Mcast` `weight` channel, `McastFixedSenderConfig(SHARDED_B_INJECTOR, Staggered)`, so R3's column injectors sit on different grid rows) reads each B round from DRAM once. It multicasts the round into every group core's reserved `cb_weight_tiles` slot. The slot address is the same everywhere because the walk is uniform, so the CB pointers stay in lockstep.
    - **Rounds.** A B round carries `g` K-blocks (`SHARDED_B_ROUND_BYTES` = 32 KB, `g | num_k_blocks`, shrunk until the 2g-deep weight CB fits L1). The g slots are reserved together and pushed one per K-block.
    - **Stall-shadow and ordering.** The injector issues the round's reads before the A round. Every core runs A round → B round per K-block, so the two channels cannot deadlock.
    - **Where the channel is off.** Batched B (it varies along the rows), single-core groups, and R2 core sets the multicast cannot cover (more than 3 rectangles: `_sharded_b_channel` catches the helper's refusal) keep the per-core reads.
  - **First bias / gate row on the B channel.** Every core reading the same row tiles was a DRAM bank hotspot (~2.4 µs after the K-walk, which the epilogue waited on). The injector now reads the row beside B (its own transaction id) and multicasts it in one row round right after the first B round, overlapping the matmul. Later blocks of a multi-block walk keep the per-core row read, because `cb_row` holds one row.
  - **R3 traffic gate** (`BS_B_MCAST_MIN_SAVED_BYTES` = 3 MiB). In R3 the B round shares the reader's serial chain with the A row exchange. It pays only when the B re-reads it saves, `(rows − 1)·K·N·b`, are large. Measured: 0.97–2.1 MB saved ran 0–9% slower (res4 reduce / expand, ASPP projection); 3.9 MB and up ran 3–61% faster. R2 has no A exchange and wins at every size, so it is ungated.
  - **Knobs parked at their measured best.**
    - `SHARDED_B_MAX_CHUNKS = 1`. The chunked, trid-pipelined round (read chunk c+1 while sending chunk c) stays live, but each chunk is a full handshake + write-ack round (~1.2 µs on 86 cores). res3 reduce measured 6.9 / 10.3 / 15.1 µs for 1 / 4 / 8 chunks.
    - `SHARDED_B_ROUND_BYTES`: one all-K round lost on res4 expand (compute waits for all of B); per-K-block rounds lost on res4 reduce5 and ASPP projection. Both shapes are now gated off.
  - **Reused**: R3's walk, topology, blocking, CBs and compute kernel; R1a's line-delivery pattern (fixed sender, lockstep slot); R4's trid read tagging; the `ttnn.Mcast` attachment.
  - **Added**:
    - reader CT args 36–39 (`bs_b_mcast`, `b_chunks`, `b_chunk_k`, `b_round_kblocks`) and two small face templates;
    - host `_sharded_b_channel`, `sharded_b_round`, `sharded_b_chunks`, `_sharded_b_depth`, and four knobs;
    - zones `rd_b_issue`, `rd_b_round`, `rd_row_round`.
- Accuracy achieved: bit-identical to the per-core-read stepping stone, since only where B's bytes come from changes. `test_genify_matmul_sharded_b_mcast.py` checks `torch.equal` with the multicast on and off for every case. All PDL sharded golden loose cases pass at their gates (bf8b LoFi: 0.999 / 0.06).
- Golden test progress:
  - `-k "(test_op_loose and (height_sharded or block_sharded)) or (test_op and not loose and (height_sharded or block_sharded) and (bias_relu or addcmul))"`: 323 passed, 60 xfailed (the permanent fp32 × 16-bit-DEST EXCLUSION), 0 failed. Re-run after the R3 gate: loose sharded + block_sharded bias cells, 173 passed, 30 xfailed.
  - Unit regression net: `test_genify_matmul`, `_mcast`, `_dram_sharded`, `_epilogue`, `_extended`, `_residency`, `_stream_mcast`, `_height_sharded`, `_block_sharded`, `_sharded_b_mcast`: 152/152.
- Perf (device ns, AICLK 1350 MHz = the reference clock, so `achievable_ns` is unscaled; one fresh run each; `perf/test_genify_matmul_perf_sharded.py`, every PDL sharded LOOSE case, multicast on (`mc1`) / off (`mc0`)):

  | PDL case | before | after | achievable | after / before | after / achievable |
  |---|---|---|---|---|---|
  | HS res2 shortcut 32768×128×256 | 18.9 | 10.7 | 9.8 | 0.57× | 1.09 |
  | HS res2 reduce 32768×128×64 | 6.3 | 4.9 | 4.0 | 0.78× | 1.23 |
  | HS res2 expand 32768×64×256 | 13.2 | 8.9 | 8.7 | 0.67× | 1.02 |
  | HS res2 reduce 32768×256×64 | 9.9 | 6.1 | 5.0 | 0.62× | 1.22 |
  | HS res3 reduce 32768×256×128 | 16.7 | 10.3 | 9.4 | 0.62× | 1.10 |
  | HS res3 expand 8192×128×512 | 23.5 | 9.9 | 8.2 | 0.42× | 1.21 |
  | HS res3 reduce 8192×512×128 | 20.5 | 8.4 | 6.9 | 0.41× | 1.22 |
  | HS decoder skip 8192×512×64 | 9.3 | 5.9 | 4.7 | 0.63× | 1.26 |
  | HS decoder skip 32768×256×32 | 5.7 | 4.4 | 3.6 | 0.77× | 1.22 |
  | HS semantic logits | 5.8 | 4.4 | 3.6 | 0.76× | 1.22 |
  | HS heads 32768×32×32 | 5.3 | 4.3 | 3.5 | 0.81× | 1.23 |
  | HS res3 shortcut 8192×256×512 (bf16 A) | 41.3 | 15.7 | 13.1 | 0.38× | 1.20 |
  | BS res4 reduce 8192×512×256 | 19.0 | 18.9 | 16.6 | 0.99× (gated) | 1.14 |
  | BS res4 expand 2048×256×1024 | 13.2 | 13.0 | 13.6 | 0.98× (gated) | 0.96 |
  | BS res4 reduce 2048×1024×256 | 12.1 | 12.2 | 9.4 | 1.01× (gated) | 1.30 |
  | BS res5 shortcut 2048×1024×2048 | 78.0 | 30.4 | 72.3 | 0.39× | 0.42 |
  | BS res5 reduce 2048×1024×512 | 19.3 | 14.9 | 10.6 | 0.77× | 1.41 |
  | BS res5 expand 2048×512×2048 | 44.5 | 24.0 | 24.2 | 0.54× | 0.99 |
  | BS res5 reduce 2048×2048×512 | 40.5 | 22.9 | 16.1 | 0.57× | 1.42 |
  | BS ASPP 1×1 2048×2048×256 | 19.8 | 18.4 | 13.8 | 0.93× | 1.33 |
  | BS ASPP pool 1×2048×256 (1 grid row: no group) | 9.4 | 9.4 | 6.9 | 1.00× | 1.36 |
  | BS ASPP projection 2048×1280×256 | 18.6 | 18.5 | 16.8 | 0.99× (gated) | 1.10 |
  | BS res4 shortcut 2048×512×1024 (bf16 A) | 17.6 | 17.2 | 16.2 | 0.98× | 1.06 |

  - **Guard set (old = commit a75cc721cfa's kernel + descriptor, same session).**
    - PERF FOCUS 1: 392.1 / 392.1 µs.
    - PERF FOCUS 2: 29.9 / 29.9.
    - PERF FOCUS 3: median of 3, 69.7 → 67.0; the first single run read 67.3 → 71.0, which was noise.
    - PERF FOCUS 4 (256×6144×18432 swiglu): 622.4 → 616.9.
    - Upsample W pass: 12.8 / 12.9.
    - 2048×256×512 (5 variants) and 4096³ (2): all within ±1%.
- **Remaining gap (measured floors).**
  - **HS (1.02–1.26×).** Two handshake rounds on the critical path: B, then the first row, ~1.4 µs each on a 103-core group (102 serialized acks to the injector, the multicast, the loopback write barrier). Plus the injector's serial issue of B's tile reads: 2.0 µs for 64 tiles. heads (1-tile B) is 4.3 µs: B round done at +1.7, row round at +3.2, epilogue +0.5.
  - **R3, ablation on the multicast build.** res5 reduce 2: full 22.8 µs; A payload stubbed (sync kept) 17.1; matmul skipped (`SKIP_COMPUTE`) 20.9; both 14.6. So the A row exchange's bytes (~5.7 µs) and handshakes, plus 4 B rounds on the same reader chain (3.3 µs), bind it.
  - **res4 reduce** stays at 16.0 µs with the matmul and the A bytes both stubbed. block_k = shard width = 2 tiles, so every K-block pays a full fp32 partials pack of the 32×1 block, plus an A round. B is not on its critical path, which is why the multicast does not move it.
  - **ASPP pool** has one grid row, so there is no group to multicast to.
- **Next (not done; each is its own structural change).**
  - Move the B channel onto the writer RISC (BRISC / NoC1), so B rounds overlap the A rounds instead of queueing behind them on NCRISC. That is the R1b / R4 split-reader pattern, and it would also lift the R3 gate. It needs the writer to produce `cb_weight_tiles` and `cb_row`.
  - Pipeline the A exchange: two rounds in flight on the depth-2 `cb_act_tiles`.
  - Coarser R3 K-blocks: several peer shards per K-block, to cut the per-K-block partial packs.
- Issues encountered:
  - `if constexpr` inside `kernel_main` does not discard `->` on the `int` placeholder of an absent pipe face; moved into small templates.
  - Bias / gate early-issue attempts each regressed one side:
    - under the K-block barrier, the hotspot latency gated the push;
    - on the per-core path, it lost its overlap with the matmul;
    - issued early by every core, it collided with the injector's B reads.

    Hence the row round.
  - An R2 M split is never chosen: splitting a single-K-block walk double-buffers B. The multi-block test uses R3 for the M split.
- Tests added:
  - `test_genify_matmul_sharded_b_mcast.py` (10 cases, multicast on/off bit-identical, with spies on the descriptor's decisions):
    - HS on PDL's 86-core 2-rectangle set with bias_relu;
    - HS addcmul and swiglu;
    - R3 res5 with multi-K-block rounds;
    - a ragged last K shard with empty grid columns, at one and at all K-blocks per round;
    - L1-forced multi-block walks (HS N split, R3 M split, R3 N split);
    - a 4-rectangle scattered HS set that falls back;
    - the R3 traffic gate on both sides.
  - Perf bench `perf/test_genify_matmul_perf_sharded.py`: every PDL sharded LOOSE case × {mc1, mc0}, with ids `hs_<case>` / `bs_<case>` and a `GM_KNOBS` override hook for sweeps. It is the guard representative of the new path.

## Refinement 11 — Interleaved split for narrow N and mid M (FLUX.2 proj_out, text-encoder projections, single-block projections, mid-M prefill)
- Date: 2026-11-17
- What was done:
  - **Measured breakdown first** (`/perf-measure`; bench `tests/.../perf/test_genify_matmul_perf_r11.py`, 15 model cases at their own config). The plans showed the split ranking, not the kernels, as the main gap: proj_out on a 64×1 R1 split re-reading all of B per core (50 MB); text-encoder qkv on R1b capped by its single A injector (5.2 MB at ~23 GB/s ≈ 230 µs); the FLUX single blocks on 81–90 cores with padded 12-wide blocks (7×20 regions computed as 7×24).
  - **Split cost model** (`SPLIT_COST_MODEL`, `split_cost` / `split_cost_terms`, `tile_k_cycles`, `padded_work`; knobs `FPU_CYCLES_PER_TILE_K`, `UNPACK_CYCLES_PER_TILE`, `DRAM_BYTES_PER_CYCLE`, `CORE_READ_BYTES_PER_CYCLE`, `CORE_WRITE_BYTES_PER_CYCLE`). Every candidate is blocked and ranked by max(padded K-walk, DRAM stream, busiest core's reads, busiest core's writes); W-ordered with an early stop. Math fidelity joins `Geometry` (`plan(..., math_fidelity=)`). Candidates also use `q_n_min`; the S3 block search minimizes the same cost (padding vs re-reads). The legacy ranking is `SPLIT_COST_MODEL = False`.
  - **R1a K-walk granularity** (`MIN_K_BLOCKS = 8`): a plain R1a helper walk (no head ramp) of fewer K-blocks is re-split, staying above the R1a gate. Not applied under the raw walk's head ramp (384×1536×1024 bias +5% with both) nor to R1b (it broke R1b's 32 KB gate on M = 1 shapes).
  - **R1a injector stall-shadow** (`R1A_INJECTOR_AHEAD = 2`, `R1A_PREFETCH_DEPTH = 3`, `R1A_PREFETCH_MIN_RATIO = 2`; reader `GM_R1A_AHEAD` / `GM_R1A_DEPTH`): the injector reserves two slots and issues K-block kb+1's reads under trid `(seq % 2) + 1` before retiring and multicasting kb. Gated to walks whose busiest-core read term is ≥ 2× the K-walk.
  - **Epilogue-none raw K-walk** (`NONE_RAW_WALK`, `NONE_RAW_WALK_MIN_RATIO = 1.25`; compute `GM_NONE_RAW_WALK`): `kw::matmul_accumulate_interm` into `cb_partials` + an identity `CopyTile → PackTile` chain, replacing `matmul_block<Out>`'s last-K-block reload; gated to compute-bound splits.
  - Reused: R1 / R1a descriptor, CBs, `kw::matmul_accumulate_interm`, the eltwise chain, the multicast channels, the trid read helpers. Added: host cost model + 4 knob families, a reader prefetch branch, a compute identity-epilogue branch. No new CB, kernel file or descriptor branch.
- Device ns (before → after; stock): proj_out 160.5 → 57.2 (61.0); te qkv 232.7 → 111.4 (139.2); te o_proj 96.1 → 85.2 (79.4); te down 541.6 → 524.7 (508.6); te gate 509.5 → 500.2 (501.0); single QKV 1381.7 → 1116.9 (1244.6); single out 1977.6 → 1426.1 (1688.5); single FF1 2588.0 → 2155.2 (2398.2); Qwen qkvg 4096 409.4 → 372.5 (371.9); in_proj 1024 147.1 → 137.2 (139.0); M = 512: in 113.4 → 100.2 (108.8), out 54.0 → 52.1 (52.1), gate_up 200.8 → 194.3 (214.4), down 120.9 → 101.2 (111.7), qkvg 109.6 → 94.8 (101.3).
- Measured nulls / regressions kept as gated knobs: the prefetch at depth 3 everywhere (te o_proj +4.6%, te gate +4.5%, Qwen M = 512 +4..8%, te down flat — its read issue blocks the multicasting RISC); the epilogue-none raw walk on DM-bound splits (+2–4%).
- Remaining gap (floor): te o_proj / down, DM-only 82.3 / 509.5 µs (compute-only 60.3 / 405.2, sync 14.7 / 37.9): the R1a line injector chain. Next lever: the B-line injection on the writer RISC.
- Guard set (140 cells, `agent_logs/r11_guard_cmp.txt`, base = this phase's start via `GENIFY_MATMUL_KNOBS=SPLIT_COST_MODEL=False,MIN_K_BLOCKS=1`, which reproduces the old plans): geomean 0.93×, worst +2.9% (identical-plan R1b / M = 1 cells, noise); 4096³ −16% / −11%, 9472×5120×3456 −8% / −6%, FLUX addcmul 2048×6144×3072 −8% / −15%, 2048×5120×4160 bf4b −10%, 384×1536×1024 −13..−22%; PERF FOCUS 1–4 within ±1%.
- Accuracy achieved: PCC ≥ 0.99974, relative RMS ≤ 0.0089 on the 15 model shapes against the quantized-weight reference (`GENIFY_MATMUL_CHECK=1`).
- Golden test progress: interleaved slice (`test_golden.py -k interleaved`) 429 passed / 75 xfailed / 0 failed; `test_translated.py` + `test_regression.py` 212 passed, 20 skipped, 80 xfailed, 14 failed (the pre-existing `test_matmul_1d_multiple_output_blocks_per_core` program-cache-count asserts, unchanged since Perf 1). Unit: interleaved 650/650, sharded 52/52.
- Issues encountered: the first cost model under-priced few-core R1 splits (per-core read rate) and the 32-core PERF FOCUS 3 split (per-core write rate) — both added as terms; `MIN_K_BLOCKS` initially pushed R1b K-blocks under the R1b gate (restricted to R1a). Plan-pinning unit tests (R1a orientation / padded tail, single-walk split, PERF FOCUS 3 split) now pin those plans under the legacy ranking, which still builds them.
- Tests added: `perf/test_genify_matmul_perf_r11.py` (15 model cases; `GENIFY_MATMUL_FORCE` / `_PLAN` / `_CHECK`), `perf/conftest.py` (`GENIFY_MATMUL_KNOBS` / `GENIFY_MATMUL_TRACE` for the whole perf dir); `test_genify_matmul_mcast.py`: `test_r1a_min_k_blocks`, `test_split_cost_model_r1a` (proj_out, te qkv), `test_none_raw_walk` (both walks).

## Refinement 12 — Hang: bf16 batched-weight matmul (PDL upsample H pass in bf16)
- Date: 2026-11-17
- What was done: root-caused and fixed the dispatch-timeout hang of `genify_matmul(A (1,32,512,128) bf16, B (1,32,128,1024) bf16 batched)`. Triage: every writer in `tail_split::wait_at_least`, every reader in `cb_reserve_back`. The Perf 2 tail write split held the core's whole LAST block's output batches until the reader served its NoC0 share, but the reader serves only after its K-walk. In bf16 the split chooser plans R1a with `block_batches = 32` (one 32-batch block per core), so the writer held batch 0 while the reader still had batches 1..31 to read: the output CB (DEPTH_OUT row groups) filled, compute stalled, the input CBs never drained, deadlock. bfp8 plans `block_batches = 1` (the tail starts after the walk) and so ran. The hang is a property of `block_batches > 1` with any CB depth / K-blocking / dtype, not of this shape. Fix (writer + reader, no host change): the split now covers only the tail block's LAST batch, whose output is produced only after the reader's final K-block push, so the reader can always serve it. Earlier batches of a multi-batch tail are written by the writer alone. The residual-loop mid-walk serve is unchanged (it now only serves batches of that last block-batch).
  Reused: the tail-split handshake, its semaphores and the `serve_one` / `serve_before` walk. Added: the `bi + 1 == block_batches` scope (writer) and the `t_bi` constant (reader).
- Accuracy achieved: helpers.TOLERANCES pass (run_genify_matmul) on the H pass bf16 / bfp8 × B DRAM / L1, the R1a multi-K multi-batch (1,8,512,512)² and the plain-R1 multi-batch (1,128,64,128)@(1,128,128,64) shapes under none / bias / addcmul.
- Golden test progress: the two target cases (`test_op_loose[...dtype=BFLOAT16...]`, `test_op_feature[..._weight_l1...]`) pass, plain and `--dev`. Golden slice `-k "batched and interleaved"`: 78 passed, 15 xfailed, 0 failed. Unit regression (test_genify_matmul / residency / mcast / stream_mcast / epilogue / extended): 105/105.
- Perf: bfp8 H pass (`test_op_loose` bfp8), A/B on one box, 4 fresh runs each: base 66.8 / 65.7 / 67.0 / 66.3, fix 67.9 / 67.2 / 67.2 / 68.2 us (medians 66.6 -> 67.6, +1.5%: inside the noise band; same handshake at block_batches = 1). bf16 H pass: hang -> 136.0 us. That is about 2x bfp8; its R1a 32-batch plan has not been profiled (left for perf rounds).
- Issues encountered: none beyond the root cause. For multi-batch tails the NoC0 share now covers 1/block_batches of the last block, which matters only for the tail of multi-batch blocks.
- Tests added: `tests/ttnn/unit_tests/operations/genify_matmul/test_genify_matmul_tail_split.py`: the H pass (bf16 / bfp8 × B DRAM / L1; asserts the bf16 plan is multi-batch), and multi-batch tails on R1a (2 K-blocks) and plain R1 × none / bias / addcmul (asserting block_batches > 1 and the walk mode). 10 cases.

## Refinement 13 — Split output: `chunks` (FLUX.2 QKV ×3, time_mod ×17)
- Date: 2026-11-18
- What was done: `genify_matmul(..., chunks=c)` returns a list of `c` tensors, chunk i = columns [i·w, (i+1)·w) of epilogue(A @ B), w = N_out / c, each with `memory_config` and `dtype`. The same single program writes every chunk directly: no full-width intermediate, no extra op. Host: the entry point checks the contract (`chunks` an int ≥ 1, N_out % c == 0, w % 32 == 0, else `ValueError`), raises `UnsupportedAxisValue` for `chunks > 1` on a sharded placement, allocates the c tensors and passes them all last to `generic_op`. The R1 descriptor plans on the logical output (`_geometry` sums the chunk widths; `l1_cb_budget` counts every chunk). Chunk 0 stands in for the per-tensor properties (dtype, page size, accessor args). Every chunk's address goes to the writer and to the reader (tail share) RT args, and `GM_OUT_CHUNKS` is defined on both. Kernels: the new `kernels/out_chunks.hpp` (`Table`, `locate`, `write_row`) maps output tile column gc to chunk gc / w, local column gc % w. It locates the chunk once per block and writes a straddling block row as one run per chunk, through one `TensorAccessor` per run. The writer's store loop and the reader's tail-share loop both call `write_row`. The split choice, blocking, CBs and compute kernel are unchanged. R3 / R4 writers compile `GM_OUT_CHUNKS = 1`, which is the old page arithmetic.
  Reused: the R1 / R1a / R1b / R1d descriptor and both dataflow kernels' store paths, the tail-split handshake, the output accessor args. Added: `out_chunks.hpp`, the chunk address table (RT), `_out_list` / `_check_chunks`, and the `GM_OUT_CHUNKS` define. No new CB, kernel or descriptor branch.
- Accuracy achieved: per chunk, double QKV img / txt and single QKV PCC 0.999994, relative RMS 0.0044–0.0046; time_mod ×17 PCC ≥ 0.999991, RMS ≤ 0.0062; 384×1536×3072 ×3 / bias ×2 PCC 0.999999, RMS 0.0017.
- Golden test progress: the six `test_op_feature[..._chunksN-...]` cases pass (6/6). Unit regression (every file except the 540-cell precision matrix and the new one): 172/172.
- Perf (device ns, the same call without `chunks` = its LOOSE_CASES entry, one profile run each): double QKV img 963.9 → 962.8 us; single QKV 1122.1 → 1121.1 us; time_mod 1602.7 → 1605.5 us (+0.2%). Double QKV txt varied by run: unsplit 319.8 / 325.4 / 328.4 us, `chunks` 335.5 / 321.4 / 328.8 us. Medians 325.4 vs 328.8 us (+1.0%) sit in the same 320–336 us noise band. Every FLUX.2 `chunks` case is within 2% of its unsplit call: the split is free.
- Issues encountered: `run_safe_pytest.sh --profile` cannot take a multi-word `-k` (Tracy's shell splits it). Pass explicit node ids instead.
- Tests added: `tests/ttnn/unit_tests/operations/genify_matmul/test_genify_matmul_chunks.py` (23 cases): every epilogue × chunks {2, 3, 16}; chunk width 1 tile (every block row straddles chunks); M = 1; folded leading dims; batched weight (R1d); a large multi-block walk; time_mod's fp32-A / bf16-bias / bf16-out mix ×17; L1 A + output; the `ValueError` contract and the chunks = 1 / > 1 return types.

## Refinement 14 — A as a list of K segments (FLUX.2 single-block out = [attention | MLP])
- Date: 2026-11-18
- What was done: `genify_matmul([A_0, .., A_{s-1}], B, ...)` computes epilogue(concat(A_i, dim=-1) @ B) in the same single program, without building the concatenation. Host: `_check_segments` enforces the contract (same leading dims / M, dtype, layout and memory config; K_i % 32 == 0; Σ K_i == B[-2]; else `ValueError`); a one-element list is the single-tensor call; a sharded segment list raises `UnsupportedAxisValue`. `validate()` / `_check_contract` see the logical concat shape (`_logical_a_shape`), segment 0 stands in for the per-tensor axes. The R1 descriptor plans on the logical A (`_a_list`: `_geometry` sums the segment widths, `l1_cb_budget` counts every segment), appends the segment table `[kt_0 .. kt_{s-1}, addr_1 .. addr_{s-1}]` to the reader RT args (after `own_kb`) and to the R1b writer's A-stream args, and defines `GM_A_SEGMENTS` on both. Kernels: the new `kernels/a_segments.hpp` (`Table`, `locate`, `read_row`) maps K tile kt to its segment and local tile; `read_act` (R1 / R1a / R1d, residency and head-ramp walks) and the R1b injector's A read locate the segment once per K-block and read a straddling row as one run per segment. Without the define both keep their old loops verbatim. Split, blocking, CBs, compute unchanged.
  Reused: the R1 / R1a / R1b / R1d descriptor, `read_act`, the R1b injector read, A's accessor args (segment 0's). Added: `a_segments.hpp`, the segment table (RT), `GM_A_SEGMENTS`, `_a_list` / `_a_segment_table` / `_check_segments` / `_logical_a_shape`. No new CB, kernel or descriptor branch.
- Accuracy achieved: all three golden `concatK` cases within helpers.TOLERANCES (bf16, HiFi2 / HiFi4, fp32 DEST); unit cases on every epilogue, R1 / R1a / R1b / R1d, fp32 A, L1 A, with chunks.
- Golden test progress: the three `test_op_feature[..._concatK...]` cases pass (3/3). Unit regression (test_genify_matmul, chunks, stream_mcast, mcast, epilogue, height_sharded, tail_split): 132/132.
- Perf (device ns, one profile run each, vs the same shape's single-tensor LOOSE call): single out M = 2304, [3072 | 9216] x 6144: 1425.9 vs 1426.4 us; final block M = 2048: 1202.3 vs 1201.7 us. Both within 0.05%: the concat is free. Perf guard: `perf/test_genify_matmul_perf_flux.py::test_perf_flux_segments` (segmented + single twin).
- Issues encountered: None.
- Tests added: `tests/ttnn/unit_tests/operations/genify_matmul/test_genify_matmul_segments.py` (19 cases): every epilogue; one-tile edge segments, one segment per K tile (8), a one-element list; M = 1; folded leading dims; batched weight (R1d); a large multi-block walk; R1a (asserted) and R1b (asserted, none + swiglu); fp32 A; L1 A + output; segments with chunks; the `ValueError` contract.

## Refinement 15 — Small-M weight streaming at the DRAM roofline (Qwen3.8 M = 128, FLUX.2 prompt stream, M = 1)
- Date: 2026-11-18
- What was done:
  - **Breakdown first** (`/perf-measure`, new `ABLATE_DEFINES` host knob: SKIP_COMPUTE / GM_ABLATE_A / GM_ABLATE_A_SEND / GM_ABLATE_B payload stubs).
    - Qwen M = 128 gate_up, 109 µs: no-A 107, no-B 69, B alone 93, A alone 59, sync floor 10. The B stream binds: 25 MB of 576 B bfp4 tile reads at ~300 GB/s.
    - Per-core B finish times spread 3–4× by grid row, whichever bank the core read. Coalesced reads alone gave 78 µs; NoC1 reads 174 µs.
    - The flat 68-core split is two multicast rectangles, so the injector sent A twice (multicast 41 µs, reads 23 µs).
  - **R1b bank-local split** (`ACT_STREAM_BANK_LOCAL`, the `bank` split candidate).
    - Interleaved DRAM page p is in bank p % 8, so B columns n, n+8, ... of a K row are consecutive pages of one bank. Each core owns a run of such units (B columns; swiglu (gate, up) pairs), read as one bank run per K row (`GM_BANK_STRIDE` / `GM_BANK_RUNS`).
    - The output (and the bias / gate / residual reads) follow the same stride (`GM_OUT_STRIDE`, `out_chunks::write_row(stride)`).
    - Placement `BANK_LOCAL_PLACEMENT = "row"`: a bank's units go to the cores of its NoC0 home row (the optimal reader's row), so responses never turn into a column. Down's B stream: 34 µs (~430 GB/s) vs ~95 µs with "near" / "spread" placements.
    - `BANK_LOCAL_ROW_FILL`: the spare rows join the most loaded class. Banks 0 / 4 share a home row; time_mod 2037 → 1363 µs with fill.
    - The split is the whole grid, one rectangle, so A is multicast once.
    - Priced in `choose_split` by its own DRAM rate, injector cost, the busiest core's B stream and a per-round A floor (`BANK_LOCAL_*`).
    - swiglu stays on the flat split (`BANK_LOCAL_SWIGLU = False`): a pair spans two banks with different home rows.
  - **Rotating A injectors** (`ACT_STREAM_INJECTORS = 3`, `GM_ACT_INJECTORS`, bank split only): `McastRotatingSenderConfig`, K-block r from injector r % 3.
    - An injector issues its next turn's read before sending r when the slots are free, else after. Issuing it after the send held its ack of r + 1: 13.6 µs rounds. Issuing it before with a blocking reserve stalled sends on compute: FLUX attn 107 → 135.
    - Down 46 → 41 µs, attn 113 → 107, ctx 260 → 251.
  - **Reused / Added.**
    - Reused: the R1b reader / writer / `act` channel / CBs / compute, `out_chunks::write_row`, the split cost model, the R4 optimal-reader map.
    - Added: the column-stride defines, `bank_layout` / `bank_candidates` / `bank_core_columns`, the rotating-injector branch of the writer's A stream, the `wr_a_barrier` / `wr_a_send` / `wr_a_issue` / `wr_a_rotate` zones.
    - No new CB, kernel file or descriptor branch.
- Accuracy achieved: PCC 0.99998–1.000000, relative RMS 0.0044–0.0083 on every R15 shape (GENIFY_MATMUL_CHECK) [128×{5120×8704 swiglu, 5120×4160, 4352×5120, 1536×5120, 5120×3584} bfp4 LoFi; 256×{6144×9216, 6144×3072 addcmul, 6144×18432 swiglu, 9216×6144, 15360×3072} HiFi2; 1×6144×6144 bf16 / fp32 A; 1×6144×52224 bias; 1×256×6144; 2048×128×3072; 2048×3072×128].
- Golden test progress:
  - Interleaved cartesian slice: 375 passed, 75 xfailed.
  - R15 loose slice (M = 128 / 256 / 1, te_gate, x_embedder, proj_out): 19/19.
  - Feature (chunks / segments): 10/10.
- Perf, device µs before → after (roofline):
  - Qwen M = 128: gate_up 109.1 → 106.8 (56.8), in_proj 63.6 → 48.2 (28.3), down 55.9 → 41.5 (30.0), out_proj 22.9 → 22.8 (12.4), qkvg 81.0 → 49.8 (27.2).
  - FLUX.2 M = 256: QKV 331 → 283 (263), attn + addcmul 148 → 107 (96), FF1 631 → 614 (509), FF2 332 → 278 (263), ctx 352 → 251 (225).
  - FLUX.2 M = 1: linear_2 fp32 / bf16 205 / 209 → 168 / 173 (167 / 166), time_mod 1600 → 1352 (1409), linear_1 10.5 → 10.5 (7.7).
  - x_embedder 47.7 → 47.5 (30.2), proj_out 58.3 → 58.1 (30.2).
  - Also: text-encoder gate 504 → 479.
  - Guard: perf_guard / R1b guard / R11 cells show no regression beyond ±3%. 64×16384×256 (16-bit DEST) needed the round floor in the cost model and the bank-only rotation: it was 134, then 92, now back to 84.
- Issues encountered:
  - The helper's rotating senders need a rectangular receiver group (flat splits keep the fixed injector).
  - The remote JIT farm's SFPI mismatched (`__xtt_vector8`), so all runs compiled locally.
  - The NoC0 per-row B-rate effect is a measured property; its root cause (arbitration vs. response routing) was not established.
- Tests added:
  - `tests/ttnn/unit_tests/operations/genify_matmul/test_genify_matmul_bank_local.py` (14 cases): selection + correctness on bfp4 / bf16, 130 columns (not a bank multiple), ragged M / N, every row-epilogue, fp32 A, L1 A, the "near" / "spread" placements, injectors 1 / 2, fill off, swiglu under its knob.
  - `perf/test_genify_matmul_perf_r15.py`: the 16 R15 cases at their golden configs, the perf-guard extension for the new path.
  - `test_r1b_act_resident` now pins the flat split, since the bank split takes that shape.

## Refinement 16 — Mid-M compute efficiency (text encoder at 512 tokens, Qwen3.8 prefill M = 512 / 1024)
- Date: 2026-11-18
- What was done:
  - **Breakdown first** (`/perf-measure`; bench `tests/.../perf/test_genify_matmul_perf_r16.py`, the 15 listed cases at their golden configs; new `GM_ABLATE_DM` stub = reads + line deliveries replaced by reserve / push, beside `SKIP_COMPUTE`). Before (full / data-movement-only / compute-only): q512 in_proj 101.6 / 97.3 / 60.5 µs, te o_proj 86.6 / 82.0 / 61.5, te down 525.5 / 510.3 / 405.2, q1024 down 134.4 / 129.1 / 110.7: the R1a operand lines bound the mid-M walks. Zones: a B-line injector spent 50 µs of the 97 issuing its line's 1920 bfp4 tile reads (~35 RISC cycles each), and every core's reader received A(kb) before it could ack B(kb).
  - **R1a split lines** (`R1A_SPLIT_LINES`, `kernels/line_stream.hpp`). The A row lines move to the writer RISC (BRISC / NoC1, `cb_act_tiles`' producer, writer `act_mode = 3`); the B column lines stay on the reader. Both use rotating injectors (`McastRotatingSenderConfig` over the line: round r from line core r % length). An injector reads its turn into a one-K-block staging CB (6 / 7) in K-row chunks over the `R1A_LINE_WINDOW = 2` rounds before it, with round 0 read alone at round 0 (`R1A_LINE_STARTUP_SHARE = 1`), then multicasts staging → the ring slot (loopback into its own slot).
    - Measured on the way (q512_in): reading turns into the depth-2 ring slot 84 µs (the slot frees too late); all turns at once 107 (round 0 landed at ~40 µs behind the whole prefetch); whole turn 2 rounds ahead 89 (it delays the ack); chunks over every round between turns 86; window 2 + round 0 alone 78.5; + startup share 75.6.
  - **Head ramp on the split walks** (`R1A_SPLIT_RAMP`): the Perf 2 schedule on the helper walk too (the writer's A line follows it; compact A on the raw walk). q1024 down 132 → 126, q512 gate_up 167 → 161, flat elsewhere.
  - **Gates** (`_r1a_split_shape`): `0.7 ≤ busiest-core read / K-walk < 2.0`, A line ≤ B line, staging fits without shrinking the output block.
    - out_proj M ≥ 1024 (0.52–0.55) measured flat to +3 %; Qwen qkvg M = 4096 (0.65) +6 %.
    - FLUX proj_out at HiFi4 (1.95, A line 6× B) 69 → 84 µs.
    - The staged solve shrank out_proj M = 4096's block (226 → 268 µs) until the same-block rule.
  - **R1b ring fallback** (`ACT_STREAM_MIN_DEPTH = 3`): text-encoder gate (PERF FOCUS 1, bank split) solved two 8-row M-blocks at ring depth 4, streaming its 89 MB of bfp8 B twice. Depth 3 keeps one 16-row block: 477 → 404 µs, the roofline.
  - **Operand-aware subblocks** (`SUBBLOCK_UNPACK_AWARE`): cost per tile-K = max(FPU, unpack(A)/sb_w, unpack(B)/sb_h), unpack(t) = bytes/100 + 3 cycles, in the R1 subblock choice and the split ranking. Calibrated on compute-only runs (bf16 A, LoFi): 2×1 subblocks ~24 cycles per tile-K, 3×1 23.6, against 17.5 for 1×4. q512 qkvg 77 → 73 µs (8×10 split, 1×4), q2048 out_proj 136 → 129 µs (1×3 instead of 4×1).
  - **Reused / Added.**
    - Reused: the R1a descriptor / walk / CBs / compute; R1b's writer-produces-`cb_act_tiles` structure; the Mcast helper's rotating senders; the kschedule head ramp; `_choose_blocking` / `split_cost_terms`.
    - Added: `line_stream.hpp`, two staging CBs, the writer's `act_mode = 3` branch, the reader's B-line branch, knobs `R1A_SPLIT_LINES` / `R1A_SPLIT_MIN_RATIO` / `R1A_SPLIT_RAMP` / `R1A_LINE_WINDOW` / `R1A_LINE_STARTUP_SHARE` / `ACT_STREAM_MIN_DEPTH` / `SUBBLOCK_UNPACK_AWARE`, and `GM_ABLATE_DM`.
    - No new kernel file beside the header, no new descriptor branch.
- Device µs, before → after (roofline; ×roofline after):

  | Group | Case | Before | After | Roofline | × roofline |
  |---|---|---|---|---|---|
  | Text encoder | gate | 478.6 | 403.1 | 403.5 | 1.00 |
  | Text encoder | down | 525.5 | 430.1 | 403.5 | 1.07 |
  | Text encoder | qkv | 113.3 | 93.7 | 75.7 | 1.24 |
  | Text encoder | o_proj | 86.6 | 68.9 | 50.4 | 1.37 |
  | Qwen3.8 M = 512 | in_proj | 101.6 | 75.1 | 51.2 | 1.47 |
  | Qwen3.8 M = 512 | gate_up + SwiGLU | 195.7 | 161.0 | 107.2 | 1.50 |
  | Qwen3.8 M = 512 | down | 102.6 | 81.8 | 53.6 | 1.53 |
  | Qwen3.8 M = 512 | qkvg | 96.3 | 72.9 | 44.1 | 1.65 |
  | Qwen3.8 M = 512 | out_proj | 53.0 | 45.0 | 24.4 | 1.84 |
  | Qwen3.8 M = 1024 | out_proj | 76.2 | 75.7 | 39.2 | 1.93 |
  | Qwen3.8 M = 1024 | qkvg | 120.9 | 107.2 | 88.3 | 1.21 |
  | Qwen3.8 M = 1024 | in_proj | 139.6 | 125.0 | 102.4 | 1.22 |
  | Qwen3.8 M = 1024 | down | 134.4 | 125.2 | 107.2 | 1.17 |
  | out_proj | M = 2048 | 136.4 | 129.0 | 75.7 | 1.70 |
  | out_proj | M = 4096 | 226.4 | 226.2 | 151.3 | 1.50 |
- Measured floors after (compute-only = `GM_ABLATE_DM` / data-movement-only = `SKIP_COMPUTE`, µs):

  | Case | Compute-only | Data-movement-only |
  |---|---|---|
  | te down | 413 | 291 |
  | te qkv | 82 | 64 |
  | te o_proj | 63 | 49 |
  | q512 in | 62 | 64 |
  | q512 gate_up | 139 | 118 |
  | q512 down | 68 | 69 |
  | q512 qkvg | 61 | 59 |
  | q512 out | 38 | 39 |
  | q1024 out | 69 | 71 |
  | q1024 qkvg | 94 | 85 |
  | q1024 in | 114 | 87 |
  | q1024 down | 114 | 96 |
  | q2048 out | 121 | 102 |
  | q4096 out | 213 | 183 |

  - Every case runs within 1.05–1.2× of its larger floor.
  - What keeps the floors above the roofline:
    - M = 512 (16 tile rows) caps the 2-D split at 80–88 cores; the roofline assumes 110. On 88 cores the K-walk alone is ~1.25× the roofline.
    - out_proj (K = 48 tiles, output 2–3× the operands) writes its whole output after the last K-block. Splitting the region into M-blocks to overlap the writes measured worse (q1024 out 2×16 blocks 90.5, 1×16 152 vs 75.9 µs): the B re-reads cost more than the overlap.
  - Both are findings for the perf rounds, not open work: a non-rectangular split filling 110 cores at M = 16 tiles; an out_proj walk with B resident across M-blocks, delivered in K-blocks during the first.
- Guard (knobs off = before; `agent_logs/r16_guard.sh`): perf_guard 62 ops, perf_r11 15, perf_r15 17, perf_flux 20, perf_sharded 46, perf_focus 1. No op slower beyond ±3 % on re-run (single-run flags on 5–12 µs cells and two R1b shapes whose plans did not change came back within noise). 10 R11 / FLUX cells −15 … −26 %, 384×1536×1024 −14 … −35 %, PERF FOCUS 1 (Qwen gate_up 2048) unchanged.
- Accuracy achieved: PCC ≥ 0.99988, relative RMS ≤ 0.0086 on all 15 R16 shapes (`GENIFY_MATMUL_CHECK`). The split lines with the ramp off are bit-identical to the fixed-injector walk; the pacing knobs are bit-identical to each other.
- Golden test progress:
  - interleaved loose + feature (`test_op_loose and interleaved or test_op_feature`): 65/65 passed.
  - Interleaved cartesian slice: 375 passed, 75 xfailed (unchanged).
  - Unit (every file but the 540-cell precision matrix): 242/242.
- Issues encountered:
  - A profiler zone-hash collision (16-bit hash of file:line:name) after kernel line numbers shifted made `--profile` runs throw at teardown (`profiler.cpp:321`). `agent_logs/r16_zone_hash_check.py` checks the op's zones against every known location.
  - The writer's K schedule must be the identity for writers whose K args are placeholders (R4 / absent A role).
- Tests added:
  - `test_genify_matmul_split_lines.py` (14 cases): the split plan asserted + helpers.TOLERANCES on every epilogue, ragged M / K, folded lead dims, bfp4 LoFi and bfp8 HiFi2; bit-identity vs the fixed injectors; pacing-knob invariance; segments + chunks.
  - `perf/test_genify_matmul_perf_r16.py`: the 15 cases.
  - `perf/test_genify_matmul_perf_r11.py`: `GENIFY_MATMUL_BLOCK` (force an output blocking).

## Refinement 17 — Decode weight streaming (dram_sharded) at the DRAM roofline
- Date: 2026-11-18
- What was done:
  - **Breakdown first** (`/perf-measure`; bench `tests/.../perf/test_genify_matmul_perf_r17.py`: the 10 Qwen3.8 decode cases at their golden config, `GENIFY_MATMUL_CSWEEP` forces cores_per_bank per run; `agent_logs/r17_sweep.sh`, `r17_core_ends.py` per-core reader end times on the grid). The `ABLATE_DEFINES` stubs now reach R4 too (`GM_ABLATE_B` in the B ring, `GM_ABLATE_A_SEND` 32-byte rounds, new `GM_ABLATE_A_ROUND` = no round at all).
    - qkvg (PERF FOCUS 3) at the Refinement 6 split (C = 2, 7-column slices, 16 workers): compute stubbed 23.9 µs, B + A payload stubbed (compute only) 30.4 µs — compute-bound (1120 tile products per core with 1×1 subblocks).
    - A cores_per_bank sweep (C = 2 … 13) showed the B stream alone varies ±8 % with C, and fewer slice readers per bank stream fastest (LM head compute-stubbed: C = 3 98 µs, C = 4 106).
    - The C = 3 cliff (lm_last 101 µs, lm_head 119) was placement: logical row 9 holds banks 0 (col 0) and 4 (col 6); the old "nearest" placement put bank 4's third slice reader at col 5, whose NoC0 responses wrap the row through bank 0's links; row 9's cores ended 86–98 µs vs ~78 elsewhere.
  - **Ragged bank slices**: `cw = ceil(sw / C)`, C any value whose slices are all non-empty (was: C divides `sw`). The reader fetches only a slice's valid columns per K row (`cols_valid`, also for the last bank's padding); the CB slot keeps the `cw` stride, the stale tail columns are never written out. `worker_cols` clips to the bank boundary.
  - **Home-row segment placement** (`DS_PLACEMENT = "segment"`, `"near"` = Refinement 6): a bank's slice readers fill its home row from the primary in NoC0's +x direction up to the row's next bank primary; beyond it, the nearest free cores.
  - **Compute-hiding cores_per_bank rule** (`DS_COMPUTE_SHARE = 0.65`, `DS_COMPUTE_FIXED_CYCLES`, `DS_BANK_BYTES_PER_CYCLE`): the smallest C whose per-core compute (K · M · cw at `tile_k_cycles` for the plan's fidelity and subblock) + fixed cost is ≤ 0.65 × the busiest bank's stream, within the segment cap (5 on this board) when one qualifies; else the cheapest. `plan_dram_sharded` takes `math_fidelity` (the descriptor passes the compute config's; HiFi4 when absent). `DS_COMPUTE_SHARE = None` restores the narrowest-slice rule.
  - **Reused / Added.** Reused: the R4 descriptor, reader B ring, writer A rounds, compute, CBs, `tile_k_cycles` / `choose_subblock`. Added: topology field `core_n_tiles`, `_ds_segment_cap`, the cost rule in `ds_cores_per_bank`, knobs `DS_PLACEMENT` / `DS_COMPUTE_SHARE` / `DS_COMPUTE_FIXED_CYCLES` / `DS_BANK_BYTES_PER_CYCLE`, R4 ablation defines. No new CB, kernel file or descriptor branch.
- Device µs, before → after (roofline; × roofline after):

  | Case | Before | After | Roofline | × | C, slice |
  |---|---|---|---|---|---|
  | qkvg 1×5120×3584 (PERF FOCUS 3) | 31.9 | 24.2 | 22.4 | 1.08 | 4, 4 |
  | LM head last chunk 1×5120×12928 | 100.9 | 83.9 | 80.8 | 1.04 | 3, 17 |
  | GDN in_proj 1×5120×4608 | 33.1 | 30.3 | 28.8 | 1.05 | 3, 6 |
  | out_proj 1×1536×5120 | 11.3 | 11.6 | 9.6 | 1.21 | 5, 4 |
  | MLP down 1×4352×5120 | 29.5 | 29.3 | 27.2 | 1.08 | 4, 5 |
  | gate_up + SwiGLU 1×5120×9216 | 62.4 | 60.1 | 57.6 | 1.04 | 3, 12 |
  | LM head 1×5120×16384 | 104.1 | 100.7 | 102.4 | 0.98 | 3, 22 |
  | gate_up batch 32 | 62.0 | 60.3 | 57.6 | 1.05 | 3, 12 |
  | down batch 32 | 30.1 | 29.6 | 27.2 | 1.09 | 4, 5 |
  | out_proj batch 16 | 11.3 | 11.2 | 9.6 | 1.17 | 5, 4 |

  - out_proj's floor (measured): compute + A payload stubbed 10.8 µs, B stubbed 10.9, every payload stubbed 9.3 (the 8 handshaked A rounds + launch), 3.5 with no rounds. The B stream, the rounds and compute each take ~10–11 µs alone; the 11.2–11.6 µs run is within ~5 % of the B-stream floor.
  - `DS_B_AHEAD = 2` (two K-blocks in flight) on the new splits: qkvg 23.6, in_proj 29.9, lm_last 89.3 (+6 %), lm_head 103.8 (+3 %): parked at 1.
  - Guard (`test_perf_dram_sharded`, before → after): PERF FOCUS 2 29.0 → 29.1, LM head 104.4 → 100.6, gate_up batch 32 62.3 → 61.2, cartesian 384×1536×1024 bf16 HiFi4 137.2 → 63.6 (C = 4 one-column slices; was one core per bank). No other regime touched.
- Accuracy achieved: PCC 0.999989–0.999996, relative RMS 0.0046–0.0088 on all 10 decode cases (`GENIFY_MATMUL_CHECK`).
- Golden test progress: `test_golden.py -k dram_sharded`: 310 passed, 150 xfailed (the permanent EXCLUSIONs); `test_op_loose and dram_sharded` 10/10; `test_translated.py` DRAM-sharded 57/57.
- Issues encountered: the per-bank stream rate depends on where its slice readers sit and how many there are (±8 % compute-stubbed); the rule takes the smallest hiding split rather than modeling it. The root cause of the remaining spread (e.g. bank 4's primary in row 9 finishing last even with "segment") was not established.
- Tests added:
  - `test_genify_matmul_dram_sharded.py`: `test_cores_per_bank_split` gains a ragged split (C = 2 over 3-column banks); `test_plan_blocking` pins PERF FOCUS 2 at LoFi (C = 4, 5-column slices, every slice reader in its bank's row) and qkvg (14 columns → 4, 4, 4, 2).
  - `perf/test_genify_matmul_perf_r17.py`: the 10 decode cases.

## Refinement 18 — PDL: sharded 1×1 convs and the bilinear upsample
- Date: 2026-11-18
- What was done:
  - **Fixed cost first** (`/perf-measure`, per-core zone timelines; scripts in this phase's commits). PDL heads (32768×32×32, 103 cores, 1-tile B): the B round ended at +1.9 µs, the bias-row round at +3.4 µs, the epilogue at +4.0 µs. Both rounds were handshake + multicast. Dropping only the handshake (counter signal) left the receivers' arrival times unchanged: the cost was the injector's serial sends (two rectangles one after another, then a second delivery for the row), not the 102 acks.
  - **R2 one-shot weight channel** (`SHARDED_B_ONE_SHOT`, reader `GM_BS_ROW_TAIL`). A walk of exactly one B round in one chunk (every PDL HS conv) lands in slots no core has used, so the channel runs handshake-free with one linked-flag delivery per walk. The first bias / gate row rides in that delivery from `cb_weight_tiles`' tail (`row_tail` extra pages), and every core copies it L1 → L1 into `cb_row`. A counter signal (two deliveries) needs a write barrier per rectangle: measured 0.9–4.7 µs slower.
  - **Injector early push**: the injector hands its landed K-block to compute before multicasting it (the send only reads the slot). The injector had been the op's last core.
  - **R2 B groups** (`SHARDED_B_RECT_GROUPS`, `sharded_b_groups`): each rectangle of the HS core set is its own B group with its own `ttnn.Mcast`, injector and reader `KernelDescriptor`, all multicasting concurrently. Before, one injector sent B to the rectangles in turn: res3's 9-core partial row got B 1.35 µs after the 77-core block. `SHARDED_B_GROUP_ROWS` (bands of 1 / 2 / 4 rows) measured worse (more injectors contend for DRAM), so it is parked at `None`.
  - **R2a B-read helper** (`SHARDED_B_READ_HELPER`, reader `GM_BS_ROLE 3` + `GM_BS_B_HELPER_SEM`). The injector's B read issue is RISC-serial (~27 ns a tile: 2.2 µs for res3 expand's 80 tiles before the first byte goes out). A reader instance on the injector's BRISC, which has no writer work since the output is in place, issues the last floor(K / 2) K rows into the same slot and posts a local semaphore.
  - **R3 split roles** (`BS_B_ON_WRITER`, reader `GM_BS_ROLE 1 / 2`). This is the R10-recorded next lever. On in-place R3 walks a second reader instance on BRISC / NoC1 runs the B half (B rounds or reads, the bias / gate row), and NCRISC keeps the A row exchange and the residual. The writer is not placed. Under the split:
    - one K-block per B round (`BS_SPLIT_B_ROUND_BYTES = 1`; 32 KB / 64 KB / all-K measured worse);
    - R10's 3 MiB traffic gate lifted (`BS_SPLIT_B_MCAST_MIN_SAVED_BYTES = 0`; res4 reduce 1024 → 256: 12.2 → 9.7 µs);
    - without a channel, the split only for heavy A rounds or a single grid row (`BS_SPLIT_MIN_A_ROUND_BYTES = 64 KB`, `BS_SPLIT_SINGLE_ROW`; res4 expand / res4 reduce5 per-core on NoC1 were +6 / +4 %);
    - the per-core B half issues the block's row with its first K-block.
  - **Measured and parked (correct, live knobs, unit-tested):**
    - `BS_A_ONE_SHOT`: the whole R3a A exchange at once on a counter. ASPP pool 9.3 → 15.9 µs; eight concurrent multicasts over one 1×8 row serialize on its links, and the matmul loses its overlap.
    - `BS_SPLIT_B_DEPTH` / `BS_SPLIT_B_LOOKAHEAD`: per-core B read-ahead on the split, flat on ASPP pool.
  - **Reused / Added.**
    - Reused: the R2 / R3 walk, CBs, compute kernel, `ttnn.Mcast` channels, `tail_split`'s local-word helpers, the reader kernel itself (its other instances run the B half / the helper).
    - Added: reader defines `GM_BS_ROW_TAIL`, `GM_BS_ROLE`, `GM_BS_A_ONE_SHOT`, `GM_BS_B_HELPER_SEM`, `GM_BS_B_AHEAD`; host `sharded_b_one_shot`, `sharded_b_groups`, and the knobs above; zones `rd_b_helper`, `rd_b_ahead`, `rd_a_exchange`.
    - No new kernel file, no new CB index.
- Device µs (one fresh run each; before = this phase's start; stock `achievable_ns`):

  | Case | Before | After | Stock |
  |---|---|---|---|
  | HS res2 shortcut | 10.7 | 10.0 | 9.8 |
  | HS res2 reduce 128→64 | 4.9 | 3.8 | 4.0 |
  | HS res2 expand | 8.9 | 8.2 | 8.7 |
  | HS res2 reduce 256→64 | 6.1 | 5.1 | 5.0 |
  | HS res3 reduce 32768×256×128 | 10.3 | 9.3 | 9.4 |
  | HS res3 expand | 9.9 | 8.3 | 8.2 |
  | HS res3 reduce | 8.4 | 6.6 | 6.9 |
  | HS decoder skip 8192×512×64 | 6.0 | 4.7 | 4.65 |
  | HS decoder skip 32768×256×32 | 4.4 | 3.4 | 3.6 |
  | HS semantic logits | 4.4 | 3.4 | 3.6 |
  | HS heads | 4.4 | 2.8 | 3.5 |
  | HS res3 shortcut (bf16 A) | 15.7 | 12.6 | 13.05 |
  | BS res4 reduce 512→256 | 18.9 | 18.4 | 16.6 |
  | BS res4 expand | 13.1 | 12.9 | 13.6 |
  | BS res4 reduce 1024→256 | 12.1 | 9.7–10.0 | 9.4 |
  | BS res5 shortcut | 30.3 | 28.7 | 72.4 |
  | BS res5 reduce 1024→512 | 14.9 | 11.6 | 10.6 |
  | BS res5 expand | 24.0 | 22.8 | 24.2 |
  | BS res5 reduce 2048→512 | 23.0 | 16.6 | 16.1 |
  | BS ASPP 1×1 | 18.4 | 13.9–14.1 | 13.8 |
  | BS ASPP pool | 9.3 | 8.8–9.4 | 6.9 |
  | BS ASPP projection | 18.6 | 16.7 | 16.8 |
  | BS res4 shortcut (bf16 A) | 17.2 | 14.2 | 16.2 |
  | upsample H pass (×3) | 66.6 | 66.8 | 123.0 (roofline 43.5) |
  | upsample W pass (×3) | 12.4 | 12.7 | 15.4 |

- **Measured floors for what is still above stock / roofline:**
  - **H pass.** Write-only (reads + compute stubbed, all sync kept) is 62.6 µs vs 67 µs full: 17.8 MB of 1088 B tile writes at ~285 GB/s. The 1.2× roofline target (52 µs) is below that floor. Every write lever measured flat or worse:
    - writing all output over both NoCs (single block + tail split at 25 / 36 / 50 %): 65–82 µs;
    - more cores, NoC0, bank stagger, batching: Refinement 8.
  - **R3 (ablation: compute stubbed / B stubbed / both, handshakes kept).**
    - res4 reduce 512→256: 18.5 / 18.5 / 15.9. The handshaked A row exchange alone is 15.9 (stock 16.6), and it overlaps imperfectly with the 32×16-tile-K matmul.
    - res5 reduce: 11.7 / 10.1 / 9.0.
    - res5 reduce 2: 16.7 / 14.9 / 12.3.
    - res4 reduce 1024→256 (before the gate lift): 12.1 / 9.9 / 9.9.
    - ASPP pool: 9.2 / 6.7 / 6.5. Its A rounds alone run 6.5 µs (stock 6.9); with B traffic in flight they stretch from ~0.6 to ~1.0 µs each (contention, not B latency: read-ahead measured flat).
  - **HS.** Within 1–3 % of stock on res2 shortcut / reduce 256→64, res3 expand and decoder skip 8192: injector read issue (now split over two RISCs) + multicast + an FPU-bound LoFi matmul + the bias epilogue pass.
- Guard (base = this phase's start commit's descriptor + reader, same session; `/tmp` harness swapping the two files):
  - perf_guard 62 ops: worst +4.0 % on 2048×256×512 HiFi2, which re-ran 3× at −2.2..+2.0 % (noise); PDL sharded rows −6..−25 %.
  - r17 (R4) worst +0.9 %, r15 (R1b) +1.8 %, r16 (R1a split lines) +2.0 %, r11 +1.4 %, PERF FOCUS 1 0.0 %.
  - ASPP pool's single-reader path measured +4 % from reader codegen alone (host descriptor byte-identical); the single-row split restored it.
- Accuracy achieved: every multicast, split-role, one-shot and helper path is bit-identical to the per-core-read single-reader walk (`torch.equal`, `test_genify_matmul_sharded_b_mcast.py`). Golden PDL loose cases pass at their helpers.TOLERANCES gates (bf8b LoFi).
- Golden test progress: `test_golden.py -k "(test_op_loose and (height_sharded or block_sharded)) or (test_op and not loose and (height_sharded or block_sharded) and (bias_relu or addcmul or bias)) or (batched and interleaved)"`: 551 passed, 105 xfailed (the permanent EXCLUSIONs), 0 failed. Unit: every file but the precision matrix, 252/252.
- Issues encountered:
  - `static_assert` inside `if constexpr` in `kernel_main` (non-template) is still evaluated; hoisted as a conditional assert.
  - The test spy recorded only the last `_sharded_b_channel` call. R2 now builds one channel per group, so the spy records "any group multicasts" plus a channel count.
  - The B-read helper first assumed a core list on R3 single-K-shard grids (ZeroDivisionError). It is R2-only.
- Tests added (`test_genify_matmul_sharded_b_mcast.py`):
  - `test_hs_one_shot_rect_groups` (heads, res3 expand, addcmul, swiglu: channel count per rectangle, one-shot + row tail);
  - `test_bs_split_roles` (res5 reduce, ASPP pool, addcmul × lookahead 0 / 2: both halves present, bit-identical);
  - `test_bs_a_one_shot_parked` (ASPP pool, res5 reduce);
  - `test_bs_traffic_gate` now pins the gate on one reader and the multicast under the split.
