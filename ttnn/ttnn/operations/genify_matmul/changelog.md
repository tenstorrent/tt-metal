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
