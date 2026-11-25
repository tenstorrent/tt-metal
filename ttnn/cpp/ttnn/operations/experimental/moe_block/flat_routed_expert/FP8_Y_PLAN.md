# Flat routed expert: fp8 y, quantized on the down cores (plan)

Status: plan only, nothing implemented. Target: GLM-5.3 (H 6144, I 2048), one Blackhole chip (a p150, or one chip of
a Galaxy), the standalone `ttnn.experimental.moe_block.flat_routed_expert`. The combine is out of scope here.

## Goal

Let the flat expert emit y as **e4m3 fp8 plus power-of-two block scales** instead of bf16, quantized inside the down
compute, with (ideally) **no added down-core time**: the quantization runs on the PACK thread's SFPU while the MATH
thread runs the next pass's matmul.

Why: the end goal is fp8 on the combine's fabric wire (half the bytes; GLM 4x4 SP is bisection bound - see "Later:
the combine" below). The sender side of that is this quantization, and it can be built and measured on one chip,
writing y to DRAM, before the combine is touched.

## Today's down path (what changes)

* `device/kernels/down_compute.cpp` (TRISC): per sub-block row tile, `y[32 rows, this core's columns] = h @ W_down`,
  accumulated in DEST over all K tiles; with `y_row_major` it `pack_untilize_dest`s each pass into out CB `c_16`
  (`Float16_b`, `flat_routed_expert_program_factory.cpp:712`).
* Passes: `pass_columns = largest_divisor(columns, DEST_AUTO_LIMIT)`. The down compute runs with `down_fp32` (fp32
  DEST) **and full sync** (`cp("down_compute.cpp", ..., cfg.down_fp32, cfg.down_fp32)`): 8 fp32 tiles, so GLM's 6
  columns are one pass - but MATH and PACK take turns on the whole DEST: the pack never overlaps the matmul.
* `device/kernels/down_weight_reader.cpp` (NCRISC, NoC 0) streams the down weights and polls the row-major y writer
  (`y_row_major.hpp`): one NoC write per token row, `columns * 64` bytes (bf16), to `y[row, first_column * 32 ..]`.
* GLM on 32 down cores: 192 tile columns / 32 = **6 columns a core**, K = 64 tiles.

## Design

### Numerics

* Format: e4m3 (`DataFormat::Fp8_e4m3`; max 448). The packer converts fp32 -> e4m3 by **truncation**, so the SFPU
  rounds to nearest even first (and saturates at 448), as `per_token_cast_to_fp8` does since #66091.
* Scales: one per (token row, block), a **power of two**: `s = 2^ceil(log2(max(amax, 1e-4) / 448))`, `q =
  RNE_e4m3(y / s)`. Power-of-two scales make `y / s` exact (an exponent shift) and make the later dequant an exponent
  add (the combine's int32 snap stays exact and deterministic).
* **Block = one DEST pass = 3 tile columns = 96 values** (GLM: 2 blocks a core a row, 64 a row). A block never spans
  two DEST halves, so the pack thread never waits for another pass to know a scale. A per-token (1 x H) scale would
  need all 32 down cores to agree on a row's max (an all-reduce per row tile): rejected.
* Golden (torch): the fp32 reference y, split in 96-wide blocks, the same scale rule, `torch.float8_e4m3fn` with
  round-to-nearest-even and saturation. Expect exact codes for nearly every element (the device's fp32 accumulation
  order differs from torch's, so a few values on a rounding boundary may differ by one code).

### DEST sync: half sync, two passes of 3

* Quantized y runs the down compute in **fp32 half sync**: 4 tiles a half; `largest_divisor(6, 4) = 3`, so each row
  tile is **2 passes of 3 columns** (the existing pass loop already handles this; only the full-sync flag changes).
* The MATH thread is unchanged: per pass, the K loop into its half, `tile_regs_commit`.
* The PACK thread, after `tile_regs_wait`, works on the other half (below), then `tile_regs_release`. This is the
  pattern `gate_up_compute.cpp` already uses for the GLU activation (`activate()`): point the SFPU at the packer's
  half (`PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_packer_dest_offset()))`), run the SFPU,
  `PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU))`, pack.
* Cost of the split: each row tile's h is unpacked twice (64 more h tile unpacks against 384 weight tile unpacks,
  ~+14% unpack). It only costs time if the down matmul is unpack bound - measured in M0.
* The SFPU's registers are one set for all threads: the MATH thread must not use the SFPU (the down compute does
  not today - keep it so).

### The pack thread's work per pass (3 tiles, 32 token rows)

SFPU facts used (tt-isa-documentation, Blackhole `VectorUnit.md`, `Dst.md`): one instruction a cycle into the SFPU, 32
lanes; `SFPLOAD` reads `Dst[R:R+4, even or odd columns]` (4 DEST rows x 8 columns: lane / 8 is the row, lane % 8 the
column); a 32x32 fp32 tile is 64 DEST rows (faces 0..3, 16 rows each) = 32 vectors. A token row r < 16 is DEST rows r
(face 0) and 16 + r (face 1) of each tile; r >= 16 is faces 2 and 3.

| Step | What | Cycles (est.) |
|---|---|---|
| 1. amax | per group of 4 token rows: 3 tiles x 2 faces x 2 column parities = 12 `SFPLOAD` + `SFPABS`, folded with `SFPSWAP` (max); then the 8 lanes of each row: rotations by 1, 2, 4 lanes (`SFPSHFT2`) + `SFPSWAP`, and back out to the row's 8 lanes. ~70 instructions x 8 groups | ~560 |
| 2. scale | clamp, `ceil_power_of_two` (from `per_token_cast_to_fp8`), 1/s as an exponent; keep 1/s per group in an LReg or a spare DEST row | ~80 |
| 3. quantize | per vector: `SFPLOAD`, multiply by 1/s, RNE to 3 mantissa bits + saturate (`calculate_round_to_e4m3_rne`'s body), `SFPSTORE`. ~10 instructions x 96 vectors | ~960 |
| 4. pack | `pack_untilize_dest` of the 3 tiles into the fp8 out CB (fp32 DEST -> e4m3 L1, truncating, exact after step 3) | ~300-500 |
| 5. scales out | the scales' exponents written into the half's spare 4th tile by the SFPU, read by TRISC2 through the RISC view of DEST (`0xFFBD_8000`, `Dst.md`), written into a scale CB page | ~150 |
| | stalls, semaphores | ~100 |
| | **total** | **~2,000-2,500** |

Against it, the MATH thread's pass: 64 K tiles x 3 columns = 192 tile matmuls; at LoFi one `MVMUL` (8x16 @ 16x16) a
cycle, 16 a tile matmul: **3,072 cycles at the FPU's ideal rate** (more in practice: unpack, waits on weights). So the
pack thread needs ~65-80% of the ideal math time - it should hide, with a thin margin if the matmul really runs at
the ideal rate. Step 3 is the largest; `SFPLOADMACRO` (running the SFPU's load / simple / MAD / round / store
sub-units in parallel) could cut it 2-3x if the margin is too thin.

Same work on the MATH thread instead (no overlap) would add ~4,500 cycles to a row tile's 6,144 ideal (~+70%).

Reusable code:
* `calculate_round_to_e4m3_rne`, `calculate_ceil_power_of_two`:
  `deepseek_prefill/per_token_cast_to_fp8/device/kernels/compute/compute_per_token_cast_to_fp8.cpp` (main).
* SFPU per-row max of |x| and the fold over a block's tiles (`FAST_BLOCK4`, `pow2_inv_scale_sfpu.hpp`): branch
  `pjosipovic/k3-moe-fp8-wire` (Oct 13, not on main; 1x128 = 4 tiles, adapt to 3).
* Pack-thread SFPU calls: `common/kernels/expert_activation.hpp`, `gate_up_compute.cpp` (`activate()`),
  `api/compute/experimental/sfpu_binary_bcast_pack.h`.

### DEST hazards checked (`Dst.md`)

* FPU on one half and SFPU / packer on the other run concurrently; a half has one client at a time.
* "Writes from different clients in the same cycle that set valid bits of rows in the same 256-row group": the two
  fp32 halves are DstBits rows 0-511 and 512-1023, different groups - no race between the matmul's writes and the
  SFPU's stores.
* `ZEROACC` racing a Vector Unit write loses the write's valid bit even on the other half: the half's clear after the
  pack must come from the PACK thread, after its SFPU work, not from MATH (it does: Blackhole
  `llk_pack_common.h` `_llk_pack_dest_section_done_` issues `ZEROACC CLR_HALF` on the packer's half).
* `SFPLOADMACRO`-scheduled stores have extra hazards with FPU writes in the **same** half only (if step 3 uses it).

### Out CB, writer, outputs

* Out CB `c_16`: `Fp8_e4m3`, one page a row tile: 32 rows x `columns * 32` bytes (half of today's).
* `y_row_major.hpp`: `segment_bytes = columns * 32`; the trailer offset in the combine hand-off becomes `H` (later).
* Scales: a scale CB on the down core (one page a row tile: 32 rows x 16 bytes); the y writer writes them next to the
  rows.
* Outputs of the op with quantized y:
  * `y`: `[rows, H]` `FP8_E4M3` ROW_MAJOR DRAM interleaved (one page a row, `H` bytes).
  * `y_scales`: fp32, laid out **[down core, rows, 4]** (a 16-byte cell a core a row: its 2 block scales + 2 unused).
    A core writes its 32 rows' cells of a row tile as one contiguous 512-byte write (rows of a row tile are
    consecutive), DRAM aligned; the host reorders it to `[rows, H / 96]` for checks. The 16-byte cell is also the
    unit the combine slot would carry (one cell a down core a row).

### Host / API

* `FlatRoutedExpertConfig`: `y_fp8` (implies `y_row_major`, `down_fp32`; full sync off for the down compute). In the
  program hash (`FlatRoutedExpertParams`).
* `flat_routed_expert(...)`: `y_fp8 = false` argument; with it, the output is fp8 and a second output `y_scales`
  (optional preallocated, like `output`). `validate` accepts `FP8_E4M3` ROW_MAJOR y.
* `flat_options.hpp`: `y_fp8`, the block width in tiles (= `pass_columns`), the scale CB.
* Plan (`flat_routed_expert_plan.cpp`): the down core's L1 (out CB halves, a scale CB is added); a GLM pass width of 3
  (assert `columns % pass_columns == 0`).

## Milestones (single chip)

**M0 - measure the room (no quant code).** GLM cases of `test_flat_routed_expert_checks.py` (`glm53-8x4-5120-uniform`,
`glm53-4x4-5120-uniform`), bf16 row-major y, `down_fp32`:
1. Device-profiler zones on the down cores: MATH time per row tile vs PACK time, and the cores' idle (waiting on
   weights). This sets the real margin: the 3,072-cycle pass is the FPU ideal; the op is 529 us (8x4) / 1006 us
   (4x4) on a Galaxy chip, against ~180 / ~365 us of ideal down FPU time (40 / 80 row tiles) - the down cores may
   already have slack.
2. Same with the down compute in half sync (passes of 3): the price of the split.

**M1 - prove the overlap with a dummy.** On the PACK thread, before the pack, a dummy SFPU loop over the half
(load / multiply / store) of N cycles per pass; sweep N (1k, 2k, 3k, 4k): the largest N with no change in the op's
device time is the real budget for steps 1-5.

**M2 - the quantize sequence, bit exact.** A small single-core compute test: random fp32 tiles (incl. zeros, tiny
values, values near 448 x 2^k, rows with one outlier) -> the pack-thread sequence -> fp8 + scales, checked against
the torch golden (scales exact; codes exact except rounding-boundary cases). Validates the lane mapping, the
cross-lane max, the RNE and the TRISC2 DEST read of the scales.

**M3 - integrate.** `down_compute.cpp` (pack-thread quantize, fp8 out CB, scale CB), `y_row_major.hpp` (fp8 segments,
scale cells), program factory / plan / options, host API and validation.

**M4 - checks.** In `test_flat_routed_expert_checks.py`, GLM cases with `y_fp8`:
* accuracy: scales and codes against the golden; dequantized y against the fp32 reference (PCC, and the max
  relative error per block, bounded by e4m3's half ulp, 2^-4);
* determinism: bit-identical y and scales over `CALLS` calls;
* device time against the bf16 row-major baseline (`FLAT_US`): the target is **no regression**.

## Risks / open questions

* **Margin.** If M0 shows the matmul near the FPU ideal, steps 1-5 (~2-2.5k cycles) are close to the pass (3,072);
  then use `SFPLOADMACRO` for step 3, or merge steps 1 and 3 where possible.
* **`pack_untilize_dest` in fp32 half sync with the pack-thread SFPU.** `gate_up_compute.cpp` uses `pack_tile` after
  its pack-thread SFPU; untilize from the packer's half needs checking (M1 covers it).
* **TRISC2 reading DEST** (step 5): needs `RISC_DEST_ACCESS_CTRL_SEC` set for 32-bit reads, and the SFPU's stores
  finished (the `STALLWAIT` above). Fallback: pack the spare tile to a small fp32 CB (a packer reconfigure a pass).
* **Docs gap.** tt-isa-documentation describes packer fp8 as e5m2 (Wormhole); e4m3 packing on Blackhole is known from
  `per_token_cast_to_fp8` (truncating, #66091), not from the docs.
* **Other models.** Pass widths with 4 fp32 tiles a half: K3 4 columns (one pass, 1x128), K2.7 7 and DSv4.1-Flash 5
  would need unequal passes (4+3, 4+1): `pack_untilize_dest` needs a per-pass column offset. Not needed for GLM.

## Later: the combine (not in this plan)

* Wire row: `[H fp8][one 16-byte scale cell a down core][32-byte trailer]`: GLM 6144 + 512 + 32 = 6688 bytes, 54%
  of today's 12320-byte bf16 row; twice the landing slots in the same L1.
* Accumulator: unpack the fp8 row to fp32 DEST, one SFPU pass: x (gate x scale / q) per block, then the existing snap
  (replaces `mul_tiles_bcast_scalar`); the factor is per block, so per `SFPLOAD` (4 DEST rows x 8 columns of a flat
  row) a two-valued vector.
* `row_bytes` (`combine_host.cpp`) splits into the wire row and the output row (bf16).
* Where it pays (Nov 6 measurements): GLM 4x4 SP 10240 is X-line bisection bound (traffic ~1590 of 1986 us): fewer
  bytes is the lever. GLM 8x4 5120 is flat bound (small gain); K3 is weight bound (none).

## Results so far (Nov 25, one p150b; work in progress)

**M0 / M1 (device time, median of 5; GLM, row-major bf16 y, `down_fp32`).** Experiment hooks (env, not in the
program hash): `TT_FLAT_EXP_DOWN_HALF_SYNC`, `TT_FLAT_EXP_DOWN_PACK_DUMMY`, `TT_FLAT_EXP_DOWN_PACK_QUANT` (1: 1 x pass
width blocks, 2: 1 x 32; perf only - y is not valid). Harness: `test_flat_fp8_y_experiments.py`.

| | 8x4-5120 | 4x4-5120 |
|---|---|---|
| full sync (today) | 517.7 us | 997.2 us |
| half sync, passes of 3 | 526.8 | 1029.9 |
| + e4m3 quantize 1x96 | 535.5 (+3.4%) | 1033.8 (+3.7%) |
| + e4m3 quantize 1x32 | 555.4 (+7.3%) | 1072.0 (+7.5%) |

Device profiler, a pass: matmul ~3,800 cycles; quantize ~3,960 (1x96) / ~5,130 (1x32) - no margin; it hides only
because the down cores wait on weights.

**M2: bit exact** (`test_flat_y_fp8_quant.py`, 1x96 and 1x32: codes and scales vs the torch golden). Scales leave
through DEST: TRISC2 reads the spare tile (RISC view, `0xFFBD8000`). Blackhole findings:
* `pack_untilize_dest` of an odd block width (3) into an 8-bit format writes one tile more a row: the out CB row
  stride is padded (9 tiles for GLM).
* The packer's e4m3 subnormals are wrong (exponent 0 keeps the mantissa without the implicit 1); the unpacker
  flushes e4m3 subnormals to 0. So the format should flush |q| < 2^-6 to 0 (drop the SFPU's subnormal branches).

**Combine consume microbench** (`routed_experts_combine/test_combine_fp8_consume_bench.py`): one accumulator core,
rows landed in L1, snap 2^-14. Wire row `[H codes][32 cells x 16 B: 12 e8m0 bytes, one per 16 columns][trailer]` =
6688 B (bf16: 12320). The rows as flat 1024-code tiles; 16 codes = one DEST row; an FP16 factor per DEST row (srcB
column 0, ELWMUL SRCB_BCAST_COL as sparse_sdpa's fused dequant; fp8 unpacks to FP16 and the FPU takes one exponent
family): bf16 today 5,090 cycles a row, fp8 4,400; both within 1 bf16 ulp of their golden.

**Accuracy: e4m3 is the weak point.** Relative RMS error of y: bf16 0.17%, bfp8_b (old TtMoe, "acceptable")
0.7-1.3%, e4m3 + pow2 scale 2.7% (any block size: with pow2 scales e4m3 is scale invariant). Sign + 7-bit magnitude
(int8 sign-magnitude, a bfp8_b mantissa) under a pow2 block scale: 1x16 = bfp8_b, 1x32 0.8-1.7%, 1x96 0.9-2.9%.

**Next:** switch to int8 sign-magnitude + pow2 block scale (1x32 / 1x96): same wire bytes, the exponent byte =
log2(s) + 133 (bfp8_b's shared exponent), and the combine reads a flat tile as a native Bfp8_b tile (64 exponent
bytes, then the 1024 values) - today's gate broadcast multiply unchanged. To verify: the packer's fp32 -> Int8 is
sign-magnitude (and Int8 pack_untilize with a pass width of 3); the fidelity the gate x mantissa product needs.

## Implemented (Nov 25): quantized routed rows, fp8 and int8, end to end

**Format** (`common/kernels/quantized_row.hpp`): `[H values][16-byte cell a down core: one exponent byte a 16 values]`
then the trailer; fp8 = e4m3 x e8m0 (subnormals flushed), int8 = sign-magnitude x a power of two (read as Bfp8_b).
GLM: 6688 B a row (bf16 12320). Torch reference: `tests/.../moe_block/quantized_row_golden.py` (bit exact).

**Producer** (flat expert): `y_quant.hpp` on the down cores' PACK thread, `flat_routed_expert(..., y_quant=1|2,
y_quant_block=0|32|64|96)` -> y `[rows, H + 16 x down cores]` uint8. Checks: `test_flat_y_quant.py` (bit exact),
`test_flat_routed_expert_checks.py -k quant` (accuracy, determinism, perf). GLM 8x4 / 4x4 5120 device time vs bf16
row-major: int8 1x96 +3.5% / +4.2% (1.0% error vs the bf16 y), fp8 1x96 +8% (2.7%).

**Consumer** (combine accumulators): `quantized_row_stage.hpp` (reader: `RowStager`, issue / finish for local or remote
slots) and `quantized_row_weigh.hpp` (compute: `RowWeigher`); `accumulator_reader.cpp` / `accumulator_compute.cpp` use
them when `row_format != 0`. Checks on one chip: `test_combine_quantized_row_bench.py` (bench compute and the real
accumulator_compute.cpp; cycles a row: bf16 5090, fp8 4409, int8 4371). The accumulator reader compiles for every
format (run without a combine around it); the fused op itself needs a Galaxy - not run yet.

**Switch it on**: `MoEBlockConfig.routed_y_quant` (0 bf16, 1 fp8, 2 int8) and `routed_y_quant_block`; in the model
`MoEBlockFfnOptions(routed_y_quant="int8", routed_y_quant_block=32)`, or for any block / combine test
`MOE_ROUTED_Y_QUANT=int8 MOE_ROUTED_Y_QUANT_BLOCK=32` (read by `block_stages.py` and `moe_block_ffn.py`), e.g.

    MOE_ROUTED_Y_QUANT=int8 pytest tests/ttnn/unit_tests/operations/experimental/moe_block/routed_experts_combine/test_experts_combine_checks.py

Recommendation: int8 (1x32 or 1x96). fp8 is 2.5-4x the error of bfp8 (the old TtMoe y) and costlier on the producer.
