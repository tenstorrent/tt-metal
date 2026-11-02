# Perf 1 tournament — shared brief (read this first)

Repo root: the tt-metal checkout containing this file. Op: `ttnn/ttnn/operations/conv2d_nhwc/`
(baseline = git HEAD `ed24abbe3ec`, which carries permanent `MaybeDeviceZoneScope` zones).
Read `op_design.md` (Blocking Model, regime R4) and `changelog.md` (Refinements 6 and 9 explain the
BLOCK path, the act exchange, the weight ring and what was already tried and measured).

## Focus case (mandatory primary target)
SDXL UNet level-1 3x3: input (1,64,64,640) NHWC, C_out 640, k3 p1 s1, BLOCK_SHARDED 10x10 grid,
shard (410, 64), bf16 activations x bf8b weights, bias, ROW_MAJOR. Achievable target 149,049 ns
(= 201.2k cycles at 1350 MHz). Runner: `tests/ttnn/unit_tests/operations/conv2d_nhwc/test_conv2d_nhwc_perf_probe.py::test_perf_probe_block_sdxl_l1_bf16_bf8b`.

Plan the host picks (regime R4): S=10 channel slices (64 ch each) along grid x, g_m=10 along y;
per core bm=13 tile-rows, bn=2 out tiles, K_t=180, bk=3, num_k_blocks=60 (6 own per slice),
tilize_w=3, act_recv_depth=2, act_send_depth=1, act_tilize_lead=2, act_channels=1,
weight_depth=6, rotating weight sender, tap_major=False. Act round = 13x3 bf16 tiles = 80 KB
multicast to the 9 other cores of the row. Weight block = 3x2 bf8b tiles (6.5 KB) down the column.

Precision contract is FIXED: LoFi, fp32_dest_acc_en=True (COMPUTE_POLICY_BY_WEIGHT), dtypes as
above. Never change fidelity / fp32 acc / approx / dtypes. Gate: PCC >= 0.995, rel-RMS <= 0.05
(run_sharded asserts it).

## Measured baseline (BH p150b, 1350 MHz, per-core max TRISC_1 KERNEL cycles, one fresh run)
full 237.3k. Cumulative payload ablation (sync kept): -act payload 196.5k; -weights 197.1k;
-gather 179.6k; -tilize 167.3k; -matmul (everything stubbed) 84.8k. Solo: -matmul 224.4k,
-weights 224.0k, -gather 217.2k.
Zones (mean per core): compute_matmul 232k (whole helper call); writer_act_wait 137.6k over 54
rounds; reader_rm_reserve 90k of reader_gather_kblock 187k (reader is mostly blocked on a full
cb_act_rm); compute_tilize_own 6 x 4.5k = 27k; bias+untilize 5k; writer_act_reserve 2.8k.
Reading: matmul payload ~82k (~17.6 cyc/tile, ~90% of LoFi peak — near roofline). An 85k
handshake floor (60 rounds x ~1.4k with zero payload) runs SERIALLY with the matmul instead of
hiding it. act_ack_lead is hard-wired to act_channels (=1) in the writer.

## How to experiment (rules)
- Work ONLY in your own dir `ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/<your_slug>/`.
  Never edit the real op, `ttnn/cpp/ttnn/kernel_lib/`, tests, or anything outside your dir. Never commit.
- Easiest isolated bench: copy the op package (`__init__.py`, `conv2d_nhwc.py`,
  `conv2d_nhwc_program_descriptor.py`, `kernels/`) into `<your_slug>/variant/` (relative imports
  work; KERNEL_DIR follows __file__), import it under a unique module name via importlib, and write
  a pytest in your dir that runs baseline (real op) and candidate (your variant) on the focus
  case with run_sharded-equivalent correctness checks. If you need a kernel_lib change (e.g. mcast
  protocol), copy that header into your dir and include it by path from your variant kernels.
- Measure: `scripts/run_safe_pytest.sh --profile <test> -k <case>` then
  `python3 -I ttnn/ttnn/operations/conv2d_nhwc/perf_experiments/tools/zones.py` (per-core max
  KERNEL cycles per RISC + zone totals; check the marker count < 250). The ops CSV durations are
  broken by cross-core clock skew on this box — use the per-core max TRISC KERNEL cycles.
  One fresh run per variant (no trial loops). Device runs serialize on a flock with other agents —
  be economical. Foreground only. If the remote JIT server errors (`__xtt_vector8`), local compile
  still works; pass `--no-jit-server` if it blocks.
- Domain sweep (guard set, same R4/R5 path): ResNet BLOCK `test_perf_probe_resnet_block` (env
  `R9_CASE` filters: layer3 downsample / layer3 3x3 / layer4 3x3), `test_guard_block_bf16_nobias`,
  `test_guard_width_strided`, `test_perf_probe_block_28x28x256`. Report baseline vs candidate on each
  you run; report where the idea is incorrect / inexpressible / measurably slower. Flat or
  untested is NOT an exception.
- Return (concise): verdict WIN/NULL/REGRESSION/BLOCKED; baseline vs candidate cycles on the focus
  case and each guard you ran; precision (PCC, rel-RMS); domain + exceptions; exact diff summary of
  what to change in the op (files/functions) so the coordinator can graduate it; any raw-LLK use
  with its justification; your artifact dir.
