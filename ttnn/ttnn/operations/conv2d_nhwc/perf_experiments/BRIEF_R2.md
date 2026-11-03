# Perf 2 tournament — shared brief (read this first, then BRIEF.md for rules/tools)

Op: `ttnn/ttnn/operations/conv2d_nhwc/` at git HEAD `22c5b903433` (Perf 1 graduated: fast gather,
weight ring on the reader RISC / NOC_0). Read `changelog.md` "## Perf 1" and `BRIEF.md` (same rules:
work only in your own `perf_experiments/<slug>/`, never edit the real op / kernel_lib / tests, never
commit, foreground device runs, one fresh profiled run per variant, per-core max TRISC_1 KERNEL cycles
via `tools/zones.py`, marker budget < 250 per RISC).

## Focus case (unchanged, mandatory)
SDXL level-1 3x3, (1,64,64,640) -> 640, BLOCK 10x10, shard (410,64), bf16 act x bf8b weights, bias,
ROW_MAJOR. Precision contract FIXED: LoFi, fp32_dest_acc_en=True, fp32 partials (cb_partials Float32),
dtypes. Gate PCC >= 0.995, rel-RMS <= 0.05 (current: 0.999776 / 0.02017). L1 peak 590,336 B,
target 618,992 B (graph-capture peak; do not exceed the target).
Plan: S=10 slices, bm=13, bk=3, bn=2, 60 K-blocks (6 own/slice), act round = 13x3 bf16 tiles = 80 KB
row multicast from the rotating owner (writer BRISC, NOC_1), cb_act_tiles landing ring depth 2
(160 KB), cb_act_send depth 1 (80 KB), cb_act_rm = one whole own K-block (80 KB), partials 106 KB.

## Round-2 measured breakdown (BH p150b 1350 MHz, cycles, one fresh run each)
Full: 215.4k (TRISC_1 max); BRISC 207.9k; NCRISC 193.8k.
Cumulative payload peel (sync kept):
| step | TRISC_1 max |
|---|---|
| full | 215.4k |
| - act payload (64 B sends) | 193.4k |
| - weights (64 B reads/sends) | ~192k |
| - gather | 177.6k |
| - tilize | 174.3k |
| - matmul (everything stubbed) | 62.6k |
Solo: - matmul 165.1k; - gather 198.8k.
Zones (mean/core): compute_matmul 208k (whole helper call), writer_act_wait 169k over 54 rounds,
reader_rm_reserve 61k, compute_tilize_own 27k, bias+untilize 5k.

Reading: the handshake floor fell to 62.6k (~1.04k/round) now the weights are off the writer. The
matmul, peeled LAST, adds 112k over that floor, versus a math floor of 4680 tile-matmuls x ~16 cyc
~= 75k. So compute and the exchange are serialising per K-block (ring depth 2, ack lead 1), and/or
the per-K-block compute (78 matmuls + 26 fp32 packer-L1-accumulate packs at bk=3) costs far more than
the math. Then act payload 22k, gather startup bubble 15k, tilize 3k, weights ~1k.
Data-movement ceiling: each core ingests 54 x 80 KB = 4.3 MB of act at <= 64 B/cyc ~= 67k; the
whole op ideal is roughly max(matmul ~75-82k, act ingest ~67k) + startup.

## Ablation bench you can copy
`perf_experiments/r2_breakdown/` = the HEAD op copied into `variant/` with stub defines
(STUB_ACT, STUB_W, STUB_GATHER, STUB_TILIZE, STUB_MATMUL) and a module-level `DEFINES` list in the
variant descriptor; `run.sh <label>` with env HF_IMPL / HF_DEFINES / HF_SET / HF_CASE
(focus | l3ds | l3 | l4 | g28 | g28nb | width). Copy it into your dir (rename the module name
`r2b_conv2d_variant` to something unique) rather than editing it.
Guard set: HF_CASE l3ds / l3 / l4 / g28 / g28nb / width, plus
`tests/ttnn/unit_tests/operations/conv2d_nhwc/test_conv2d_nhwc_perf_probe.py::test_perf_probe_block_sdxl_l0_ds_bf16_bf8b`.

## Return (concise, <= 40 lines)
verdict WIN/NULL/REGRESSION/BLOCKED; focus baseline vs candidate cycles (and per option if a menu);
precision (PCC, rel-RMS) per option; L1 peak delta; guard results; domain + exceptions (only
incorrect / inexpressible / measured-regression); exact diff summary for graduation (files, functions,
descriptor knobs); raw-LLK use + justification + which helper and why (capability vs ergonomics, what
exactly was missing); artifact dir.
