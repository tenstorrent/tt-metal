# one_mblock_nsplit (Perf 19): N split on the one-M-block-per-core programs

On small interleaved convs the K-blocked solve gives each core one M-block. For example, 66x128 128->512 (P15, idx 34)
has 256 output tile-rows, so it runs 86 cores with 3 rows x 16 columns each, and one sender streams the whole [K, N]
weight matrix to all of them. Here the work grid splits into g = 2 N groups. Each group has its own multicast group and
sender for its [K, N/2] slice. A core takes ceil(M / 55) tile-rows x N/2 columns, so P15 becomes 104 cores x 5 x 8.
The bk is unchanged and so are the dot products and their order, so the output is bit-identical.

## Change (`integration.patch`, op_base -> op_copy, host only)
- `select_blocking` runs after the resident N split. When the solved walk has one M-block per core, the input is
  interleaved, there is no K split / fold / N slice, and the stride is 1, it tries each g in `ONE_MBLOCK_NSPLIT_GROUPS`.
  The solve is `m_split(g, cores / g)` at the base bk, with bm = core_m (one M-block) and bn = N/g; the weights are not
  resident (`solve(..., streamed_split=True)`). A split is taken when:
  - the busiest core's tile-MACs do not grow (core_m * N/g <= base core_m * N), and
  - N/g >= `ONE_MBLOCK_NSPLIT_MIN_BN` (8).
- The descriptor's N split assert now also accepts streamed weights with one M-block per core.
- No kernel changes. The kernels already take the per-core matmul column start (writer RT arg 6, used by the weight
  round reads and the store), the bias / store column start, and one multicast group per N group (Perf 17).
- Knobs: `ONE_MBLOCK_NSPLIT_GROUPS = (2,)`, `ONE_MBLOCK_NSPLIT_MIN_BN = 8`, `ONE_MBLOCK_NSPLIT_FORCE` (None: the solve
  decides, 0: off, g: force g groups).

## Results
Blackhole p150b, 11x10 grid, bf16 HiFi4 fp32 DEST, bias, l1_small 65536. Device kernel us, median of 3 calls.
Every variant was bit-identical to base (sha1 of the output bytes). PCC vs torch is 0.9999985 on every case.

| case | base blocking | split blocking | base | g=2 | g=4 |
|---|---|---|---|---|---|
| 34 RM 66x128 128->512 (P15) | 86 c, 3 x 16 | 104 c, 5 x 8 | 140.1 / 139.3 / 140.1 | 119.3 / 119.1 / 118.7 (-15%) | 129.4 |
| t34 TILE 66x128 128->512 | 86 c, 3 x 16, bk 4 | 104 c, 5 x 8 | 144.0 | 127.6 (-11.4%) | |
| x0 1x1 TILE 128x128 32->512 (FLUX idx 0) | 103 c, 5 x 16 | 104 c, 10 x 8 | 112.6 / 111.9 | 107.8 / 106.8 (-4.5%) | 108.1 |
| rB RM 66x128 64->512 | 86 c, 3 x 16 | 104 c, 5 x 8 | 92.4 | 83.5 (-9.6%) | |
| rC RM 66x128 32->512 | 86 c, 3 x 16 | 104 c, 5 x 8 | 68.2 | 62.4 (-8.4%) | |
| rD RM 34x64 128->512 | 64 c, 1 x 16 | 64 c, 2 x 8 | 72.4 | 52.1 (-28%) | 56.8 (noisy) |
| rA RM 66x128 128->256 (bn 4: not taken) | 86 c, 3 x 8 | 104 c, 5 x 4 | 75.4 | 76.0 (flat) | |
| 32 RM 130x256 32->128 (P16, bn 2: not taken) | 103 c, 10 x 4 | 108 c, 19 x 2 | 70.9 | 87.7 (+24%) | 116.1 |
| t32 TILE 130x256 32->128 (not taken) | 103 c | 108 c | 62.8 | 74.5 (+18.6%) | |
| 31 RM 66x128 32->128 (P17, bn 2: not taken) | 86 c, 3 x 4 | 104 c, 5 x 2 | 25.5 | 29.1 (+14%) | 38.3 |
| t31 TILE 66x128 32->128 (not taken) | 86 c | 104 c | 23.9 | 24.9 (+4.4%) | |

With the gate (`final` run, base / ns alternating), the split is taken on 34 (140.1 -> 118.7) and x0 (111.9 -> 106.8).
On 32 and 31 the program is base's (70.6 / 70.4 and 25.6 / 25.7).

How the win divides:
- The busiest core's tile-MACs drop where more cores join (P15: 1728 -> 1440, -17%, about -14 us of HiFi4 matmul).
- The weight stream splits over g senders, each sending rounds 1/g the size. This is the whole gain on rD and x0, where
  tile-MACs per core are equal: 34x64 128->512 -28% on the same 64 cores.
- The cost is per-row work. Each core's tile-rows grow by about g x cores_before / cores_after, so the fill, tilize, bias
  and store per row grow too. Where a group keeps only 2 or 4 output tiles (C_out 128 / 256), the split adds as much as
  it saves, or more. Hence the gate on N/g >= 8.

## Domain
- **Reached:** interleaved (DRAM ROW_MAJOR and TILE) stride-1 programs where the K-blocked solve runs one M-block per
  core, C_out >= 512 (N/2 >= 8 tiles), and the split's one-block blocking fits L1 at the base bk.
  - Among the 17 focus cases that is only P15 (idx 34). The others run more than one M-block (all but 31 / 32 / 34), or
    fall to the bn gate (P16 and P17, C_out 128).
  - Among the FLUX 1x1 single-device TILE cases it is only idx 0 (128x128 32->512). idx 1 (512->32) has N = 1. idx 8
    and idx 16 (512->512) run 5 M-blocks.
- **Exceptions:**
  - N/g < 8: measured regression. P16 +24%, P17 +14%, TILE versions +18.6% / +4.4%; 128->256 flat. The gate excludes
    these.
  - g = 4: measured slower than g = 2 on P15 (129.4 vs 119.3) and no faster on x0 / rD, so only g = 2 is tried.
- **Untested:** stride 2. The patch keeps the existing N split's `stride == 1` gate; no FLUX stride-2 case runs one
  M-block per core. Also untested: LoFi on interleaved input. The matmul share is smaller there, but the weight-stream
  share is unchanged.
- **Sharded programs (Panoptic): unreachable.** The gate requires `not g.sharded_output and not g.k_split`. With
  `n_groups == 1` every descriptor change is the identity.
- **L1:** the footprint shrinks on every reached case (P15 1282 -> 945 KB, x0 740 -> 682 KB, TILE P15 784 -> 656 KB).
  The blocking solve changes only on the reached cases.

## Re-run
`run.sh <label> <cases> <variants> [ENV=...]`, for example `run.sh f 34,x0 base,ns OMN_ORDER=alt`.
- Cases: `<n>` (bh2x4 RM), `t<n>` (bh2x4 TILE), `x<n>` (1x1 TILE), `rA..rD` (extra RM regimes, see `test_omn.py`).
- Variants: base, ns (gated), ns2 / ns4 / ns8 (forced), off. `base_abl` / `ns2_abl` (`OMN_ABL=1
  CONV2D_3X3_ABLATE=...`) need `op_base_abl/` and `op_copy_abl/`: copies of op_base / op_copy with
  `ablation_r19/apply_hooks.py` applied.
- Host-only blocking per case: `python3 -I omn_plan.py` (focus) and `python3 -I omn_plan.py reach` (every FLUX case).
