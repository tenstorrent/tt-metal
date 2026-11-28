# ttnn.bringup.grouped_matmul (grouped_matmul_ttnn)

## What it does
Since NC.1 (2026-11-27) the native C++ op `ttnn.grouped_matmul` (ttnn/cpp/ttnn/operations/grouped_matmul: device
operation, planner, program factory, kernels under device/kernels; the device program cache, no prepared program);
`ttnn.bringup.grouped_matmul` calls it. The pre-port Python route (prepared ProgramDescriptor) is kept for tests only as
`ttnn.bringup.grouped_matmul_ttnn.generic.grouped_matmul_generic`. Originally a new op (Python ProgramDescriptor, no
TTNN source). y[g] = x[g] @ w[g] for x [1, G, M, K], w [1, G, K, N] -> [1, G, M, N]
(TILE, DRAM interleaved, any tile dtypes). One grid row a group, GX = min(grid width, N / 32) columns; core (cx, g)
computes all M rows x PN(cx) column tiles of group g:
- x: the row's column-0 core reads each [M x block_w] K block from DRAM and multicasts it along the row (READY / VALID
  semaphores; every wait bounded);
- w: every core reads its own [block_w x PN] weight blocks (no weight tile read twice);
- compute: fp32 DEST accumulates a block's products, the packer adds them to Float32 partials in L1 (an fp32 sum over
  the blocks); one copy + pack to the output dtype at the end; output written by each core.
Always dispatched as a prepared program (ttnn.bringup.prepare_program / run_prepared; x, w, y addresses patched).

## Consider it instead of
- `ttnn.matmul` with one weight per batch at small M (2D multicast, fuse_batch=False): it splits only M over the grid
  rows and loops the batches on every core (55 cores at M = 160 on an 11 x 10 grid).

## Options
- `dtype` (default x's): output dtype. `compute_kernel_config`: math fidelity, fp32 DEST, approx (default HiFi2, fp32).
- `output_tensor`: the caller's buffer (exact spec), returned as is. `memory_config`: of a new output.
- `max_grid` (CoreCoord or (x, y), default the whole grid): cores from (0, 0) .. (max_grid - 1): a Galaxy's 110-core
  matmul cap (agent rule 18).
- `block_w` (default the widest of 32, 16, 8, 4, 2, 1 that fits 1 MB of L1): K tiles a block.
- `depth` (default 2): K blocks of x and of w buffered a core.
- `rt` (default 1): row tiles a DST block (rt x PN tiles; full-sync DEST when it exceeds half of DEST). Bank path
  (FIX.42): a core whose rt x PN exceeds DEST (8 tiles at fp32 DEST) runs its columns in sub-blocks of DEST / rt tiles
  (define GMM_N_SUB; before FIX.42 refused); shapes within DEST keep the kernel path they had. The interleaved path
  still refuses rt x PN over DEST.
- `supported(x_shape, w_shape, x_dtype, w_dtype, out_dtype, device, max_grid=None, block_w=None)`: whether a call has a
  layout (cached).
- `w_layout` (default None: the interleaved path above): a `bank.BankLayout` (`bank.plan(device, G, K, N, max_cores,
  cores_per_bank=12, shape="band" | "row" | "rect", sender="start" | "anchor", order="row" | "col", contiguous=False, grid=None)`,
  None when the shape has no even split / does not fit; "rect" = today's core set, one rectangle from (0, 0), grid row
  b = bank b; `order` the order a rectangle's cores take its columns; `contiguous` a side's bands on consecutive rows;
  `grid` the cores the layout may use from (0, 0), e.g. a matmul-capped 11 x 10 compute sub-device); `w` is then `bank.upload(w_torch, layout, mesh)` (WIDTH_SHARDED DRAM, one bank a 32-wide shard): every core
  reads only its own DRAM bank, one contiguous read a K block; same tiles, K order and output bytes as the interleaved
  path at the same block_w. Bank-path knobs: `w_inflight` (default 1, <= min(depth, 8)) weight K blocks in flight;
  `x_ahead` (default 1, <= min(3, depth - 1)) x blocks the sender reads ahead when their slots are free; `one_group`
  (default False) one kernel group for every core width (the core's column tiles a runtime arg, define GMM_PN_RT):
  fewer core ranges, the launch's host time back to the interleaved path's (WA.3).
- `diag` (diagnostic, timing probes only, wrong output): `("no_w", "no_x", "no_compute")` stubs.

## Measured
- 8x4 Galaxy, wo_a of deepseek_v41_flash_d_p (8 x [160, 4096] @ [4096, 1024] bfloat8_b in / out, 88 cores), slowest
  chip: stock 2D 0.175 ms; block_w 4, depth 4: 0.133 ms (block_w 8: 0.140, 32: 0.150); bit-identical to the stock call.
- Bank path (w_layout, 2026-11-27, WA.2, same probe style, tests/woa/test_woa_probe.py): 256 rows (block_w 2) 151.3 ->
  125.0 us (band, 10 cores a bank, depth 4, w_inflight 1, x_ahead 1); 160 rows (block_w 4) 131.5 -> 106.9 us (band 10,
  depth 4, w_inflight 2, x_ahead 1); host a call ~124-134 vs ~100 us (not trimmed yet); bit-identical. At 256 rows band 10 is compute-bound (each stub ~116-121 us).
- WA.3 (tests/woa/test_woa_layout_probe.py, model-like call with the output buffer passed; host = median of 5
  alternating rounds of 50 calls): band 10 with `one_group` 124.9 us (256 rows) / 104.1 us (160 rows), host a call
  (run_prepared alone) 81.0 / 52.9 us vs the interleaved path's 77.3 / 54.8 (two kernel groups: 108.8 / 99.5 us, 24
  core ranges); the Python wrapper 18 / 11 us either path. Today's core set ("rect", 11 a bank) 147.8 / 121.1 us.
  On the matmul-capped 11 x 10 grid (`grid=max_grid`: the east bands in columns 6-10, as a model inside a gather-hide
  sub-device needs): band 10 one group 130.3 / 114.2 us, today 151.1 / 131.7, host 84.3 / 53.9 vs 80.2 / 51.2 us.
- NC.1 native C++ op vs the Python prepared route (8x4, tests in deepseek_v41_flash_d_p tests/woa/test_gmm_native_layer.py,
  the model's wo_a knobs, output buffer passed): bit-identical (max abs diff 0, repeats 0, no program-cache growth on
  fresh buffers); host a warm call 20-26 us vs 63-85 us (interleaved 256 / 160 rows: 21.0 / 23.2 vs 85.2 / 63.2; bank:
  25.5 / 25.9 vs 68.9 / 67.7); device equal (149.97 / 130.59 / 129.48 / 112.28 vs 149.93 / 130.61 / 129.86 / 112.29 us).

- FIX.42 (8x4, op level, tests in deepseek_v41_flash_d_p tests/woa/test_qbob_bank_probe.py): G = 1 bank layouts (N
  split over the 8 banks; every cluster's sender reads all of x) for the attention's replicated projections, vs the
  stock call the model runs (slowest chip, 20 calls): wq_b [rows, 1280] @ [1280, 32768] bfp8 -> bf16, 256 rows 238.8 ->
  202.9 us (rect 11, block_w 4, depth 4), 160 rows 195.6 -> 151.4 us (row 11, block_w 4, depth 3); rel vs fp64 equal
  to the stock call's (1.68e-3), repeats 0. wo_b [rows, 8192] @ [8192, 5120]: slower (205.99 vs 157.7 / 161.2 vs
  139.2 us): with G = 1 each of the 8 clusters reads the whole 2.2 MB x (no-x stub 121.8 us at 256 rows).

## Limits
- G <= the grid's rows; batch shape [1, G]; M, K, N multiples of 32; interleaved DRAM tensors.
- The x sender reads x alone (one core a row); the weight reads go one block in flight a core (a read-ahead version
  measured slower).
- Cores: G x GX (88 on the 8x4). Bank path: banks x cores_per_bank (80 / 96 on the 8x4), refused above max_cores.
- Host a launch grows with the program's core ranges x kernel groups (WA.3: 24 ranges +23-45 us a call on 32 chips):
  keep a bank layout's kernels to few rectangles (`one_group`).
- Bank path: G x N / 32 must split evenly over the DRAM banks with each bank's share inside one group's columns as a
  whole number of rectangle columns; every bank's cores in a band (or a row) on its anchor's side; the x sender must sit
  at the rectangle's first core (sender="anchor" wraps NoC-0 multicasts around the chip: 2x slower); weights must go
  through `bank.upload` (a `from_torch` straight to the sharded config quantizes bfloat8_b differently).
- Bank path with G = 1 (one group over all banks): every cluster's sender reads ALL of x from DRAM and multicasts it
  to its own cluster only: 8 x the x traffic. Fine for a small x (wq_b, 0.35 MB); a wide-K x (wo_b, 2.2 MB) makes the
  call slower than the stock 1D in0 multicast (FIX.42). A grid-wide x multicast is not built.

## How to call
```python
from ttnn.bringup.grouped_matmul_ttnn import supported
if supported(tuple(x.shape), tuple(w.shape), x.dtype, w.dtype, ttnn.bfloat8_b, mesh, grid):
    y = ttnn.bringup.grouped_matmul(x, w, dtype=ttnn.bfloat8_b, compute_kernel_config=ckc, max_grid=grid, block_w=4)
```

Bank path:
```python
from ttnn.bringup.grouped_matmul_ttnn import bank
lay = bank.plan(mesh, G, K, N, max_cores=110, cores_per_bank=10)      # once, at load
wb = bank.upload(w_torch, lay, mesh)                                     # [1, G, K, N] torch -> bank tensor
y = ttnn.bringup.grouped_matmul(x, wb, w_layout=lay, dtype=ttnn.bfloat8_b, compute_kernel_config=ckc, block_w=2,
                                depth=4, w_inflight=1, x_ahead=1)
```

## Used by
- deepseek_v41_flash_d_p: tt/attention.py `_out_wo` (ATTN_ROOFLINE, P.81), the grouped wo_a; the bank path only by the
  WA.2 probe (tests/woa/test_woa_probe.py) so far.
