# grouped_matmul_ttnn changelog

Source: none (new op), deepseek_v41_flash_d_p P.81 (OWNER 2026-11-18: attention at its roofline; the grouped wo_a).

## Changes

### NC.1: the native C++ op (2026-11-27, deepseek_v41_flash_d_p NC.1)
- What: the tt_ops_code_gen native_migrate port applied as is (native checkout, git diff f39977aa64b 49384eeffde):
  ttnn.grouped_matmul in ttnn/cpp/ttnn/operations/grouped_matmul (device operation, planner, program factory, nanobind
  binding with the bank layout cast; kernels moved to device/kernels with a shared args header grouped_matmul_args.hpp);
  grouped_matmul.py keeps `plan` / `supported` / `column_split` and aliases `grouped_matmul = ttnn.grouped_matmul`. The
  pre-port Python route copied to generic/ (grouped_matmul_generic, kernels; test only). The port's pre-migration suite
  (tests/ttnn/unit_tests/operations/grouped_matmul/pre_migration) opens its mesh with FABRIC_2D (owner rule).
- Why: OWNER 2026-11-27: native C++ ops instead of Python prepared programs (host time a call, no Python program build).
- Measured (8x4, tests/woa/test_gmm_native_layer.py): bit-identical to the Python route at the model's wo_a calls (256 /
  160 rows, interleaved and bank, today's knobs), repeats 0, 0 program-cache entries added on fresh buffers / new data;
  host 63-85 -> 21-26 us a call; device unchanged.
- Cores: unchanged (interleaved G x GX = 88; bank 80 on the 8x4, <= 110).
- Files: ttnn/cpp/ttnn/operations/grouped_matmul/*, ttnn/cpp/ttnn/operations/CMakeLists.txt,
  ttnn/cpp/ttnn-nanobind/__init__.cpp, grouped_matmul.py, generic/, README.md, ../INDEX.md,
  tests/ttnn/unit_tests/operations/grouped_matmul/pre_migration/. Used by: deepseek_v41_flash_d_p tt/attention.py
  `_out_wo` (unchanged code).

### Initial op (2026-11-19)
- What: `ttnn.bringup.grouped_matmul(x, w, dtype=None, compute_kernel_config=None, output_tensor=None,
  memory_config=None, max_grid=None, block_w=None, depth=2, rt=1)`: y[g] = x[g] @ w[g] for x [1, G, M, K], w
  [1, G, K, N] (TILE, DRAM interleaved), one grid row a group, GX columns (N split as evenly as possible); the row's
  column-0 core reads x's [M x block_w] K blocks and multicasts them along the row (READY / VALID semaphores, bounded
  waits); each core reads its own [block_w x PN] weight blocks; fp32 DEST per block, Float32 partials accumulated by the
  packer in L1, one copy + pack to the output dtype at the end. `supported(...)`: whether a call has a layout.
- Why: the stock 2D multicast batched matmul splits only M over the grid rows: at M = 160 it runs on 5 x 11 = 55 cores,
  compute-bound (0.175 ms for deepseek_v41_flash_d_p's wo_a vs ~0.1 ms of DRAM). Here 8 x 11 = 88 cores.
- Measured (8x4, real-time profiler, slowest chip, 20 iterations; models/demos/deepseek_v41_flash_d_p/tests/perf/
  test_attn_proj_probe.py): wo_a 8 x [160, 4096] @ [4096, 1024] bfloat8_b: stock 0.175 ms; block_w 32 0.150, 16 0.144,
  8 0.140, 4 0.133 (depth 2 and 4 alike); rt 2 (full-sync DEST) 0.135. A reader issuing depth - 1 blocks ahead under
  NoC transaction ids: 0.143-0.155 (slower, not kept). 42.6 MB moved a call: ~320 GB/s at 0.133 ms.
- Tests: tests/unit/test_grouped_matmul.py (5 cases: the 8x4's wo_a at two block widths, one group, a 4 x 3 grid with
  N not divisible by the columns, a 1 x 2 grid; vs the stock batched ttnn.matmul (bit-identical on all) and float64;
  every chip equal; a cache hit on fresh buffers into a caller's output: the same bytes; refusals). Run on the model's
  mesh through the resident runner: models/demos/deepseek_v41_flash_d_p/tests/test_grouped_matmul_wo_a.py: passed.
- Needed by: deepseek_v41_flash_d_p P.81 (ATTN_ROOFLINE)
- Files: grouped_matmul.py, __init__.py, kernels/gmm_reader.cpp, kernels/gmm_compute.cpp, kernels/gmm_writer.cpp,
  tests/unit/test_grouped_matmul.py, README.md; ../__init__.py (PYTHON_OPS), ../INDEX.md

### Option w_layout: bank-local weights (2026-11-27)
- What: `grouped_matmul(..., w_layout=None, w_inflight=1, x_ahead=1, diag=())`. `w_layout` (default None: today's
  path, unchanged) = a `bank.BankLayout` from the new `bank.py`: `plan(device, G, K, N, max_cores, cores_per_bank=12,
  shape="band", sender="start")` (the G x NT (group, column) units an equal run a DRAM bank; a bank's cores a
  cores_per_bank / 2 x 2 rectangle ("band") or one grid row ("row") on its optimal NoC-0 worker's side, nearest its row;
  each core's PN columns K-major in its bank so any K block is ONE contiguous read; refuses an uneven split, a shape
  that does not fit, more cores than max_cores; no padding, OV-3), `regions`, `upload` (the experts' bank_host_tensor
  matrix, WIDTH_SHARDED DRAM, replicated). New reader kernels/gmm_bank_reader.cpp: each core reads only its own bank
  (get_noc_addr_from_bank_id), `w_inflight` K blocks in flight under transaction ids; the cluster's sender (the
  rectangle's first core) reads x and multicasts it to the rectangle (today's READY / VALID handshake, bounded waits),
  reading up to `x_ahead` later x blocks whose slots are already free (never waits for a slot). Compute and writer
  kernels: today's. Diagnostic `diag` stubs (compile-time defines GMM_DIAG_NO_W / _NO_X / _NO_COMPUTE in today's and
  the bank reader / the compute kernel; timing probes only, wrong output).
- Why: deepseek_v41_flash_d_p WA.2 (OWNER 2026-11-27): wo_a's interleaved weight is read one 1088 B tile at a time from
  banks across the chip; step 0 of the probe showed today's call reader-bound (no weight reads 0.71x / 0.54x of today
  at 256 / 160 rows, no compute 0.99x).
- Measured (8x4, real-time profiler, slowest chip, 20 calls; models/demos/deepseek_v41_flash_d_p/tests/woa/
  test_woa_probe.py; wo_a 8 x [rows, 4096] @ [4096, 1024] bfloat8_b, HiFi2 fp32 DEST): 256 rows (block_w 2) today
  151.3 us -> best bank config (band, 10 cores a bank = 80 cores, depth 4, w_inflight 1, x_ahead 1) 125.0 us; 160 rows
  (block_w 4) 131.5 -> 106.9 us (band 10, depth 4, w_inflight 2, x_ahead 1). Host a call ~124-134 us vs today's ~100
  (the layout's program-cache key is a digest computed once; the rest not trimmed yet). Bit-identical to today's path at the same
  block_w on every chip for all 870 swept configs, repeats bit-identical. Lessons: a NoC-0 multicast from a sender
  inside the rectangle (at the bank's anchor) wraps around the chip: 2x slower x stream (342 vs 155 us); an x
  read-ahead that blocks on a slot at depth 2 stalls the multicast on the sender's own compute (hence the non-blocking
  x_ahead); `from_torch(..., memory_config=<WIDTH_SHARDED>)` quantizes bfloat8_b differently from the plain TILE
  conversion (max |d| 0.0625), so `upload` converts plain and then places the tensor (`.to(mesh, mc)`).
- Tests: tests/unit/test_grouped_matmul_bank.py (8 device cases smallest first: G 8 [32, 256] @ [256, 1024] at several
  knob sets, band / row, start / anchor senders, a G 4 case (two banks a group), [96, 1024] float32 output at 10 cores a
  bank, the 8x4's wo_a at 160 and 256 rows; vs today's path at the same knobs bit-identical on every chip, vs torch,
  a repeat on fresh buffers (cache hit) bit-identical; refusals; a CPU packing test on a fake Galaxy grid). Run on the
  model's mesh through the resident runner (models/demos/deepseek_v41_flash_d_p/tests/woa/test_woa_fork_bank.py): 12
  passed. The existing cases (option off) were not re-run: the brief forbids fork regression suites on the 8x4; today's
  path is unchanged when no diag define is set (its kernels only gained #ifdef blocks), and today's call measured the
  same as before (151.2 / 131.6 us, PB.MM256 151.0, P.81 133.0).
- Cores: bank path 8 x cores_per_bank (96 at 12, 80 at 10; <= max_cores, the 110-core matmul cap, agent rule 18).
- Needed by: deepseek_v41_flash_d_p WA.2 (probe only; not called by the model yet: WA.3)
- Files: bank.py, grouped_matmul.py, __init__.py, kernels/gmm_bank_reader.cpp, kernels/gmm_reader.cpp,
  kernels/gmm_compute.cpp, tests/unit/test_grouped_matmul_bank.py, README.md; ../INDEX.md

## 2026-11-27: bank path host cost and core sets (deepseek_v41_flash_d_p WA.3)
- What: (1) option `one_group` (default False: the behaviour before): the bank program as one kernel group for every
  core width, the core's column tiles a runtime arg of the reader, writer and compute kernels (compile-time define
  GMM_PN_RT; without it every kernel is unchanged). (2) `bank.plan` shape "rect" (today's interleaved path's core set:
  one rectangle from (0, 0), grid row b = bank b), `order` ("row" default | "col") and `contiguous` (default False).
  `bank.plan(grid=...)` (default None: the device's grid, as before): the cores a layout may use from (0, 0) (the
  model's bank cores must stay in its matmul-capped 11 x 10 compute sub-device: a packed chunk's wo_a runs inside a
  gather-hide region, where a core in column 11 fails "Programs must be executed on a single sub-device"; 11 wide:
  band 10 one group 130.3 / 114.2 us vs 124.9 / 104.1 on the full 12-wide grid).
  (3) The prepared-program key no longer carries str() of the three memory configs (x and out are checked
  INTERLEAVED DRAM, w INTERLEAVED or the layout's bank tensor, so they add nothing) and keys the dtypes as values; the
  ckc key is cached per config object; the bank weight's shard check runs once a tensor object (`_check_bank_w`).
  Same programs, same cache hits.
- Why: OV-78 / OV-79 (owner 2026-11-27): the bank call cost +24-34 us host a call in WA.2; the owner asked whether
  today's core set would do. Measured split (tests/woa/test_woa_layout_probe.py, 8x4): the Python wrapper is equal
  (18 / 11 us at 256 / 160 rows), str() of a memory config only ~1-2 us; the delta is run_prepared's enqueue, which
  grows with the core ranges: band 10 as two kernel groups = 24 core ranges, +23-45 us; `one_group`: 4 ranges, 81.0 /
  52.9 us vs the interleaved path's 77.3 / 54.8 (noise ~+-8 us). Device: band 10 one group 124.9 / 104.1 us (two
  groups 125.2 / 106.1); "rect" 11 147.8 / 121.1 us (today 151.4 / 131.8): the clusters are what buys the gain.
  `order="col"` 8 ranges, host between; `contiguous` bands 4 ranges but 128 / 123-128 us device (worse).
- Tests: tests/woa/test_woa_layout_probe.py (every config bit-identical vs today's call at 256 and 160 rows, a repeat
  bit-identical); the model's off == on test tests/woa/test_woa_offon.py (WA.3 gate). Fork regression suites not run
  on the 8x4 (brief); the interleaved path's kernels are unchanged without GMM_PN_RT.
- Cores: unchanged (bank path banks x cores_per_bank: 80 on the 8x4 with 10 a bank, <= 110).
- Files: grouped_matmul.py, bank.py, kernels/gmm_bank_reader.cpp, kernels/gmm_compute.cpp, kernels/gmm_writer.cpp,
  README.md; ../INDEX.md. Used by: deepseek_v41_flash_d_p tt/attention.py `_out_wo` under WOA_BANK=on.

### 2026-11-28: bank-path column sub-blocks (GMM_N_SUB) for G = 1 layouts (deepseek_v41_flash_d_p FIX.42)
- What: the C++ op's bank path accepts a core whose rt x column tiles exceed DEST: gmm_compute.cpp under define
  GMM_N_SUB=<SN> cuts the core's PN columns into ceil(PN / SN) sub-blocks (as even as possible, wider first), each its
  own DST block (rt x width, matmul_block_init on a width change), packed to the same fp32 partial slots; SN = DEST /
  rt (8 / rt at fp32 DEST; full sync as before). The program sets the define only where the shape was refused before
  (planner.hpp bank_n_sub, device_operation check_bank_dest_fit, program_factory bank_descriptor), so every shape that
  ran keeps its compiled kernel unchanged; rt alone over DEST is still refused; the interleaved path is unchanged.
- Why: OWNER 2026-11-27, the wq_b / wo_b bank-layout probe (G = 1, N split over the 8 banks): wq_b's 1024 column
  tiles give 12-16 a core under the 110-core cap (> 8): "rt 1 x 13 column tiles exceed DEST" before this.
- Measured (8x4, tests/woa/test_qbob_bank_probe.py in deepseek_v41_flash_d_p): wq_b 256 rows 238.8 -> 202.9 us, 160
  rows 195.6 -> 151.4 us vs the stock call; wo_b slower (per-cluster x reads); see README "Measured".
- Tests: tests/unit/test_grouped_matmul_nsub.py (5 cases: rt 1 vs an rt that needs sub-blocks bit-identical, one and two
  kernel groups; cores over DEST at rt 1 vs fp64 of the device's inputs and a repeat, wq_b's real shape included): 5
  passed on the 8x4 (resident runner via tests/woa/test_gmm_fork_nsub.py). Full fork regression not run on the 8x4
  (OV-25); the gate's NC.1 units (test_acceptance, test_grouped_matmul, test_gmm_native_layer) re-run.
- Cores: unchanged (the layout's; 88 for row / rect 11, 80 for band 10, <= 110).
- Files: ttnn/cpp/ttnn/operations/grouped_matmul/device/{kernels/gmm_compute.cpp, grouped_matmul_planner.hpp,
  grouped_matmul_device_operation.cpp, grouped_matmul_program_factory.cpp}; tests/unit/test_grouped_matmul_nsub.py;
  README.md; ../INDEX.md. Used by: deepseek_v41_flash_d_p tests only (op-level probe; no model wiring).

### Brought to model_bringup (2026-11-28, deepseek_v41_flash_d_p MB.NC)
- What: the bank option (w_layout, bank.py; WA.2 / WA.3) and the native C++ ttnn.grouped_matmul (NC.1), taken as is from dnijemcevic/model_bringup_batching (the entries above); the pre-port Python route stays
  in `generic/` for the op test only.
- Why: OWNER 2026-11-27, one batch of the batching branch's C++ ports on model_bringup (no packed path here).
- Measured (8x4, 5120 chunk shapes, models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/results/MB.NC_layer.md):
  native == the Python prepared route bit for bit, repeats 0, no program-cache growth on fresh buffers.
- Files: as the entries above; tests models/demos/deepseek_v41_flash_d_p/tests/native/, tests/woa/test_gmm_native_layer.py.

### Brought to model_bringup (2026-11-28, deepseek_v41_flash_d_p MB.QB)
- What: the GMM_N_SUB bank-path column sub-blocks (FIX.42 above), taken as is from dnijemcevic/model_bringup_batching (47b34099c19; the C++ op byte-identical to that branch), and its first model caller: wq_b's G = 1 bank call (QB.1 3ca2cf6a70d, WQB_BANK, default on) at 160 rows a chip (row 11 bw 4 depth 3 w_inflight 1).
- Measured (8x4, models/demos/deepseek_v41_flash_d_p/tests/native/test_native_mb_qb.py, results/MB.QB_layer.md): wq_b at 160 rows device 177.8 -> 152.0 us vs model_bringup's stock call (MM_TUNED160's 1D config), host 92.0 -> 53.4 us; bit-identical to the stock call on these inputs, rel vs fp64 1.680e-3 both; repeat 0; 0 program-cache growth. Cores: 88 (<= 110).
- Tests: the gate case above; tests/unit/test_grouped_matmul_nsub.py brought (via the model's tests/woa/test_gmm_fork_nsub.py), not re-run here (the C++ is the batching branch's, 5 passed there).
- Used by: deepseek_v41_flash_d_p tt/attention.py `_wqb_mm` (WQB_BANK=on) and `_out_wo` (WOA_BANK=on).
