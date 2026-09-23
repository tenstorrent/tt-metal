# noc_region: static per-region NoC assignment (tilize perf tournament, one idea)

WH B0 n150, 64 Tensix cores, AICLK 1000 MHz (cycles == ns). All variants bit-exact (`torch.equal`).

## What was tried
The real op is untouched. `noc_region.py` swaps `tilize_program_descriptor.ttnn` for a proxy module
(test-scoped `monkeypatch`). At ProgramDescriptor time the proxy splits the reader and writer
`KernelDescriptor`s into one descriptor per region: same source, same CT args, that region's RT args.
Each descriptor gets its own static `DataMovementConfigDescriptor(processor, noc, DM_DEDICATED_NOC)`.
Processors stay put (reader on NCRISC, writer on BRISC). "swap" = reader on NoC1 and writer on NoC0.
The kernels use `noc_index` everywhere, so the kernel sources are unchanged.

Region rules are written in NoC0 physical coordinates (logical x 0..7 -> NoC0 x {1,2,3,4,6,7,8,9},
logical y 0..7 -> NoC0 y {1,2,3,4,7,8,9,10}). Note: `device.worker_core_from_logical_core` returns
WH *virtual* coordinates (18+), not NoC0 coordinates. My first sweep used it and was invalid.

Secondary lever (`noc_region_split.py`): after the op builds its row split, the RT args
(row_start, core_row_tiles) are rewritten so each Tensix core gets a weighted number of tile-rows.
CT args are not changed. Schemes: `oracle` (rows ~ 1/T_core, with T_core taken from the measured
BRISC end grid of the focus shape), `oracle_sq`, `geo_tail`, `geo_tail2`.

## Run
```
NOCR_RULES=unpatched,swap_all,swap_tl NOCR_ASSIGN=-      NOCR_CASES=focus scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/tilize/test_tilize_perf1_noc_region.py
NOCR_RULES=unpatched               NOCR_ASSIGN=-,oracle NOCR_CASES=focus,h16k_w32 ...
python ttnn/ttnn/operations/tilize/perf_experiments/noc_region/analyze.py <report_dir> <labels> [--grid N]
```
Rules and cases are listed in `noc_region.py` (RULES) and in the test (CASES). Collection order is
case, then rule, then assign.

## Result: REGRESSION (primary) / shape-specific NULL (secondary)
Focus shape [1,1,16384,64]. The same-session baseline is 23.8-24.5k ns. Every static swap that moves
real traffic is slower: swap_all 51.3k, swap_tl 24.9k, swap_tail9 27.7k, checker 35.2k. A swapped
core's own reads (NoC1) and writes (NoC0) run against the DRAM-column geometry, so the swapped core
becomes the new tail. A large swapped set also slows the whole grid.

Both RISC-Vs of one Tensix core on the same NoC is inexpressible in DM_DEDICATED_NOC. Each NoC's
transaction counters are shared per NoC, so one RISC-V's barriers would count the other's traffic.
That combination needs DM_DYNAMIC_NOC, which is out of scope.

**Measurement artifact (important).** On small ops, *any* split of reader/writer into two kernel
groups lowers DEVICE KERNEL DURATION by 5-13%. The `split_one` control (two groups, no NoC change)
shows the same drop. The cause is that in the one-group program the data-movement RISC-Vs start about
300 cycles after the TRISCs. DEVICE FW DURATION is flat to slightly worse (about +0.7 us firmware
overhead per extra group). The drop is not a real speedup. Judge kernel-group changes on FW duration.

Row rebalance `oracle`: -2.5% on the focus shape (median of 5 runs, 23576 vs about 24100 ns, in both
kernel and FW duration). It regresses [1,1,16384,32] by 11% (16.0k vs 14.4k ns) and is flat
elsewhere. Its weight table was fitted to one shape, so it is not a rule. The geometry rule
`geo_tail2` was flat on the focus shape (median 24287 vs about 24082 ns).
