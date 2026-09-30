# sinkhorn_sfpu_fast (E2, perf round 2): a faster SFPU Sinkhorn at the same fp32 precision

Measured on BH p150 at 1350 MHz, in-process DEVICE KERNEL DURATION. The precision contract is unchanged for every
variant: fp32_dest_acc_en=True, HiFi4, approx off, fp32 SFPU math, 2 Newton steps per reciprocal, 20 iterations.

## Layout
| path | what |
|---|---|
| `bench/sinkhorn_variants.hpp` | every Sinkhorn variant (v0 = the op's code verbatim .. v7) on one coefficient-major DEST tile |
| `bench/sinkhorn_bench_compute.cpp`, `bench/test_sinkhorn_bench.py`, `bench/run.sh` | the 1-core, 1-tile micro-bench. `test_slot_map` checks the (slot, lane) -> tile map. `test_correct` checks bitwise vs v0 and the error vs an fp64 torch Sinkhorn (8 logit regimes, scales 1..40, uniform rows, +60 offsets). `test_perf` reports the slope over REPS=1 and REPS=11 |
| `bench/gen_sinkhorn_lm.py` -> `bench/sinkhorn_lm_gen.hpp` | the generator of the SFPLOADMACRO pass schedule, plus a rule checker (see below) |
| `op/` | a copy of the op's descriptor and kernels, with the graduated Sinkhorn (v6) in `op/kernels/mhc_pre_compute.cpp` |
| `bench/skb_op_plugin.py` | a pytest plugin that runs the real op entry point with `op/` (golden / unit suites) |
| `bench/test_op_perf.py`, `bench/op_perf.sh` | whole-op base-vs-grad runs, interleaved, with bitwise / precision checks |
| `bench/dst_raw_check.py` | a static Dst store->load distance check on a TRISC1 disassembly |
| `graduation.patch` | the patch against the real `kernels/mhc_pre_compute.cpp` (the diff of `op/kernels`). `git apply --check` is clean |

Commands:
- `bench/run.sh correct|perf|slot_map` (env `SKB_VARIANTS`, `SKB_ITERS`, `SKB_HALF=1` = second DEST half)
- `bench/op_perf.sh` (env `SKB_SHAPES`, `SKB_DTYPES`, `SKB_OPS`, `SKB_KNOBS`, `SKB_WDT`)
- `PYTHONPATH=<bench> scripts/run_safe_pytest.sh eval/golden_tests/mhc_pre/ -p skb_op_plugin`

## Option menu (isolated, one tile, 20 iterations; per-Sinkhorn time includes the 0.11 us copy_tile)
| v | technique | us | precision |
|---|---|---|---|
| v0 | the op's code (baseline) | 4.97 | reference |
| v1 | 2.0 and eps in the programmable constant regs (no SFPLOADI or injected runtime immediate per use) | 4.70 | bitwise == v0 |
| v2 | v1 + the 4 sum / reciprocal chains interleaved | 3.66 | bitwise |
| v3 | v2 + fused passes: each scaling pass also accumulates the other direction's sums (drops 32 loads per iteration) | 3.53 | bitwise |
| v4 | v3 + softmax j loops unrolled (v3's exp loop injected its dst addresses at runtime) | 3.49 | bitwise |
| v5 | v4 + row max via SFPSWAP (`sfpi::max`) | 3.49 | bitwise |
| **v6** | v5 + **hand-scheduled SFPLOADMACRO passes** (raw TTI): one issued instruction does an entry's load, multiply and write-back. Only the sums' adds are issued besides. 142 -> ~105 ns per iteration | **2.73** | **bitwise** (also in the second DEST half, and in-op, see below) |
| v7 | scaling-vector form m = diag(r) K diag(c), eps folded exactly | 3.61 | NOT bitwise: max \|d\| vs v0 2.4e-7. vs fp64 it is 0.85-2.0e-7, where v0 is 1.1-1.9e-7. Dominated by v6 |

vs fp64 (bench), v0..v6 identical: max abs 1.1-1.9e-7, max rel 0.7-4.5e-6. torch fp32 on the same logits: 1.1-1.9e-7.

The SFPU turned out to be issue-bound (~1 instruction per cycle, no visible MAD-latency stalls), so every win is a
cut in instruction count. The bitwise floor without macros is ~184 slots per iteration (v5). Each entry needs two
roundings of multiplication per iteration, and with only 8 LREGs a single pass cannot hold 4 multipliers, 4
accumulators and the row. SFPLOADMACRO moves the multiply and the store off the issue slots.

### v6 schedule (raw-LLK)
- Registers are pinned: L0..L2 macro temps (the VD of an SFPLOADMACRO must be L0..L3 at even DEST addresses), L3 the
  accumulator, L4..L7 the four multipliers, L9 = 0, L10 = 1, L12 = 2.0, L13 = eps.
- Macro q is template q = `SFPMUL(L(4+q), VD, L9)` at delay 0, plus a store to the loaded address 2 issued
  instructions later (WaitForElapsedInstructions).
- Pass C runs row-major (macro j = column); pass R runs column-major (macro i = row). So each group's sum is the
  plain formulation's operand order.
- The DFS scheduler takes 40 slots for the 16 entries plus 12 adds plus 3 scratch stores (v5 took 68). The 4th group sum
  stays in L3. The reciprocal phase takes 25-28 slots.
- The rules are re-checked by `gen_sinkhorn_lm.py::verify`:
  - no issued MAD-unit op in the macro-MAD slot, and no issued SFPSTORE in the macro-store slot;
  - no temp is reused before its store and add have read it;
  - the accumulator's RAW distance is at least 2;
  - a scratch store is at least 5 slots before its reload.

## Whole op (grad = graduation.patch, OWNER_C_DISCOUNT = 7; medians of 5-7 interleaved calls)
| cell | base us | grad us | delta |
|---|---|---|---|
| 640x1792 bf16 X / fp32 W | 43.88 | 42.91 | -0.97 (-2.2%) |
| 640x7168 bf16 / fp32 | 145.66 | 144.95 | flat |
| 1280x4096 bf16 / fp32 | 137.88 | 138.04 | flat (noise +-2) |
| 640x1792 fp32 X / fp32 W | 118.70 | 113.66 | -5.0 (-4.2%) |
| 64x4096 bf16 / fp32 | 42.88 | 40.79 | -2.1 (-4.9%) |
| 4096x1792 bf16 / fp32 | 228.8 | 229.4 | flat |
| 4096x1792 fp32 / fp32 | 556.5 | 518.9 | -37 (bimodal cell: base samples 543-559, grad 516-538) |
| 2048x5120 bf16 / fp32 | 303.3 | 301.2 | flat |
| 2048x5120 fp32 / fp32 | 701.4 | 697.9 | flat |
| 1x7168 bf16 / fp32 | 46.1 | 44.5 | -1.7 |
| 1x7168 fp32 / fp32 | 114.6 | 114.5 | flat |
| 32x128 bf16 / fp32 | 17.3 | 14.8 | -2.5 (-14%) |
| 32x128 fp32 / fp32 | 23.0 | 20.5 | -2.5 |
| 640x1792 bf16 / bf16 W | 44.2 | 41.1 | -3.1 (-7%) |
| 640x1792 fp32 / bf16 W | 114.1 | 107.9 | -6.1 |
| 1280x4096 bf16 / bf16 W | 183.3 | 180.9 | -2.4 |
| 1280x4096 fp32 / bf16 W | 340.6 | 340.4 | flat (bimodal) |
| 64x4096 bf16 / bf16 W | 35.5 | 32.9 | -2.6 |
| 64x4096 fp32 / bf16 W | 64.2 | 60.9 | -3.2 |

Zones (640x1792 bf16, KERNEL_PERF_ZONES):
- `c_owned` TRISC_0 goes from 6.69 to 4.54 us (-2.15 us). TRISC_1 goes from 7.15 to 4.99 us.
- The TRISC max drops from 42.4 to 40.2 us, but the wall only from 44.2 to 43.9 us. The critical core moves off the
  owner, to (1,2), whose BRISC / writer tail now bounds the op.

OWNER_C_DISCOUNT with v6:

| cell | 7 (current) | 6 | 5 | 4 | 3 |
|---|---|---|---|---|---|
| 640x1792 | 42.77 | **41.98** | 42.93 | 43.01 | 43.23 |
| 640x7168 | **143.98** | 145.88 | 146.77 | 147.05 | 147.50 |

The discount's effect is dominated by DRAM-bank aliasing of the rank c_starts (the descriptor comment already notes
this), not by the Sinkhorn time. No single value is better on both cells, so **no change to OWNER_C_DISCOUNT**
(keep 7). A shape-keyed discount of 6 for 640x1792 is a separate option (-0.8 us there).

## Precision / "bitwise" findings (important)
- In isolation (bench) v1..v6 are bitwise identical to v0, in both DEST halves.
- In the op, grad comb != base comb by <= 2.4e-7 (<= 12 ulp; post and y are identical). This is not a hazard in
  the new code:
  - An in-op self-check ran the old Sinkhorn on a copy of the real logits in DEST tile 1 and compared it lane-exactly
    with v6. It found 0 mismatches against v0, v4 and v5 as compiled from the bench header, across 640x1792 / 32x128.
  - The only mismatch was against `recip_pos` compiled in the op's own context. There, sfpi (-ffast-math) emits the
    Newton residual `2 - x*y` as **SFPMUL + SFPADDI** (two roundings) instead of one SFPMAD.
  - This choice flips with unrelated code. Enabling KERNEL_PERF_ZONES alone changes the base op's post / comb by the
    same few ulp. So "bitwise identical to the current binary" is not a stable property of this op.
  - v6 pins the fused form: raw TTI, or an L12 operand that cannot become an SFPADDI immediate.
- Against fp64 Sinkhorn on logits dumped from the op (6 cells), grad and base are equivalent:
  - max abs: grad 1.19-2.01e-7, base 1.20-1.96e-7;
  - mean abs: grad 2.20-2.35e-8, below base's 2.34-2.41e-8 in all 6 cells;
  - torch fp32 on the same logits: 1.13-1.69e-7.
- The golden suite is 206/206 and the unit tests (test_mhc_pre + blocking) are 24/24 on the patched copy.
- `--dev` (watcher) passes on the bf16 cells. For fp32 X under `--dev`, base and grad both exceed the kernel config
  buffer (83360 / 84496 > 70656 B), which is pre-existing. Non-dev is fine.

## Domain
It applies everywhere the Sinkhorn runs: every dtype, shape, block geometry, and both DEST halves. There are no
exceptions: no cell is incorrect, none is inexpressible, and no measured regression goes beyond noise. The static
assert limits it to n = 4, which is the op's only n: W is (nC, 24).
