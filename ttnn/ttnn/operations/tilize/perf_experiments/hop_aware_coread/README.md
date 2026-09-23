# hop_aware_coread — choose WHICH sticks each RISC-V co-reads by DRAM-bank geometry

Perf 2 part-optimizer experiment for `tilize`. Box: WH B0 n150, 64 Tensix cores, AICLK 1000 MHz
(cycles = ns). Metric: `DEVICE KERNEL DURATION [ns]`. Every run is bit-exact (golden `comp_equal`,
bf16 -> bf16 and fp32 -> fp32, default precision; nothing in the precision contract changes).

**Verdict: WIN.** The geometry is the lever that made Refinement 8's co-read lose at large
segments. The graduation candidate (`graduate/`) wins by 5..24 % wherever the NoC is loaded. It is
flat on the light-load walks: there it keeps HEAD's exact binaries.

## Mechanism (measured)

- **Geometry** (`geom_probe/`, a device dump of `NOC_NODE_ID` and the bank table):
  - DRAM bank b sits at a fixed physical NoC0 (x, y):
    (0,11) (0,1) (0,5) (0,7) (5,1) (5,11) (5,2) (5,9) (5,8) (5,3) (5,5) (5,7).
  - Logical Tensix (x, y) maps to physical x in {1,2,3,4,6,7,8,9} and physical y in {1,2,3,4,7,8,9,10}. Row 5 is harvested on this box.
  - `my_x` / `my_y` and `worker_core_from_logical_core` return only translated coordinates (18..25).
- **Hops.** NoC0 routes east then south; NoC1 routes west then north (10 x 12 torus).
  - A read's request + response is always the full loop on either NoC (22 hops). Only the length of the **data (response) path** differs between NoCs.
- **The lever.** At equal split sizes, the inverted preference costs +24..80 % over the preferred one:
  - `Winv` vs `Wbal`: [1,1,2048,128] +24 %, LOOSE_CASES[8] +75 %.
- **Zones on LOOSE_CASES[8]** (1 KiB segments, reads only):

  | Split | BRISC reads (sticks) | BRISC `writer_coread` p50 (cycles) |
  |---|---|---|
  | R8 positional | 16 | 15.6 k |
  | inverted | 16 | 18.1 k |
  | geometric | 13.5 on average | 7.7 k |

  With co-read off, NCRISC reads all 32 sticks on NoC0 in 8.3 k cycles.
- **NoC1 penalty.** Sticks with equal hops are cheaper on NoC0. Sending the ties to NoC1 cost +13 % on case 8 (`tl4` vs `geo`), so the split charges NoC1 reads a 2-hop penalty.
- **Light load** (8–16 Tensix cores, 64-B segments; case 3, [1,1,256,64]): the walk is issue-bound (~45 cycles per read on one RISC-V).
  - Balance is what matters there; geometry does not.
  - An unbalanced geometric split costs +8..13 %.
- **Kernel overhead of a list** (zones, case 3), relative to R8's constant-trip loop:

  | Encoding | Overhead |
  |---|---|
  | per-read L1 byte load | ~8 cycles per read |
  | mask walk | ~5.5 cycles per step |
  | packed words (4 sticks per word, unrolled) | ~4 cycles per read |
  | physical-row select | ~65 cycles |
  | launch (bigger binaries) | ~90 cycles per RISC-V |

  So the candidate compiles the list path only where the model predicts a gain of at least 250 cycles (define `CO_READ_LISTED`). Otherwise it keeps HEAD's binaries.
- **No gain at ≥ 512 B DRAM → DRAM.** There the op is at the DRAM roofline (~175 GB/s), and the writer RISC-V's own NoC1 tile writes contend with its reads. The geometric co-read is only +0..6 % there (the positional one is +30..37 %), but there is nothing to gain.
  - With an L1-interleaved output it is +6..14 %.
  - A 2-D split at 256 B (column groups read the same 32 sticks, so 8 of 12 banks are hot, i.e. bank-bound) is +2..6 %.

## The split (host, `_co_read_split`)

- BRISC takes the k most NoC1-favoured sticks (score = NoC0 data hops − NoC1 data hops).
- k minimizes `max(n0, n1) * 45 + w * (Σ data hops + 2 * n1)`, with `w = 2 * flits(segment) * cores / 64`.
- Under light load this gives the balanced 16 / 16 split. Under heavy load it approaches the pure geometric split.
- The kernel picks its own physical row with a raw `NOC_NODE_ID` read. The host precomputes lists for 3 candidate rows, because WH harvests rows and the bindings expose no physical coordinates.
  - A selection probe gave the non-matching rows the inverted split: the time matched `Wtp8_2` (10964 vs 10787 ns), not `Winv` (21514), so the kernel picked the right row.
  - Approximating the physical rows from logical ones instead lost about half of the case-8 win (−3.7 % vs −8.2 %).

## Menu (median ns, bit-exact everywhere; same-session A/B)

The prefixes are kernel encodings (see the harness docstring). `W` is the packed-word list, used for all rows below except `grad`.

| variant | c3 [1,1,128,64] | c5 [1,1,32,2048] | w64 [1,1,2048,64] | w128 | c4 [1,1,32,8192] | c8 (→HS L1, 1 KiB) | w512 | notes |
|---|---|---|---|---|---|---|---|---|
| head (R8) | 2389 | 3517 | 5661 | 8321 | 7158 | 12161 | 23252 | co-read on at ≤128 B, positional |
| off | 2971 | 3782 | 5723 | 8373 | 7349 | 11983 | 23701 | |
| pos (R8 split, gate removed) | 2357 | 3553 | 5372 | 8683 (+4 %) | 8384 (+17 %) | 18399 (+51 %) | 31918 (+37 %) | the R8 regression |
| inv (control) | 2366 | 3666 | 5785 | 9683 | 9085 | 21665 (+78 %) | 34510 | geometry inverted |
| bal (geo, 16/16) | 2375 | 3470 | 4980 (−12 %) | 7797 (−6 %) | 7963 | 12388 | 24865 | |
| geo (pure preference) | 2582 (+8 %) | 3531 | 4732 (−16 %) | 7565 (−9 %) | 7534 | 11412 (−6 %) | 24622 | unbalanced: hurts light load |
| tp8_2 (model, gate removed) | 2485 | 3614 | 4939 (−11 %) | 7453 (−14 %) | 7643 | 10739 (−11 %) | 24490 (+2 %) | |
| **grad** (candidate, medians of 6) | 2311 (flat) | 3494 (flat) | 5144 (−5 %) | 7589 (−9 %) | 7373 (flat, gated) | 10849 (−10 %) | 23515 (flat, gated) | the patches |

Graduation candidate vs HEAD (`logs/grad_final_medians.txt`, 6 sessions). Identical-program pairs spread −5.5..+4.8 %: that is the noise floor.

After the last host restructure (the list decision moved ahead of the CT args, and R8's 128-B window was restored wherever the lists are not sent), 3 more sessions (`logs/grad_final2_medians.txt`) show the same picture:
- LOOSE_CASES[8] −9.3 %
- [1,1,2048,128] −10.1 %
- fp32 128 B −14.5 %
- resident outputs −6..−24 %
- [1,1,32,4096] −9.3 %
- tile_h 16 [1,1,1024,128] −10.0 %
- LOOSE_CASES[3] / [4] / [5] / [0] and all positional or off shapes: flat.

| shape (bf16 unless noted) | engages | HEAD | grad | Δ |
|---|---|---|---|---|
| LOOSE_CASES[8] DRAM → HEIGHT_SHARDED L1, 1 KiB | listed | 12091 | 10849 | −10.3 % |
| [1,1,2048,64] DRAM → HS L1 (128 B) | listed | 3513 | 2683 | −23.6 % |
| [1,1,2048,256] → HS L1 (512 B) | listed | 6548 | 6129 | −6.4 % |
| [1,1,2048,1024] → HS L1 (2 KiB) | listed | 23790 | 20831 | −12.4 % |
| [1,1,2048,128] DRAM → DRAM (256 B, row split) | listed (new gate) | 8321 | 7589 | −8.8 % |
| [1,1,2048,128] DRAM → L1 interleaved | listed (new gate) | 7510 | 6972 | −7.2 % |
| fp32 [1,1,2048,32] (128 B) | listed | 5299 | 4659 | −12.1 % |
| fp32 [1,1,2048,64] (256 B) | listed (new gate) | 8957 | 8479 | −5.3 % |
| [1,1,2048,64] (128 B) | listed | 5396 | 5144 | −4.7 % |
| [1,1,32,4096] (128 B, 2-D split) | listed | 5359 | 5021 | −6.3 % |
| [1,1,64,2048] | listed | 5219 | 5115 | −2.0 % |
| [1,1,2048,32] / LOOSE_CASES[5] | listed | 3452 / 3419 | 3392 / 3494 | flat |
| LOOSE_CASES[3], [1,1,256,64], [1,1,512,64] | positional (HEAD binaries) | 2381 / 2455 / 2773 | 2311 / 2407 / 2782 | flat |
| LOOSE_CASES[4], [1,1,256,1024] (2-D, 256 B) | off (gated) | 7709 / 7526 | 7373 / 7398 | flat (identical programs) |
| [1,1,2048,{256,512}] DRAM → DRAM | off (gated) | 12664 / 23968 | 12182 / 23515 | flat (identical programs) |
| L1-interleaved input [1,1,2048,256] | positional (no DRAM geometry) | 10763 | 11257 | +4.6 % (identical programs: noise) |
| LOOSE_CASES[0] focus [1,1,16384,64] | not engaged | 23580 | 24014 | flat (identical programs) |
| tile_h 16: [1,1,1024,64] / [1,1,1024,128] | listed | 3293 / 5581 | 3059 / 4585 | −7 % / −18 % (1 run each) |

Golden, run against the candidate via `graduate/hac_graduate_plugin.py` (`single_tile or short_wide or small or loose or program_cache`):
- 699–701 passed, 18 xfailed.
- 53 programs took `CO_READ_LISTED` and 192 the positional split, all bit-exact.
- The 1–3 failures are the pre-existing flaky padded bfloat4_b / bfloat8_b cells: `1x1x50x50` and the rank-0 `pad_auto` case. They reproduce on HEAD (1, 3 and 2 failures in 3 runs, `logs/golden_head_flaky*.log`). Those cells are padded, so co-read never engages there.

## The gate becomes

`CO_READ_SEGMENT_BYTES[DRAM]` changes from `(0, 128)` to `(0, 256)`, with these adjustments:
- **Unbounded** when the output is resident (`CO_READ_RESIDENT_OUTPUT_UNBOUNDED`: no NoC writes, so the writer RISC-V's NoC1 carries only its reads).
- **Capped at 128 B** on a 2-D split (`CO_READ_SHARED_STICK_MAX_BYTES`: the column groups share sticks, so the op is bank-bound).
- L1 inputs are unchanged (unbounded, positional split).
- Within the gate, the geometric lists are used only when the model's mean per-core gain is at least 250 cycles (`CO_READ_LIST_MIN_GAIN_CYCLES`). Otherwise the program is R8's, byte for byte.
- Without the lists (not WH, an L1 / sharded / paged input, or a light-load walk), R8's 128-B DRAM window applies unchanged (`CO_READ_POSITIONAL_DRAM_MAX_BYTES`). A DRAM segment past it is not co-read.
- Final golden subset against the candidate: 701 passed, 1 failed (the flaky rank-0 `pad_auto` bfloat8_b cell, see above). 53 programs listed, 184 positional.

## Files

- `make_variants.py`: generates the git-ignored kernel variants:
  - `kernels_mask`: step bit-mask
  - `kernels_list`: byte list
  - `kernels_list16`: constant count
  - `kernels_listw`: packed words, the `W` variants
  - `kernels_listsel`: physical-row select
- `make_graduate.py`: builds `graduate/kernels/` and `graduate/tilize_program_descriptor.py` from the op's CURRENT files, plus **`graduate/kernels.patch`** and **`graduate/host.patch`** (the exact diff that would graduate).
- `graduate/hac_graduate_plugin.py`: a pytest plugin that runs any tilize test against the candidate.
- `geom_probe/`: the geometry dump kernel and script, and `geometry.json` (used by the harness).
- `medians.py`: medians over several logs (pass their reports via `REPORTS=`).
- `logs/`: trimmed session logs.
- Harness: `tests/ttnn/unit_tests/operations/tilize/test_tilize_perf2_hop_aware_coread.py`. Its docstring lists the variants and the case syntax `w<W>[_h<H>][_l1][_f32][_rs|_ol1][_t<N>]`.

## Repro

```bash
python3 ttnn/ttnn/operations/tilize/perf_experiments/hop_aware_coread/make_variants.py
python3 ttnn/ttnn/operations/tilize/perf_experiments/hop_aware_coread/make_graduate.py
TILIZE_PERF_EXPERIMENTS=1 HAC_CASES=3,4,5,8,0,w64,w128,w64_rs HAC_VARIANTS=head,grad \
  scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/tilize/test_tilize_perf2_hop_aware_coread.py -s > log
python3 ttnn/ttnn/operations/tilize/perf_experiments/p2_breakdown/label_ns.py log generated/profiler/reports/<this run>/
# golden against the candidate
PYTHONPATH=ttnn/ttnn/operations/tilize/perf_experiments/hop_aware_coread/graduate \
  scripts/run_safe_pytest.sh --run-all -p hac_graduate_plugin eval/golden_tests/tilize/test_golden.py -k "short_wide or small or loose"
```

Always pass the run's own report dir to `label_ns.py` / `medians.py`: sibling agents write to the same `generated/profiler/reports/`.

## Raw LLK / helper bypass

- `select_co_read_list` reads `NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID)` directly. No dataflow API returns the physical NoC coordinate: `my_x` / `my_y` are translated.
- The stick reads themselves are still plain `noc_async_read` through the `TensorAccessor`.

## Untested (not exceptions)

- Blackhole: the candidate falls back to the positional lists and HEAD binaries there, because the geometry tables are WH's.
- Geometric lists for L1-interleaved or DRAM-sharded inputs (both positional today).
- Integer dtypes. They use the same byte path.
