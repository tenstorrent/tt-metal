# onepos_pipeline (Perf 2): stream the output in column sub-blocks

**Verdict: WIN, graduation-ready, rebased on HEAD 975178ffaca.** Only the output-streaming half is
delivered. The earlier resident-output split-read half (−2 % on `LOOSE_CASES[8]`) is dropped: HEAD's
geometric co-read already takes that case from 12098 to 10813 ns.

All numbers: Wormhole B0 n150, 64 Tensix cores, AICLK 1000 MHz, `DEVICE KERNEL DURATION [ns]`.
Every variant runs under the user's precision config unchanged, and the output is bit-exact against
HEAD's program on every case.

## Mechanism
Compute tilizes each tile-row as column sub-blocks of `SUB_BLOCK_TILES` = 2 tiles (a trailing 1-tile
remainder merges into the sub-block before it). It pushes each sub-block's output pages as soon as
they are packed. The writer (BRISC, NoC1) waits for the pages cumulatively and writes each sub-block
at once.

- **Write order.** The writer reproduces `store_rows`' exact tile order: rotated start
  `t0 = stick_rotation mod valid_width`, then wrapping.
- **Production order.** Production starts at the sub-block that holds `t0`, so the first writes of
  the Tensix cores still spread over all DRAM banks.

On a one-position walk the writer otherwise idles through the whole tilize. On `LOOSE_CASES[7]`,
the BRISC zone `writer_wait` drops from 1361 to 385 cycles (p50).

The gain is smaller than that drop suggests. The slowest Tensix core is write-congestion bound:
`writer_issue` max is 13.6k cycles (head) and 14.5k (grad). So starting the writes earlier only
partly shortens the op.

## Graduate (apply-ready)
- `graduate.diff`: against HEAD. `git apply --check` is clean. It touches `kernels/tilize_compute.cpp`,
  `kernels/tilize_writer.cpp`, the new `kernels/tilize_sub_blocks.hpp` and `tilize_program_descriptor.py`.
- `graduate/`: full replacement files. `tilize_reader.cpp` and `tilize_stick_reads.hpp` are
  identical to HEAD; they are only there so that `graduate/kernels` is a complete `KERNEL_DIR`.
- **OFF switch:** `SUB_BLOCK_TILES = 0`. It builds exactly HEAD's programs: no define and no extra
  runtime (RT) arg. The harness checks this with a program digest (kernel files, defines,
  compile-time (CT) args, RT args, CB sizes), which matched HEAD on all 58 cases. Coordinator A/B
  after applying the diff:
  `TILIZE_P2_VARIANTS="head,head@SUB_BLOCK_TILES=0"`.
- **Kernel changes:**
  - Compute: `tilize_sub_blocks::tilize_rows`, raw WH LLK (next section). Compute RT arg 2 is
    `stick_rotation`.
  - Writer: `store_row_sub_blocked`.
  - Both are compiled only under the `TILIZE_SUB_BLOCK_TILES` define and `ARCH_WORMHOLE`. The
    experiment-only switches are gone.
- **Zones** (`MaybeDeviceZoneScope`, 3 per tile-row, the same budget as `store_rows`):
  - Compute keeps `compute_tilize`.
  - The writer keeps `writer_wait` (the first sub-block), `writer_issue` (the writes, plus the waits
    on later sub-blocks, which interleave with them), `writer_flush` and `writer_barrier`.

### Engagement rule (host, `create_program_descriptor`)
The path engages on **every walk, one position or many**, when all of these hold:
- `SUB_BLOCK_TILES > 0`;
- the device is Wormhole B0;
- the output is not resident and there is no split reader;
- `write_ahead == 1` and `write_noc_split == 0`;
- `block_width >= 4`.

Each carve-out and its reason:

| carve-out | reason class | reason |
|---|---|---|
| not Wormhole | inexpressible as written | The raw LLK copies the WH branch. The BH fast-tilize LLK has another signature (unit chunks, row begin / end), so BH needs a port; untested. |
| `block_width <= 3` (LOOSE 0, 1, 2, 3, 5, 6) | inexpressible | Two sub-blocks of at least 2 tiles each are impossible. A 1-tile fast-tilize unit on a wider row is illegal (tilize.h WH branch). |
| resident output (LOOSE 8, 9, 10) | nothing to overlap | Compute packs straight into the shard and the writer issues no NoC write. |
| split reader, `write_ahead > 1`, write NoC split | inexpressible as written | These are parked knobs, all off by default. They use other store paths (two input CBs, `TileStorer`, a dynamic NoC). |
| one-position walks only | **removed** | Multi-position walks measured flat or faster (table below). |

No carve-out comes from a measured regression.

HEAD's co-read (the writer RISC-V co-reads sticks) composes with this path: the writer runs its
co-read share first, then the sub-block writes.

- DRAM co-read, `[1,1,2048,128]` (`CO_READ_LISTED`): +0.8 % (n = 4) and +0.4 % (n = 8), i.e. flat.
  It is kept (one path).
- L1-interleaved co-read: −1.3 % and −3.1 %.

### Raw LLK (helper bypass)
| helper | kind | what is missing | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::tilize` (and `fast_tilize_block` / `tilize_block` under it) | compute: tilize a column slice of a wider tile-row | The unpacker row stride is tied to the width being tilized (`full_dim = block`; `llk_unpack_tilize_block` passes `block_c_tiles` as the stride), and there is no column offset inside the row. A 2-tile sub-block needs stride = `block_width` with width = 2 at a column offset. | LOOSE 7: 16997 (n = 6), 16950 (n = 8) | 15568, 15856 | `kernels/tilize_compute.cpp:89` (comment), `:109` `tilize_cols_fast`, `:163` `tilize_cols_slow` |

- TRISC_2 kernel span: 1672 cycles (helper) → 1878 cycles (raw), hidden under the writes.
- Init / uninit and the fast-vs-lossless choice mirror the helper's `InitAndUninit` call.
- There is no reconfig, matching HEAD's `NoReconfigure`.

## Numbers (same-session interleaved A/B, alternating order, median [min–max])
### LOOSE targets, n = 6 (`logs/ab_loose.log`), plus an OFF A/A control
| case | head | grad | Δ | OFF (A/A) |
|---|---|---|---|---|
| 7 [1,1,2048,512] `HEIGHT_SHARDED` L1 → DRAM | 16997 [16219–17443] | 15568 [14759–16046] | **−8.4 %** | +0.2 % |
| 8 [1,1,2048,512] DRAM → `HEIGHT_SHARDED` L1 (not engaged: resident output) | 10919 | 10873 | −0.4 % | −0.9 % |
| 4 [1,1,32,8192] | 7489 | 7323 | −2.2 % | +0.7 % |
| 5 [1,1,32,2048] (not engaged: `block_width` 1) | 3518 | 3480 | −1.1 % | −1.8 % |
| 3 [1,1,128,64] (not engaged) | 2376 | 2358 | −0.8 % | −0.4 % |
| 0 [1,1,16384,64] (not engaged: `block_width` 2) | 23675 | 23877 | +0.9 % | +1.5 % |

Repeat sessions of case 7:
- `ab_multi_n8.log`: 16950 → 15856 (−6.5 %, n = 8).
- `s1_loose_multi.log`: 16594 → 16131 (−2.8 %, n = 4).
- `s2_case7_old.log`: 16471 → 16022 (−2.7 %, n = 6).

Across sessions HEAD moves 16.5–17.0 k and grad 15.6–16.1 k.

`s2` also ran the pre-rebase candidate (whose compute still reconfigured the data formats): 15826
(−3.9 %), the same as grad within noise.

### Multi-position walks (sub-blocks on every position), n = 8 (`logs/ab_multi_n8.log`)
| shape | walk | head | grad | Δ |
|---|---|---|---|---|
| [1,1,4096,256] DRAM → DRAM | 2 positions, `block_width` 8 | 23458 | 23269 | −0.8 % |
| [1,1,8192,256] DRAM → DRAM | 4 positions, `block_width` 8 | 43732 | 43277 | −1.0 % |
| [1,1,16384,128] DRAM → DRAM | 8 positions, `block_width` 4, 2 rows per quantum | 46212 | 45971 | −0.5 % |
| [1,1,4096,512] DRAM → DRAM | 2 positions, `block_width` 16 | 45671 | 45638 | −0.1 % |
| [1,1,4096,512] `HEIGHT_SHARDED` (64-row shard) → DRAM | 2 positions | 31406 | 30063 | **−4.3 %** |

More multi-position regimes, n = 4 (`logs/ab_domain.log`):

| shape | Δ |
|---|---|
| [1,1,8192,512] | −0.1 % |
| [1,1,4096,1024] | −0.9 % |
| hs [1,1,8192,256] (4-row shard) | −1.7 % |
| hs [1,1,4096,512] → L1 | −4.6 % |
| L1-interleaved [1,1,4096,512] | −5.2 % |
| fp32 [1,1,4096,512] | −0.5 % |
| tiny16 [1,1,4096,512] | −0.9 % |
| tiny1 [1,1,512,512] | −2.5 % |
| retile 32→16 [1,1,2048,512] | −0.8 % |
| low_l1 [1,1,8192,512] | −0.2 % |
| mixed 1 / 2-position grid [1,1,3200,512] | −1.9 % |
| [1,1,2048,2048] (R5 2-position pipeline) | −1.3 % |

The only positive values seen:
- d4096x512: +1.7 % (n = 4), then −0.1 % (n = 8).
- d16384x128: +1.1 % (n = 4), then −0.5 % (n = 8).

Both are inside the noise. The noise floor is ±2–4 %: identical-program pairs have spread up to
±3.7 % within one session.

### One-position domain, n = 4 (`logs/ab_domain.log`, head → grad)
| regime | shapes | Δ |
|---|---|---|
| resident input → DRAM | hs [1,1,2048,256] / 224 (`block_width` 7, merged 3-tile tail) / 1024 | −8.1 / −4.5 / −6.2 % |
| | fp32 hs [1,1,2048,512] | −6.4 % |
| | int32 hs [1,1,2048,512] | −7.2 % |
| | bf16 → bfp8_b hs [1,1,2048,512] | −10.3 % |
| resident input → L1 | [1,1,2048,512] | −6.6 % |
| L1-interleaved source | [1,1,2048,256] / 512 | −1.3 / −3.1 % |
| DRAM → DRAM | [1,1,32,16384] / [1,1,64,8192] / [1,1,64,4096] | −3.3 / −3.1 / −4.2 % |
| | [1,1,2048,128] (co-read) / 160 / 256 / 512 / 1024 | +0.8 / −1.3 / +0.3 / +0.2 / −1.5 % |
| | fp32 [1,1,2048,512] | +0.2 % |

## Correctness
- `logs/corr_final.log`: head, grad and grad OFF on all 11 `LOOSE_CASES` and 47 domain / coverage
  cases (58 cases × 3 variants). All pass the golden contract, and every OFF digest equals HEAD's.
  The coverage cases include:
  - pad auto (negative fill), ragged 2-D split;
  - tiny tiles 16 / 8 / 1, retile 32 → 16, `low_l1`;
  - int32, uint16, uint8, fp32, bf16 → bfp8_b, fp32 → bf16;
  - odd `block_width` 5 / 7, a mixed grid;
  - every multi-position shape above.
- `logs/corr_bitexact.log` (`TILIZE_OP_BITEXACT=1`): on all 58 cases, grad's output is `torch.equal`
  to HEAD's program run on the same input, including the cast cases the contract only checks by PCC.
- `logs/dev_final.log` (`--dev`: watcher, NoC sanitizer, LLK asserts): 15 cases, no hang, no assert.
  They include case 7, case 4, co-read composition, L1 co-read, odd widths, tiny tiles, multi-position,
  fp32, int32, mixed, retile, pad and ragged.
- Pre-rebase mutation check (the same raw functions): dropping the column offset makes the fast path
  fail at PCC 0.25 and the slow path fail int32 / tiny16 / fp32. So both raw paths really run.

## Menu (every variant tried; the same precision config; bit-exact)
| variant | case 7 | notes |
|---|---|---|
| head (helper, whole row) | 16.5–17.0 k | baseline |
| **2-tile sub-blocks, rotated start (graduated)** | −2.7 .. −8.4 % | every walk |
| 4-tile sub-blocks | −2.4 % (pre-rebase) | |
| 2 sub-blocks of 8 tiles | −5.2 % (pre-rebase) | |
| column-order start | −3.6 % (pre-rebase) | **+8.6 % on LOOSE 4**, +13 % on [1,1,64,4096]: the first sub-block of every Tensix core then covers only 6 of 12 DRAM banks |
| one-position walks only | = graduated on one-position walks | dropped: multi-position measured flat or faster |
| resident-output split reads (2 parts ≥ 512 B) | n/a | case 8 −2 % pre-rebase; superseded by the geometric co-read (dropped) |

## Domain
The path applies everywhere it is expressible, and nothing is excluded for a measured regression.

Exceptions:
- **Inexpressible:**
  - `block_width` ≤ 3;
  - not Wormhole (as written);
  - the parked split-reader / `TileStorer` / write-NoC-split paths.
- **Nothing to overlap:** a resident output.

Untested:
- Blackhole / Quasar;
- `WIDTH_SHARDED` / `BLOCK_SHARDED` / ND-sharded inputs with a streamed output;
- `bfloat4_b` output;
- `fp8_e4m3` input (Blackhole only).

## Repro
```
V=$(for r in 1 2 3 4 5 6; do [ $((r%2)) = 1 ] && echo -n "head#$r,grad#$r," || echo -n "grad#$r,head#$r,"; done); V=${V%,}
TILIZE_PERF_EXPERIMENTS=1 TILIZE_OP_CASES=7,8,4,5,3,0 TILIZE_OP_VARIANTS=$V \
  scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/tilize/test_tilize_perf2_onepos_pipeline.py -s > log
python3 ttnn/ttnn/operations/tilize/perf_experiments/onepos_pipeline/stats.py log   # uses the log's PROFILER CSV line
```
- Tokens are `head` | `grad`, optionally followed by `#<rep>` and then `@KNOB=value+...` (knob
  overrides on the graduate descriptor module).
- `TILIZE_OP_BITEXACT=1` adds the bit-exact comparison against HEAD. Use it for correctness runs
  only, because it adds generic ops.
- Zones: set `TT_METAL_KERNEL_PERF_ZONES=1`, then run
  `perf_experiments/breakdown/zones.py <report_dir>`.
- `superseded/` holds the pre-rebase generators and diffs (the split-read half included), kept for
  the record. They target the pre-975178ffaca kernels, so do not apply them.
