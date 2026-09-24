# hop_aware_noc — per-(Tensix core, DRAM bank) write NoC choice by hop distance (Perf 2)

WH B0 n150, 1 GHz AICLK (cycles = ns), up to 64 Tensix cores. Every number is DEVICE KERNEL
DURATION [ns]. Rows are same-session medians (n = 4 unless noted), both variant orders alternated
per repeat. Noise is about ±3 % (identical binaries have shown up to ±5 % on shapes under 3 µs).
Every run was bit-exact (`torch.equal` / golden contract check). Precision was not touched.

## Status: rebased onto HEAD be093526489 (column sub-blocks) — WIN

The deliverable is `graduate/`, plus `graduate.diff` against HEAD (`git apply --check` passes):

- Changed: `tilize_program_descriptor.py`, `kernels/tilize_reader.cpp`, `kernels/tilize_writer.cpp`
  and `kernels/tilize_stick_reads.hpp`.
- `kernels/tilize_compute.cpp` and `kernels/tilize_sub_blocks.hpp` are byte-identical HEAD copies.
  They are there so `KERNEL_DIR` is complete.
- The pre-rebase `graduate_writer_hop.patch`, `graduated_descriptor.py` and `kernels_dedg_hop/`
  are superseded.

On the focus case LOOSE_CASES[7] [1,1,2048,512] `HEIGHT_SHARDED` L1 → DRAM (sub-block path),
HEAD 15808 → 14086 ns (−10.9 %, n = 10 over two sessions: −9.1 % n = 4 and −12.7 % n = 6).

## What changed vs the pre-rebase patch

1. **Both store paths take the NoC choice.**
   - `store_rows` (`tilize_stick_reads.hpp`) is used by the multi-row / narrow-block walks and by
     the split reader's `store_next`.
   - HEAD's new `store_row_sub_blocked` (`tilize_writer.cpp`, its `write_cols`) is the sub-block
     path.
   - Both pick the NoC per page by the page's DRAM bank (bank = page mod 12, tracked incrementally)
     and flush both NoCs.
   - The only path that does NOT take it is `TileStorer` (`write_ahead > 1`, the parked
     `WRITE_WINDOW_MIN_TILES` knob). Its per-quantum transaction ids live on one NoC, so it is
     inexpressible as written; the host gate and a `static_assert` exclude it.
   - The split reader is no longer excluded, because `store_next` is `store_rows`. It is verified
     under `--dev` through `test_tilize_knobs` `split*` on [1,1,1600,96].
2. **The OFF switch yields exactly HEAD's binaries.**
   - All hop code sits under `#ifdef TILIZE_HOP_WRITE_MIN_SAVING`. The host adds that define plus
     `TILIZE_HOP_SEM`, and the semaphore, only when engaged.
   - `HOP_WRITE_MIN_SAVING = 0` or a non-engaged shape gives HEAD's programs.
   - `elf_compare.py` checked this: 44 reader and 41 writer off-configs have `.text` / `.data`
     byte-identical to a HEAD build (release and `--dev` builds). Every engaged build differs.
   - A constexpr-`hop_t` form was functionally equal but only register-allocation-identical, so it
     was replaced.
3. **The reader re-sync is an RAII guard (`HopReaderResync`).** Its destructor runs on every return
   path of `kernel_main`, so HEAD's reader body is unchanged.
4. **The carve-outs were re-derived on HEAD** (next section). The old `HOP_WRITE_MIN_TILES = 256`
   measured the wrong axis. The old blanket DRAM-input carve-out was too wide.
5. **Zone compile fix.** Under `TT_METAL_KERNEL_PERF_ZONES=1`, HEAD's writer does not compile on the
   sub-block path: two `writer_barrier` zones share `kernel_main`'s scope. The graduate scopes the
   sub-block one.

## Engagement rule (`hop_write_t`, graduate descriptor)

The rule is ON wherever it is expressible, minus two measured carve-outs.

**Expressible, all required:**

- Wormhole B0. The kernel's 10 × 12 NoC torus and untranslated-DRAM model are Wormhole's.
  Blackhole is untested.
- A DRAM `TensorMemoryLayout::INTERLEAVED` output. Page p is in bank p mod 12. An L1 or sharded
  output has Tensix-core banks, which a 12-bit mask cannot express.
- Output not resident (a resident output issues no writes).
- `write_ahead == 1` (see `TileStorer` above).
- No parked `READ_NOC_SPLIT` / `WRITE_NOC_SPLIT` (both are DM_DYNAMIC_NOC one-NoC schemes).
- Not `bank_coalesced` with `BANK_COALESCE_SCATTER_WRITE`. That path is NCRISC NoC0 writes, which
  would share NIU 0's write counters, so it would be incorrect.

**Carve-out 1: `HOP_WRITE_MIN_CORES = 32` writing Tensix cores** (`measured-regression`). NoC1's DRAM
write path congests only with many writers. Head → hop, carve-out lifted:

| writers | cases | Δ |
|---|---|---|
| 2 | [1,1,512,512] HS (256 tiles) / [1,1,64,512] HS | +41..45 % / +24..28 % |
| 4 | hsb (128 tiles / core) / hs (16 tiles / core) | +36..41 % / +26..30 % |
| 8 | hsb / hs | +28 % / +15..20 % |
| 16 | HS ×2, BS 4×4, WS 4×4, L1 [1,1,256,64] | +2..3 %, +5..8 %, +6..9 %, +6..10 % |
| 20 | HS (16 tiles / core) / HS (128 tiles / core) | −3 % / +3..5 % |
| 24 | HS ×3 / BS 6×4 [1,1,512,768] | −4..−7 % / +1..+7 % |
| 32 | HS ×2, BS 8×4 | −2.5..−8 % |
| 48 / 64 | HS | −10 % / −9..−17 % |

- Small outputs on many Tensix cores are flat: 64 tiles on 64 Tensix cores +0.2 / +1.3 %;
  [1,1,32,2048] DRAM → DRAM −0.4 %.
- So the old tile-count carve-out (`HOP_WRITE_MIN_TILES = 256`) was measuring the wrong axis:
  [1,1,512,512] HS on 2 Tensix cores has 256 tiles and loses +42 % on HEAD, because the sub-block
  path changed the few-writer picture (it was −5.6 % pre-rebase).

**Carve-out 2: `HOP_WRITE_DRAM_INPUT_MAX_BYTES_PER_CORE = 8192`** (`measured-regression`). It is
narrowed from "every DRAM input".

- DRAM read responses ride NoC0 through the steady state.
- **> 8 KiB input per Tensix core, carve-out lifted:**
  - [1,1,16384,64] (LOOSE 0) +25 %;
  - [1,1,16384,32] (LOOSE 1) +34..36 %;
  - [1,1,32768,64] (LOOSE 2) +28..32 %;
  - [1,1,8192,256] +32..34 %; [1,1,16384,128] +18..19 %; [1,1,2048,1024] +9 %;
  - [1,1,2048,256] +18..19 %; [1,1,4096,128] +12..18 %; [1,1,1024,1024] +11..14 %;
  - [1,1,16384,32] fp32 +19..23 %; [1,1,4096,64] fp32 +8..9 %; [1,1,32,16384] +12..15 %.
- **≤ 8 KiB per Tensix core, now ENGAGED:**

  | case | Δ |
  |---|---|
  | LOOSE 4 [1,1,32,8192] | −7.3 % |
  | [1,1,4096,64] | −10.8 % |
  | [1,1,2048,64] | −9.1 % |
  | [1,1,1024,256] | −7.5 % |
  | [1,1,2048,64] fp32 | −13.4 % |
  | [1,1,4096,32] fp32 | −11.4 % |
  | [1,1,32,4096] | −5.9 % |
  | LOOSE 5 [1,1,32,2048] | flat, −0.4 % |

- **Left on the table at 16 KiB** (not separable from the 16 KiB losses):
  - LOOSE 6 [1,1,8192,32] fp32 −6..−10 %;
  - [1,1,8192,64] −2..−6 %;
  - [1,1,64,8192] flat.

## A/B, final code (`logs/final_ab_medians.txt`; head vs graduate vs graduate@HOP_WRITE_MIN_SAVING=0)

| case | path | head | graduate | Δ | off Δ |
|---|---|---|---|---|---|
| LOOSE 7 [1,1,2048,512] HS L1 → DRAM | sub-block | 15720 | 14290 | −9.1 % | +0.8 % |
| LOOSE 7 (second session, n = 6) | sub-block | 15882 | 13857 | −12.7 % | |
| [1,1,8192,256] HS L1 → DRAM | sub-block | 30444 | 26272 | −13.7 % | +0.8 % |
| [1,1,1024,1024] BS L1 → DRAM | sub-block | 16110 | 14138 | −12.2 % | −1.6 % |
| [1,1,2048,1024] HS L1 → DRAM | sub-block | 30794 | 26034 | −15.5 % | −3.6 % |
| [1,1,2048,512] fp32 HS | sub-block | 32604 | 29288 | −10.2 % | +3.2 % |
| [1,1,16384,64] L1 interleaved → DRAM | store_rows | 22700 | 19734 | −13.1 % | −1.8 % |
| [1,1,2048,256] L1 interleaved (co-read) | sub-block | 10254 | 9400 | −8.3 % | +0.1 % |
| [1,1,2048,1024] L1 interleaved | sub-block | 40604 | 35170 | −13.4 % | +1.8 % |
| [1,1,16384,64] HS | store_rows | 16750 | 14675 | −12.4 % | −3.2 % |
| [1,1,256,2048] WIDTH_SHARDED | store_rows | 7640 | 7024 | −8.1 % | +1.1 % |
| [1,1,1000,100] L1 padded (pad auto) | store_rows | 8610 | 8264 | −4.0 % | +1.0 % |
| [1,1,4000,500] L1 padded | sub-block | 45414 | 41534 | −8.5 % | (noisy) |
| [1,1,16384,64] L1 low_l1 off / on | store_rows | 22576 / 22464 | 19394 / 19579 | −14.1 / −12.8 % | ±0 |
| [1,1,16384,64] L1, tile_h 16 | store_rows | 21496 | 20348 | −5.3 % | −1.0 % |
| [1,1,4096,64] L1 retile 32 → 16 | store_rows | 10806 | 10879 | +0.7 % (flat) | +1.0 % |
| HS 32 Tensix cores [1,1,2048,512] | sub-block | 15952 | 14677 | −8.0 % | −2.4 % |
| HS 40 / BS 32 / HS 32 Tensix cores | sub-block | 10022 / 11240 / 8275 | 9066 / 10568 / 8064 | −9.5 / −6.0 / −2.5 % | |
| LOOSE 4 [1,1,32,8192] DRAM → DRAM | sub-block | 7307 | 6774 | −7.3 % | −1.9 % |
| [1,1,4096,64] / [1,1,2048,64] fp32 DRAM → DRAM | store_rows | 7754 / 8315 | 6920 / 7198 | −10.8 / −13.4 % | |
| LOOSE 0 / 1 / 2 / 6 / 8 / 9 / 10 (not engaged) | — | 24054 / 13358 / 45462 / 14408 / 10862 / 1872 / 1818 | identical programs | −0.8..+2.3 % | |

The guards (LOOSE 0, 1, 2, 6, 8, and 3, 9, 10) are byte-identical programs to HEAD.

- LOOSE 3 [1,1,128,64] on 8 Tensix cores showed +5.5 % at 2.3 µs with identical binaries. A
  re-run gave +2.0 / −1.3 %, so it is noise.
- Carve-out cases with the graduate defaults are identical programs:
  - [1,1,8192,256] DRAM → DRAM −1.9 %;
  - [1,1,2048,256] DRAM +2.5 %;
  - HS on 8 Tensix cores −1.6 %;
  - BS on 16 Tensix cores +1.4 %;
  - BS on 24 Tensix cores −0.2 %;
  - HS on 2 Tensix cores −0.5 %.

**Zones** (`TT_METAL_KERNEL_PERF_ZONES=1`, LOOSE 7, p50 cycles per Tensix core, hop / off):

| zone | hop | off |
|---|---|---|
| writer_hop_init | 451 | — |
| writer_wait | 36 | 394 |
| writer_issue | 8739 | 8774 |
| BRISC-KERNEL max | 13807 | 15337 |

- The mask and counter snapshot overlap with compute's first sub-block, so the net critical-path
  cost is about 90 cycles.
- `reader_hop_resync` is NCRISC idling until BRISC finishes; it is not on the critical path.

## Correctness and robustness

- **`--dev`** (watcher, NoC sanitizer, ebreak / LLK asserts):
  - `logs/dev_final.log`: 41 / 41 pass. That covers all 11 LOOSE cases plus 30 domain cases, both
    store paths engaged.
  - `logs/dev_final2.log`: 102 / 102 pass (head, off and graduate on 34 cases).
- **Back-to-back on one device under `--dev`** (`test_hop_back_to_back`, `logs/dev_b2b.log`,
  30 steps, all bit-exact):
  - Engaged tilizes are interleaved with non-engaged tilizes (few-core HS, DRAM → DRAM LOOSE 0 / 1
    / 2 / 6, resident-output LOOSE 8 / 9) and with a `ttnn.add`.
  - It includes repeated program-cache hits: LOOSE 7 six times, [1,1,16384,64] L1 twice,
    [1,1,2048,64] DRAM twice, and the cache-entry count does not grow.
  - No hang and no idle ASSERT across any program boundary.
- **Negative control** (`kernels_neg_noresync`, NCRISC re-sync removed, `logs/dev_negative.log`):
  `--dev` hangs with all 64 Tensix cores' NCRISC at `NKFW`, the `ncrisck.cc:86` nonposted-writes
  ASSERT. So the idle check is live, and the handshake is what satisfies it.
- **Unit nets against the graduate** (`graduate/hop_graduate_plugin.py` overlays the descriptor so
  `pd` knob monkeypatches still apply):

  | net | release | `--dev` | hop programs |
  |---|---|---|---|
  | test_tilize_sharded | 26 / 26 | 26 / 26 | 1..2 |
  | test_tilize_padding | 60 / 60 | 60 / 60 | 10..20 |
  | test_tilize_knobs | 301 / 306 | 252 / 252 (`-k "not co_read_open_gate and not window"`) | 78..182, incl. split reader |

  The excluded knob failures are **pre-existing at HEAD and unrelated to hop**; hop is off in both:
  - `co_read_open_gate` × 5: `TypeError min(None, 128)` at HEAD `tilize_program_descriptor.py:917`,
    from co-read commit 975178ffaca.
  - `*window*` knobs under `--dev`: HEAD hangs, BRISC at `NKFW` on the parked `TileStorer`
    `write_ahead > 1` path (reproduced with no plugin on `[1x1x16384x64-windows8]`).
  - The unit nets never engage hop together with sub-blocks (their engaged shapes have
    block_width < 4). The perf harness covers that combination (LOOSE 7, HS / BS / L1 wide).

## Helper-bypass row

| helper | kind | what was missing | helper ns | raw ns | file:line |
|---|---|---|---|---|---|
| none (dataflow API) | physical-coordinate query | no API returns the Tensix core's physical NoC0 xy (`my_x` / `my_y` are translated) | — | mask + snapshot 451 cycles, overlapped (~90 net) | `graduate/kernels/tilize_stick_reads.hpp` `hop_write::init` (NOC_CMD_BUF_READ_REG(NOC_NODE_ID)) |
| `DM_DYNAMIC_NOC` (the supported way to write on both NoCs from one RISC-V) | NoC mode | the dedicated-mode counter bookkeeping has no API | dyn mode alone +13..18 % on issue-bound shapes (pre-rebase: 2 Tensix cores × 128 tiles 13546 → 15904, hs_tiny 3446 → 3909) | dedicated + handshake: LOOSE 7 15808 → 14086 | `graduate/kernels/tilize_writer.cpp` `noc_local_state_init(1 - noc_index)` in kernel_main; `tilize_stick_reads.hpp` `HopReaderResync` (`noc_local_state_init(noc_index)`) |

## Mechanism (pre-rebase, still the basis)

1. **Coordinates.** Physical NoC0 xy comes from NIU 0's `NOC_NODE_ID`, and DRAM endpoints come
   from `dram_bank_to_noc_xy[0][b]`. Both NoCs share each bank's endpoint.
2. **NoC0 DRAM writes have far less capacity.** In the LOOSE 7 writes-only ablation, all writes on
   NoC0 take 41861 ns against 15573 on NoC1. Any 50/50 split loses.
3. **Picking pairs by hops is the lever, at about NoC0's capacity share.** Moving a bank only when
   that saves ≥ 6 hops puts 28 % of pairs on NoC0. At that same 28 % share: hop T6 14426, random
   16537, anti-T6 18509, head 17167.
4. **NoC0 must be free of heavy DRAM read-response traffic.** That is carve-out 2.
5. **The dedicated mode plus handshake avoids the DM_DYNAMIC_NOC tax.** The two-RISC-V "duo" split
   was no better (the NoC path is the limit, not the issue rate).

The reader twin (NCRISC reads on NoC1 for a resident output) is an option, not in the graduate:

| case (head → T4) | head | T4 |
|---|---|---|
| LOOSE 8 | 12048 | 11437 (−5.1 %) |
| [1,1,8192,256] → HS | 22479 | 20584 (−8.4 %) |
| 4 Tensix cores | 3682 | 4660 (+26.6 %) |

See the variant menu in `make_variants.py`.

## Reproduce

```bash
# A/B (cases: LOOSE indices or EXTRA keys of the harness; "grad2" = graduate/, knobs after "@", "+"-joined)
TILIZE_PERF_EXPERIMENTS=1 HOP_REPEAT=4 HOP_CASES=7,hs256,l1_64,0 \
  HOP_VARIANTS="head,grad2,grad2@HOP_WRITE_MIN_SAVING=0" \
  scripts/run_safe_pytest.sh --profile --run-all tests/ttnn/unit_tests/operations/tilize/test_tilize_perf2_hop_aware_noc.py -s > log
python3 ttnn/ttnn/operations/tilize/perf_experiments/hop_aware_noc/label2.py log generated/profiler/reports/<dir printed as PROFILER CSV>/
# carve-outs lifted: grad2@HOP_WRITE_MIN_CORES=0, grad2@HOP_WRITE_DRAM_INPUT_MAX_BYTES_PER_CORE=None
# --dev + back-to-back + negative control
TILIZE_PERF_EXPERIMENTS=1 HOP_CASES=0,7,l1_64 HOP_VARIANTS=grad2 scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/unit_tests/operations/tilize/test_tilize_perf2_hop_aware_noc.py
TILIZE_PERF_EXPERIMENTS=1 HOP_B2B=1 scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/unit_tests/operations/tilize/test_tilize_perf2_hop_aware_noc.py::test_hop_back_to_back -s
TILIZE_PERF_EXPERIMENTS=1 HOP_CASES=7 HOP_VARIANTS="grad2@KERNEL_DIR='hop_aware_noc/kernels_neg_noresync'" scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/unit_tests/operations/tilize/test_tilize_perf2_hop_aware_noc.py
# unit nets against the graduate
PYTHONPATH=$PWD/ttnn/ttnn/operations/tilize/perf_experiments/hop_aware_noc/graduate HOP_GRADUATE_COUNT=1 \
  scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/tilize/test_tilize_knobs.py -p hop_graduate_plugin
# off switch == HEAD binaries (after a run that built both)
python3 ttnn/ttnn/operations/tilize/perf_experiments/hop_aware_noc/elf_compare.py
```

After `git apply graduate.diff`, the coordinator's harness A/B is
`TILIZE_P2_VARIANTS="head,head@HOP_WRITE_MIN_SAVING=0"`.
