<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
SPDX-License-Identifier: Apache-2.0
-->

# tilize — self reflection (advisory, post-blind)

## Summary

Blind final: **988 passed / 884 skipped / 13 failed / 2 error / 2 xfail** over 1889 cells. The
registry surface is spotless — all 940 `test_golden.py` cells pass, skip or xfail, and
`verify_supported` reports `supported_fail: 0`, `xpass_drift: 0`. **Every one of the 15
failures lives outside `test_golden.py`** (11+2 in `test_golden_main_tests.py`, 1 in
`test_golden_main_trace.py`, 1 in `test_translated.py`).

The single most important finding is framework-level, not op-level: the identical 14
main-suite failures were already present at `golden_refinement_4` and are **byte-identical**
through refinements 5 and 6, both perf rounds, and the blind pass — while every changelog
"Golden test progress" section reported `0 failed`, because the loop's gate is computed over
`test_golden.py` only. Three refinements and two perf tournaments ran on top of a frozen,
unreported failure set.

Everything else follows from one shape: **golden's axis model is one-sided and origin-anchored**,
so the four failure clusters sit in regions the taggers cannot even name. The perf phase
(2 rounds, 7 graduations) produced unusually good helper feedback, including one latent
**correctness** bug in a shared helper.

---

## 1. Golden coverage → `eval/golden_tests/tilize/feature_spec.py`

Note: `feature_spec.py` declares no `LOOSE_CASES` at all (700 lines, `TARGET`/`INPUTS`/`INVALID`
only), so all four proposals below are the first entries in that list.

### F1 — The `use_multicore=False × sharded` region contains **zero** golden cells (8 of 13 failures)

**What.** 8 blind failures are `test_tilize_nd_sharded[False-…]` — a sharded call with
`use_multicore=False`. Both axis values are in `TARGET` (`use_multicore: [False, True]`,
`shard_api: ["none","legacy_2d","nd"]`), and the op declares the pair in `EXCLUSIONS`. But golden
never pairs them: of 940 cells with axes, **0** have `use_multicore=False ∧ shard_api != "none"`
— every one of the 34 sharded scenarios in `INPUTS` hardcodes `use_multicore: True`. The
exclusion is therefore *unfalsifiable inside golden*: it can never surface as `supported_fail`
(not SUPPORTED) nor as `xpass_drift` (no cell exists). The upstream acceptance suite exercises it
and fails 8 times.

**Evidence.**
- `golden_blind_final/verifier_report.json` — 940 cells with axes, `use_multicore=False ∧ shard_api!=none` → **0**.
- `eval/golden_tests/tilize/feature_spec.py:291,297,303,311,317` — every sharded scenario is `"use_multicore": True`.
- `changelog.md:271` — the rationale: "`use_multicore=False` × sharded (a shard is inherently multi-core)".
- Failure: `test_golden_main_tests.py::test_tilize_nd_sharded[False-True-False-shard_core_grid0-tensor_shape0-…]`
  → `ExcludedCell: tilize: unsupported combination (refinement candidate): {'use_multicore': False, 'shard_api': 'nd'}`.

**Recommendation.** Add a `LOOSE_CASES` cell that pins the region so the exclusion becomes
testable (it will xfail until promoted, and flip to `xpass_drift` the moment it works) — minimal
representative, 2 cores:
`({"input_shape": [1,1,64,64], "use_multicore": False, "shard_api": "legacy_2d", "in": _sh(_L1, _crs(((0,0),(1,0))), (32,64), _ROW, _HEIGHT), "out": _sh(_L1, _crs(((0,0),(1,0))), (32,64), _ROW, _HEIGHT)},)`.
Consider a spec rule: **an `EXCLUSIONS` pair must have at least one golden cell**, otherwise it is
an unmeasured assertion. **Confidence: high.**

### F2 — A **sharded DRAM source** is invisible to the taggers (axis-blind) *and* `dram_to_l1 × WIDTH_SHARDED` is untested (axis-covered)

**What.** The one translated failure crashes inside tilize's program build on a DRAM
width-sharded input. Golden looks covered — 60 cells carry `buffer=dram_to_l1 ∧ shard_api ∈
{legacy_2d, nd}` — but **all 60 have an *interleaved* DRAM source**: `feature_spec.py` contains
`_sh(_DRAM` **zero** times. No tagger reads the *input* spec's kind or scheme (`tag_out_scheme`
reads `scenario["out"]` only; `tilize.py:53-57`), so "the DRAM side is itself sharded" is not an
axis at all. Separately, at the plain axis level `out_scheme=WIDTH_SHARDED` exists only with
`buffer=l1_to_l1` (60/60 cells) — so `dram_to_l1 × WIDTH_SHARDED` is a straightforward missing
axes-tuple too.

**Evidence.**
- `test_translated.py::test_tilize_width_sharded_dram_input_to_l1_sharded_output_49107` →
  `RuntimeError: No core coordinate found at location: (8, 0, TENSIX, LOGICAL)`, frame 4
  `tt::tt_metal::TensorAccessorArgs::get_compile_time_args()` — i.e. inside the op, not the test.
- The test's grid is derived from the **DRAM** grid: `test_translated.py:270` `num_cores = device.dram_grid_size().x`, `:276` `CoreRange(CoreCoord(0,0), CoreCoord(num_cores-1,0))`.
- `verifier_report.json`: `WIDTH_SHARDED` cells → `Counter({'l1_to_l1': 60})`; `dram_to_l1 ∧ sharded` → `Counter({('dram_to_l1','legacy_2d','HEIGHT_SHARDED'): 40, ('dram_to_l1','nd','nd'): 20})`.
- `grep -c "_sh(_DRAM" feature_spec.py` → `0`.

**Recommendation.** (a) `LOOSE_CASES` entry with a genuinely DRAM-*sharded* source:
`({"input_shape": [32, 256], "use_multicore": True, "shard_api": "legacy_2d", "in": _sh(_DRAM, _crs(((0,0),(3,0))), (32,64), _ROW, _WIDTH), "out": _sh(_L1, _crs(((0,0),(3,0))), (32,64), _ROW, _WIDTH)},)`.
(b) Promote the input side to an axis — add `tag_in_scheme` mirroring `tag_out_scheme`
(`"interleaved" | TensorMemoryLayout | "nd"`) so the input's placement is no longer collapsed into
`buffer`. This one tagger also disambiguates `shard_api`, which today is a single hand-written
scenario key for a two-sided property (`feature_spec.py:531` tags an nd-in→legacy-out scenario
`"nd"`; `:540` tags legacy-in→nd-out `"legacy_2d"`). **Confidence: high.**

### F3 — Every golden shard grid is anchored at core `(0,0)`; a non-origin / reserved-core grid crashes

**What.** The trace-mode failure is a `TT_FATAL` "Kernels cannot be placed on dispatch cores" for
`tilize_compute`. The discriminative comparison is clean: the *same* test with the *same*
`device_params` passes on `interleaved` and fails on `width_sharded` — so the trigger is not trace
mode, it is the **caller-supplied shard grid**. The op adopts that grid verbatim as its kernel
placement (`tilize_program_descriptor.py:797-799` "Cores are the cores that HOLD the shards …
`cores = shard["cores"]`") without intersecting it against the device's legal worker grid; the
accessor path does respect `compute_with_storage_grid_size()` (`:516`), the sharded path does not.
All **34** golden shard grids are single rectangles anchored at the origin
(`grep -o "_crs(((<x>, <y>)" → 34 × "(0, 0)"`), so no golden cell can express a grid that starts
elsewhere or overlaps a reserved core.

**Evidence.**
- `test_golden_main_trace.py::test_deepseek_v3_mla_tilize_trace_mode[device_params0-100-10-width_sharded-wo_tilize-32]`
  → `TT_FATAL @ program.cpp:148: not on_dispatch_core | Illegal kernel placement for tilize_compute`.
  Same file's `interleaved` variant passes.
- The grid has a non-zero origin: `test_golden_main_trace.py:82` `CoreRange(CoreCoord(1, 0), CoreCoord(4, 1))`; `device_params` sets `dispatch_core_axis=DispatchCoreAxis.COL` (`:54`).
- `tilize_program_descriptor.py:797-799` vs `:516`.

**Recommendation.** `LOOSE_CASES` entry with an **origin-offset** grid (cheap, catches the whole
class):
`({"input_shape": [1,1,64,128], "use_multicore": True, "shard_api": "legacy_2d", "in": _sh(_L1, _crs(((1,0),(2,0))), (64,64), _ROW, _WIDTH), "out": _sh(_L1, _crs(((1,0),(2,0))), (64,64), _ROW, _WIDTH)},)`.
Consider a `grid_origin: ["zero","offset"]` axis. Op-side (not golden's job, noted for the human):
validate() should reject / clamp a placement grid that is not a subset of the available worker
grid instead of reaching `program.cpp`. **Confidence: high.**

### F4 — ND shard shape whose last dim does not divide the tensor's last dim (3 failures)

**What.** 3 failures are `test_tilize_nd_sharded_to_legacy_sharded[*-tensor_shape2-shard_shape2-*]`
— all three output layouts, one shape. The discriminative conjunction is exact: of the three
parametrized shapes, only `([7,128,128], [2,64,96])` has `shard_shape[-1] ∤ tensor_shape[-1]`
(96 ∤ 128); `[2,64,64]` and `[2,32,64]` divide 128 and both pass. The non-dividing ND shard pads
the tensor's **physical** width to 192, after which a legacy output shard spec derived from the
*logical* width 128 is unsatisfiable — and the op propagates it into `TensorSpec` instead of
refusing. No tagger sees this: `tag_alignment` (`tilize.py:88-107`) measures `input_shape`'s last
two dims against the tile, never shard-shape divisibility. Golden exercises non-dividing widths
only on the nd **output** side (`feature_spec.py:540-545`, out shard `(3,96,96)` on `[3,128,128]`),
never on the nd **input** side crossed with a legacy output.

**Evidence.**
- `test_golden_main_tests.py:250-254` — the three shapes; only `([7,128,128],[2,64,96])` fails.
- Failure messages name the padded physical width directly: "Shard width 128 must match physical
  width 192 for height sharded"; "Number of shards along width 6 must not exceed number of cores 4";
  "Number of shards along width 3 must not exceed number of columns 2" — all `TT_FATAL @ tensor_spec.cpp:153`.

**Recommendation.** `LOOSE_CASES` entry pinning nd-in (non-dividing) → legacy-out:
`({"input_shape": [7,128,128], "use_multicore": True, "shard_api": "nd", "in": _sh(_L1, _crs(((0,0),(1,1))), (2,64,96), _ROW, None), "out": _sh(_L1, _crs(((0,0),(1,1))), (224,128), _ROW, _HEIGHT)},)`,
and consider an axis for the facet, e.g. `shard_divides: [True, False]`. If a human judges the
request genuinely unsatisfiable, the right home is `INVALID` — but then the op still owes a clean
rejection rather than a `TT_FATAL`. **Confidence: med** (the pass/fail split is unambiguous; the
padded-width mechanism is inferred from the error text).

### F5 — Test-universe hygiene: the trace test is duplicated, and the duplicate errors on setup

**What.** 2 of the 15 are pure harness errors, not op behavior:
`use_module_device` + `parametrize("device_params")` is illegal. The pipeline already knew — it
created `test_golden_main_trace.py` *solely* to work around it — but never removed the copy from
`test_golden_main_tests.py`, so the broken pair errors in every phase.

**Evidence.** `test_golden_main_trace.py:6-21` — "This file exists for one reason: the case below
parametrizes `device_params` … **DO NOT** add `pytestmark = pytest.mark.use_module_device` here."
Yet `test_golden_main_tests.py::test_deepseek_v3_mla_tilize_trace_mode[…]` errors:
`ValueError: Cannot use @pytest.mark.use_module_device with @pytest.mark.parametrize('device_params', ...)`.

**Recommendation.** Delete the `test_deepseek_v3_mla_tilize_trace_mode` copy from
`test_golden_main_tests.py` (it lives in `test_golden_main_trace.py`); consider making the
test-splitting step remove the original. **Confidence: high.**

---

## 2. SUPPORTED honesty → `tilize.py` `SUPPORTED` / `EXCLUSIONS`

### H1 — The report is clean, but it is blind to 50% of the suite; recommend no `SUPPORTED` edit on its evidence alone

**What.** `verify_supported` on the blind dir: `supported_pass: 346`, **`supported_fail: 0`**,
**`xpass_drift: 0`**, `xfail_wrong_mode: 0`, `invalid_unexpected: 0`. Within the registry surface
the declarations are honest — there is no over-claim to fix or demote and no under-claim to
promote. But `no_axes_found: **949**` (50.2% of 1889) is exactly `235 main_tests + 685 translated
+ 27 regression + 2 trace`, i.e. **the entire non-registry surface is uncategorized**, and all 15
failures are in it. So "0 `supported_fail`" measures the axis model, not the op's API.

**Evidence.** `golden_blind_final/verifier_report.json` `summary`; 940 of 1889 cells carry axes.

**Recommendation.** No `SUPPORTED`/`EXCLUSIONS` edit is justified by the verifier report itself.
Consider having `verify_supported` report `no_axes_found` **and the failure count inside it** as a
loud category, so a 50%-blind run cannot read as a clean bill of health. **Confidence: high.**

### H2 — Promote (fix), don't keep excluding: `use_multicore=False × sharded`

**What.** The one declaration this run gives evidence against is the F1 exclusion pair. Its stated
rationale — "a shard is inherently multi-core" (`changelog.md:271`) — is a design assumption that
the reference API contradicts: in the upstream suite `use_multicore` is a **performance hint**
passed alongside a sharded `memory_config` (`test_golden_main_tests.py:241`
`ttnn.tilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=use_multicore)`),
not a placement constraint. 8 blind failures.

**Recommendation.** Propose **fix → promote**: implement single-core sharded (gather/scatter
through one core, correctness over speed) and move `{use_multicore: False, shard_api: nd}` /
`{…, legacy_2d}` out of `EXCLUSIONS`. If a human decides to keep the exclusion, it should be
recorded as a **known reference-suite divergence** in `op_requirements.md` rather than a silent
`EXCLUSIONS` row, since the current form makes 8 real failures look like correct refusals.
**Confidence: high** (that the region is untested and fails; **med** on which of fix/document is
the better call).

---

## 3. Helper / reference docs → helper docstrings + `.claude/references/`

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `compute_kernel_lib::tilize` → `ckernel::tilize_block` (Perf 2, `tilize_compute.cpp:65`, 13,435→7,428 ns) | capability: no DEST-window knob; batched form only inside `fast_tilize_block` | **missing** — confirmed, but **one layer below the claim**: the gap is in the compute API, not `kernel_lib` | `tt_metal/hw/inc/api/compute/tilize.h:178-195` — `for (t < block) { llk_math_wait_for_dest_available … llk_pack(0 /*tile index*/…) … llk_math_dest_section_done }`, i.e. one DEST round trip per tile on slot 0; `fast_tilize_block` batches by chunk at `:375-397`. `tilize_helpers.inl:246` faithfully forwards `block_width_tiles` to `tilize_block`, so the row's "controls the CB handshake width, not the DEST window" is accurate | add a `tiles_per_dest` parameter to `tilize_block` in `api/compute/tilize.h` (sketch below); `compute_kernel_lib::tilize` then forwards it. Fixing the wrapper alone cannot help |
| `dataflow_kernel_lib::read_sticks_for_tilize` (3 bypass sites: `tilize_reader.cpp:347` 16,371→13,949 ns, `:577` 24,721, `:731` 8,981) | capability: no trid / in-flight window / multi-stick merge / page-stride | **missing** — confirmed, but the *gather* row is narrower than written | `.inl:117-127` — `cb_reserve_back` … `for (row…) noc_async_read(…)` … `noc_async_read_barrier(); cb_push_back` — one plain read per stick, one barrier per block, CB handshake owned internally. `.hpp:88-93` signature is `(accessor, total_num_rows, row_bytes, start_page, byte_offset_within_page)` — no policy argument exists. **However** the column-slice half of the `:731` gather row **is** present and documented (`.hpp:73-75` "Combined with `byte_offset_within_page` this selects a chunk along W"), so what is actually missing there is only the **page-index map** | one `StickReadPolicy` argument (trid / in-flight window / coalesce hint / barrier ownership) plus an optional page-index functor (sketch below). Re-scope the `:731` row to "page-index map" — the W-slice is already there |
| *(none covers it)* — L1→L1 block move (Perf 1, `tilize_reader.cpp:513`, 99,849 rv32 → 41,902 loopback) | capability: no local L1→L1 block-move helper at all | **missing (batched move) + `undocumented` (the enabling primitive)** | `l1_helpers.hpp` contains only `addr_to_l1_ptr`, `zero_tile`, `prepare_zero_tile` — and **`local_noc_addr(addr, noc_id)` at `:30-36`, "Create NOC source/destination args for a local L1 address on this core"**: exactly the loopback primitive this bypass needed. The kernel never found it — `tilize_reader.cpp` includes `tilize_helpers_dataflow.hpp` and `dataflow_api.h` but **not `l1_helpers.hpp`**, and wrote raw `get_noc_addr(...)` at `:516,:574` | two separate, much cheaper items: (1) doc line on `local_noc_addr` — "a local NoC read is the fast path for L1→L1 block moves; far cheaper than an rv32 store loop" + cross-reference from the dataflow reference; (2) the genuinely missing batched strided move (sketch below) |
| `dataflow_kernel_lib::write_sticks_after_untilize` (Perf 1, `tilize_writer.cpp:392`, 141,271 ns raw) | capability: inverse direction; and no way to source a page from an L1 address other than the CB slot | **missing** — confirmed | `.hpp:130-135` — `(accessor, total_num_rows, row_bytes, start_page, byte_offset_within_page)`; the payload is the template's `cb_id` and there is no source-address parameter, so the pre-stamped pad tile cannot be named. Direction confirmed by the name/contract (tiles → sticks) | add a `write_pages_from(accessor, …, uint32_t src_l1_addr)` overload (or a `payload_src` parameter) to the page-writer family |
| `compute_kernel_lib::tilize` (split alternation, `tilize_compute.cpp:131`) | **not bypassed** — recorded as the place a bypass was expected and did not happen | **correct, and worth keeping in the format** | 14,768 (helper) vs 14,780 (raw) — the two ns are ~equal *and no bypass was taken*, which is the good case: the documented `InitOnly`/`Neither`/`UninitOnly` lifecycle expressed the two-CB alternation for free | none. Recommend the perf prompt keep requiring these negative rows — they are what makes the table's positive rows credible |
| — | *unrecorded bypasses* | **record is complete** | grepping all four kernel files for `llk_*` / low-level calls yields exactly one raw-LLK site, `tilize_compute.cpp:66-87` (`tilize_block_wide`), which **is** row 1; `noc_async_read_one_packet` at `tilize_reader.cpp:149` is public dataflow API inside the recorded `read_sticks_for_tilize` family. No silent bypass found | none — noting it because a clean result here is itself evidence |

**API sketches** (signature + call site only; each derived from the raw sequence actually written).

```cpp
// (1) Batched-DEST regular tilize — derived from tilize_compute.cpp:62-91 (`tilize_block_wide`),
//     which is tilize_block plus a DEST window. Proposed in api/compute/tilize.h:
ALWI void tilize_block(uint32_t icb, uint32_t block, uint32_t ocb,
                       uint32_t input_tile_index = 0, uint32_t output_tile_index = 0,
                       uint32_t tiles_per_dest = 1);   // 1 == today's behavior, bit-identical
// call site becomes (replacing the op's whole 30-line raw loop):
tilize_block(icb, block, ocb, 0, 0, /*tiles_per_dest=*/window);
// and through the wrapper: compute_kernel_lib::tilize<W, in_dfb, out_dfb, /*tiles_per_dest=*/4>(nblk);

// (2) read_sticks_for_tilize policy — derived from tilize_reader.cpp:143-156 (`issue_tile_row`,
//     which is the helper loop plus `coal` and `one_packet`) and the op's per-block trid deferral:
struct StickReadPolicy { uint8_t trid = 0; uint8_t inflight_blocks = 1;
                         uint32_t coalesce_sticks = 1; bool own_barrier = true; };
read_sticks_for_tilize<cb_in>(accessor, rows, row_bytes, start_page, byte_off, policy);
// and for the gather at :731, the only missing piece — a page-index map:
read_sticks_for_tilize<cb_in>(accessor, rows, row_bytes, PageMap{/*row_pages=*/rp}, byte_off, policy);

// (3) Local strided block move — derived from tilize_reader.cpp:565-584: n runs of run_bytes,
//     strided source and strided destination, caller owns ONE barrier for the batch:
void local_copy_strided(uint32_t dst_l1, uint32_t dst_stride,
                        uint32_t src_l1, uint32_t src_stride,
                        uint32_t run_bytes, uint32_t n_runs);   // issues only; no barrier
// call site: the triple loop collapses to one call per row, then the existing
// noc_async_read_barrier() at :584 stays exactly where it is.
```
`confidence: low` on (2)'s exact shape and (3)'s parameterization — both are extrapolated from this
op's single call site each; (1) is a direct transcription of code that already exists and is measured.

### D1 — Latent **correctness** bug in `compute_kernel_lib::get_dest_limit()` / `DEST_AUTO_LIMIT`

**What.** The highest-severity helper finding of the run, and it is not a bypass row — it is
buried in Perf-2 prose. `get_dest_limit()` keys DEST capacity on `DST_ACCUM_MODE` and sync mode
**only**; but a 32-bit *input datum* occupies a 32-bit DEST slot regardless of the accumulation
flag. So with `fp32_dest_acc_en=false` it reports 8 where the true capacity is 4, and any caller
trusting it to size an integer-format DEST fill corrupts silently. The header's own capacity table
states the rule as a pure function of `DST_ACCUM_MODE`, i.e. the docstring is *actively* wrong,
not merely silent. Shared by `untilize_helpers.hpp`, `tilize_helpers.hpp` "and other kernel
libraries".

**Evidence.** `dest_helpers.hpp:89-97` — `constexpr uint32_t get_dest_limit() { constexpr bool
is_fp32_accum = get_fp32_dest_acc_enabled(); … return is_fp32_accum ? 8 : 16; }`; the table at
`:22-26` ("SyncHalf + 16-bit (DST_ACCUM_MODE=false): 8 tiles"). Measured by the perf phase:
`changelog.md:2046-2050` "uint32 with `fp32_dest_acc_en=false` is **not bit-exact** at the limit it
reports (8) and is exact at 4". The op carries the corrected rule and says so:
`tilize_compute.cpp:233-239` "NOT `compute_kernel_lib::DEST_AUTO_LIMIT`: that halves the capacity
on `DST_ACCUM_MODE` alone, but a 32-BIT INPUT DATUM …".

**Recommendation.** Make `get_dest_limit()` take the datum width (or the input DFB) into account,
and correct the capacity table at `dest_helpers.hpp:18-26` to say **"capacity is set by the widest
of the accumulation mode and the input datum width"**. Until then, a warning line in the
docstring. **Confidence: high** (measured, and the code matches the claim).

### D2 — `.claude/references/` documents the TRID API but not the rule whose absence hung the device

**What.** Adding a write TRID caused a whole-grid hang; the cause was that a TRID left set at
kernel exit trips a firmware assert. The reference documents the four TRID entry points in a table
and says nothing about clearing them — a missing invariant, not a wrong one.

**Evidence.** `.claude/references/data_transfer_analysis_reference.md:653-660` — the table lists
`noc_async_write_set_trid(trid, noc)` / `noc_async_read_set_trid` and closes with only "TRIDs allow
fine-grained synchronization". Breadcrumb `ttnn-implementer_breadcrumbs.jsonl`:
`hang_detected` on `probe_019` → hypothesis "`brisck.cc:91 ASSERT(ncrisc_noc_packet_tags_cleared)`
tripped because `noc_async_write_set_trid` left …" → `fix_applied`:
"`noc_async_write_set_trid(0)/noc_async_read_set_trid(0)` before kernel exit, hang gone".

**Recommendation.** Add one line under that table: "**A kernel must reset every TRID it set
(`…_set_trid(0)`) before returning — firmware asserts `ncrisc_noc_packet_tags_cleared` at exit and
the core hangs otherwise.**" **Confidence: high.**

### D3 — Nothing documents that an 8-bit datum requires `fp32_dest_acc_en` (silent all-zeros)

**What.** `uint8` tilize returned **all zeros** (8159/8192 wrong) — a silent wrong-answer, not a
crash. The rule (8-bit datums need dest accumulation enabled) is documented nowhere in
`.claude/references/`; the implementer recovered it only by reading an in-tree LLK test. This is
the same missing concept as D1 — the relationship between *datum width* and DEST configuration —
so one doc fix can serve both.

**Evidence.** `ttnn-implementer_breadcrumbs.jsonl`: `probe_021` "uint8 returns ALL ZEROS
(8159/8192 wrong)"; hypothesis "the in-tree LLK 8-bit tilize test … only passes with
`DestAccumulation.Yes`"; `fix_applied` "`fp32_dest_acc_en=True` when either side is an 8-bit datum
dtype". `grep -rln "8-bit\|EIGHT_BIT" .claude/references/` → no hit in
`ttnn-op-constraints.md` / `precision_convention.md`.

**Recommendation.** One line in `.claude/references/ttnn-op-constraints.md` (and in the tilize
helper docstring): "**An 8-bit datum on either side requires `fp32_dest_acc_en=true`; without it
the LLK tilize path silently emits zeros.**" **Confidence: high.**

### D4 — `compute_kernel_lib::tilize` did not compile at all; the library fix shipped inside an op commit

**What.** `has_unpack_to_dest_fp32` was defined twice (byte-identical), so *any* kernel including
`tilize_helpers.hpp` failed to compile on all three TRISCs — the helper was unusable out of the
box for every consumer. It cost a full compile-fail debug cycle, and the fix landed inside the
op's Phase-0 commit, where it is invisible as a library repair.

**Evidence.** Breadcrumb `ttnn-implementer` friction, `ref: ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl:47-81`
— "defined TWICE … `compute_kernel_lib::tilize` was unusable out of the box … blocks every future
op that uses the tilize helper (`toy_tilize_untilize` is equally broken)". Commit `ae597debd7`
("[ttnn-implementer] PASS: tilize Phase 0") body: "Also fixes a pre-existing blocker:
`tilize_helpers.inl` had `has_unpack_to_dest_fp32` defined twice"; its diff removes the duplicate
block. Now 1 definition (`tilize_helpers.inl:48`).

**Recommendation.** Add a compile smoke test per helper header in `kernel_lib/tests/` so a header
that cannot be included is caught by the library, not by the next op. Consider a convention that
shared-library repairs are split into their own commit so they are attributable. **Confidence: high.**

### D5 — The op template cannot express an explicit padded output shape

**What.** Every padding op needs an output whose padded shape differs from
tile-round(logical) — the tilize contract requires it — and the reference template shows only the
allocation form that derives padding from the logical shape. The working mechanism exists but is
documented nowhere.

**Evidence.** Breadcrumb `ttnn-implementer` friction, `ref: .claude/references/generic_op_template/template_op.py:47-54`
— "shows only `allocate_tensor_on_device(shape, dtype, layout, device, mem_config)`, which derives
the padded shape from the logical one … The working mechanism is `ttnn.reshape(t, logical_shape,
padded_shape)`, which is a zero-cost view (same buffer address, verified by probe) but is
documented nowhere in the references and reads like a data-movement op."

**Recommendation.** Add to `template_op.py` (comment) and the op-constraints reference:
"**To give an output an explicit padded shape, allocate logically then `ttnn.reshape(t,
logical_shape, padded_shape)` — a zero-cost view, not a data movement.
`TensorLayout::fromPaddedShape` has no Python binding.**" **Confidence: high.**

### D6 — `assert_with_pcc` docstring contradicts its return value

**What.** The docstring promises `(pcc_passed: bool, pcc_message: str)`; in this tree the second
element is a float, so the precision-baseline recording the verifier prompt asks for fails with
`AttributeError` on the first device run.

**Evidence.** Breadcrumb `incremental-verifier` friction, `ref: tests/ttnn/utils_for_testing.py:110-126`
— "`comp_pcc` returns a float so the second element is the measured PCC as a float … following the
docstring (str ops on the second element) fails with `AttributeError` on the first device run."

**Recommendation.** Fix the docstring to `-> (bool, float)` (and mention it is the measured PCC).
**Confidence: high.**

---

## 4. Agent prompts → `.claude/agents/*.md`, `eval/prompts/*`

### P1 — The refinement/perf gate is computed over `test_golden.py` only, so 14 acceptance failures froze in place for 5 phases

**What.** The most consequential process finding of the run. The identical 14 failures/errors
(8 × `test_tilize_nd_sharded[False-…]`, 3 × `nd_sharded_to_legacy_sharded[…tensor_shape2…]`,
2 trace setup errors, 1 trace `TT_FATAL`) are present at `golden_refinement_4` and **unchanged**
at refinement 5, refinement 6, `golden_perf_1`, `golden_perf_2` and the blind pass. No refinement
targeted them, because each phase's own report only ever counted `test_golden.py` and the unit
directory. Two full perf tournaments were then run on a tree with a known-broken acceptance
surface.

**Evidence.**
- `golden_refinement_4/5/6`, `golden_perf_1/2` `test_results.json`: each has exactly
  `Counter({'test_golden_main_tests': 13, 'test_golden_main_trace': 1})` failures/errors — the same
  nodeids in the same order.
- `changelog.md:1462-1474` (Refinement 6 "Golden test progress") reports only
  "`-k "legacy_2d or explicit or l1_to_l1"` → **214 passed, 346 skipped, 0 failed, 0 xpass**" and
  "Whole tilize unit directory: **333 → 338 passed**" — `test_golden_main_tests.py` is never named.

**Recommendation.** In the refinement/perf prompts, make the phase report state the **full**
suite's failed count (`test_golden.py` + `test_golden_main_tests.py` + `test_golden_main_trace.py`
+ `test_regression.py`), and require that a phase which leaves a *pre-existing* non-registry
failure unchanged say so explicitly. A frozen failure set should be loud, not invisible.
**Confidence: high.**

### P2 — A refinement may add an `EXCLUSIONS` row that the acceptance suite contradicts, with no cross-check

**What.** Refinement 1 added `use_multicore=False × sharded` to `EXCLUSIONS` on a design
assumption; 8 acceptance cells exercise exactly that pair (F1/H2). Nothing in the process
required checking a new exclusion against the reference suite, and because golden has no cell
there, no later gate could catch it either. This is the mechanism behind the largest failure
cluster, and it is generic: any exclusion whose region golden does not populate is
self-confirming.

**Evidence.** `changelog.md:271` (the exclusion + rationale) vs
`test_golden_main_tests.py:241` (the acceptance call passing `use_multicore=use_multicore`
alongside a sharded `memory_config`); `verifier_report.json` — 0 golden cells in the region.

**Recommendation.** In the planner/verifier prompts: **before adding an `EXCLUSIONS` entry, grep
the acceptance + translated suites for cells in that region; if any exist, the entry needs an
explicit "known divergence" justification in `op_requirements.md` and at least one golden cell
pinning it.** **Confidence: high.**

### P3 — A new-axis refinement was not required to cross-test the caller-supplied placement it inherited

**What.** Refinement 1 introduced the sharded surface and, with it, the rule "launch only where
the data is" — the op adopts the caller's shard grid as its kernel placement. That grid is
attacker-controlled input, but the gate for the refinement was built entirely from
origin-anchored grids on a default device, so neither a non-origin grid nor a reserved-core
device configuration was ever crossed against the new axis (F3). The verifier signed off
(`verifier_report.json`, `supported_fail: 0`).

**Evidence.** `tilize_program_descriptor.py:797-799` (`cores = shard["cores"]`, no intersection
with the worker grid) vs `:516` (the accessor path does call
`compute_with_storage_grid_size()`); all 34 golden grids at origin `(0,0)`; the resulting
`TT_FATAL @ program.cpp:148`.

**Recommendation.** Verifier prompt: when a refinement adds an axis whose values are
**caller-supplied resources** (core grids, buffers, memory configs), require one cell that
supplies a *hostile-but-legal* value — here, a grid not anchored at `(0,0)`. **Confidence: med.**

### P4 — The documented `verify_supported` invocation suppresses the report the prompt then tells you to read

**Evidence.** Breadcrumb `incremental-verifier` friction, `ref: eval/verify_supported.py:441-446`
— "documented as `python3 -m eval.verify_supported <results_dir> <op_module> --output …` and then
says \"The CLI emits a categorized report. Read every loud category\" — but `--output`
**suppresses** the human-readable report (the `render_text` branch is the `else` of `if
args.output`). Following the prompt verbatim yields zero stdout."

**Recommendation.** Either make `--output` also render to stdout, or change the prompt to invoke
twice / add `--report`. **Confidence: high.**

### P5 — The Output-Summary table implies persisting a `verifier_report.json` that trips the repo's own pre-commit hook

**Evidence.** Breadcrumb `incremental-verifier` friction, `ref: eval/prompts/*: Output Summary
table (verifier_report.json row)` — "1.2 MB (940 registry cells x full axes dict), which trips the
repo pre-commit hook `check for large files (>500 KB)` … Two aborted commits before trimming it to
summary + per-category counts + blocking-axis histogram + sample cells."

**Recommendation.** Prompt should specify the **trimmed** shape (summary + per-category counts +
blocking-axis histogram + sample cells) for the committed artifact, keeping the full JSON in the
results dir only. **Confidence: high.**

### P6 — Two authorities disagreed on Phase-0 scope; the op could have shipped failing its own gate

**Evidence.** Breadcrumb `ttnn-implementer` friction, `ref: eval/prompts/tilize.txt (Phase 0
SUPPORTED) vs tests/ttnn/unit_tests/operations/tilize/test_tilize.py` — "`eval/prompts/tilize.txt`
pins Phase 0 SUPPORTED to `use_multicore=False, pad_mode=none, dtype=bfloat16, rank=4`; the
acceptance test the task prompt calls THE SPEC (and requires to pass) exercises multicore, both CB
depths, fp32/bf8b casts, ranks 2/3/5, and all three pad modes. Gating to the narrow rectangle
would have failed the acceptance test outright … a future agent can just as reasonably read it the
other way and ship a Phase 0 that fails its own gate."

**Recommendation.** State the precedence rule once in the implementer prompt: **the acceptance
suite is binding; a per-op `eval/prompts/<op>.txt` Phase-0 rectangle is a starting suggestion and
must be widened to whatever the acceptance suite exercises.** **Confidence: high.**

### P7 — `--profile` is documented as the way to get device ns but does not work in this checkout

**Evidence.** Breadcrumb `ttnn-implementer` friction, `ref: scripts/run_safe_pytest.sh:574-584 vs
.claude/skills/perf-measure/SKILL.md (Layer A1)` — "it never wraps `PYTEST_CMD` with `python -m
tracy` … prints \"WARNING: --profile set but this run produced no ops_perf_results CSV\" …
Burned ~4 device runs before I found the working recipe … only discoverable by reading
`tests/ttnn/unit_tests/operations/examples/test_double_buffer.py`."

**Recommendation.** Fix the script or update `perf-measure/SKILL.md` to document the in-process
recipe (`TT_METAL_DEVICE_PROFILER` / `MID_RUN_DUMP` / `CPP_POST_PROCESS` at module import +
`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`) as Layer A1.
**Confidence: high.**
