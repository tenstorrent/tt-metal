# Quasar Uplift Report — `ttnn.linear` (matmul op family)

**Date:** 2026-09-10
**Branch:** `vsureshTT/quasar_uplift_round_2` (uncommitted; delete this file before merge)
**Op directory:** `ttnn/cpp/ttnn/operations/matmul/`
**Driving test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_linear.py`
**Recipe followed:** `quasar_porting.md` §1 workflow → step 1 (confirm the Gen1 Metal 2.0 port) → RED-stop.

---

## Status: RED — not Metal 2.0 on Gen1 yet (uplift stopped at step 1, no code changed)

`quasar_porting.md` §1 step 1 and the "RED status" list are explicit: *"Not Metal 2.0 on Gen1 yet — factory
still `create_descriptor`/`ProgramDescriptor`. Do the Metal 2.0 port first."* Both program factories the
linear test can reach are on `create_descriptor` → `tt::tt_metal::ProgramDescriptor`, and every kernel they
bind is still on the legacy device API (`dataflow_api.h`, positional `get_compile_time_arg_val`, CB-index
`uint16_t` ids). There is nothing for an `ARCH_QUASAR`-guarded uplift to attach to. Per the recipe a RED
here is the audit succeeding — it stops a bad port — so this report records the evidence and stops.

### How the test maps onto the op (scope determination)

`test_linear.py` has 8 captured cases; each carries a `program_config`, and
`MatmulDeviceOperation::select_program_factory`
(`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2187-2220`) dispatches on its type:

| Cases | `program_config.kind` | Factory selected | Concept today |
|---|---|---|---|
| 00, 01, 02, 03, 04, 07 (1872 of 1968 captured calls) | `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` | `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` | `create_descriptor` → `ProgramDescriptor` (`device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.hpp:14`, `.cpp:937`) |
| 05, 06 (96 calls) | `MatmulMultiCoreReuseMultiCastProgramConfig` | `MatmulMultiCoreReuseMcast2DProgramFactory` | `create_descriptor` → `ProgramDescriptor` + `override_runtime_arguments` (`device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp:34-45`, `.cpp:3382-3392`) |

The only Metal 2.0 factory in the family is `MatmulMultiCoreProgramFactory`
(`create_program_artifacts`, `device/factory/matmul_multicore_program_factory.hpp:14`, landed in #55224),
selected only for `MatmulMultiCoreProgramConfig` — a config the linear test never uses. The remaining
factories (`ReuseOptimized`, `Mcast1D`, `MeshWorkloadMcast1D`, `BatchedHSDRAMSharded`) are also legacy
descriptor / mesh-workload and are off the test path; they were not audited further.

### Kernel-side confirmation (recipe §1 step 1: "a kernel still on the legacy device API is not ported")

Kernels bound by the two on-path factories (paths from the factory `kernel_source` assignments):

| Kernel (`device/kernels/...`) | Bound by | Legacy-API evidence |
|---|---|---|
| `dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` | DRAM-sharded (`.cpp:428`) | `#include "api/dataflow/dataflow_api.h"`; 21 legacy-idiom hits (`get_arg_val` / `get_compile_time_arg_val` / `noc_async_*`), no `dfb::`/`args::`/`tensor::` |
| `dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` | DRAM-sharded (`.cpp:443`) | same shape; 27 legacy hits |
| `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` | 2D (`.cpp:764`) | same; 47 legacy hits |
| `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | 2D (`.cpp:782`) | same; 79 legacy hits; also `#include "api/remote_circular_buffer.h"` + `#ifdef ENABLE_GLOBAL_CB` (GlobalCB path, see Deferred) |
| `dataflow/reader_bmm_tile_layout_in0_receiver.cpp` | 2D (`.cpp:818`, `:850`) | same; 11 legacy hits |
| `dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | 2D (`.cpp:800`, `:832`) | same; 35 legacy hits |
| `dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` | 2D (`.cpp:723`, block-sharded in0 path) | same; 32 legacy hits |
| `compute/bmm_large_block_zm_fused_bias_activation.cpp` | both (`dram_sharded.cpp:494`, `2d.cpp:903`) | mixed: constructs `DataflowBuffer` objects but from positional/named CTAs (`get_compile_time_arg_val(0..18)`, `get_named_compile_time_arg_val("cb_in0")` …), and writes `get_local_cb_interface(...).fifo_rd_ptr` at `:115-116` |

None of these kernels has an `experimental/kernel_args.h` include or a `dfb::` / `args::` / `tensor::`
token. The `_metal2` forks that do exist in this directory (`compute/bmm_metal2.cpp`,
`dataflow/reader_bmm_8bank_output_tiles_partitioned_metal2.cpp`) belong to `MatmulMultiCoreProgramFactory`
and are not bound by either on-path factory.

---

## Files changed

**None.** This report is the only artifact. No edit under `ttnn/cpp/ttnn/operations/matmul/`, no
`_metal2` fork created anywhere, no build/CMake/doc change. The op's directory and namespace are untouched;
nothing tempted a move or rename.

---

## How the user's note ("linear — try the experimental/quasar") was interpreted

`ttnn/cpp/ttnn/operations/experimental/quasar/matmul/` exists (a whole-op copy: `device/{config,factory,
kernels,sparse,utilities}`, `matmul.cpp`, `matmul_nanobind.cpp`). I read the note as *"there is a Quasar
copy of matmul; see whether that helps."* The recipe answers that directly, and I followed it:

- `quasar_porting.md` hard rule 1 and §2: *"Do not copy the op into `experimental/quasar/`… Do not import
  anything from `experimental/quasar/` — not a name, not a construct, not as 'evidence it's portable.'"*
  The uplift must land **in place** in `ttnn/cpp/ttnn/operations/matmul/`.
- `metal2_port.md` / `metal2_audit.md` / `pass_procedure.md` all forbid reading, citing, or forking that tree.

So I **listed the directory's file names only** (to confirm what the note refers to) and **did not open any
file in it**. Nothing from it informs this report. Two consequences for the user:

1. That copy cannot be the deliverable — the recipe's deliverable is the mainline op, reviewed via PR.
2. If the goal is *"get `ttnn.linear` running on the emulator for the llama32_1b_quasar model"*, the
   recipe-sanctioned path is: Metal 2.0 pre-port audit → in-place Metal 2.0 port of the two on-path factories
   → post-port passes → this Quasar uplift. The `experimental/quasar/matmul` copy is, per the recipe,
   bring-up scaffolding whose *lessons* have already been distilled into `quasar_porting.md` §7–§12 (e.g. the
   2D-mcast rectangle normalization it mentions); its *structure* is not to be reused.

If the user intended instead that the test should call `ttnn.experimental.quasar.*` matmul, that is a
model-test change outside this op's scope and outside the recipe; I did not make it.

---

## Gotchas (§7–§8, §11) — applied vs. considered

**Applied: none.** The recipe says §7–§8 fixes are *reactive* ("apply one only when its symptom actually
fires") and are `ARCH_QUASAR` guards on top of an already-Metal-2.0 op. With no Metal 2.0 factory on the
path and no device run, applying any of them would be manufacturing changes on a legacy descriptor factory
— exactly what the RED rule exists to prevent.

**Considered and deferred (pre-audit heads-up for whoever does the Metal 2.0 port + uplift).** These are
cheap grep-level observations on the on-path code; they are *not* a substitute for the Metal 2.0 pre-port
audit (`metal2_audit.md`) or the kernel audit (`cb_dfb_quasar_audit_helper.md`), which must still be run.

| Recipe item | Finding on the linear path | Why deferred |
|---|---|---|
| `quasar_audit.md` check 2 — **non-zero-init semaphore** | DRAM-sharded factory creates `in0_mcast_sender_valid_semaphore_id` with `.initial_value = VALID` (`= 1`, `hostdevcommon/common_values.hpp:14`) at `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:657-658`. The 2D factory's four semaphores are all `INVALID` (`= 0`) (`2d.cpp:1128-1135`). | Explicit Quasar-uplift blocker requiring an **op-owner change**; belongs in a separate PR before/with the Metal 2.0 port. |
| §11 ⚠️ — **mcast rectangle must stay ascending on Quasar** | Both factories carry the WH/BH NOC_1 start/end swap: `dram_sharded.cpp:106-113` (`if (in0_noc == NOC_1) std::swap(start_core_noc, end_core_noc)`), `2d.cpp:1161-1163, 1199-1208` (in0 and in1 swaps), and `2d.cpp:2656-2712` (legacy `matmul_multi_core_reuse_mcast_2d_optimized_helper` path). `in0_noc`/`in1_noc` come from `preferred_noc_for_dram_write/read(arch)`, which the recipe warns returns NOC_1 on Quasar too. | This is the recipe's "repeat offender" and will need arch-normalization to `[min..max]` at every site during the uplift — but only after the factories are Metal 2.0. |
| §11 — **degenerate-grid mcast corner clamp** | 2D factory uses `left_core_plus_one` / `top_core_plus_one` as mcast start (`2d.cpp:1177-1178, 1199-1208`); on a 1-wide/1-tall emulator grid this names a nonexistent core (`No core at (0,1)` class). | Uplift-time fix; irrelevant until the factory is Metal 2.0. |
| §7 — **bare `wait_front`→`pop_front` pairs (TEN-4746)** in compute | `compute/bmm_large_block_zm_fused_bias_activation.cpp:458-459` and `:468-469` — `mm_partials_dfb.wait_front(n); mm_partials_dfb.pop_front(n);` back-to-back in a loop, under `#ifdef PACKER_L1_ACC` (BARE shape). Other pairs have real UNPACR/PACR work between them (`:54/60` transpose, `:110/123` `copy_block`, `:140/163` `copy_tile`, `:328-329/482-483` `matmul_block`). | Uplift item (`dummy_unpack` post-#54948, or the `copy_tile` drain idiom); needs a full three-shape control-flow audit at uplift time, not a grep. |
| `cb_dfb_quasar_audit_helper.md` **GATE** — `get_local_cb_interface(...).<field>` write | `compute/bmm_large_block_zm_fused_bias_activation.cpp:115-116`: `UNPACK((get_local_cb_interface(mm_partials_reload_dfb_id).fifo_rd_ptr = get_local_cb_interface(mm_partials_dfb_id).fifo_rd_ptr));` under `#ifdef MM_PARTIALS_RELOAD_ALIAS_CB`. | Hard GATE for the CB→DFB kernel port (must become `evil_set_read_ptr` on WH/BH; Class 5 partials RMW → Quasar redesign debt). A Metal 2.0 pre-port concern, not an uplift edit. |
| §7 — **`partials_cb_uses_output` / borrow-with-offset** | DRAM-sharded factory aliases `c_5` (intermediate) onto the `c_4` (output) CB via two `format_descriptors` on one `CBDescriptor` when formats match (`dram_sharded.cpp:581-619`). This is a same-base *aliased CB* (Metal 2.0 `alias_with`), not an `address_offset` borrow — so the §7 offset-clobber rule does not obviously apply. `c_2` (in0) and `c_6` (resharded out) are tensor-backed (`cb_desc.tensor = &in0_tensor` / `&out_tensor`) → `borrowed_from` at port time. | Port-time classification; confirm in the Metal 2.0 audit. |
| §4 — `unpack_modes` / `hw_config` / `opt_level` | Not evaluated: there is no `KernelSpec` yet. `gen2_hardware_configs.md` is a post-port pass on a Metal 2.0 factory. | Not applicable pre-port. |
| §7 — **Quasar formats** (no uint16/uint32; Gen2 "replaces BFP with MXFP" per `gen2_hardware_configs.md`) | Every linear case uses `BFLOAT8_B` or `BFLOAT4_B` weights (and two cases `BFLOAT8_B` output). | Model-level dtype decision, not an op edit; flag to the llama32_1b_quasar owners — the captured dtypes may need a Quasar variant regardless of the op's state. |
| §5 — `fifo_page_size` via `get_local_cb_interface` | No `fifo_page_size` / `fifo_num_pages` reads found in the on-path kernels (only the `fifo_rd_ptr` write above). | Nothing to do. |
| §6 — DM self-loop / sync-free DFBs | Not classifiable until the Metal 2.0 port declares `dfb_bindings`. | Post-port style/semantic passes. |

---

## Deferred / blocked items (with exact symptom)

1. **BLOCKER — Metal 2.0 port of the two on-path factories has not happened.**
   Symptom: `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory::create_descriptor` and
   `MatmulMultiCoreReuseMcast2DProgramFactory::create_descriptor` return `tt::tt_metal::ProgramDescriptor`;
   no `create_program_artifacts` / `ProgramArtifacts` in either. Route: run `metal2_audit.md` on
   `ttnn/cpp/ttnn/operations/matmul` (per-factory), then `metal2_port.md`, one factory at a time. Note for
   that audit: the 2D factory has `override_runtime_arguments` → target is
   `CustomProgramSpecFactoryConcept`; the 2D in1 sender kernel carries an `ENABLE_GLOBAL_CB` /
   `api/remote_circular_buffer.h` path (Appendix A GlobalCircularBuffer, UNSUPPORTED) that the audit must
   scope (it is compiled out unless the define is set, but the kernel is shared with the 1D GCB factory);
   and the compute kernel is bound by six factories (`ReuseOptimized`, `BatchedHSDRAMSharded`,
   `DRAMSharded`, `Mcast2D`, `Mcast1D`, `sparse_matmul_1d`) — a **lent/intra-op shared kernel**, so the
   port will take the `_metal2` fork rung, not convert in place.
2. **Op-owner change — non-zero-init semaphore** (`dram_sharded.cpp:657-658`, `VALID`). Separate PR.
3. **Kernel GATE — `get_local_cb_interface(...).fifo_rd_ptr` write** in the shared compute kernel
   (`:115-116`). Must be resolved (→ `evil_set_read_ptr`) in the Metal 2.0 kernel port; Class 5 debt for
   Quasar.
4. **Test-harness geometry (not an op item, but it will look like one):** every linear case carries an L1
   shard grid of 32 or 64 cores (`[0,0,7,3]`, `[0,0,7,7]`) and a 12-bank DRAM shard grid (`[0,0,11,0]`); the
   two 2D cases have a 1024-row activation. `graph_ops/conftest.py` tags **none** of them `emulator`
   (`_EMU_MAX_CORES = 8`, `_EMU_MAX_ROWS = 128`), and `graph_case.build_memory_config` will `pytest.skip`
   any case whose captured grid exceeds the device (`graph_case.py:189-203`). On a 2-node Quasar emulator
   `test_linear.py` is therefore expected to **skip all 8 cases** even after a successful uplift. A
   Quasar-sized linear case (or a hand-written `tests/ops/test_linear.py`-style test) is needed to exercise
   the op there.
5. **Model-level dtype:** BFLOAT8_B / BFLOAT4_B weights in every case — confirm Quasar format support before
   expecting PCC.

---

## Parity claim (WH/BH)

**Trivially preserved: the diff to `ttnn/cpp/ttnn/operations/matmul/` is empty** (only this uncommitted
report was added). WH and BH take exactly the code path they took before this session. Per §9 ("Auditing
without a device run? … argue it structurally"): a zero diff ⇒ no behaviour change. The control commands
below let the user confirm on hardware.

---

## Commands for the user (recipe §9: user runs all builds/tests; order BH → WH → Quasar)

Run from the repo root with the venv active. `TT_METAL_FORCE_JIT_COMPILE=1` is unnecessary here (no
kernel changed) but harmless.

### Blackhole (control — must be unchanged)
```bash
# DRAM-sharded in1 path (cases 00-04, 07 shape family)
pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k "in1_dram_sharded" -v
pytest tests/ttnn/unit_tests/operations/matmul/test_linear.py -k "dram_sharded_in1" -v
pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py -v
# 2D mcast path (cases 05, 06)
pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k "2d" -v
pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_1d_2d.py -k "2d" -v
# ttnn.linear surface
pytest tests/ttnn/unit_tests/operations/matmul/test_linear.py -v
```

### Wormhole (control — must be unchanged)
```bash
pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py -k "in1_dram_sharded or 2d" -v
pytest tests/ttnn/unit_tests/operations/matmul/test_linear.py -v
pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py -v
pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_1d_2d.py -k "2d_wh" -v
```
(`test_matmul.py:484` has an `is_blackhole()` branch in the tiny-tile DRAM-sharded test; expect the BH and
WH parametrizations to differ, not to fail.)

### Quasar emulator (expected outcome today: not runnable via this op — see Status)
```bash
# The model-level test named for this op. With the op still on ProgramDescriptor, expect the Metal 2.0
# path to be absent; and per Deferred item 4 the harness will skip cases whose grids exceed the emulator.
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_linear.py -v
# Emulator-tagged subset (currently selects zero linear cases; included so the skip is visible):
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_linear.py -m emulator -v
# Shape/dtype/finiteness only, once the op is uplifted:
TTNN_GRAPH_OPS_NO_GOLDEN=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_linear.py -v
```
Run the Quasar command both with `TT_METAL_LLK_ASSERTS` set and unset (§9); DPRINT requires it unset.

---

## Definition-of-done checklist (§10) — state at stop

- [ ] Op uplifted in place — **not started (RED at step 1)**; nothing copied into `experimental/quasar/`.
- [ ] Factory is `create_program_artifacts` — **no** (both on-path factories are `create_descriptor`).
- [ ] Kernels use `dfb::`/`args::`/`tensor::` — **no** (legacy device API throughout).
- [ ] opt_level / data_format_metadata / sync-free / implicit-sync / borrow-offset / mcast-normalization /
      re-init / bare-pair / semaphore items — **deferred to post-Metal-2.0-port**, with heads-ups above.
- [x] BH and WH unchanged — **zero diff**.
- [ ] Quasar builds and runs — **not attempted**.
- [x] No DIAG/debug leftovers — none added.
- [x] Missing-prereq work flagged for separate PRs — items 1–3 above.
- [x] `QUASAR_UPLIFT_REPORT.md` written with RED status, changed-file list (none), parity claim.
