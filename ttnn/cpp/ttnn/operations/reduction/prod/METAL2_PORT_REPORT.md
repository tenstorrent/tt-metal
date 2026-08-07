# Metal 2.0 Port Report — `reduction/prod`

*Opened at the start of the port; entries captured as they occurred, polished at the end.*

## Outcome

`PORTED` — both factories (`ProdAllProgramFactory`, `ProdNcProgramFactory`) converted to
`MetalV2FactoryConcept`; confirmed against the baseline test set (see Verification below).

## Provenance

- **Recipe docs (this port):** `37f03926088 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` — recorded at audit time. After the branch was rebased onto `main`, `git log` over `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` returns nothing (the recipe docs are not tracked in main's history — they live on a separate doc branch), so the version can't be re-pinned from this checkout.
- **Audit docs (inherited):** `37f03926088 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Verification

Baseline captured pre-port and re-run after the port **and again after rebasing the branch onto `main`
tip**; results **identical** across all three (no behavior change):

| Test | Pre-port | Post-port (on `main`) |
|---|---|---|
| `test_prod_all.py` | 18 passed | 18 passed |
| `test_prod_nc.py` | 32 passed | 32 passed |
| `test_reduction.py::test_prod` | 56 passed, 24 skipped | 56 passed, 24 skipped |

Device: Blackhole (`-mcpu=tt-bh` JIT). Build: `./build_metal.sh --build-tests` — compile + link **clean**
(reduction unity TUs + `unit_tests_ttnn` rebuilt/relinked, no compiler errors). The only build failure is
an **unrelated tt-train `_ttml.so` install-time RPATH rewrite** (a uv-Python path mismatch in the
environment: `/home/.../.local/share/uv/...` vs `/usr/local/share/uv/...`); it does not touch prod and does
not affect the tests. Confirmed the baseline set with the invoker before relying on it (the two direct
device-op tests + `reduce/test_reduction.py::test_prod`). Kernels JIT-compile at test runtime (the host
build does not compile them), so the green test run is the real kernel-side validation — including the
`-O3` compute build and the new eltwise LLK init API (below).

## Rebase onto `main` — upstream integrations

The branch was rebased forward onto `main` tip. Two upstream changes overlapped prod's files and were
blended (details in-line where they landed):

- **DFB migration `#49173`** ("[Cleanup] Migrate MM/Fused/Reduce Kernels from CircularBuffer to
  DataflowBuffer") — a mechanical `CircularBuffer`→`DataflowBuffer` rename of prod's compute/reader
  kernels. Superseded by this port's fuller named-binding rewrite; the blend kept the port's `dfb::`
  tokens and adopted upstream's `dfb_*` object-variable naming for consistency.
- **Eltwise LLK init cleanup `#50745`** ("Eltwise binary + broadcast init cleanup") — deprecated the
  legacy inits. Adopted the new API in both compute kernels: `binary_op_init_common` →
  `compute_kernel_hw_startup`, and `binary_dest_reuse_tiles_init/tiles<ELWMUL, …>` →
  `mul_reuse_dest_init/tiles<…>`. Required (the deprecated forms are `-Werror` on `main`).
- **Shared writer fork already upstream** — resolved to *reuse*, not duplicate (see Open items → Cross-op).

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for both factories — each `create_descriptor` replaced by a static
`create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`. No change
to the surrounding device-op class (`validate_*`, `compute_output_specs`, `create_output_tensors`).

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (neither op had one).
- Pybind entry points removed: **none** (`prod_nanobind.cpp` never bound a factory entry point).

### Open items
- **Relaxation candidates:** none applied. The `TensorParameter`s use strict `TensorSpec` matching
  (default). The op's kernels iterate tile-by-tile, so a `dynamic_tensor_shape` relaxation *might*
  be tolerable, but the legacy factory declared no `ArgConfig::Runtime*` shape flag (grep clean), so
  there is nothing to mirror — left strict per the "don't self-decide a relaxation" rule.
- No op-owned tensors, no `GlobalSemaphore`, no multi-program need — the base concept fits cleanly.

## Handoff points

- **Port capitulation:** none — both factories ported.
- **Boundary-rule assumption violations (`sem::`/`tensor::` crossing out-of-op):** none.
- **Kernel-lib / framework gaps:** none hit.
- **Removed pybind surface:** none — `prod_nanobind.cpp` never bound a factory entry point.

## Successes

- **[Shared-dataflow-kernel fork Caution](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)**
  worked as designed across *independent* ports. The two `eltwise/unary` donor kernels are broadly-shared
  (~29 / ~12 co-borrowers), so the recipe's fork-with-`_metal2`-suffix path applies. When the branch was
  rebased onto `main`, the **writer** fork had already been established upstream by sibling Metal 2.0 ports
  (added to main by Gelu Backwards `#51771`; also used by Typecast `#51397`), producing an add/add conflict
  that resolved to **reusing the canonical upstream fork** — prod made no change to the file and conformed
  its own binding to the fork's frozen `tensor::dst` interface. The **reader** fork had no upstream twin, so
  prod contributes it with the matching `tensor::src` convention. This is the intended convergence: sibling
  ports land on one shared fork instead of divergent copies. `eltwise/unary`'s `GLOB_RECURSE
  device/kernels/*.cpp` auto-picks up the reader fork — no CMake edit needed.
- **[Hardware-config discipline](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#hardware-configuration)**
  caught three silent-regression traps: `dst_full_sync_en=false` → `double_buffer_dest=true` (the
  inversion); the FP32-only required `unpack_modes` entry; and the compiler optimization level —
  `KernelSpec::CompilerOptions` defaults compute to `-O2`, but the legacy `ComputeConfig` defaults compute
  to `-O3`, so both compute KernelSpecs set `opt_level = O3` explicitly to match `main` (DM kernels stay O2,
  matching the legacy `DataMovementConfig` default). All three would have compiled and passed tests while
  silently shifting perf/precision. The before/after value diff (Style B, direct `ComputeGen1Config`)
  surfaced them.
- **Named-arg model naturally shed the four dead legacy args** (audit "Misc anomalies") with no
  cleanup action — binding exactly the kernel's reads leaves them out by construction.
- **Unity-build hygiene applied proactively.** prod_all/prod_nc factories declare the same
  `DFBSpecName`/`TensorParamName`/`KernelSpecName` constants; since `reduction` is a unity-build target,
  anon-namespace duplicates risk a redefinition error if the two `.cpp` land in one unity group. The
  constants are declared **function-local** instead — per the [Unity-build hygiene pattern](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols).

## Friction

- **Gap — the recommended reference port is API-stale.** The accumulation reference on
  `akertesz/porting-experiment-accumulation-jun10` (the recipe's "first worked end-to-end" shape
  reference) predates the current headers in several load-bearing ways: it uses
  `create_program_spec` (not `create_program_artifacts`), `#include "ttnn/metal2_artifacts.hpp"`
  (now `metal_v2_artifacts.hpp`), `DataMovementHardwareConfig{.role = RoleHint::READER}` (the
  `.role`/`RoleHint` field no longer exists — the current path is the `ttnn::create_reader/writer_datamovement_config(arch)`
  helpers), and the old flat `ComputeHardwareConfig{.math_fidelity, .fp32_dest_acc_en, .dst_full_sync_en,
  .math_approx_mode, .unpack_to_dest_mode}` struct (now `ComputeGen1Config{.fpu_math_fidelity,
  .sfpu_precision_mode, .enable_32_bit_dest, .double_buffer_dest, .unpack_modes}` with the
  bool→`Precision` and `dst_full_sync_en`→`double_buffer_dest` transforms). Its `ProgramRunArgs`
  RTA shape is also node-first-list, not the current name-first `Table`. The recipe already warns
  "don't lean on already-ported ops as templates"; this port confirms that warning is load-bearing
  even for the *designated* reference — I followed the current headers + migration guide over the
  reference wherever they disagreed. **Suggestion:** either refresh the accumulation reference to
  the current API, or have the recipe name a more recently-landed reference.
- **Confusion (minor) — `ProgramRunArgs` "must specify KernelRunArgs for ALL kernels" vs the recipe's
  "omit the entry if no RTAs."** `program_run_args.hpp` says a `KernelRunArgs` must exist for every
  kernel; the recipe says the entry "may be omitted entirely" when a kernel has no RTAs (prod_all
  compute). Resolved by including a values-less `KernelRunArgs{.kernel = COMPUTE}` — satisfies both
  readings (an entry exists; its empty schema means nothing to set). Tests pass, so an empty entry
  is accepted. Worth a one-line reconciliation in the recipe.
- **Confusion (minor) — DFB page-size getter units.** The CB→DFB whitelist maps
  `get_local_cb_interface(...).fifo_page_size` → §B `get_entry_size()`, and separately notes "TRISC
  size getters return bytes." For the DM writer/reader forks, `get_entry_size()` is the right
  bytes-valued replacement for the legacy `fifo_page_size` fed to `noc.async_write/read`; tests
  confirm. A one-line "DM `get_entry_size()` == legacy `fifo_page_size` (bytes)" note would remove
  the moment of doubt.

## Open items for downstream

### Cross-op kernel touches (shared `eltwise/unary` donor forks)
prod binds Metal 2.0 `_metal2` forks of two broadly-shared donor kernels. After the rebase onto `main`,
the writer fork already existed upstream, so prod converges onto the shared file rather than owning a
private copy:

- **`writer_unary_interleaved_start_id_metal2.cpp` — REUSED, not created/modified by prod.**
  - Established upstream (added to `main` by Gelu Backwards `#51771`; also bound by Typecast `#51397`).
  - Consumed by: prod_all + prod_nc writer KernelSpecs, which conform to the fork's frozen interface —
    `dfb::out`, `tensor::dst`, RTAs `num_pages`/`start_id`. prod's diff does **not** include this file.
  - Legacy copy `writer_unary_interleaved_start_id.cpp` retires when its remaining ~29-family co-borrower
    set finishes migrating (not prod's to sunset).
- **`reader_unary_interleaved_start_id_metal2.cpp` — CONTRIBUTED by prod (first Metal 2.0 consumer).**
  - No upstream twin existed; prod_all establishes the fork. Canonical interface follows the same
    kernel-vocabulary convention as the writer fork — `dfb::in`, `tensor::src`, RTAs `num_pages`/`start_id`
    (documented in the file's header so later co-borrowers inherit it).
  - Consumed by: prod_all reader KernelSpec. Remaining legacy-copy consumers: ~12 op families; sunset the
    legacy reader when the last migrates.

> Note: prod_nc's reader is its *own* kernel (`reader_prod_nc.cpp`, in the op directory), not the shared
> fork — it keeps `tensor::input`/`dfb::in` internal to prod and is unaffected by the shared-fork convention.

### Pre-existing anomalies left for the ops team (NOT acted on)
- prod_nc dead reader RTA `dim` (`prod_nc_program_factory.cpp` legacy RTA 7), dead writer RTA
  `is_dram` (legacy RTA 3), dead compute CTA `num_cols_per_core_group_*`; prod_all dead compute
  CTA `per_core_block_size` (CTA[1]). None are read by any kernel; they have no representation in
  the named-arg model, so the port neither carries nor "cleans up" the legacy factory lines —
  behavior is unchanged. The anomalies themselves remain for the ops team.
- Output CB at `c_3` (not the `c_16+` output convention). Cosmetic; invisible under DFB naming.
