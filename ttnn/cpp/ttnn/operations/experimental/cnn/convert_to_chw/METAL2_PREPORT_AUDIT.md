# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw`

- **`ConvertToCHWDeviceOperation`** — `device/convert_to_chw_device_operation.hpp:17-26` (`program_factory_t = std::variant<ConvertToCHWProgramFactory>` at `:22`)
  - `ConvertToCHWProgramFactory` — `device/convert_to_chw_program_factory.cpp` (declared in `device/convert_to_chw_program_factory.hpp:17-27`). It has a `create_descriptor` (`.cpp:42-148`) and an `override_runtime_arguments` (`.cpp:150-167`).
- **Kernels bound by the factory.** All three live in this op's `device/kernels/` directory, only this factory binds them, and all three were audited in full:
  - reader: `reader_convert_to_chw.cpp` (`ReaderConfigDescriptor`, factory `.cpp:112-118`)
  - writer: `writer_convert_to_chw.cpp` (`WriterConfigDescriptor`, `.cpp:120-126`)
  - compute: `convert_to_chw.cpp` (`.cpp:128-139`)
  - The op directory has no unreferenced kernel files.
- **Code outside the op directory that this op exercises:**
  - the pool-family kernel header `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp`, included by the reader and the writer;
  - the family-level host helper `ttnn/cpp/ttnn/operations/experimental/cnn/cnn_cb_descriptor.hpp` (`make_cnn_circular_buffer`), which is shared with `convert_to_hwc`.
- **Not bundled:** `experimental/cnn/convert_to_hwc` is a sibling op directory with its own DeviceOperation and its own kernels. It shares only the host helper above, so it gets its own audit.
- **What the op does:** it turns a height-sharded, tiled `[1, 1, HW, C]` tensor (`C ≤ 32`) into a width-sharded, row-major `[1, 1, C, HW]` tensor. The compute kernel transposes and pack-untilizes each input tile into an intermediate CB. The writer then scatters 64-byte sticks into the output shard using NoC self-reads.

Shorthand used below:
- `reader:` / `writer:` / `compute:` are the three kernel files.
- `.cpp:` / `.hpp:` are `device/convert_to_chw_program_factory.{cpp,hpp}`.
- Every other file is named in full.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** ``c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form``

**Code audited:** worktree branch `iwrosz/convert_to_chw_2.0` at `4c3e12db692`, which is recipe branch `c41f7f4ef0a` plus `origin/main` `d70d725e238` (2026-10-30). This was a static read only: nothing was built, submodules are not initialized, and nothing was run on hardware.

**Readiness sheet:** the *"Operations analysis"* sheet (file ID `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`), re-fetched **2026-10-30 12:04 UTC** by the launching session following `ttnn_op_porting_readiness.md` (`download_file_content`, `text/csv`, base64-decoded). This op has one row, at CSV line 171. Columns are referenced by header name, and the header row (28 columns) is the same in both fetches.
- **History:** the first pass of this audit used the 11:39 UTC fetch, where this row still read `Concept` = `legacy device-op`, so that pass reported RED (spreadsheet broken).
- **Correction:** Diego then corrected the row, and the GREEN verdict below rests on the 12:04 UTC fetch.
- **Code:** not re-audited, because HEAD (`4c3e12db692`) and `origin/main` (`d70d725e238`) are unchanged since the first pass.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw` |
| **Overall** | **GREEN**: every gate clears, and the brief is issued (`METAL2_PORT_BRIEF.md`). |
| **DOps / Factories** | `ConvertToCHWDeviceOperation` → `ConvertToCHWProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** for the reader, writer, compute kernel and the donor helpers they call. One judgment call is recorded: a hand-built loopback endpoint inside the pool helper (see Gate detail). |
| *Prereqs* — Cross-op escapes | Ok. There is one cross-family header (pool `experimental_device_api.hpp`), no resource handle crosses it, and there are no borrowed kernel files. |
| *Feature Support* — overall | GREEN: every Appendix A entry is N/A. |
| *Feature Support* — Variadic-CTA | Ok: no CTA varargs. This row has no Appendix A entry (see Recipe notes). |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes**: the sheet reads `yes`, and the primary cross-check is clean. The 11:39 UTC fetch had a stale row, which Diego has since corrected (see Gate detail). |
| *TTNN Readiness* — Concept (current) | `descriptor` (the sheet and the code agree) |
| *TTNN Readiness* — Execution Model | `SPMD` on the sheet, which matches the code: one `ProgramDescriptor` and nothing that varies per coordinate. `Secretly SPMD Workload?` is N/A because this is not a mesh-shaped concept. |
| *TTNN Readiness* — Custom hash | No. The sheet says `no` for both the hash and the backdoor, and the code confirms it. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No. The sheet says `no`, and the code confirms it. |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; the porter translates it): `.cpp:150-167`. The sheet says `yes`. |
| *TTNN Readiness* — Pybind `create_descriptor` | No. The sheet says `no`, and `convert_to_chw_nanobind.cpp:25-26` binds only the free function. |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `CustomProgramSpecFactoryConcept` (the sheet's `Porting Target`). Translate the `override_runtime_arguments` at `.cpp:150-167`. |
| *Port work* — Offset base pointer | None. No address is passed as an RTA, CRTA or CTA anywhere. |
| *Port work* — Tensor bindings (per binding) | `input` clean (borrowed DFB) · `output` clean (borrowed DFB) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | None. No accessor passes a 3rd arg; the op has no `TensorAccessor` at all. |
| *Port work* — CB endpoints | `c_0` legal 1:1 · `c_2` legal 1:1 · `c_1` **self-loop** (writer, DM) |

**Subject coverage: all twelve subjects were run.**
- In the first pass (11:39 UTC sheet), the seven purely informational subjects (TTNN porting shape, TensorParameter relaxations, TensorParameter analysis, CB endpoints, Out-of-directory coupling, RTA varargs, Incidental anomalies) ran under the recipe's *"RED that clears without touching the op's code"* exception. I judged that blocker to be **elsewhere**, because it was a fix to the readiness-sheet data.
- After the sheet correction (12:04 UTC), I re-ran only the TTNN factory concept prerequisite and TTNN porting shape against the new row. The code is unchanged, so every other subject's findings stand as written and feed the brief.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`).

- **Every gate clears:** Device 2.0 ✓ · Feature compatibility ✓ (no Appendix A feature) · TTNN factory concept ✓ (`Is able to port?` = `yes` on the 12:04 UTC fetch, cross-check clean) · Offset base pointers ✓ · TensorAccessor 3rd arg N/A (the op has no `TensorAccessor`).
- **Target concept:** `CustomProgramSpecFactoryConcept`. The porter translates the descriptor-style `override_runtime_arguments` at `.cpp:150-167`.
- **One open caveat:** the Device 2.0 gate cleared on a judgment call that is still open with the Device 2.0 team (the pool helper's hand-built loopback endpoint; Questions 2). If they overturn it before the port starts, the gate flips to RED and the brief no longer applies.
- **History:** the first pass of this audit (11:39 UTC sheet) was RED, solely because the readiness sheet still listed the op as `legacy device-op`. Diego corrected the row, and the code did not change.

## Gate detail

### TTNN factory concept (`Is able to port?`) — GREEN

On the 12:04 UTC fetch (CSV line 171), `Is able to port?` reads exactly `yes`, which clears this prerequisite. The cross-check of the row against the code is clean:

| Column (by header) | Sheet value | Code evidence | Check |
|---|---|---|---|
| `Concept` | `descriptor` | `ConvertToCHWProgramFactory::create_descriptor(...)` returns `tt::tt_metal::ProgramDescriptor` (`.hpp:18-19`, `.cpp:42-148`). There is no `cached_program_t` or `create()`, so it cannot satisfy `ProgramFactoryConcept` (`ttnn/api/ttnn/operation_concepts.hpp:26`); it satisfies `ProgramDescriptorFactoryConcept` (`:73-74`). It is `descriptor` rather than `descriptor (per-coord)` because `create_descriptor` takes no `MeshCoordinate`. | ✓ |
| `Override runtime args method? (PD only)` | `yes` | A descriptor-style override is present: `override_runtime_arguments(Program&, …, const std::optional<MeshCoordinate>&)` (`.hpp:21-26`, `.cpp:150-167`). It sits on a `descriptor` factory, so this is the target-concept signal, not the legacy signature. | ✓ |
| `Custom hash (compute_program_hash)` | `no` | `ConvertToCHWDeviceOperation` has no `compute_program_hash` (`convert_to_chw_device_operation.hpp:17-26`). | ✓ |
| `Backdoor custom hash (attribute_values / to_hash)` | `no` | `ConvertToCHWParams` has neither (`convert_to_chw_device_operation_types.hpp:11-14`). | ✓ |
| `Runtime-args update (get_dynamic_runtime_args)` | `no` | The device-op has no such hook. | ✓ |
| `Pybind descriptor (nb::class_ of device op)` | `no` | `convert_to_chw_nanobind.cpp:25-26` binds only `ttnn::experimental::convert_to_chw`. | ✓ |
| `Op-owned tensors?` | `no` | The concept is `descriptor`, and there is no `WorkloadDescriptor`. | ✓ |
| `Secretly SPMD Workload?` / `Why secretly SPMD?` | blank | Not a mesh-shaped concept, so N/A. | ✓ |
| Factory-set match | one row: `ConvertToCHWProgramFactory` | `program_factory_t = std::variant<ConvertToCHWProgramFactory>` (`convert_to_chw_device_operation.hpp:22`) | ✓ (no phantom or missing row) |
| Cross-column invariants | — | None violated: `get_dynamic_runtime_args` is `no`, and `Op-owned tensors?` is `no` on a non-`WorkloadDescriptor` row. | ✓ |

**Derived cells (read, not vetted):**
- `Op Classification` = `PD Op (custom)`
- `Is able to port?` = `yes`
- `Execution Model` = `SPMD`
- `Porting Target` = `CustomProgramSpecFactoryConcept`

**Blocking signals:** none. `get_dynamic_runtime_args` is `no`, there is no smuggled pointer, `Known op issues` is blank, the relaxation is `none`, and the concept is not legacy.

**Other cells (unchanged from the first fetch):**
- `Known op issues` blank
- `TensorParameter relaxation` = `none`
- `Smuggled pointer` = `no` (consistent with the code: no address args)
- `Diego validation` = `yes`
- `Model` = `other`
- `ProgramFactory used in llama?` = `no`

**History (first pass, 11:39 UTC fetch):**
- **What the row said then:** `Concept` = `legacy device-op`, `Override runtime args method?` = `n/a`, `Op Classification` = `Legacy Op`, `Porting Target` = `ProgramSpecFactoryConcept` and `Is able to port?` = `yes (with PD step)`.
- **Why that was RED:** both primary cells conflicted with the code, because the factory has been on `ProgramDescriptor` since PR #57513 (`3a09d019edd`, merged 2026-10-02). The gate was therefore RED as spreadsheet-broken and routed to the readiness-sheet owner, not to the legacy-API short-circuit (see Recipe notes).
- **How it was resolved:** Diego corrected the row. The sibling `convert_to_hwc` row, which the same PR migrated, was updated as well; it is out of scope here.

### Device 2.0 (every kernel used) — GREEN

| Kernel | Device 2.0 idioms in use | Free functions keyed on a CB index |
|---|---|---|
| reader (`reader_convert_to_chw.cpp`) | `experimental::CB` (which is `CircularBuffer`) `push_back` (`reader:11-12`) | none |
| writer (`writer_convert_to_chw.cpp`) | `Noc` (`writer:23`)<br>`experimental::CB` FIFO calls and `get_read_ptr` / `get_write_ptr` peeks (`writer:24-29, 35-36, 48, 52-53, 61, 64`)<br>`noc.async_read_barrier()` (`writer:47, 60, 63`)<br>NoC self-reads through pool helpers that wrap `Noc::set_async_read_state` and `Noc::async_read_with_state` (`writer:29, 41, 56`) | `get_tile_size(cb_in_transpose)` (`writer:19`) is **sanctioned**, so it is not flagged |
| compute (`convert_to_chw.cpp`) | `CircularBuffer` FIFO calls (`compute:13-15, 22, 24, 29`) | LLK compute APIs that take CB ids (`transpose_tile`, `pack_untilize_*`, `compute_kernel_hw_startup`, `transpose_init`). These are compute APIs with no wrapper-method replacement, not Device 2.0 holdovers. |
| donor `pool/device/kernels/experimental_device_api.hpp` (only the functions this op calls) | `set_read_state` (`:43-49`) calls `Noc::set_async_read_state`.<br>The raw `read_with_state` overload (`:54-59`) calls `Noc::async_read_with_state` with a `CoreLocalMem`.<br>`using CB = CircularBuffer` (`:24`). | none |

There are no raw `noc_async_*` calls, no address generators, no raw semaphore addresses, and no `get_read_ptr(cb_id)`-style holdovers.

**Judgment call (not gated; flagged for the Device 2.0 team in Questions).** Both pool helpers build their NoC-side source endpoint with `experimental::local_addr()` (`experimental_device_api.hpp:37-39`). That function creates a `UnicastEndpoint` from `my_x[noc]` / `my_y[noc]` plus a raw L1 address. Since #61229 (`f388f4c73ce`, 2026-10-20), `device_api_migration_guide.md:524-552` labels exactly this shape **"Avoid** (Device 2.0 API, but hand-built loopback endpoint)". I did not mark it RED, for three reasons:

1. The guide itself calls the shape Device 2.0 API. The recipe's two RED shapes are a CB-index holdover that has a wrapper-method replacement, and broad Device 1.0 usage. This is neither.
2. The guide's prescribed replacement puts a local handle on the NoC side. Here the NoC-side source is a CB entry (`cb_transpose.get_read_ptr()` plus offsets). The guide says a `CircularBuffer` or `DataflowBuffer` "can be the local side of such a copy, but not the NoC side" (`:532`). #61229's description also explicitly defers "DFB-to-DFB copies … to upcoming PRs", and this copy is CB-to-CB (`c_2` → `c_1`).
3. Kernels on main that are already ported to Metal 2.0 use the same helpers on raw addresses taken from a DFB:
   - `data_movement/fold/device/kernels/dataflow/writer_dfb2s_row_major.cpp:44, 60`
   - `experimental/padded_slice/device/kernels/dataflow/writer_unary_sharded_padded_tiled.cpp:153, 163, 178`
   - `experimental/padded_slice/device/kernels/dataflow/writer_unary_sharded_padded_rm.cpp:60`

If the Device 2.0 team rules otherwise, this gate flips to RED. That would be an isolated fix in the donor pool header, which 45 files outside the quasar tree include, not a fix in this op.

### Feature compatibility — GREEN (no gate fired)

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No GCB type, include, `remote_index` or `remote_cb` appears in the op or the donor header. The CB descriptors come from `make_cnn_circular_buffer` (`cnn_cb_descriptor.hpp:31-40`), which never sets `.global_circular_buffer` (default `nullptr`, `tt_metal/api/tt-metalium/program_descriptors.hpp:86`). |
| CBDescriptor `address_offset` (non-zero) | N/A | Never set, so it keeps its default of `0` (`program_descriptors.hpp:85`). The override's descriptors (`.cpp:164-165`) set only `.buffer`. |
| GlobalSemaphore | N/A | The op uses no semaphores of any kind. |

### CB endpoints (GATE-free) — `c_0` legal · `c_2` legal · `c_1` self-loop

There is only one config: the factory sets no defines and has no branches, and the kernels have no `#ifdef`s. The census below is therefore the same for every config. All three kernels run on every node of the input shard grid.

| CB | Backing / sizing | Reader | Compute | Writer | Census / node | Disposition |
|---|---|---|---|---|---|---|
| `c_0` `cb_in` (`.cpp:82-86`) | **borrowed from input**; `total_tiles_per_core` × input tile; input dtype | `push_back(total_tiles)` (`reader:12`), so locked producer | `wait_front` / `pop_front` (`compute:15, 22`); LLK reads at `compute:19` and inits at `compute:39-41`, so locked consumer | — | 2 (1P + 1C) | Legal 1:1. Reader is PRODUCER, compute is CONSUMER. |
| `c_1` `cb_out` (`.cpp:88-94`) | **borrowed from output**; `C` entries × (shard width × element size); output dtype | — | — | `reserve_back(1)` (`writer:27`), `get_write_ptr()` (`writer:28`) plus NoC self-read writes (`writer:41, 56`), `push_back(1)` (`writer:64`), so locked producer | **1** | **Self-loop**: the writer binds as both PRODUCER and CONSUMER. This is a DM self-loop, which is legal on Gen1. |
| `c_2` `cb_in_transpose` (`.cpp:96-106`) | own L1; 16 × `Float16_b` tile | — | `reserve_back` / `pack_untilize_dest` / `push_back` (`compute:24, 26, 29`); inits at `compute:39-40, 43, 52`, so locked producer | `wait_front` / `pop_front` (`writer:35, 48, 52, 61`); `get_read_ptr()` peeks (`writer:29, 36, 53`); `get_tile_size` (`writer:19`), so locked consumer | 2 (1P + 1C) | Legal 1:1. Compute is PRODUCER, writer is CONSUMER. |

- **Dead CBs:** none. Every index has a toucher.
- **Multi-binding faces:** none found.
  - (a) No hidden second writer: the writer's only raw writes land in `c_1`, which it already produces through the FIFO.
  - (b) No CB has more than one reader.
  - (c) No dual-instance work split: the three kernel sources are all distinct.
- **Naming trap:** compute's helper `transpose<BATCH_SIZE>(cb_in, cb_out)` is called with `cb_transpose_in` (`compute:46, 50`). Its local `cb_out` is therefore **`c_2`**, not `c_1`. The compute kernel never touches `c_1`.

### Offset base pointers — GREEN

- **No address of any kind reaches a kernel through an argument.**
  - The only RTA is `total_tiles_per_core`, and it is the same for all three kernels (`.cpp:27-39, 142`).
  - The CTAs are CB ids plus `C` (`.cpp:108-110`).
- **Tensor memory reaches the kernels only through the two borrowed CBs, at the buffer base.** See `make_cnn_circular_buffer(…, input_buffer | output_buffer, …)` at `.cpp:85-86, 93-94`; no `address_offset` is set.
- **The dated triage doc `2026-07-19_offset_base_pointers.md` does not list this op.** The reconciliation outcome is therefore *no fold, not in the tables*, which means clean.

### TensorAccessor 3rd argument — N/A

- **None of the three kernels constructs a `TensorAccessor`, and neither do the donor functions they call.** This subject therefore never fires. Note that "no sites" is a different finding from "sites found and classified Class 2".
- **The dated triage doc `2026-07-06_tensor_accessor_3rd_arg_triage.md` does not list this op.**

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings:**
  - `input` is **clean**: it is the borrowed-memory DFB `c_0`, and compute reads it through `wait_front` plus LLK calls.
  - `output` is **clean**: it is the borrowed-memory DFB `c_1`, and the writer fills it through its write pointer.
  - Both are **borrow-only `TensorParameter`s**. Each is named by `DataflowBufferSpec::borrowed_from` and has no kernel `TensorBinding`, which is the validator's stated exception (`migration_guide.md:366`).
  - There are no address RTAs, CRTAs or CTAs, and no `Buffer*` RTA bindings.
- **TensorParameter relaxation:** none (the sheet says `none`).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:**
  - Self-loop `c_1` (`cb_out`) on the writer, in every config.
  - `c_0` and `c_2` are legal 1P+1C, with roles fixed by their FIFO calls.
  - No multi-binding flag, no dead-CB drop, and no conditional DFB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none of the three kernel sources is shared.
  - The filename census (`grep -rl <filename> ttnn/cpp/ttnn/operations/`) finds only this factory. The other hit, `sources.cmake:10`, is the same-named *host* file in a build list, so it is discarded.
  - No `_metal2` sibling exists. The shared-kernel caution does not apply, so convert all three kernels in place.
  - **Sunset list:** empty.
- **RTA varargs:** none.
  - Each kernel reads exactly one RTA at a fixed index: `total_tiles` (`reader:9`, `writer:13`, `compute:33`), which has the same value on every node.
  - All CTAs are read at fixed indices, and `C` (writer CTA 2) is the only CTA that is not a CB id.
- **Anything else:**
  - **Keep the pool helper.** Keep including `pool/device/kernels/experimental_device_api.hpp` and keep calling `experimental::set_read_state` / `experimental::read_with_state` on the raw `get_read_ptr()` / `get_write_ptr()` addresses. `cb_dfb_api_whitelist.md:138` keeps peeks for helpers that take raw L1. Ported precedent: `fold/.../writer_dfb2s_row_major.cpp:7-9, 37-44, 60` includes both `dataflow_buffer.h` *and* the pool header.
  - **`experimental::CB` is an alias.** It is `using CB = CircularBuffer` (`experimental_device_api.hpp:24`). The CB→DFB swap applies to `reader:11` and `writer:24-25`, even though the token `CircularBuffer` never appears in those files.
  - **Keep the `constexpr` tile size as a free function.** `writer:19` is `constexpr uint32_t in_transpose_tile_size = get_tile_size(cb_in_transpose);`. Because it is `constexpr`, it keeps the free-function form with the binding token (`get_tile_size(dfb::<name>)`) rather than the member getter (`cb_dfb_api_whitelist.md:93-107`). The token form is Gen1-only, so record this site in the port report.
  - **The compute helper takes raw CB ids.** `transpose<BATCH_SIZE>(uint32_t cb_in, uint32_t cb_out)` (`compute:11-30`) builds `CircularBuffer` wrappers from those ids for its FIFO calls, and passes the same ids to LLKs.
    - The LLK arguments can take `dfb::` tokens directly (`port_patterns.md` → "Pass DFB handles directly to LLKs").
    - The FIFO wrappers need real `DataflowBuffer` objects. Because they drive FIFO calls, this is not the temp-wrapper anti-pattern. Building them from a raw id inside the helper would use `DataflowBuffer`'s low-level constructor, which `dataflow_buffer.h:155-156` marks "prefer DFBBindingToken".
    - How the helper receives them is the porter's call.
  - **Swap the direct `circular_buffer.h` include.** The compute kernel includes `api/dataflow/circular_buffer.h` directly (`compute:9`). Swap it per the whitelist.
  - **The reader pushes without reserving.** `reader:12` calls `push_back(total_tiles)` on the borrowed `c_0` with no `reserve_back`. This is the usual signal that the data is already resident; keep it.
  - **The writer peeks before its first wait.** `writer:29` peeks `cb_transpose.get_read_ptr()` before its first `wait_front`, only to program the NoC read state. It is a peek on the writer's consumer binding; keep it.
  - **The override refreshes exactly {`c_0` ← input buffer, `c_1` ← output buffer} and no runtime args** (`.cpp:163-166`).
    - CBs are matched by position against the order `create_descriptor` pushes them (`tt_metal/impl/program/program_descriptors.cpp:231-243`).
    - In Metal 2.0, that set is exactly the two borrow-only `TensorParameter`s.
    - The target is `CustomProgramSpecFactoryConcept`, so translate the override rather than deleting it. It must return `ProgramRunArgs`, not `void` (`ttnn_factory.md:104`), with `tensor_args` for **both** `INPUT` and `OUTPUT` and no `kernel_run_args`. Leaving one out freezes that borrowed DFB's address (`ttnn_factory.md:118-133`).
    - The ported-from override ignores `mesh_dispatch_coordinate` (`.cpp:155`), so the translated one can ignore `coord`.
    - Cache-hit coverage already exists: `test_convert_to_chw.py::test_convert_to_chw_with_program_cache` (`:94-116`) runs eight iterations over two specs and two dtypes, with a dummy L1 allocation shifting addresses each time, and expects 4 cache entries. It exercises exactly this refresh.
    - The op already has a `program_factory_t`, so this is a method swap inside the existing struct; the direct-descriptor exception does not apply.
  - **Leave the shared host helper alone.** `cnn_cb_descriptor.hpp` (`make_cnn_circular_buffer`) is also called six times by `convert_to_hwc` (`convert_to_hwc/device/convert_to_hwc_program_factory.cpp:313-355`). Stop using it from this factory, but do not edit or delete it.
  - **Borrowed DFBs fit Metal 2.0's check.** `ValidateBorrowedMemory` (`tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp:34`) needs an L1, non-per-core-allocated backing tensor whose per-bank slice is at least `entry_size × num_entries`. For the op's documented L1 height-sharded input (`convert_to_chw_nanobind.cpp:19`), `c_0` is at most the input shard and `c_1` equals the output shard.
  - **Carry the kernel configs over.**
    - The reader uses `ReaderConfigDescriptor{}` and the writer uses `WriterConfigDescriptor{}` (`.cpp:118, 126`); keep their RISC/NoC choice.
    - Compute uses `HiFi4`, `fp32_dest_acc_en = false`, `dst_full_sync_en = false` and `math_approx_mode = false` (`.cpp:134-139`).
    - With `enable_32_bit_dest` left at false, the Float32 `unpack_modes` validator rule (`migration_guide.md:302`) does not trigger.
  - **No build-file change is needed.** Kernel files are picked up by the family glob in `cnn/CMakeLists.txt:16-20`.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.**
- **Function-call escapes:** one cross-family donor, `pool/device/kernels/experimental_device_api.hpp` (class 6, pool family), included by the reader and the writer.
  - No resource handle crosses this boundary. The functions called take `Noc` by value plus raw L1 `uint32_t` addresses, and `CB` is only a type alias the op uses for its own objects.
  - Kernels already ported to Metal 2.0 call the same functions unchanged.
- **File-path escapes:** none.
- **Host side:** the in-family helper `cnn_cb_descriptor.hpp`, shared with `convert_to_hwc`.

| Op kernel | Donor file | Class | Status |
|---|---|---|---|
| `reader_convert_to_chw.cpp` | `api/dataflow/dataflow_api.h` | 1 `tt_metal/*` | ✓ |
| `reader_convert_to_chw.cpp` | `pool/device/kernels/experimental_device_api.hpp` | 6 cross-family (pool) | ✓ (uses only the `CB` alias) |
| `writer_convert_to_chw.cpp` | `api/dataflow/dataflow_api.h` | 1 | ✓ |
| `writer_convert_to_chw.cpp` | `pool/device/kernels/experimental_device_api.hpp` | 6 | ✓ |
| `convert_to_chw.cpp` (compute) | `api/compute/pack_untilize.h`, `api/compute/transpose.h`, `api/dataflow/circular_buffer.h` | 1 | ✓ |

Per-call detail is omitted because everything rolls up ✓. For the record, these are the pool functions this op reaches:
- the `CB` alias (`:24`);
- `set_read_state<transfer_size>(Noc, uint32_t)` (`:43-49`);
- `read_with_state<transfer_size = 1>(Noc, uint32_t, uint32_t)` (`:54-59`).

Both functions call `local_addr` (`:37-39`), which is the Device 2.0 judgment call above.

This op does not reach `set_read_trid` (`:81`), `async_read_barrier_with_trid` (`:85-87`), or the typed-`Dst` overloads of `read_with_state` (`:62-75`). FYI for the Device 2.0 team: `set_read_trid` wraps the `dataflow_api.h` free function `noc_async_read_set_trid` (`dataflow_api.h:2518`) inside this "Device 2.0 convenience header". That does not affect this op, but it may matter for the header's other includers.

**Borrowed kernel files:** none. This op owns all three sources and is the only one that binds them. No `_metal2` fork exists, and none is needed.

### Relaxation candidates

None: there is no custom hash to mine.

### TTNN factory analysis

**Facts, with evidence:**
- **Concept:** `descriptor` (the sheet and the code agree).
- **Op-owned tensors:** none.
- **Execution model:** `SPMD`. The override ignores `mesh_dispatch_coordinate` (`.cpp:155`), and nothing varies per coordinate.
- **Pybind `create_descriptor`:** none, so the port removes no binding. There is no other risky pybind either: `bind_function` exposes only the public free function.
- **Custom hash:** none. The default reflection hash covers `ConvertToCHWParams{memory_config, dtype}` and the input tensor.
- **`get_dynamic_runtime_args`:** none.
- **`override_runtime_arguments`:** **yes** (`.hpp:21-26`, `.cpp:150-167`). It refreshes only the two borrowed CB addresses, using a CB-only descriptor plus `apply_descriptor_runtime_args`, and is tagged `// override-rebuild-ok: cb-addr-only` per `scripts/detect_override_rebuild.py`.

**Gate conjuncts:**
- relaxation `none`: clears;
- `get_dynamic_runtime_args` absent: clears;
- concept `descriptor`, not legacy: clears. (The 11:39 UTC fetch said legacy; it was corrected by 12:04 UTC.)

#### TTNN ProgramFactory  *(block per `ttnn_factory.md`)*

- **Concept:** `CustomProgramSpecFactoryConcept` (the sheet's `Porting Target`).
  - The override to translate is `override_runtime_arguments` at `.cpp:150-167` (declared `.hpp:21-26`). This agrees with `ttnn_factory.md`'s one-question rule (`:26`).
  - It refreshes exactly the two tensors behind the borrowed DFBs (`INPUT` → `c_0`, `OUTPUT` → `c_1`) and no runtime args.
  - The port is not blocked on any missing framework capability.
- **Fit:**
  - Execution model: `SPMD`.
  - Op-owned device resources: none.
  - Tensor-arg matching: strict (relaxation `none`).
  - Shape of the factory being ported from: 1:1, with one `create_descriptor` mapping to one `ProgramSpec`.
- **Custom `compute_program_hash`:** none; the default reflection-based hash applies.
- **Stop signals:** none. The overall audit result is GREEN.

## Misc anomalies  *(team-only, non-gating, for the ops team; both come from static reading and neither was reproduced)*

- **The writer assumes a 2-byte output element.**
  - **Where:** `writer:9-10` hard-codes `ELEMENT_SIZE_BYTES = 2` (so `STICK_SIZE = 64`). The writer copies raw sticks out of `c_2`, which is always `Float16_b` (`.cpp:79`).
  - **The mismatch:** the output CB is sized and paged from the output dtype the user selects (`.cpp:88-92`). That dtype defaults to the input's dtype (`convert_to_chw_device_operation.cpp:88`), and nothing restricts it (`:15-39`, `:62-92`).
  - **Consequence:** a 4-byte output dtype such as `float32` (which is also the default for a `float32` input) would get bf16 bit patterns written at 2-byte strides into a shard paged in 4-byte elements.
  - **In-tree callers:** every one passes `dtype=ttnn.bfloat16`:
    - `test_convert_to_chw.py:41, 85`
    - `models/experimental/functional_unet/tt/unet_shallow_ttnn.py:543`
    - `models/demos/stable_diffusion_xl_base/vae/tt/tt_decoder.py:214`
    - `models/demos/vision/segmentation/vanilla_unet/tt/model.py:184`
  - **Possible fix:** validate `dtype == BFLOAT16`, or derive the stick size from the output format.
- **The writer's channel stride assumes the minimal shard height.**
  - **Where:** the writer strides channels by `total_tiles × 64` bytes (`writer:31`), where `total_tiles` = `div_up(HW/32, num_cores)` (`.cpp:66-67`).
  - **The mismatch:** each output shard row is as long as the input shard height (`convert_to_chw_device_operation.cpp:76-79`; the page size is set at `.cpp:92`). The two agree only when the input shard height equals `div_up(HW/32, num_cores) × 32`.
  - **Why the checks don't catch it:** they still allow taller shards. One check requires the height to be divisible by 32 (`device_operation.cpp:32-34`); the other requires `div_up(HW, shard_h) == num_cores` (`.cpp:62-64`).
  - **Example:** `HW = 2048` on 2 cores with a shard height of 1280 passes both checks. The op then processes only 32 of core 0's 40 tiles, and writes channels at a 2048-byte stride into 2560-byte rows.
  - **Test coverage:** every tested config satisfies the equality (`test_convert_to_chw.py:12-25, 50-69`).
  - **Possible fix:** validate the equality, or derive the stride and tile count from the shard shape.

## Questions for the user

1. **Resolved: readiness-sheet reconciliation (Diego).**
   - On the 12:04 UTC re-fetch, row 171 reads `Concept` = `descriptor` and `Override runtime args method?` = `yes`.
   - The re-derived cells read `Op Classification` = `PD Op (custom)`, `Is able to port?` = `yes` and `Porting Target` = `CustomProgramSpecFactoryConcept`.
   - That target agrees with `ttnn_factory.md:26`, so the target-concept question is settled.
   - The sibling `convert_to_hwc` row was updated too; it is out of scope here.
2. **Device 2.0 team: is the pool helper's hand-built loopback endpoint compliant for CB-sourced local copies, until a DFB-to-DFB loopback exists?**
   - The construct is `experimental::local_addr()` building `UnicastEndpoint{my_x, my_y, raw addr}` (`pool/device/kernels/experimental_device_api.hpp:37-39`), as used by `set_read_state` and `read_with_state`.
   - I treated it as compliant (Device 2.0 GREEN), for the reasons in Gate detail.
   - If it is not compliant, this op's Device 2.0 gate flips to RED. The fix would be an isolated change in the donor pool header, which 45 files outside the quasar tree include.

## Recipe notes

1. **Two routing rules collide, and the recipe doesn't rank them.**
   - `Is able to port? == yes (with PD step)` sends the op to the legacy-API short-circuit, which skips the seven informational subjects.
   - A `Concept` primary-column conflict showing that the op is *already* on `ProgramDescriptor` sends it to "spreadsheet is broken". That is an elsewhere blocker, so the seven subjects run.
   - The Routing list doesn't say which rule wins. I applied spreadsheet-broken, because the short-circuit's stated reason (the code is still imperative and about to change under a PD migration) is false here: the migration landed four weeks before the sheet was fetched.
   - Suggested addition: *"if the code is already on `ProgramDescriptor` while the sheet says `legacy`, route as spreadsheet-broken; the short-circuit does not apply."*
   - A related gap: TTNN porting shape says to read `Porting Target` "once the TTNN factory concept prerequisite clears". It does not say what to record when the subject runs under the elsewhere exception and that cell is derived from the conflicting columns. I recorded the cell verbatim, marked it *pending reconciliation*, and gave the code fact.
2. **The Device 2.0 gate doesn't say whether the guide's "Avoid" forms gate.**
   - The gate defines RED as a CB-index holdover with a wrapper replacement, or broad Device 1.0. It also says compliance is whatever the migration guide entails.
   - Since #61229 (2026-10-20), the guide carries items marked "Avoid (Device 2.0 API, but …)", starting with hand-built loopback endpoints.
   - Here that forced a judgment call (see Gate detail), and the question will recur. The pool header that produces the shape is included by 45 files outside the quasar tree, including kernels already ported to Metal 2.0. Every future audit of a conv, pool or data-movement op that copies through it will hit the same question.
   - Suggestion: state the rule. For example, sanction CB/DFB-sourced loopbacks until the deferred DFB-to-DFB copy support lands, using the same *"unseat only when the Device 2.0 surface changes"* logic as `get_tile_size`.
3. **The Case-2 bridge the recipe names no longer exists.**
   - TensorParameter analysis names `TensorAccessor::get_bank_base_address` as the sanctioned Case-2 bridge, both in the Case 2 text and in the audit and brief templates.
   - #61451 (`d097b50e356`, 2026-10-21) removed that getter. It replaced raw-base uses with `BankView` / `LocalTensorAccessor`, plus `TransferAccess::bank_base` at a few same-address sites. There are no hits for it under `tt_metal/hw/inc/api/`.
   - This op has no Case-2 binding, so it is unaffected. The next audit with a Case-2 binding, however, will point the porter at an API that no longer exists.
4. **One status-summary row looks vestigial.** The row *"Feature Support — Variadic-CTA (Ok / Unsupported)"* has no Appendix A entry, and the RTA varargs subject documents CTA varargs as supported (`compile_time_varargs`). I filled it in as `Ok`.
5. **Two templates ask for the same TTNN facts.** In `ttnn_factory.md`, "Audit report deliverable" asks the auditor to add a `## TTNN ProgramFactory` block (Concept / Fit / Custom hash / Stop signals) to this file. The audit template has no such section; its Team-only "TTNN factory analysis" bullet covers the same facts. I included both, with the block nested under that bullet. One home for these facts would be cleaner.
