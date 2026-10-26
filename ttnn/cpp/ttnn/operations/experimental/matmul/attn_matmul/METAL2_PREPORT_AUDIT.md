# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul`

One device operation, one program factory, declared in the `program_factory_t` variant of
`device/attn_matmul_device_operation.hpp:23`:

- **`AttnMatmulDeviceOperation`** (`device/attn_matmul_device_operation.{hpp,cpp}`)
  - `AttnMatmulProgramFactory` (`device/attn_matmul_program_factory.{hpp,cpp}`) — `create_descriptor` only
    (`attn_matmul_program_factory.cpp:19`); no `override_runtime_arguments`, no custom hash.

Two public entry points share the factory: `ttnn::experimental::attn_matmul` and
`ttnn::experimental::attn_matmul_from_cache` (`attn_matmul.cpp:13`, `:34`); the second only adds
`num_tokens` / `transpose_hw` to the attributes. Same kernels, same CBs — one audit covers both.

Kernels, all referenced (no dead kernel files in the directory):

| Kernel | Bound by | Owner |
|---|---|---|
| `device/kernels/dataflow/reader_transformer_attn_matmul.cpp` | reader (`attn_matmul_program_factory.cpp:182-184`) | this op |
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | writer (`:191-192`) | **borrowed** (`eltwise/unary` shared pool; `_metal2` fork exists) |
| `device/kernels/compute/transformer_attn_matmul.cpp` | compute (`:203-204`) | this op |

> `ttnn/cpp/ttnn/operations/experimental/quasar/` holds **no** copy of this op (its `matmul/` entry is the
> generic matmul). Nothing under that tree was read for this audit, and the porter must not read it either.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `akertesz/op-porting-recipe` @ `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
(the recipe branch is an orphan history and cannot be merged into `main`; the audit ran on a verbatim local import of its `docs/…/metal_2.0/` tree, which this branch does not carry)

**Code audited:** `origin/main` @ `fd25f0f4aac` (2026-10-26). Line numbers below are against that commit.
Provenance of the shape audited: ProgramDescriptor migration PR #45074 (2026-06-02), custom hash removed by
PR #47232 (2026-06-23), `Buffer*` bindings by PR #49134 (2026-07-07), Device 2.0 kernels by PR #39552
(2026-03-10), runtime-arg core mapping fixed by PR #59789 (2026-10-15).

**Readiness sheet:** fetched live this session (CSV export of *"TTNN Operations analysis"*); 1 row for
`experimental/matmul/attn_matmul` (sheet row 192; the sibling `experimental/matmul/group_attn_matmul` is row
193 and is a separate op, not audited here).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `AttnMatmulDeviceOperation` → `AttnMatmulProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — reader on `Noc` / `CircularBuffer` / `TensorAccessor` / `UnicastEndpoint` / `CoreLocalMem`; borrowed writer already on `DataflowBuffer`; compute is a TRISC kernel (DM migration N/A) whose FIFO ops are already wrapper methods |
| *Prereqs* — Cross-op escapes | Ok — the compute kernel calls two kernel-lib helpers (`compute_kernel_lib::untilize` / `tilize`) that take `uint32_t` CB ids as template parameters; the reader includes only `tt_metal` API headers; one borrowed kernel file (fork exists) |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | Ok — every compile-time-arg read is at a constexpr index |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (`Diego validation = yes`) |
| *TTNN Readiness* — Concept (current) | `descriptor` — cross-check ✓ (`create_descriptor` at `attn_matmul_program_factory.cpp:19`; PR #45074) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (sheet cell blank; the factory returns a plain `ProgramDescriptor`) |
| *TTNN Readiness* — Custom hash | No — sheet `no` (and `Backdoor custom hash = no`); no `compute_program_hash` / `attribute_values` / `to_hash` in the op ✓. Sheet `Formerly custom hashed? = yes` ✓ — removed by PR #47232 |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — sheet `no`; absent from the device op ✓ |
| *TTNN Readiness* — `override_runtime_arguments` | No — sheet `no`; the factory declares only `create_descriptor` (`attn_matmul_program_factory.hpp:19`) ✓ |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `attn_matmul_nanobind.cpp:20` and `:37` bind only the two public wrappers; sheet `no` ✓ |
| *TTNN Readiness* — Op-owned tensors | No — sheet cell blank (N/A on a `descriptor` row); plain `ProgramDescriptor` ✓ |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (sheet `Porting Target` agrees) |
| *Port work* — Offset base pointer | **none** — the three address RTAs are `Buffer*` bindings (`attn_matmul_program_factory.cpp:237-238`, `:260`); tile indices ride as separate scalars |
| *Port work* — Tensor bindings (per binding) | `input_tensor_a`: Case 1 · `input_tensor_b`: Case 1 · `output`: Case 1 · no clean (borrowed-memory) bindings, no Case 2 |
| *TTNN Readiness* — TensorParameter relaxation | `none` → clears |
| *Port work* — TensorAccessor 3rd arg | **none** — no accessor in the op passes a 3rd argument |
| *Port work* — CB endpoints | legal 1:1 × 5 (`in0`, `in1`, `intermed1`, `intermed2`, `out`) · self-loop × 1 (`intermed0`, compute) — one config |

**CB endpoints** are dispositions, not gates: the one self-loop has its port-time resolution recorded in
*Gate detail*.

## Result

**GREEN → brief issued.** Device 2.0, Appendix A feature compatibility, the readiness sheet
(`Is able to port? = yes`, every cheaply-checkable column matching the code), offset base pointers, the
TensorAccessor 3rd argument and the `TensorParameter` relaxation column (`none`) all clear. The port can begin
on explicit go-ahead: a single factory, three kernels, six CBs, three Case-1 tensor bindings, and a borrowed
writer whose `_metal2` fork already exists.

**Subject coverage.** All twelve subjects were run in full; nothing was skipped or deferred.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** — the row reads `yes`, every cheaply-checkable
  primary column matches the code, cross-column invariants hold, factory-set match ✓ (1 row ↔ 1 factory, no
  phantom, no missing):

  | Factory (sheet row) | `Concept` | `Op Classification` | `Override…?` | `Custom hash` | `get_dynamic…` | `Pybind descriptor` | `Smuggled pointer` | `Known op issues` | `TensorParameter relaxation` | `Is able to port?` | Cross-check |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | `AttnMatmulProgramFactory` (192) | `descriptor` | `PD Op (pointer-patching)` | `no` | `no` | `no` | `no` | `no` | *(empty)* | `none` | `yes` | ✓ `create_descriptor` `.cpp:19`; no override; address RTAs delivered as `Buffer*` (`.cpp:237-238`, `.cpp:260`), patched by the framework |

  Other cells, quoted for the record: `Execution Model = SPMD`, `Porting Target = ProgramSpecFactoryConcept`,
  `Backdoor custom hash = no`, `Diego validation = yes`, `Model = other`, `ProgramFactory used in llama? = no`,
  `Uses llama kernels? = yes` (the borrowed `writer_unary_interleaved_start_id.cpp`), `Op-owned tensors?` /
  `Secretly SPMD Workload?` / `Why secretly SPMD?` / `Provisional relaxation finding` blank,
  `Pointer patching perf issue? = suspect perf regression (+ fixed latent bug)`, `Formerly custom hashed? = yes`.
  The last two are informational columns outside the blocking table; see *Team-only → Relaxation candidates*
  for what they most likely refer to. No `analyses/relaxations/experimental_matmul_attn_matmul.md` exists and
  none is needed (`none`).

- **Device 2.0 (every kernel used):** **GREEN.** Per kernel:

  | Kernel | Idioms | Verdict |
  |---|---|---|
  | reader (`reader_transformer_attn_matmul.cpp`) | `Noc` (`:36`), `CircularBuffer` wrappers (`:37-40`) with `reserve_back` / `push_back` / `wait_front` / `pop_front` methods (`:82`, `:87`, `:92`, `:96`, `:99`, `:109`, `:118`, `:124`), `TensorAccessorArgs` + `TensorAccessor` (`:46-49`), `noc.async_read(accessor, cb, …)` (`:85`, `:97`), `noc.async_read_barrier()` (`:86`, `:98`, `:117`), `UnicastEndpoint` + `CoreLocalMem` loopback read (`:67`, `:110-116`); `get_tile_size(cb_id)` (`:43-44`) is on the **sanctioned** list | ✓ |
  | writer (borrowed `writer_unary_interleaved_start_id.cpp`) | already `DataflowBuffer` (`:30`), `Noc` (`:29`), `TensorAccessor` (`:42`), `noc.async_write(dfb, s, …)` (`:52`); `get_local_cb_interface(cb).fifo_page_size` (`:27`) sanctioned | ✓ (and the port binds the fork anyway) |
  | compute (`transformer_attn_matmul.cpp`) | TRISC kernel: `CircularBuffer` wrappers for every FIFO op (`:34-39`, `:54`, `:56`, `:60`, `:64`, `:68`, `:82`), LLK free functions over cb ids (`compute_kernel_hw_startup` `:43`, `matmul_init` `:44`/`:80`, `matmul_tiles` `:58`, `pack_tile` `:66`, `reconfig_data_format*` `:79`/`:88-89`, `matmul_block_init` `:90`), kernel-lib `untilize` / `tilize` (`:71-77`, `:86`) | N/A — not a data-movement kernel |

  No `InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_*` / `cb_*` free-function FIFO calls / raw
  semaphore addresses in any kernel the op binds. **One breadcrumb, not a gate:** the reader's L1→L1 row copy
  builds its loopback source from the firmware coordinate globals — `my_x[noc.get_noc_id()]` /
  `my_y[noc.get_noc_id()]` into a `UnicastEndpoint` (`:65-67`, `:115`) with a `CoreLocalMem` destination
  (`:110`). The Device 2.0 migration guide
  (`docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md:526-551`)
  labels exactly this form *"Avoid (Device 2.0 API, but hand-built loopback endpoint)"* and prefers passing
  local handles as both endpoints. It is Device 2.0 API — the gate's Red bullets name legacy idioms (CB-index
  holdovers, raw `noc_async_read`, addr-gens), and this is none of them — so it clears; the preferred form is a
  transfer rewrite, which a port does not do. Recorded for the porter under *Heads-ups* (keep as is) and for
  the recipe maintainer under *Recipe notes*. A checked-in `_metal2` fork on `main` keeps the same hand-built
  loopback (`data_movement/sharded/device/kernels/dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp:133-137`),
  so there is precedent for carrying it through a port unchanged.

- **Feature compatibility:** every Appendix A entry `N/A`:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `.global_circular_buffer` field on any of the six `CBDescriptor`s (`attn_matmul_program_factory.cpp:104-169`), no `remote_cb` / `remote_index` idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset`, no `.buffer` (no borrowed-memory CB at all), no `UpdateDynamicCircularBufferAddress` |
  | GlobalSemaphore | N/A | the op declares no semaphores of any kind (`desc.semaphores` never touched) |

- **CB endpoints (GATE-free):** classified per CB, per node. Every CB has one instance per node over the
  **full device compute grid** (`all_device_cores`, `attn_matmul_program_factory.cpp:75-76`), and so does
  every kernel (`:186`, `:194`, `:206`). There is a single configuration for the census: `transpose_hw`
  and `fp32_dest_acc_en` change compile-time args, data formats and tile sizes, never which kernel touches
  which CB. Idle cores run the same kernels with zeroed runtime args (`:228-230`) — the census is identical
  there, every loop simply runs zero times.

  | CB (index) | Built at | Touchers | Census | Disposition |
  |---|---|---|---|---|
  | `in0` (`c_0`, `Kt` tiles) | `:103-112` | reader: `reserve_back(Kt)` `:82`, `noc.async_read` into it `:85`, `push_back(1)` × `Kt` `:87` → locked PRODUCER; compute: `wait_front(kt + 1)` `:54`, `matmul_tiles` reads `:58`, `pop_front(Kt)` `:82` → locked CONSUMER | 1P + 1C | **legal 1:1** |
  | `in1` (`c_1`, 2 tiles) | `:114-124` | reader: `reserve_back(1)` / `noc.async_read` / `push_back(1)` `:96-99` → locked PRODUCER; compute: `wait_front(1)` `:56`, `pop_front(1)` `:60` → locked CONSUMER | 1P + 1C | **legal 1:1** |
  | `intermed0` (`c_2`, 1 tile) | `:126-135` | compute only: `reserve_back(1)` / `pack_tile` / `push_back(1)` `:64-68`, then `compute_kernel_lib::untilize<1, c_2, c_3, …>(1)` `:71-77` consumes it (`in_dfb.wait_front` / `pop_front`, `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl:221-238`) | 1 toucher | **self-loop** on the compute kernel (bind it PRODUCER and CONSUMER; legal on Gen1) |
  | `intermed1` (`c_3`, 1 tile) | `:137-146` | compute: `untilize` packs into it (`out_dfb.reserve_back` / `push_back`, `untilize_helpers.inl:235-238`) → locked PRODUCER; reader: `wait_front(1)` `:109`, `pop_front(1)` `:118` → locked CONSUMER, plus a `get_read_ptr()` peek `:58` on its own binding | 1P + 1C | **legal 1:1** |
  | `intermed2` (`c_4`, 1 tile) | `:148-157` | reader: `reserve_back(1)` `:92`, raw NoC write into it through the `get_write_ptr()` peek `:59` / `:110-116`, `push_back(1)` `:124` → locked PRODUCER; compute: `compute_kernel_lib::tilize<1, c_4, c_5>(1)` `:86` consumes it (`in_dfb.wait_front` / `pop_front`, `tilize_helpers.inl:163-189`) → locked CONSUMER | 1P + 1C | **legal 1:1** |
  | `out` (`c_5`, 2 tiles) | `:159-169` | compute: `tilize` packs into it (`out_dfb.reserve_back` / `push_back`, `tilize_helpers.inl:180-188`) → locked PRODUCER; writer: `wait_front(1)` / `pop_front(1)` (fork `writer_unary_interleaved_start_id_metal2.cpp:57`, `:60`) → locked CONSUMER | 1P + 1C | **legal 1:1** |

  No hidden second writer: the only raw write in the op (reader → `intermed2`) belongs to that CB's own FIFO
  producer, and the op has no semaphores to gate a co-fill. No dual-instance work-split (three distinct kernel
  sources, one instance each). No dead CB. No multi-binding flag anywhere.

- **Offset base pointers:** **GREEN.** Every address RTA resolved:

  | RTA | Host value | Kernel use | Verdict |
  |---|---|---|---|
  | reader RTA 0 | `src0_buffer` = `a.buffer()` as a `Buffer*` binding (`attn_matmul_program_factory.cpp:58`, `:237`) | `TensorAccessor(in0_args, src0_addr)` (`reader:48`); page ids are tile indices from scalars `itileA_start` (`reader:24`, host `:246`) walked in-kernel | **clean** — bare base |
  | reader RTA 1 | `src1_buffer` = `b.buffer()` as a `Buffer*` binding (`:59`, `:238`) | `TensorAccessor(in1_args, src1_addr)` (`reader:49`); `itileB_start` (`reader:25`, host `:247`, always `0`) | **clean** — bare base |
  | writer RTA 0 | `dst_buffer` = `output.buffer()` as a `Buffer*` binding (`:60`, `:260`) | `TensorAccessor(dst_args, dst_addr)` (legacy writer `:42`; fork: `TensorAccessor(tensor::dst)` `:48`) with `start_id` (`:21`, host `:262`) | **clean** — bare base |

  No `->address() + <offset>` fold anywhere in the factory (the only `address()`-shaped values are the three
  `Buffer*` pointers). Not in the tables of `2026-07-19_offset_base_pointers.md` → the *"no fold, op not in
  the tables"* outcome. Type 3 (`address_offset`) is N/A per Appendix A; Type 4 (`narrow`) is not used.

- **TensorAccessor 3rd argument:** **N/A — no accessor in the op passes a 3rd argument.** The reader builds
  two 2-argument accessors (`reader:48-49`); the legacy writer builds one (`:42`) and the fork a 1-argument one
  (`:48`). `2026-07-06_tensor_accessor_3rd_arg_triage.md` does not list `attn_matmul` (its `matmul` /
  `minimal_matmul` rows are other ops). Nothing to drop.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding), all three delivered today in the `Buffer*`-binding form of PR #49134:
  - `input_tensor_a` — **Case 1** (`Buffer*` at `attn_matmul_program_factory.cpp:237` → `TensorAccessor` at
    `reader:48`) → `TensorParameter` / `TensorBinding`; reader RTA slot 0 and the first `TensorAccessorArgs` CTA
    tail (`:175`, kernel `TensorAccessorArgs<2>()` at `reader:46`) disappear.
  - `input_tensor_b` — **Case 1** (`:238` → `reader:49`) → `TensorParameter` / `TensorBinding`; reader RTA slot 1
    and the second CTA tail (`:176`, `reader:47`) disappear.
  - `output` — **Case 1** (`:260` → legacy writer `:42`) → `TensorParameter` / `TensorBinding` named
    `tensor::dst` by the fork; writer RTA slot 0 and its CTA tail (`:179`) disappear.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** `intermed0` → self-loop on the compute kernel; `in0`, `in1`, `intermed1`, `intermed2`,
  `out` → legal 1:1 as tabled above.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no CB has more than two touchers and no two kernels
  share a FIFO role.
- **Cross-op / shared kernels:** `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
  → **fork exists**: `writer_unary_interleaved_start_id_metal2.cpp` beside it (issue #52228). Rung 1: bind it,
  adopt its vocabulary (`dfb::out` bound CONSUMER, `tensor::dst`, named args `num_pages` / `start_id`), do not
  touch the original. The legacy factory sets no defines (`attn_matmul_program_factory.cpp:190-196`), so none
  of the fork's `OUT_SHARDED` / `BACKWARDS` / `ENTRIES_PER_PAGE` branches applies — the interleaved forward loop with one page per entry is the one in use.
  Sunset list of other legacy binders on `main` (by bound path; quasar copies and the same-named private copies
  under `data_movement/slice` and `matmul` excluded):
  `data_movement/reshape_on_device/device/reshape_tile_program_factory.cpp`,
  `eltwise/unary_backward/device/unary_backward_program_factory.cpp`,
  `examples/example/device/multi_core_program_factory.cpp`,
  `examples/example/device/single_core_program_factory.cpp`.
  **Sunset list, not authorization to convert in place.**
- **RTA varargs:** none. The reader reads eleven distinct fields at fixed indices (`reader:15-25`), the compute
  kernel four (`compute:22-25`), the fork two named args. No CTA varargs: the reader reads CTAs 0-1 plus two
  accessor tails at constexpr offsets (`reader:27-28`, `:46-47`), the compute kernel CTA 0 (`compute:21`).
- **Anything else the porter needs:**
  - **Full-grid kernels with zero-arg idle cores.** Kernels and CBs are placed on the whole device compute grid
    (`:75-77`), and the runtime-arg loop walks every device core (`:220-221`, mapping fixed by PR #59789):
    cores in `core_group_1` / `core_group_2` get real args, every other core gets an all-zero vector
    (`:228-230`) so its kernels start and fall straight through (`blocks = 0`, `batch = 0`, `num_pages = 0`).
    Reproduce the shape: `core_ranges` = the full grid on all three `KernelSpec`s, per-core named values with
    zeros on idle cores. The `0` that today sits in the `Buffer*` slot on idle cores (`:228`, `:230`) is a
    placeholder for the address; tensor bindings are per-kernel, so it simply vanishes.
    `test_attn_matmul_compute_grid_shorter_than_device` (`tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py:979`)
    pins this behaviour.
  - **Loopback L1→L1 row copy in the reader (`reader:65-67`, `:110-117`).** Keep it exactly as it is: a
    `UnicastEndpoint` source built from `my_x[noc.get_noc_id()]` / `my_y[…]` plus the `intermed1` read pointer,
    a `CoreLocalMem<uint32_t>` destination at the `intermed2` write pointer, 64 or 128 bytes per row. It is the
    migration guide's *"Avoid"* form (Device 2.0 API, hand-built loopback endpoint; guide `:526-551`), and the
    guide's preferred form — local handles as both endpoints — is a transfer rewrite, out of port scope (a
    post-port style item at most). The port touches only the `CircularBuffer` → `DataflowBuffer` spelling of
    the two peeks that feed it.
  - **Cursor peeks taken once, before any FIFO op (`reader:58-59`).** `cb_intermed1_obj.get_read_ptr()` and
    `cb_intermed2_obj.get_write_ptr()` are captured at kernel entry and reused for every `wait_front` /
    `reserve_back` iteration; that is correct only because both CBs are exactly one page (`:139`, `:150`), so
    the cursors never move. They are public peeks (`cb_dfb_api_whitelist.md` §C) and map 1:1 onto
    `DataflowBuffer::get_read_ptr()` / `get_write_ptr()`; keep the one-page sizing (the port changes no sizes).
    `scripts/check_kernel_cb_ptrs.py` (pre-commit, issue #39432) passes on the kernel today.
  - **Compute `hw_config`: Style A with dropped fields.** The op resolves the TTNN config
    (`get_compute_kernel_config_args`, `:32-33`) but copies only `math_fidelity` and `fp32_dest_acc_en` onto the
    `ComputeConfigDescriptor` (`:208-211`). `math_approx_mode` and `dst_full_sync_en` are resolved and
    **dropped** (descriptor defaults: `false` / `false`), `packer_l1_acc` has no Metal 2.0 counterpart. Per the
    port recipe's *"check for a dropped field"* rule: translate with `to_compute_hardware_config`, then force
    `sfpu_precision_mode = Precise` and `double_buffer_dest = true` on the returned `ComputeGen1Config`
    regardless of what the caller passed — that is the legacy behaviour. Metal-only fields: `bfp_pack_precision_mode`
    stays default (legacy sets no `bfp8_pack_precise`); **`unpack_modes` needs conditional entries** — the
    legacy descriptor sets no `unpack_to_dest_mode` (`Default` → `UnpackToSrc`), and the validator requires an
    explicit entry for every Float32 DFB the compute kernel consumes while `enable_32_bit_dest` is on. The
    compute kernel consumes `in0`, `in1`, `intermed0` and `intermed2`; `in0` / `in1` are Float32 when the
    inputs are (`:35-36`; validation then forces `fp32_dest_acc_en`, `:46-49`), `intermed0` / `intermed2` are
    Float32 when `fp32_dest_acc_en && in0 == Float32` (`:36-39`). Gate each `UnpackToSrc` entry on its DFB's
    format being Float32; `test_attn_matmul_fp32` (`test_attn_matmul.py:93`) exercises the Float32 case.
  - **`opt_level`:** none set anywhere (`grep -n opt_level` on the factory: no hits), so the compute spec
    resolves to **O3** and must say so explicitly; both DM specs resolve to O2. DM configs are
    `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` (`:188`, `:196`) → the arch-agnostic reader /
    writer helpers.
  - **`constexpr` metadata:** none from CB metadata. `reader:43-44` declares `const uint32_t in0_tile_bytes =
    get_tile_size(cb_id_in0)` (runtime `const`) → the DFB getter. `bfloat16_row_bytes` (`reader:62`) is a
    CTA-derived `constexpr`, untouched.
  - **LLK and kernel-lib calls take `dfb::name` directly.** `compute_kernel_hw_startup<SrcOrder::Reverse>(…)`
    (`compute:43`), `matmul_init` / `matmul_tiles` / `matmul_block_init`, `pack_tile`, `reconfig_data_format*`
    all take `uint32_t` cb ids; the id-only kernel-lib forms `untilize<1, c_2, c_3, …>(1)` (`:71-77`) and
    `tilize<1, c_4, c_5>(1)` (`:86`) take them as non-type template parameters and build their own
    `DataflowBuffer`s inside (`untilize_helpers.hpp:175-177`, `tilize_helpers.hpp:186-189`) — the constexpr
    token conversion covers both positions (*Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers*).
  - **Stale hash comments to drop, not carry.** `attn_matmul_program_factory.hpp:14-18` and
    `attn_matmul_program_factory.cpp:99-102` say `padded_shape` is folded into `compute_program_hash()`; that
    custom hash was removed by PR #47232 and the op is on the default reflection hash (which keys on the full
    `TensorSpec`s, padded shapes included). Do not transplant the comment into the ported factory.

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean** (no `uint32_t sem_id` / sem-address /
  `TensorAccessorArgs<N>` / `CircularBuffer&` / addr-gen shapes in any called donor function).

  | Op kernel | Donor file | Class | Functions called | Shape |
  |---|---|---|---|---|
  | reader | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h` | `tt_metal/*` | `Noc`, `CircularBuffer`, `TensorAccessor`, `UnicastEndpoint`, `CoreLocalMem`, `get_tile_size`, `my_x` / `my_y` | ✓ framework API, no op donor |
  | compute | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | kernel-lib (class 2) | `compute_kernel_lib::untilize<block_width_tiles, input_dfb, output_dfb, InitUninitMode, WaitMode, ReconfigureRegisterDatatypeMode>(uint32_t num_blocks)` (`untilize_helpers.hpp:165-180`), `compute_kernel_lib::tilize<block_width_tiles, input_dfb, output_dfb>(uint32_t num_blocks)` (`tilize_helpers.hpp:176-190`) — cb ids as `uint32_t` NTTPs | ✓ `uint32_t cb_id` |
  | compute | `api/compute/tile_move_copy.h`, `api/compute/matmul.h`, `api/compute/compute_kernel_hw_startup.h`, `api/compute/tilize.h`, `api/dataflow/circular_buffer.h` | `tt_metal/*` (LLK) | `compute_kernel_hw_startup(uint32_t, uint32_t, uint32_t)`, `matmul_init`, `matmul_tiles`, `matmul_block_init`, `pack_tile`, `reconfig_data_format`, `reconfig_data_format_srca`, `pack_reconfig_data_format` | ✓ `uint32_t cb_id` |
  | writer (borrowed) | `api/dataflow/*`, `api/tensor/noc_traits.h`, `experimental/kernel_args.h` (fork) | `tt_metal/*` | — | ✓ |

  Borrowed kernel file: `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
  (broadly shared; fork present; legacy binders listed under *Heads-ups*; 25 other factories on `main`
  already bind the fork). This op lends nothing.
- **Relaxation candidates (FALLIBLE — candidates to verify, default strict):** the custom hash removed by
  PR #47232 keyed on `transpose_hw`, output memory config / dtype, both inputs' dtype / memory config /
  **padded shape** — i.e. it was already shape-strict, so the default hash is equivalent and the sheet's
  `none` is consistent. The kernels take `Mt` / `Kt` / `Nt` as runtime args and only `in0`'s CB capacity
  (`Kt` tiles, `:105`) is baked into the program, so PR #41267's observation stands: every new `kv_len` /
  `num_tokens` costs a program build. A shape relaxation on the non-`K` dims would restore reuse, **but** on
  `ProgramSpecFactoryConcept` only tensor bindings refresh on a cache hit, so the geometry RTAs would go stale
  — such a relaxation would need the custom concept and an override recomputing them. Ops-team call; the
  sheet's `Pointer patching perf issue? = suspect perf regression (+ fixed latent bug)` most likely tracks the
  same cost.
- **TTNN factory analysis:** single-program SPMD; no op-owned tensors; no custom hash (default reflection
  hash; `Formerly custom hashed? = yes`, removed by PR #47232); no `get_dynamic_runtime_args`; no pybound
  `create_descriptor` (`attn_matmul_nanobind.cpp` binds only `attn_matmul` and `attn_matmul_from_cache`); no
  `override_runtime_arguments` → **`ProgramSpecFactoryConcept`**. Tensors reachable from `tensor_args`
  (`input_tensor_a`, `input_tensor_b`, optional `preallocated_output`) and `tensor_return_value`.

  ### TTNN ProgramFactory
  #### Concept
  `ProgramSpecFactoryConcept` (ported-from factory has no `override_runtime_arguments`).
  #### Fit
  - Single vs multi-program: single — one `ProgramDescriptor` today, one `ProgramSpec` stamped across the mesh.
  - Op-owned device resources: none.
  - Tensor-arg matching: strict (sheet `none`).
  - Legacy-to-Metal-2.0 shape: 1:1 with the ported-from factory.
  #### Custom `compute_program_hash`
  none — default reflection-based hash (the custom one was removed by PR #47232).
  #### Stop signals
  None.

## Misc anomalies  *(team-only, non-gating)*

- `attn_matmul_program_factory.hpp:14-18` and `attn_matmul_program_factory.cpp:99-102` — comments claim
  `padded_shape` is folded into `compute_program_hash()`; the custom hash was deleted by PR #47232. The
  substance survives (the default hash keys on the tensor specs) but the reference is dangling.
- `attn_matmul_program_factory.cpp:32-33` vs `:208-211` — `math_approx_mode`, `packer_l1_acc` and
  `dst_full_sync_en` are resolved from the caller's `compute_kernel_config` and silently ignored; a caller
  cannot change them. Preserved behaviour, reported rather than repaired.
- `attn_matmul_device_operation.cpp:72-75` — the only `compute_with_storage_grid_size` ≤ device-grid check is
  a `TT_ASSERT` (debug builds only) while every other validation is `TT_FATAL`; in release an oversized grid
  reaches `split_work_to_cores` (`attn_matmul_program_factory.cpp:72-73`) unchecked. PR #59789 made the
  runtime-arg loop safe for a *smaller* grid; a *larger* one is still unguarded.
- `attn_matmul_program_factory.cpp:249-256` — the compute kernel is told `batch = 1`, `Mt = 1`,
  `Nt = blocks × Mt × Nt`, while the reader is told the real `blocks` / `Mt` / `Nt`; the counts agree, the
  names don't.
- `reader_transformer_attn_matmul.cpp:14` — "same arg indices as in reader_binary_diff_lengths for compat" is a
  stale coupling note; nothing else reads this layout. `:20` (`// if 0`) is a truncated comment.
- `attn_matmul_program_factory.cpp:66-67` — `num_cores` and `all_cores` from `split_work_to_cores` are unused.

## Questions for the user  *(none)*

## Recipe notes

1. **The workspace recipe still cannot be followed literally** (already raised by the `padded_slice` audit):
   `akertesz/op-porting-recipe` is an orphan history, so "branch off the recipe branch, merge main" is
   impossible; this audit branched off `main` and imported `docs/…/metal_2.0/` verbatim.
2. **The Device 2.0 gate has no wording for the migration guide's "Avoid" forms.** The guide distinguishes
   *Legacy* / *Avoid (Device 2.0 API, but hand-built loopback endpoint)* / *New API*; the audit's Green/Red
   bullets only name legacy idioms. I treated "Avoid" as compliant-but-discouraged (FYI-P, keep as is) — a
   sentence in the gate saying which side of the line the guide's "Avoid" forms fall on would remove the
   judgment call, and a post-port style recipe could own the rewrite.
3. **`unpack_modes` entries that depend on a runtime-derived DFB format.** The port recipe's required-entry rule
   is stated for a DFB *that is* Float32; here the intermediate DFBs are Float32 only under
   `fp32_dest_acc_en && in0 == Float32`, so the entry must be conditional on an *unconditional* DFB. The
   *Conditional / optional resource bindings* pattern covers conditional DFBs, not conditional `unpack_modes`
   keys; worth an explicit sentence.
4. **Idle-core placeholder zeros.** The TensorParameter subject explains the `Buffer*`-binding form but not its
   companion shape — an all-`uint32_t` zero vector on cores with no binding, where slot 0 is a placeholder for
   the address. Porters should be told the placeholder just disappears (bindings are per-kernel).
5. **Sheet row numbers.** `ttnn_op_porting_readiness.md` suggests `grep -n`; the header cells contain embedded
   newlines, so CSV line numbers are off from sheet row numbers (here by 9). Parse with a CSV reader and use
   the record index.
