# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul`

One device operation, one program factory, declared in the `program_factory_t` variant of
`device/group_attn_matmul_device_operation.hpp:173`:

- **`GroupAttnMatmulDeviceOperation`** (`device/group_attn_matmul_device_operation.{hpp,cpp}`)
  - `GroupAttnMatmulProgramFactory` (`device/group_attn_matmul_program_factory.{hpp,cpp}`) — `create_descriptor`
    only (`group_attn_matmul_program_factory.cpp:19`); no `override_runtime_arguments`, no custom hash.

One public entry point, `ttnn::experimental::group_attn_matmul` (`group_attn_matmul.cpp:39`), calls
`ttnn::prim::group_attn_matmul` (`device/group_attn_matmul_device_operation.cpp:212`) with
`num_tokens = nullopt` and `transpose_hw = nullopt` (`group_attn_matmul.cpp:79-80`); the nanobind module binds
only that wrapper (`group_attn_matmul_nanobind.cpp:105`), and nothing else in `ttnn/` or `models/` calls the
`prim`. The KV-cache attributes therefore exist on the device op (`device/group_attn_matmul_device_operation_types.hpp:134-135`)
but are unreachable today (see *Misc anomalies*); they change compile-time args and tile arithmetic, never which
kernel touches which CB, so they do not add a census configuration.

Kernels, all referenced, all owned by this op (no borrowed kernel files, no `_metal2` fork beside any of them,
no dead kernel file in the directory):

| Kernel | Bound by | RISC / NOC |
|---|---|---|
| `device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp` | reader (`group_attn_matmul_program_factory.cpp:279-290`) | `RISCV_1`, `preferred_noc_for_dram_read(arch)` (`:275`, `NOC_0` on every arch today — `tt_metal/api/tt-metalium/kernel_types.hpp:140-145`) |
| `device/kernels/dataflow/writer_transformer_group_attn_matmul.cpp` | writer (`:292-303`) | `RISCV_0`, the other NOC (`:276-277`) |
| `device/kernels/compute/transformer_group_attn_matmul.cpp` | compute (`:305-320`) | TRISC |

> `ttnn/cpp/ttnn/operations/experimental/quasar/` holds **no** copy of this op. Nothing under that tree was read
> for this audit, and the porter must not read it either.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `akertesz/op-porting-recipe` @ `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
(the recipe branch is an orphan history that cannot be merged into `main`; the audit read its
`docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` tree straight from `origin/akertesz/op-porting-recipe`
— this branch does not carry a copy, so the provenance command in the recipe prints nothing here).

**Code audited:** `origin/main` @ `0df0dde3316` (2026-10-27). Line numbers below are against that commit; the op
directory last changed in `d52909f346a` (2026-09-25, #57790). Provenance of the shape audited (recovered through
`gh`, since this clone is shallow): new-infra migration PR #32866 (2025-11-21), Device 2.0 kernels PR #39552
(2026-03-10), ProgramDescriptor migration + shape-aware hash PR #45074 (2026-06-02), custom hash removed by
PR #47232 (2026-06-23), `Buffer*` bindings by PR #49134 (2026-07-07).

**Readiness sheet:** fetched live this session (CSV export of *"TTNN Operations analysis"*); 1 row for
`experimental/matmul/group_attn_matmul` (CSV record 193 = sheet row 193; the sibling `experimental/matmul/attn_matmul`
is row 192 and a separate op).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `GroupAttnMatmulDeviceOperation` → `GroupAttnMatmulProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — reader and writer on `Noc` / `CircularBuffer` / `Semaphore<>` / `TensorAccessor` / `UnicastEndpoint` / `CoreLocalMem`; compute is a TRISC kernel (DM migration N/A) whose FIFO ops are already wrapper methods. `get_tile_size(cb_id)` sites are sanctioned. No holdovers |
| *Prereqs* — Cross-op escapes | Ok — every `#include` is `tt_metal` API; no op donor, no borrowed kernel file, no kernel-lib helper |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | Ok — every compile-time-arg read is at a literal or constexpr index |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (`Diego validation = yes`) |
| *TTNN Readiness* — Concept (current) | `descriptor` — cross-check ✓ (`create_descriptor` at `group_attn_matmul_program_factory.cpp:19`; PR #45074) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (sheet cell blank; the factory returns a plain `ProgramDescriptor`) |
| *TTNN Readiness* — Custom hash | No — sheet `no` (`Backdoor custom hash = no`); no `compute_program_hash` / `attribute_values` / `to_hash` in the op ✓. Sheet `Formerly custom hashed? = yes` ✓ — removed by PR #47232; two comments still refer to it (see *Misc anomalies*) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — sheet `no`; absent from the device op ✓ |
| *TTNN Readiness* — `override_runtime_arguments` | No — sheet `no`; the factory declares only `create_descriptor` (`group_attn_matmul_program_factory.hpp:223-226`) ✓ |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `group_attn_matmul_nanobind.cpp:105` binds only the public wrapper; sheet `no` ✓ |
| *TTNN Readiness* — Op-owned tensors | No — sheet cell blank (N/A on a `descriptor` row); plain `ProgramDescriptor` ✓ |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (sheet `Porting Target` agrees) |
| *Port work* — Offset base pointer | **none** — the three address RTAs are `Buffer*` bindings (`group_attn_matmul_program_factory.cpp:361`, `:406-407`); tile indices ride as separate scalars |
| *Port work* — Tensor bindings (per binding) | `input_tensor_b`: Case 1 interleaved / clean (borrowed `c_2`) sharded · `input_tensor_a`: Case 1 interleaved / clean (borrowed `c_0`) sharded · `output`: Case 1 interleaved / clean (borrowed `c_5`) sharded · no Case 2 |
| *TTNN Readiness* — TensorParameter relaxation | `none` → clears |
| *Port work* — TensorAccessor 3rd arg | **none** — no accessor in the op passes a 3rd argument |
| *Port work* — CB endpoints | legal 1:1 × 4 (`in0`, `intermed0`, `intermed1`, `out`) · `in1`: 1:1 on q-work nodes, **self-loop on mcast-only nodes** (`q_heads < 32` only) → expressed as a second reader `KernelSpec` over that node set · `in2`: self-loop (reader), **conditional DFB** (`IN1_SHARDED` only) · no dead CB · no multi-binding flag |

**CB endpoints** are dispositions, not gates: the `in1` per-node split and the `in2` self-loop have their
port-time resolutions recorded in *Gate detail* and *Heads-ups*.

## Result

**GREEN → brief issued.** Device 2.0, Appendix A feature compatibility, the readiness sheet
(`Is able to port? = yes`, every cheaply-checkable column matching the code), offset base pointers, the
TensorAccessor 3rd argument and the `TensorParameter` relaxation column (`none`) all clear. The port can begin on
explicit go-ahead: a single factory, three owned kernels, six CBs (one conditional), two semaphores, three tensor
bindings whose case flips with sharding, one genuine RTA vararg block, and one CB (`in1`) whose endpoint roles
differ *per node* within a single configuration — the one structural decision the brief spells out.

**Subject coverage.** All twelve subjects were run in full; nothing was skipped or deferred.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** — the row reads `yes`, every cheaply-checkable
  primary column matches the code, cross-column invariants hold, factory-set match ✓ (1 row ↔ 1 factory, no
  phantom, no missing):

  | Factory (sheet row) | `Concept` | `Op Classification` | `Override…?` | `Custom hash` | `get_dynamic…` | `Pybind descriptor` | `Smuggled pointer` | `Known op issues` | `TensorParameter relaxation` | `Is able to port?` | Cross-check |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | `GroupAttnMatmulProgramFactory` (193) | `descriptor` | `PD Op (pointer-patching)` | `no` | `no` | `no` | `no` | `no` | *(empty)* | `none` | `yes` | ✓ `create_descriptor` `.cpp:19`; no override; address RTAs delivered as `Buffer*` (`.cpp:361`, `:406-407`), patched by the framework |

  Other cells, quoted for the record: `Execution Model = SPMD`, `Porting Target = ProgramSpecFactoryConcept`,
  `Backdoor custom hash = no`, `Diego validation = yes`, `Model = other`, `ProgramFactory used in llama? = no`,
  `Uses llama kernels? = no`, `Op-owned tensors?` / `Secretly SPMD Workload?` / `Why secretly SPMD?` /
  `Provisional relaxation finding` blank, `Pointer patching perf issue? = suspect perf regression (+ fixed latent bug)`,
  `Formerly custom hashed? = yes`. The last two are informational columns outside the blocking table (see
  *Team-only → Relaxation candidates*). No `analyses/relaxations/experimental_matmul_group_attn_matmul.md`
  exists and none is needed (`none`).

- **Device 2.0 (every kernel used):** **GREEN.** Per kernel:

  | Kernel | Idioms | Verdict |
  |---|---|---|
  | reader (`reader_mcast_transformer_group_attn_matmul.cpp`) | `Noc` (`:73`), `CircularBuffer` wrappers (`:74-75`) with `reserve_back` / `push_back` / `pop_front` / `get_write_ptr` / `get_read_ptr` methods (`:168-169`, `:123`, `:323`, `:327-328`), `Semaphore<>` objects built from the two id RTAs (`:54-55`) with `.set` / `.wait` / `.up(noc, x, y, 1)` / `.set_multicast(noc, …)` (`:88`, `:234-235`, `:295-301`, `:312`, `:315-316`, `:320`), `TensorAccessorArgs<3>` + `TensorAccessor` (`:82-83`), `noc.async_read(accessor, cb, …)` (`:214-219`), `noc.async_write_multicast<…>(CoreLocalMem, cb, …)` (`:243-253`, `:257-267`, `:272-282`), `noc.async_read_barrier()` / `async_writes_flushed()` / `async_write_barrier()` (`:205`, `:228`, `:291`, `:305`), `UnicastEndpoint` + `CoreLocalMem` (`:119`, `:239`, `:271`); `get_tile_size(cb_id_in1)` (`:78`) is on the **sanctioned** list | ✓ |
  | writer (`writer_transformer_group_attn_matmul.cpp`) | `Noc` (`:51`), `CircularBuffer` wrappers (`:52-55`) with `reserve_back` / `push_back` / `wait_front` / `pop_front` / `get_write_ptr` / `get_read_ptr` (`:98`, `:113`, `:115-116`, `:131-132`, `:141`, `:153`, `:156`, `:166`, `:176`), two `TensorAccessor`s (`:65`, `:71`), `noc.async_read(accessor, cb, …)` (`:104-105`), `noc.async_write(cb, accessor, …)` (`:160-161`), `UnicastEndpoint` + `CoreLocalMem` loopback read (`:89`, `:133-139`), barriers (`:110`, `:140`, `:165`); `get_tile_size(cb)` (`:59`, `:61`, `:68`) sanctioned | ✓ |
  | compute (`transformer_group_attn_matmul.cpp`) | TRISC kernel: `CircularBuffer` wrappers for every FIFO op (`:54-58`, `:74`, `:79-80`, `:142`, `:145`, `:151`, `:162`, `:164`, `:169`, `:171`, `:174`), LLK free functions over cb ids (`compute_kernel_hw_startup` `:64`, `matmul_init` `:67`, `matmul_tiles` `:116`, `pack_untilize_dest_init` / `pack_untilize_dest` / `pack_untilize_uninit` `:83` / `:147-148`, `reconfig_data_format_srca` / `pack_reconfig_data_format` `:160-161`, `tilize_init_short_with_dt` / `tilize_block` / `tilize_uninit` `:167-168` / `:172`) | N/A — not a data-movement kernel |

  No `InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_*` / `cb_*` free-function FIFO calls / raw
  semaphore addresses (`get_semaphore`, `noc_semaphore_*`) in any kernel. **Two breadcrumbs, not gates:** both
  DM kernels build an L1→L1 loopback source from the firmware coordinate globals — `my_x[noc.get_noc_id()]` /
  `my_y[…]` into a `UnicastEndpoint` (reader `:117-119`, used by the sharded-`in1` local copy `:189-196`;
  writer `:87-89`, used by the row gather `:133-139`). The Device 2.0 migration guide
  (`docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md:526-551`)
  labels exactly this form *"Avoid (Device 2.0 API, but hand-built loopback endpoint)"*. It is Device 2.0 API —
  the gate's Red bullets name legacy idioms, and this is none of them — so it clears; the preferred form is a
  transfer rewrite, which a port does not do. Same call the `attn_matmul` audit made on the same idiom; recorded
  for the porter under *Heads-ups* (keep as is). `get_arg_addr(i)` (`reader:61`, `:63`) is positional-RTA
  access, not a data-movement idiom — it is the RTA-vararg finding below.

- **Feature compatibility:** every Appendix A entry `N/A`:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `.global_circular_buffer` field on any of the six `CBDescriptor`s (`group_attn_matmul_program_factory.cpp:163-245`), no `remote_cb` / `remote_index` idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | three borrowed-memory CBs set `.buffer` (`c_0` `:174` when `in0` is sharded, `c_2` `:203`, `c_5` `:243` when the output is sharded) and **none** sets `.address_offset`; no `UpdateDynamicCircularBufferAddress`; no Type-3 offset |
  | GlobalSemaphore | N/A | the two semaphores are plain `SemaphoreDescriptor`s (`:142-153`, ids 0 / 1 over `all_device_cores`, `initial_value = INVALID` = 0 per `tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp:13`) |
  | Variable-count compile-time arguments (CTA varargs) | N/A | reader reads CTAs 0-2 at literal indices (`reader:66-68`) plus one accessor tail at a constexpr offset (`TensorAccessorArgs<3>()` `:82`); writer CTAs 0-2 (`writer:41-43`) plus two chained tails (`:62`, `:69`); compute CTAs 0-3 (`compute:42-45`). No count-driven CTA loop anywhere |

- **CB endpoints (GATE-free):** classified per `(CB, config, node class)`. Every CB and every kernel has one
  instance per node over the **full device compute grid** (`all_device_cores`, `group_attn_matmul_program_factory.cpp:60-62`;
  CBs `:168`, `:183`, `:197`, `:212`, `:224`, `:237`; kernels `:284`, `:297`, `:310`). The census axes are the
  three sharding flags (`in0_is_sharded`, `in1_is_sharded`, `output_is_sharded`, `:159-161` — they select the
  `IN0_SHARDED` / `IN1_SHARDED` / `OUT_SHARDED` defines `:263-273` and whether `c_2` exists `:193`) and the
  **node class**, which the per-core runtime args set (`:352-353`; `num_active_cores = max(q_heads, 32)` `:97`):

  - **Q** — q-work nodes (`has_work_for_q_heads = 1`, cores `0 … q_heads-1` in `cores` order): all three kernels do work.
  - **M** — mcast-only nodes (`has_work_for_q_heads = 0`, `has_work_for_mcast_kv_heads = 1`, cores `q_heads … 31`;
    **exist only when `q_heads < 32`**): the reader runs the full mcast protocol; compute (`compute:20-21`) and
    writer (`writer:15-18`) return before touching anything.
  - **I** — idle nodes (`has_work_for_mcast_kv_heads = 0`, cores `≥ num_active_cores`): all three kernels return at once (`reader:17-20`).

  `transpose_hw` and `fp32_dest_acc_en` change CTAs, data formats and tile sizes, never which kernel touches which CB.

  | CB (index) | Built at | Touchers on **Q** nodes | On **M** nodes | Disposition |
  |---|---|---|---|---|
  | `in0` (`c_0`, `Kt` tiles interleaved / whole shard sharded, `:163-176`) | `:166-175` (borrowed from `input_tensor_a` when `in0_is_sharded`, `:174`) | writer: `reserve_back(in0_block_w)` `:98`, interleaved `noc.async_read` via `s0` into it `:104-105`, `push_back` `:113` (under `IN0_SHARDED` the reserve/push bracket no data movement) → locked PRODUCER; compute: `wait_front` `:74`, `matmul_tiles` reads `:116`, `pop_front` `:174` → locked CONSUMER | untouched | **legal 1:1** (both configs); `borrowed_from` under `IN0_SHARDED` |
  | `in1` (`c_1`, `2 · in1_block_num_tiles`, `:178-190`) | `:181-189` | reader: `reserve_back` `:168`, `get_write_ptr` peek `:169`, local copy / interleaved reads into it (`:189-196` / `:214-219`), mcast destination (`:243-253`, `:257-267`, `:272-282`), `push_back` `:323` → locked PRODUCER; compute: `wait_front` `:79`, `pop_front(num_kv_heads_skip)` `:80`, `matmul_tiles` reads `:116`, `pop_front(num_kv_heads_remaining)` `:142` → locked CONSUMER | reader **only**: `push_back` `:327` **and** `pop_front` `:328` (lockstep, keeps the write pointer in sync with the mcast grid) → locked PRODUCER **and** locked CONSUMER; compute absent | **Q nodes: legal 1:1. M nodes: one toucher in both FIFO roles → self-loop.** Because bindings follow `KernelSpec`s, express it as a **second reader `KernelSpec`** (same source, PRODUCER + CONSUMER of `in1`) in its own `WorkUnitSpec` over the M nodes — see *Heads-ups*. When `q_heads ≥ 32` there are no M nodes and `in1` is a plain 1:1 everywhere |
  | `in2` (`c_2`, whole `in1` shard, **only when `in1_is_sharded`**, `:192-205`) | `:195-204` (borrowed from `input_tensor_b`, `:203`) | reader only: `get_read_ptr()` peek `:123`; the address feeds the loopback copy source (`:185-197`) and the direct-mcast source (`:239`) — raw reads through the NoC, no FIFO op | same (M nodes are senders too) | **self-loop** on the reader (DM self-loop, legal on Gen1); **conditional DFB** — exists only under `IN1_SHARDED` (the host already allocates it conditionally, `:193`); kernel-side references are *not* gated (`:71`, `:75`, `:123`, `:239`) → see *Heads-ups* |
  | `intermed0` (`c_3`, `2 · intermediate_num_tiles`, `:207-219`) | `:210-218` | compute: `reserve_back` `:145`, `pack_untilize_dest` packs `:147`, `push_back` `:151` → locked PRODUCER; writer: `wait_front` `:131`, `get_read_ptr` peek `:132`, `pop_front` `:141` → locked CONSUMER | untouched | **legal 1:1** |
  | `intermed1` (`c_4`, `MtNt` tiles, `:221-230`) | `:222-230` | writer: `reserve_back` `:115`, `get_write_ptr` peek `:116`, loopback `noc.async_read` rows into it `:134-139`, `push_back` `:153` → locked PRODUCER; compute: `wait_front` `:162`, `tilize_block` reads `:168`, `pop_front` `:171` → locked CONSUMER | untouched | **legal 1:1** |
  | `out` (`c_5`, `MtNt` tiles interleaved / whole shard sharded, `:232-245`) | `:235-244` (borrowed from the output when `output_is_sharded`, `:243`) | compute: `reserve_back` `:164`, `tilize_block` packs `:168`, `push_back` `:169` → locked PRODUCER; writer: interleaved `wait_front` `:156`, `noc.async_write` from it `:160-161`, `pop_front` `:166`; `OUT_SHARDED` `wait_front` only `:176` → locked CONSUMER | untouched | **legal 1:1** (both configs); `borrowed_from` under `OUT_SHARDED` |

  **Hidden-second-writer hunt:** the only raw writes are the reader's into `in1` (its own PRODUCER binding — the
  remote senders' multicasts land in each node's `in1` at the local write pointer the local reader reserved, so
  they are the data for the local producer's reservation, not a second local toucher) and the writer's into
  `intermed1` (its own PRODUCER binding). The two semaphores gate the mcast handshake, not a co-fill by a
  non-producer. **Multi-reader:** `in2` has one reader-kernel; no other kernel reads any borrowed CB. **Dual-instance
  work-split:** none — three distinct kernel sources, one `KernelDescriptor` each. **Dead CB:** none — every
  index is referenced in every config in which it is allocated (`c_2` is allocated only under `IN1_SHARDED`
  and referenced in that config; see the `in2` row and *Misc anomalies* for its unguarded references in the
  other config). No multi-binding flag anywhere.

- **Offset base pointers:** **GREEN.** Every address RTA resolved:

  | RTA | Host value | Kernel use | Verdict |
  |---|---|---|---|
  | reader slot 2 (`src1_addr`) | `src1_buffer` = `b.buffer()` as a `Buffer*` binding (`group_attn_matmul_program_factory.cpp:54`, `:361`) | interleaved: `TensorAccessor(in1_args, src1_addr)` (`reader:83`); page ids are tile indices walked in-kernel from `in1_start_id` (`reader:31`, host `:368`, always `0`) with `Nt` / `in1_CKtNt` strides. `IN1_SHARDED`: unused (the accessor is `#ifndef IN1_SHARDED`, `reader:81-84`) | **clean** — bare base |
  | writer slot 1 (`src0_addr`) | `src0_buffer` = `a.buffer()` as a `Buffer*` binding (`:53`, `:406`) | interleaved: `TensorAccessor(in0_args, src0_addr)` (`writer:65`) with `in0_start_id = num_blocks_written · MtKt` (`writer:27`, host `:413`). `IN0_SHARDED`: unused (`writer:64-66`) | **clean** — bare base |
  | writer slot 2 (`dst_addr`) | `dst_buffer` = `output.buffer()` as a `Buffer*` binding (`:55`, `:407`) | interleaved: `TensorAccessor(out_args, dst_addr)` (`writer:71`) with `out_start_id = num_blocks_written · MtNt` (`writer:28`, host `:414`). `OUT_SHARDED`: unused (`writer:70-72`) | **clean** — bare base |

  No `->address() + <offset>` fold anywhere in the factory (the only `address()`-shaped values are the three
  `Buffer*` pointers). Kernel-side `in1_sharded_cb_addr + …` (`reader:185-199`, `:339`) and `cb_intermed1_addr + …`
  (`writer:125`, `:143`, `:150`) are L1 arithmetic inside a CB, not a device-tensor base fold. Not in the tables of
  `2026-07-19_offset_base_pointers.md` → the *"no fold, op not in the tables"* outcome. Type 3 (`address_offset`)
  is N/A per Appendix A; Type 4 (`narrow`) is not used.

- **TensorAccessor 3rd argument:** **N/A — no accessor in the op passes a 3rd argument.** The reader builds one
  2-argument accessor (`reader:83`), the writer two (`writer:65`, `:71`). `2026-07-06_tensor_accessor_3rd_arg_triage.md`
  does not list `group_attn_matmul` (its `matmul` / `minimal_matmul` rows are other ops). Nothing to drop.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding), all three delivered today in the `Buffer*`-binding form of PR #49134; the
  case flips with the tensor's sharding, so each binding is **conditional** on the interleaved config and the
  sharded config reaches the tensor through `borrowed_from` instead:
  - `input_tensor_b` — interleaved: **Case 1** (`Buffer*` at `group_attn_matmul_program_factory.cpp:361` →
    `TensorAccessor` at `reader:83`) → `TensorParameter` / `TensorBinding` on the reader; reader RTA slot 2 and the
    `TensorAccessorArgs` CTA tail (`:253`, kernel `TensorAccessorArgs<3>()` `reader:82`) disappear. Sharded
    (`IN1_SHARDED`): **clean** — the tensor backs `in2` via `borrowed_from` (`:203`); no reader `TensorBinding`
    (the validator counts a `borrowed_from` use as a use, `tt_metal/impl/metal2_host_api/program_spec.cpp:649-650`).
  - `input_tensor_a` — interleaved: **Case 1** (`:406` → `writer:65`) → `TensorParameter` / `TensorBinding` on the
    writer; writer RTA slot 1 and the first writer CTA tail (`:260`, `writer:62`) disappear. Sharded
    (`IN0_SHARDED`): **clean** — backs `in0` via `borrowed_from` (`:174`).
  - `output` — interleaved: **Case 1** (`:407` → `writer:71`) → `TensorParameter` / `TensorBinding` on the writer;
    writer RTA slot 2 and the second CTA tail (`:261`, `writer:69`) disappear. Sharded (`OUT_SHARDED`): **clean**
    — backs `out` via `borrowed_from` (`:243`).
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** `in0`, `intermed0`, `intermed1`, `out` → legal 1:1 as tabled above; `in1` → 1:1 on Q nodes
  and a **reader self-loop on M nodes** (second reader `KernelSpec` + `WorkUnitSpec`, `q_heads < 32` only);
  `in2` → reader self-loop, **conditional DFB** (`IN1_SHARDED`).
- **Semaphores:** two `SemaphoreDescriptor`s (`:142-153`) → two `SemaphoreSpec`s with `target_nodes` = the full
  grid, bound on the reader `KernelSpec`(s) and read as `sem::<name>` in place of the two id RTAs (`reader:54-55`,
  host `:389-390`). `initial_value = INVALID` is `0`, the Metal 2.0 default — nothing non-zero to carry.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no CB has more than two touchers on any node and no
  two kernels share a FIFO role on any node. **Do not reach for `allow_instance_multi_binding` on `in1`:** the
  reader-is-also-consumer shape is confined to the M node set, and the spec-level invariant is per node
  (`tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp:41-50` — several `KernelSpec`s
  may bind one role as long as their node sets are disjoint, they are the same kernel kind, and their binding
  parameters match; the validator tallies producers and consumers per node, `program_spec.cpp:1909-1930`).
- **`in1` on mcast-only nodes — two reader `KernelSpec`s, two `WorkUnitSpec`s.** Legacy launches all three kernels
  on every device core and steers by the `has_work_*` flags (`:352-353`). The port keeps that for Q and I nodes
  and carves the **M nodes** (`has_work_for_q_heads = 0 && has_work_for_mcast_kv_heads = 1`, computable on the host
  from the same loop `:345-356`) into their own `WorkUnitSpec` holding **only a second reader `KernelSpec`** of the
  same source: identical CTAs, defines and run-arg schema, bound PRODUCER **and** CONSUMER of `in1` (and
  PRODUCER + CONSUMER of `in2` under `IN1_SHARDED`, both semaphores, and the `input_tensor_b` binding when
  interleaved — M nodes are mcast *senders*, so they read `in1` `:208-228`). The first reader `KernelSpec` (Q + I
  nodes) binds `in1` PRODUCER only; compute binds it CONSUMER on Q + I nodes. Compute and writer are **not** placed
  on M nodes — today they return at their first statement there (`compute:20-21`, `writer:15-18`), so nothing
  observable changes. This is the sanctioned "same-source `KernelSpec`s over disjoint node sets" shape
  (`port_patterns.md` → *Anti-pattern: Demoting per-group CTA to RTA*, the *Why wrong* paragraph); the
  `WorkUnitSpec` rules it must satisfy are no overlap in `target_nodes`, ≥ 1 kernel, ≤ 1 compute and ≤ 2 DM kernels
  per work unit (`program_spec.cpp:2329-2399`). When `q_heads ≥ 32` the M set is empty — emit the second
  `WorkUnitSpec` only when it is non-empty. Kernel code is untouched. Tests that exercise `q_heads < 32`:
  `test_group_attn_matmul` (`q_heads = 16`), `test_group_attn_matmul_fp32` (`q_heads = 2`),
  `test_group_attn_matmul_with_program_cache_exhaustive` (`q_heads = 16`, `24`); production Falcon-40B on T3000
  runs 16 q-heads per chip (`models/demos/t3000/falcon40b/tt/falcon_attention.py:520`, `:575`).
- **`in2` is a conditional DFB with unconditional kernel-side references.** Host-side the CB already exists only
  under `in1_is_sharded` (`:193-205`), so the `DataflowBufferSpec` and the reader's binding are conditional on the
  same flag that emits `IN1_SHARDED` (`:268-270`). Kernel-side, the legacy reader declares `cb_id_in2` (`:71`),
  constructs `cb_in2_obj` (`:75`) and peeks it (`:123`) **outside** any `#ifdef` — today that is a harmless read of
  an unconfigured CB interface in the interleaved build, but post-port `dfb::in2` has no token there. Per
  *Pattern: Conditional / optional resource bindings*, gate `:71` / `:75` / `:123` under `#ifdef IN1_SHARDED`.
  Mind the dead branch: `in1_sharded_cb_addr` is also consumed at `:239` inside `if (mcast_in1_to_local_cb)` —
  a runtime bool that is only ever set under `IN1_SHARDED` (`:129-132`) but whose branch compiles in both builds —
  so either gate `:238-254` too or keep an `#else` fallback definition of `in1_sharded_cb_addr` for the
  interleaved build. Either is a preprocessor-level change the pattern prescribes, not kernel-logic surgery.
- **Writer reads the tile size of a CB it never binds.** `writer:59` `const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1)`
  is **unused** (no other reference in the file). Post-port there is neither a `dfb::in1` token nor a
  `DataflowBuffer` object in the writer to query, so the line cannot be rewritten per whitelist rule 7 — drop
  it and record the drop in the port report (a dead local has no behaviour). *(Question 2 below.)*
- **Cross-op / shared kernels:** none — the op owns all three kernels, no other factory on `main` references them
  (grep over `ttnn/`, `models/`, `tests/`), and no `_metal2` fork exists beside them. This port creates no fork
  and inherits none.
- **RTA varargs:** **one genuine block** — the reader's mcast-sender coordinate tables. `reader:61-64` takes
  `tt_l1_ptr uint32_t*` pointers into the arg buffer (`get_arg_addr(i)`) and advances `i` by the runtime counts
  `in1_mcast_sender_num_x` / `in1_mcast_sender_num_y` (`:59-60`; host `:393-394` = the mcast sender grid's x / y
  extents, `:123-137`, appended as two vectors `:396-399`); the tables are then read by index in the
  coordinate-precompute loops (`:96-97`, `:107-108`). Two homogeneous collections with non-literal counts →
  **vararg** (`num_runtime_varargs = num_x + num_y`, read with `get_vararg(i)` — the pointer walk becomes indexed
  reads; x entries first, then y). The two **counts** are distinct fields → **named**. Everything else is named:
  the reader's remaining 30 scalars read once each at the top (`:17-60`; slot 2 becomes the tensor binding, the
  two semaphore ids become `sem::` bindings), the writer's 15 (`:15-39`; slots 1-2 become tensor bindings), the
  compute kernel's 14 (`:20-39`). No CTA varargs.
- **Anything else the porter needs:**
  - **Full-grid kernels with flag-steered idle cores.** All three kernels, all six CBs and both semaphores are
    placed on the whole device compute grid (`:60-62`); the run-arg loop walks every device core
    (`grid_to_cores_with_noop`, `:332-337`) and sets `has_work_for_q_heads` / `has_work_for_mcast_kv_heads` per core
    (`:352-353`), with `num_output_blocks_per_core = 0` beyond `num_cores` (`:347-349`). Reproduce the shape:
    full-grid `WorkUnitSpec` (minus the M nodes, above), per-core named run-args with the same values. The
    `Buffer*` slots simply vanish — bindings are per kernel.
  - **Loopback L1→L1 copies — keep as is.** Reader `:189-196` (sharded `in1` → `in1` staging copy, under
    `IN1_SHARDED`) and writer `:134-139` (untilized row gather from `intermed0` into `intermed1`) both use a
    `UnicastEndpoint` addressed with `my_x[noc.get_noc_id()]` / `my_y[…]` plus a peeked CB address. This is the
    Device 2.0 migration guide's *"Avoid"* form (guide `:526-551`); the preferred local-handle form is a transfer
    rewrite, out of port scope. Only the peeks that feed them change spelling (`CircularBuffer` → `DataflowBuffer`).
  - **The DFB is the multicast destination.** `noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(mcast_src, cb_in1_obj, …)`
    (`:243-253`) and the two plain multicasts (`:257-267`, `:272-282`) write into `in1` at `.offset_bytes = 0`
    from a `CoreLocalMem` source built on a peeked address (`:239` the `in2` read pointer, `:271` the `in1` write
    pointer). A `DataflowBuffer` is a valid NoC destination exactly as the `CircularBuffer` was; keep the calls.
  - **Cursor peeks.** Reader `:169` (`in1` write pointer, taken after `reserve_back` each iteration) and `:123`
    (`in2` read pointer, taken once at entry — static because the borrowed shard never moves); writer `:116`
    (`intermed1` write pointer after `reserve_back`) and `:132` (`intermed0` read pointer after `wait_front`).
    All public peeks (`cb_dfb_api_whitelist.md` §C) — same method names on `DataflowBuffer`. No cursor surgery
    anywhere.
  - **Writer CTA 0 is a CB index.** `output_cb_index` rides as the writer's first CTA (`:256` → `writer:41`) →
    becomes `dfb::<out>` (whitelist rule 2), not a named arg; `out_subblock_w` and `intermediate_num_tiles`
    (`writer:42-43`) are named CTAs. Reader CTAs `transpose_hw`, `row_major`, `out_subblock_w` (`reader:66-68`) and
    compute CTAs `transpose_hw`, `out_subblock_w`, `out_subblock_num_tiles`, `intermediate_num_tiles`
    (`compute:42-45`) are named; the accessor tails (`:253`, `:260-261`) go away.
  - **Defines pair with the conditional resources.** `IN0_SHARDED` → writer (`:265-267`), `IN1_SHARDED` → reader
    (`:268-270`), `OUT_SHARDED` → writer (`:271-273`) become `compiler_options.defines`; each one's condition is
    also the condition of a `borrowed_from` DFB and the absence of the matching `TensorBinding`.
  - **Compute `hw_config`: Style A with dropped fields.** The op resolves the TTNN config
    (`get_compute_kernel_config_args`, `:34-35`) but copies only `math_fidelity` and `fp32_dest_acc_en` onto the
    `ComputeConfigDescriptor` (`:317-320`); `math_approx_mode` and `dst_full_sync_en` are resolved and dropped
    (descriptor defaults `false` / `false`), `packer_l1_acc` has no counterpart. Translate with
    `to_compute_hardware_config(device->arch(), config)`, then force `sfpu_precision_mode = Precision::Precise`
    and `double_buffer_dest = true` on the returned `ComputeGen1Config` regardless of the caller's values (legacy
    behaviour, preserved not repaired). **`unpack_modes`:** the legacy descriptor sets no `unpack_to_dest_mode`
    (`Default` → `UnpackMode::UnpackToSrc`); the validator requires an explicit entry for every Float32 DFB the
    compute kernel consumes while `enable_32_bit_dest` is on. The compute kernel consumes `in0`, `in1` and
    `intermed1`: `in0` / `in1` are Float32 when the input dtypes are (`:37-38`; validation then forces
    `fp32_dest_acc_en`, `:48-51`), `intermed1` carries `interm_data_format`, Float32 exactly when
    `fp32_dest_acc_en && in0 == Float32` (`:39-41`). Gate each entry on that DFB's format being Float32.
    `test_group_attn_matmul_fp32` (`test_attn_matmul.py:462`, `in_dtype = float32`) exercises the branch.
  - **`opt_level`:** none set anywhere (`grep -n opt_level` on the factory returns nothing) → the compute spec
    resolves to **O3** and must say `KernelBuildOptLevel::O3` explicitly; both DM specs resolve to O2.
  - **DM `hw_config`: value-match the defaults, but note the intent.** Reader = `RISCV_1` + `preferred_noc_for_dram_read(arch)`
    (`:275`, `:287-290`), writer = `RISCV_0` + the other NOC (`:276-277`, `:300-303`). The helper returns `NOC_0`
    for every arch today (`kernel_types.hpp:140-145`), so the resolved triples are exactly the reader / writer
    defaults (`DM_DEDICATED_NOC`) → the arch-agnostic `create_reader_datamovement_config` /
    `create_writer_datamovement_config` helpers apply. The legacy code *intends* an arch-dependent choice; record
    in the port report that the match is by value as of this commit.
  - **`constexpr` metadata:** none from CB metadata. `reader:78` `in1_tile_bytes`, `writer:61` `in0_tile_bytes`
    and `writer:68` `out_tile_bytes` are runtime `const` → DFB getters (`writer:59` is the dead one above).
  - **LLK calls take `dfb::name` directly** (*Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers*):
    `compute_kernel_hw_startup<SrcOrder::Reverse>(c_0, c_1, c_3)` (`compute:64`), `matmul_init` (`:67`),
    `pack_untilize_dest_init<N>(c_3)` / `pack_untilize_dest<N>(c_3)` / `pack_untilize_uninit(c_3)` (`:83`,
    `:147-148`), `matmul_tiles` (`:116`), `reconfig_data_format_srca` / `pack_reconfig_data_format` (`:160-161`),
    `tilize_init_short_with_dt` / `tilize_block` / `tilize_uninit` (`:167-168`, `:172`) — all `uint32_t` cb ids.
    No kernel-lib helpers in this op.
  - **Stale hash comments — do not carry them.** `group_attn_matmul_program_factory.hpp:217-222` and
    `group_attn_matmul_program_factory.cpp:155-158` say `padded_shape` is folded into `compute_program_hash()`;
    that custom hash was removed by PR #47232 and the op is on the default reflection hash (which keys on the full
    `TensorSpec`s, padded shapes included — so the shape-keyed CB capacities `:165`, `:180`, `:209`, `:223`,
    `:234` are still safe). `:58-59` additionally claims CB sizes are re-applied on cache hit, which `:155-158`
    itself contradicts.
  - **Tests (no-regression baseline, confirm with the invoker):**
    `tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py` — `test_group_attn_matmul` (`:228`;
    bf8 inputs, `q_heads` 32 / 32 / 16, all eight sharding combinations), `test_group_attn_matmul_with_program_cache`
    (`:327`), `test_group_attn_matmul_fp32` (`:462`; float32 + bf16, all-sharded, `q_heads` 32 / 32 / 2),
    `test_group_attn_matmul_with_program_cache_exhaustive` (`:630`; `q_heads` 16-64, every sharding / orientation /
    buffer-type / optional-output combination). The `test_attn_matmul*` cases in the same file are a different op.
    Production callers: `models/demos/falcon7b_common/tt/falcon_attention.py:698`, `:805`;
    `models/demos/ttnn_falcon7b/tt/falcon_attention.py:167`, `:230`;
    `models/demos/t3000/falcon40b/tt/falcon_attention.py:520`, `:575`.

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean** — no op donor, no kernel-lib helper, no
  borrowed kernel file; every escape is framework API.

  | Op kernel | Includes | Class | Functions / types used | Shape |
  |---|---|---|---|---|
  | reader | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h`, `api/dataflow/noc_semaphore.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h` | `tt_metal/*` | `Noc`, `CircularBuffer`, `Semaphore<>`, `TensorAccessor`, `UnicastEndpoint`, `CoreLocalMem`, `get_tile_size`, `get_arg_val` / `get_arg_addr`, `my_x` / `my_y`, `VALID` / `INVALID` | ✓ framework API |
  | writer | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h` | `tt_metal/*` | `Noc`, `CircularBuffer`, `TensorAccessor`, `UnicastEndpoint`, `CoreLocalMem`, `get_tile_size`, `my_x` / `my_y` | ✓ framework API |
  | compute | `api/compute/compute_kernel_hw_startup.h`, `api/compute/tile_move_copy.h`, `api/compute/matmul.h`, `api/compute/tilize.h`, `api/compute/pack_untilize.h`, `api/dataflow/circular_buffer.h` | `tt_metal/*` (LLK) | `compute_kernel_hw_startup`, `matmul_init`, `matmul_tiles`, `pack_untilize_*`, `reconfig_data_format_srca`, `pack_reconfig_data_format`, `tilize_*` | ✓ `uint32_t cb_id` |

  Borrowed kernel files: none. This op lends nothing (no other factory references its kernels).
- **Relaxation candidates (FALLIBLE — candidates to verify, default strict):** the custom hash removed by
  PR #47232 is not visible in this shallow clone; the surviving comments (`group_attn_matmul_program_factory.hpp:217-222`)
  say it folded `padded_shape` in, i.e. it was already shape-strict, so the default hash is equivalent and the
  sheet's `none` is consistent. The kernels take `Mt` / `Kt` / `Nt` / `KV_HEADS`-derived counts as runtime args,
  but five CB capacities are shape-keyed (`in0` `Kt` tiles `:165`, `in1` `2·KV_HEADS·Kt·out_subblock_w` `:180-182`,
  `intermed0` `2·out_subblock_w` `:209`, `intermed1` / `out` `MtNt` `:223`, `:234`) and the mcast geometry RTAs
  depend on `q_heads` — so every new `(q_heads, kv_heads, head_dim, kv_len)` costs a program build. A shape
  relaxation would need the custom concept (geometry RTAs go stale on a hit under `ProgramSpecFactoryConcept`)
  plus CB re-sizing; ops-team call. The sheet's `Pointer patching perf issue? = suspect perf regression (+ fixed latent bug)`
  most likely tracks this cost; the "fixed latent bug" could not be attributed from the available history.
- **TTNN factory analysis:** single-program SPMD; no op-owned tensors; no custom hash (default reflection hash;
  `Formerly custom hashed? = yes`, removed by PR #47232); no `get_dynamic_runtime_args`; no pybound
  `create_descriptor`; no `override_runtime_arguments` → **`ProgramSpecFactoryConcept`**. Tensors reachable from
  `tensor_args` (`input_tensor_a`, `input_tensor_b`, optional `preallocated_output`) and `tensor_return_value`.

  ### TTNN ProgramFactory
  #### Concept
  `ProgramSpecFactoryConcept` (ported-from factory has no `override_runtime_arguments`).
  #### Fit
  - Single vs multi-program: single — one `ProgramDescriptor` today, one `ProgramSpec` stamped across the mesh.
  - Op-owned device resources: none.
  - Tensor-arg matching: strict (sheet `none`).
  - Legacy-to-Metal-2.0 shape: 1:1 with the ported-from factory, plus the M-node `WorkUnitSpec` split described
    under *Heads-ups* (a placement refinement, not a second factory).
  #### Custom `compute_program_hash`
  none — default reflection-based hash (the custom one was removed by PR #47232).
  #### Stop signals
  None.

## Misc anomalies  *(team-only, non-gating)*

- `group_attn_matmul_program_factory.hpp:217-222` and `group_attn_matmul_program_factory.cpp:155-158` — comments
  claim `padded_shape` is folded into `compute_program_hash()`; the custom hash was deleted by PR #47232. The
  substance survives (the default hash keys on the tensor specs) but the reference is dangling. `:58-59` says
  the cached program "covers shape variation via CB-size re-application on cache hit", which `:155-158` directly
  contradicts (CB sizes are *not* re-applied; the hash keeps each sizing in its own entry).
- **The KV-cache path is unreachable from the public API.** `ttnn::experimental::group_attn_matmul` passes
  `num_tokens = nullopt`, `transpose_hw = nullopt` (`group_attn_matmul.cpp:79-80`); the nanobind module binds only
  that wrapper (`group_attn_matmul_nanobind.cpp:105`); nothing else calls `ttnn::prim::group_attn_matmul`. So the
  validation branch `device/group_attn_matmul_device_operation.cpp:133-157`, the `N = num_tokens` override in
  `compute_output_specs` (`:172-174`), the `transpose_hw` CTA (`reader:66`, `compute:42` → `matmul_init` `:67`),
  `in1_Kt` / `in1_KtNt` arithmetic (`group_attn_matmul_program_factory.cpp:71-75`) and the reader's
  `in1_CKtNt_mul_32` stride (`reader:29`, `:347`) are dead code from the public surface (the sibling `attn_matmul`
  exposes its equivalent as `attn_matmul_from_cache`; this op never grew the wrapper).
- `reader_mcast_transformer_group_attn_matmul.cpp:75`, `:123` — in the interleaved-`in1` build the reader
  constructs a `CircularBuffer` for index 2 and reads its pointer although no CB 2 is configured in that program;
  benign today (the value is only consumed under `IN1_SHARDED` or inside a branch only that config enables), but a
  read of an unconfigured CB interface nonetheless. Becomes a compile error after the port unless gated (see
  *Heads-ups*).
- `writer_transformer_group_attn_matmul.cpp:59` — `in1_tile_bytes = get_tile_size(cb_id_in1)` is never used; the
  writer never touches CB 1. `:57` `onetile` and `compute:60` `onetile` are also unused.
- `transformer_group_attn_matmul.cpp:90-136` — the `#ifdef ARCH_GRAYSKULL` branch (`:90-106`) is identical to the
  `#else` branch (`:107-122`); Grayskull is no longer a target. `:68` and `:123-135` carry a commented-out
  `matmul_block` path ("switch back after didt solved").
- **Runtime args that are constants.** `in1_start_id` is always `0` (`:368`); `in1_num_subblocks` is a
  `constexpr 1` passed as an RTA to the reader and compute kernels (`:80`, `:372`, `:435`); `out_subblock_h` is a
  `constexpr 1` passed to compute (`:79`, `:434`); `num_output_blocks_per_core` is forced to `1` on every active
  core by the `TT_FATAL` at `:107-110`, so the `blocks` / `batch` outer loops (`reader:142`, `writer:91`,
  `compute:70`) run exactly once. The kernels' `// TODO: Must be 1` comments acknowledge this. Preserved
  behaviour — the port names these args and changes no values.
- `group_attn_matmul_program_factory.cpp:34-35` vs `:317-320` — `math_approx_mode`, `packer_l1_acc` and
  `dst_full_sync_en` are resolved from the caller's `compute_kernel_config` and silently ignored; a caller cannot
  change them. Preserved behaviour, reported rather than repaired.
- `group_attn_matmul_program_factory.cpp:351` — the `KV_HEADS == 0` guard is unreachable: `KV_HEADS = bshape[1]`
  already served as a modulus in validation (`device/group_attn_matmul_device_operation.cpp:39`).
- `group_attn_matmul_program_factory.cpp:92-94` — `ONE_ROW_BFLOAT16_BYTES` is `128` under fp32 accumulation; the
  name describes only the bf16 case.
- `group_attn_matmul_program_factory.cpp:253` — the `in1` `TensorAccessorArgs` tail is appended to the reader
  CTAs even when `in1` is sharded and the kernel never instantiates the accessor (`reader:81-84`); likewise the
  `in0` / output tails (`:260-261`) under `IN0_SHARDED` / `OUT_SHARDED`. Harmless today; all three vanish in the port.

## Per-DeviceOperation attribution

Single `GroupAttnMatmulDeviceOperation`, single factory — not bundled; the status summary is the attribution.

## Questions for the user

1. **`in1` on mcast-only nodes — confirm the node-set split is in scope.** The audit's disposition adds a second
   reader `KernelSpec` and a second `WorkUnitSpec` (M nodes only, present only when `q_heads < 32`) and stops
   launching compute / writer on those nodes, where today they return at their first statement
   (`compute:20-21`, `writer:15-18`). Kernel code is untouched and no observable behaviour changes, but it is a
   host-side placement change the recipe does not spell out for this shape. The alternatives are
   `allow_instance_multi_binding` on `in1` (discouraged, unsafe, Gen2-forbidden —
   `advanced_options.hpp:227-244`), or binding the reader PRODUCER-only everywhere and relying on Gen1's
   role-agnostic CB lowering to let its `pop_front` (`reader:328`) go through on M nodes — the kernel-side token
   carries no role (`tt_metal/hw/inc/api/dataflow/dfb_binding_token.h`), so that would compile and run on Gen1,
   but no document sanctions a consumer FIFO op from a producer binding, and the audit did not assume it.
2. **Dropping the writer's dead `get_tile_size(cb_id_in1)` line (`writer:59`).** It cannot be rewritten per
   whitelist rule 7 (the writer has no `in1` binding, so there is neither a token nor an object) and is unused.
   Confirm a dead-local drop is acceptable port scope; the only alternative is a cosmetic CONSUMER binding of `in1`
   on the writer, which the audit does not recommend (it would be a third toucher on Q nodes).

## Recipe notes

1. **Endpoint roles that vary per *node* inside one configuration have no row in the census table.** The
   `CB endpoints` subject classifies per `(CB, config)` and resolves a one-toucher CB with a self-loop, but here
   the same CB is a plain 1:1 on q-work nodes and a reader self-loop on mcast-only nodes of the *same* program
   (`reader:322-329`). The mechanism that expresses it — same-source `KernelSpec`s with different `dfb_bindings`
   over disjoint `WorkUnitSpec`s — is documented only inside the *Why wrong* paragraph of *Anti-pattern: Demoting
   per-group CTA to RTA*. A first-class pattern entry ("per-node-set bindings") and a sentence in the audit's
   census table ("if roles differ per node set, split the `KernelSpec` by node set; never flag") would remove the
   judgment call. Mcast senders without compute work are a recurring shape in matmul-family ops.
2. **A metadata-only touch is not in the census vocabulary.** `writer:59` reads `get_tile_size(cb_id_in1)` on a CB
   the writer never accesses. The census counts "any access" to the *memory*; a metadata read is not one, yet
   post-port it needs either a binding (a token) or a drop. The recipe should say which: this audit chose "drop
   if dead, else bind cosmetically" and filed it as a question.
3. **Conditional DFB whose legacy kernel references it outside the config `#ifdef`.** *Pattern: Conditional /
   optional resource bindings* assumes the legacy kernel already gates its uses; here the object construction
   (`reader:75`), the peek (`:123`) and a dead runtime branch (`:238-254`) sit outside `#ifdef IN1_SHARDED`. The
   pattern could add: "also gate the declaration and any dead-branch references, or provide an `#else`
   fallback value, when the legacy kernel referenced the resource unconditionally".
4. **The Device 2.0 gate still has no wording for the migration guide's "Avoid" forms** (already filed by the
   `attn_matmul` audit); this op carries the same hand-built loopback endpoint twice (`reader:117-119` /
   `:189-196`, `writer:87-89` / `:133-139`), judged compliant-but-discouraged again.
5. **Provenance on a shallow clone / orphan recipe branch.** This checkout is shallow (`.git/shallow`), so
   `git log --follow` on the op shows two commits and the PD-migration / Device 2.0 PRs had to be recovered with
   `gh api …/pulls/<n>/files`; and the recipe's provenance command prints nothing because the docs live only on
   the orphan `akertesz/op-porting-recipe` branch. Both worked around; worth a line in `workspace_setup.md`.
6. **Sheet row numbers** (already filed): the header cells contain embedded newlines, so `grep -n` line numbers
   are off from sheet rows; parsing with a CSV reader and using the record index gives row 193 here.
