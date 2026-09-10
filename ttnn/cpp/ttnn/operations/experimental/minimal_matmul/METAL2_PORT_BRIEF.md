# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/minimal_matmul`

> ⚠ **PROVISIONAL — issued under invoker override, not by a passing audit.** The audit
> (`METAL2_PREPORT_AUDIT.md`, 2026-09-10) is **RED**, so the recipe would normally issue no brief.
> The RED is a stale readiness-sheet row, not a code finding: every code-side gate cleared, and the
> audit ran all seven informational subjects in full because a sheet-only RED clears without
> touching the op's code. The invoker (`iwrosz@tenstorrent.com`) instructed the port to proceed and
> accepted responsibility; see `METAL2_PORT_REPORT.md` → *Precondition* for the full record and for
> what would invalidate this brief. Content below is the audit's PORT WORK + FYI-P findings; only
> the authorization is irregular.

**Gates cleared:** Device 2.0 ✓ (see caveat below) · Features ✓ · Offset base pointers ✓ ·
TensorAccessor 3rd arg ✓ · TTNN factory concept ✗ **(sheet stale — the override)**

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry into the port report's Provenance section)*

**Device 2.0 caveat.** Cleared for every configuration this factory emits. The shared kernel files
do carry non-Device-2.0 code, but only inside `MM_WINDOW_BLOCKS` / `FUSE_AG` /
`SRS_FUSE_OP_SIGNALER` regions that **only the legacy CCL emitter defines** — this factory never
does. You will see those regions when you open the kernels; they are not yours to convert and not a
stop signal. Full evidence in the audit's Gate detail, and the alternative (per-file) reading is
recorded there as an open question.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor` — `create_descriptor` returning `ProgramDescriptor`, declared at
  `device/minimal_matmul_device_operation.hpp:29-32`, body at
  `device/minimal_matmul_program_descriptor.cpp:170`.
- **Op-owned tensors:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** (the base concept) — **an invoker-authorized
  deviation from the recipe's rule.** The recipe maps "factory has an `override_runtime_arguments`"
  → `CustomProgramSpecFactoryConcept`, and this factory has one
  (`device/minimal_matmul_device_operation.hpp:34-39`, body at
  `device/minimal_matmul_program_descriptor.cpp:951`). The override is **deleted**, not translated,
  and the framework's automatic tensor-binding refresh replaces it.

  **Why this is safe here.** The rule's stated purpose is to avoid "silently discarding every
  *non-tensor* refresh the op relied on." This override has none: every write in `:1019-1027` is a
  buffer address (in0/in1/in2/in3, both ternaries, all N outputs). It touches no scalar run-arg and
  never touches the compute kernel. The base concept refreshes exactly that set automatically, so
  the end state is identical. Verified no external caller: SRS calls the *legacy*
  `MinimalMatmulProgramFactory::override_runtime_arguments`
  (`minimal_matmul_strided_reduce_scatter_async_program.cpp:165`), a different symbol in a different
  struct, untouched by this port.

  **What a reviewer should know.** This trades mechanism fidelity for simplicity, which the port's
  zero-change invariant would normally forbid. Decision and rationale are recorded in
  `METAL2_PORT_REPORT.md` → *Concept deviation*. The concept is selected by the compiler via
  `detail::HasSpecRuntimeArgsOverride` (`ttnn/api/ttnn/operation_concepts.hpp:111-115`), keyed on
  the return type, so simply **not** declaring an override lands the factory on the base concept.
- **Factory struct already exists.** `program_factory_t = std::variant<ProgramFactory>`
  (`device/minimal_matmul_device_operation.hpp:42`), so `ttnn_factory.md` §3 (*give a
  direct-descriptor op a conventional program factory*) does **not** apply — this is a method swap
  inside the existing nested struct.
- **Gate-cleared, confirmed absent:** a `TensorParameter relaxation` that is neither `none` nor an
  analysis pointer (the cell reads `none`) · `get_dynamic_runtime_args` · a pybound
  `create_descriptor` · a custom `compute_program_hash` and its `attribute_values` / `to_hash`
  backdoor. Nothing to preserve on the cache key.

## Construct — to do

**Tensor bindings** — **all Case 1** (fed to a `TensorAccessor`; no Case 2, no borrowed-memory DFB
anywhere in this op). Express each as a `TensorParameter` / `TensorBinding`; the kernel builds
`TensorAccessor(tensor::name)` and the legacy address-via-RTA plus its `TensorAccessorArgs` chain
both disappear.

| Binding | Present when | Legacy delivery | Kernel site |
|---|---|---|---|
| `in0` (activation) | always | `Buffer*` RTA idx 0, both in0 kernels | `dm_in0_sender.cpp:68` |
| `in1` (weight) | always | `Buffer*` RTA idx 0, both in1 kernels | `dm_in1_sender_out.cpp:67` |
| `in2` (bias) | `FUSE_BIAS` | `Buffer*` RTA idx 1, all four DM kernels | `dm_in0_sender.cpp:79`, `dm_in1_sender_out.cpp:78` |
| `in3` (`optional_input_tensor`) | `IN0_VIRTUAL_CONCAT` | `Buffer*` RTA idx 2, in0 kernels only | `dm_in0_sender.cpp:195` |
| `ternary_a` | `FUSE_TERNARY` | `Buffer*` RTA idx 14 / 13 | `dm_in0_sender.cpp:117`, `dm_in1_sender_out.cpp:97` |
| `ternary_b` | `FUSE_TERNARY` | `Buffer*` RTA idx 15 / 14 | `dm_in0_sender.cpp:118`, `dm_in1_sender_out.cpp:98` |
| `out[0..N_chunks)` | always (N ≥ 1) | `Buffer*` RTA tail | `matmul_dataflow_common.hpp:43` |

All arrive as the **`Buffer*`-binding form** (`emplace_runtime_args` with a `Buffer*`, not
`->address()`) — routine port work, not the stale-address hazard. Note for the self-audit: the
factory contains **no** `->address()` in its cache-miss path, so the "no buffer address survived"
grep must search for `emplace_runtime_args` / bare `Buffer*` too, or it passes vacuously.

**TensorParameter relaxation:** `none` — nothing to apply.

**TensorAccessor 3rd arg:** drop the page-size *argument* at
`device/kernels/matmul_dataflow_common.hpp:33-34` (the only 3-arg accessor in the op; the other nine
construction sites are already 2-arg). Class 2 (inert). **Do not** set `dynamic_tensor_shape`: the
page size is compile-time-pinned by the output dtype, which is hashed.

> ⚠ **The CTA that feeds it does *not* drop with it — this op is an exception to the recipe's
> wording.** The recipe says the 3rd argument's "host-side CTA/RTA emission drops with it", which
> holds only when the CTA exists *solely* to feed the accessor. Here `out_tile_size` (CTA index 12,
> emitted at `device/minimal_matmul_program_descriptor.cpp:553, 599, 640, 681`) has **eight further
> uses in each DM kernel** — it is the `tile_size_bytes` stride for the L1 pointer walks in
> `write_block_sync`, `write_block_sync_split`, `write_block_sync_granular` and
> `write_block_sync_granular_split` (`dm_in0_sender.cpp:302, 312, 330, 340, 502, 516, 530, 540`;
> `dm_in1_sender_out.cpp:264, 274, 292, 302, 440, 454, 468, 478`). **Keep it as a named CTA**
> (`args::out_tile_size`). Dropping it would silently break every output write's addressing.
>
> Do **not** substitute `dfb::out.get_tile_size()` either: whitelist rule 7 covers metadata the
> kernel read *by cb id*, and this arrives as a CTA, so rule 7 does not apply.

**CB endpoints:**

| DFB | Legacy CB | Disposition |
|---|---|---|
| `in0` | `c_0` | plain 1:1 — in0-DM PRODUCER, compute CONSUMER |
| `in1` | `c_1` | plain 1:1 — in1-DM PRODUCER, compute CONSUMER |
| `out` | `c_2` | plain 1:1 — compute PRODUCER, output-writer DM CONSUMER **(conditional — see below)** |
| `intermediate` | `c_3` | **self-loop** — compute is the only toucher; bind it PRODUCER *and* CONSUMER |
| `in2` | `c_4` | plain 1:1 — non-writer DM PRODUCER, compute CONSUMER **(conditional ×2)** |
| `ternary_a` | `c_5` | plain 1:1 — non-writer DM PRODUCER, compute CONSUMER **(conditional)** |
| `ternary_b` | `c_6` | plain 1:1 — non-writer DM PRODUCER, compute CONSUMER **(conditional)** |

No multi-binding anywhere — **do not set `allow_instance_multi_binding`**. The census was re-derived
per node: each node hosts exactly three kernel instances (one in0-DM, one in1-DM, one compute; the
sender/receiver core ranges partition the grid, `minimal_matmul_program_descriptor.cpp:377-380`), and
no CB has more than one producer and one consumer. The two same-source `KernelDescriptor`s per DM
kernel cover **disjoint** node sets, so this is *not* the dual-instance work-split shape and there is
no two-toucher assignment to make either. Dispositions are **config-independent**:
`transpose_core_grid` swaps *which* DM kernel writes the output, but never the census.

**Conditional bindings — the bulk of the CB-side work.** Four sites reference a CB index on a kernel
instance (or in a configuration) that never touches it. All are inert today —
`CircularBuffer(cb_id)` merely stores the index — and all become compile errors or needless extra
bindings under Metal 2.0, where `dfb::name` exists only if the host binds it. Apply
*Pattern: Conditional / optional resource bindings*:

1. `dm_in0_sender.cpp:155` / `dm_in1_sender_out.cpp:136` — `cb_out` (`c_2`) is constructed
   unconditionally but every access is under `if constexpr (is_output_writer)` (CTA 17). **Promote
   that CTA gate to a define** and `#ifdef`-gate the construction. Binding `c_2` on the non-writer
   instance would give it a third binding per node and force the multi-binding flag for nothing.
2. `dm_in0_sender.cpp:157` / `dm_in1_sender_out.cpp:138` — same for `cb_in2` (`c_4`), which only the
   **non**-writer instance fills (`if constexpr (!is_output_writer)`).
3. `dm_in0_sender.cpp:111-112` / `dm_in1_sender_out.cpp:92-93` —
   `get_tile_size(cb_ternary_a_id)` / `(cb_ternary_b_id)` are evaluated on **both** instances, but
   their only consumer `read_ternary_blocks_sync` runs under `if constexpr (!is_output_writer)`
   (`dm_in0_sender.cpp:457`, `dm_in1_sender_out.cpp:401`). Under whitelist rule 7 these become
   `dfb::ternary_a.get_tile_size()`, so left where they are they force the writer instance to bind
   `c_5`/`c_6` it never touches. Sink them into the `!is_output_writer` scope.
   *(Both are legacy `constexpr`, so rule 7's carve-out applies — keep the free-function form with
   the binding token, `get_tile_size(dfb::ternary_a)`. Do not demote to `const` to fit a getter.)*
4. `compute.cpp:440` — `CircularBuffer cb_in2(in2_cb)` is constructed **unconditionally**, and
   `in2_cb` (= `c_4`) is passed to `swiglu_block` (`:535`) and `add_bias_and_addcmul_block` (`:557`),
   and re-wrapped at `:157`, regardless of `FUSE_BIAS`. Without bias the host allocates no `c_4` at
   all (`minimal_matmul_program_descriptor.cpp:437-440`), so `dfb::in2` will not exist. `#ifdef
   FUSE_BIAS`-gate all four references. `ternary_a_cb` / `ternary_b_cb` at `compute.cpp:433-434`
   need the same under `FUSE_TERNARY`.

**Semaphores:** six, all plain (`SemaphoreSpec`). The descriptor hard-codes ids 0-5 as literals
(`minimal_matmul_program_descriptor.cpp:388-408`) and passes them as CTAs 14-16
(`:555-557`); both halves are replaced by named `sem::` bindings. The in0 kernels use ids 0/1/2, the
in1 kernels 3/4/5 — each kernel binds three of the six.

**`override_runtime_arguments` — the set to mirror.** It refreshes **every tensor binding and
nothing else**: `in0`/`in2`/`in3` on the in0 kernels, `in1`/`in2` on the in1 kernels,
`ternary_a`/`ternary_b` and all N outputs on both (`:1019-1027`). It deliberately refreshes no
scalar run-args and never touches the compute kernel — everything else derives from hashed inputs
(`:946-948, 963-966, 906-907`). So the ported override's `tensor_args` names every `TensorParameter`
and its `kernel_run_args` stays **empty**. Mirror that set exactly, in both directions.

## Watch for

- **Shared kernels — this port creates the first `_metal2` forks.** All four kernel sources are bound
  by a *second* emitter that will not convert with you: `minimal_matmul_factory_helper_common` in
  `device/minimal_matmul_program_factory.cpp` (same file paths, at `:551, 596, 635, 674, 704`). This
  is the **Lent / Intra-op** shape. `ls device/kernels/` shows **no `_metal2` sibling**, so you are on
  **rung 2**: fork `dm_in0_sender.cpp`, `dm_in1_sender_out.cpp`, `compute.cpp` and
  `matmul_dataflow_common.hpp` beside the originals, convert the copies, leave the originals plus
  the pointer comment. **Sunset list — not authorization to convert in place:**
  `experimental/ccl/minimal_matmul_strided_reduce_scatter_async`.
- **A variable number of output tensor bindings — use a `TensorBindingSequence`.** `N_chunks` (the
  `chunks` attribute — 1 for `minimal_matmul`, N for `minimal_matmul_split`) sets the number of
  output tensors, and every DM kernel builds a `TensorAccessor` tuple of exactly that size
  (`matmul_dataflow_common.hpp:42-47`, via `make_tensor_accessor_args_tuple<N_chunks, …>`). The count
  is a CTA — fixed per instantiation, varying across them — so the outputs cannot be named
  individually in kernel source. **Metal 2.0 has a purpose-built mechanism for exactly this**, and it
  is a near drop-in:

  `KernelAdvancedOptions::tensor_binding_sequences`
  (`tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp:112-152`) — *"a kernel
  that wishes to express a compile-time-variadic number of tensor bindings, and therefore needs to
  access them positionally."*

  - **Host:** declare `TensorParameter`s `out0 … out{N-1}` and their `TensorBinding`s on each DM
    kernel in a loop, then add one
    `TensorBindingSequence{.sequence_name = "outputs", .members = {"out0", …}}`. Codegen emits
    `tensor::outputs` as a `std::tuple` of the binding tokens, in the declared order.
  - **Kernel:** `auto outputs_tuple = make_tensor_accessors(tensor::outputs);`
    (`tt_metal/hw/inc/api/tensor/tensor_accessor.h:679`) returns
    `std::tuple<TensorAccessor<…>, …>` — the **same shape** the kernel already has, so
    `write_block_sync_split` / `write_block_sync_granular_split` and the
    `write_tile_to_chunk` fold over `std::index_sequence_for<Accessors...>` keep working unchanged.
    The whole of `make_tensor_accessor_tuple_uniform_page_size` (`matmul_dataflow_common.hpp:25-47`)
    is deleted, and the 3rd-arg page size goes with it.
  - **No count argument needed** — `std::tuple_size_v<decltype(tensor::outputs)>` gives it. Keep the
    existing `N_chunks` CTA anyway: it is used for chunk *index arithmetic*
    (`N_tiles_per_chunk`, the `chunk_idx >= N_chunks` padding guard), not only as a tuple size, so
    removing it would be a rewrite rather than a swap.
  - **Do not** reach for `make_abstract_tensor_accessor_wrappers` (the type-erased array form). It
    would let the writer index accessors at runtime and delete the fold — a tidier kernel, but a
    behavioural rewrite outside the syntax-swap invariant, and it forfeits the concrete
    `TensorAccessor<DSpec>` dispatch the current code deliberately preserves.

  **This op appears to be the mechanism's first production user** — `grep` finds
  `tensor_binding_sequences` only in the framework itself (`kernel.cpp`, `kernel.hpp`), not in any
  ported op. Expect to shake it out, and report friction.
- **RTA varargs: none survive.** Both DM kernels read a fixed `argidx++` run of 14 (in0) / 13 (in1)
  distinct scalar fields, +3 under `FUSE_TERNARY`; compute reads 4 (+2). **All nameable.** The only
  variable-count run is the N output addresses at the tail — and those are buffer addresses, so they
  leave the runtime-arg channel entirely and become the binding set above. **Do not port them as an
  RTA vararg block.** No CTA-vararg site either: the `TensorAccessorArgs` CTA blocks are read at
  constexpr offsets and vanish wholesale.
- **Two dead legacy args — carry them, do not clean them.** `in3_tile_size` (in0 CTA index 21) is
  never read by the kernel, and `max_defer_write_k_block` (RTA idx 13, both DM kernels) is used only
  under `SRS_FUSE_OP_SIGNALER`. Both are recorded as anomalies in the audit and route to the ops
  team. They are *not* the port's to remove — dropping either is a functional change to the arg
  layout. (Contrast with the dead-CB rule, which does sanction a drop; that does not extend here.)
- **`opt_level`.** The legacy factory sets none anywhere — `grep -n opt_level` over both emitters
  returns nothing. So the compute `KernelSpec` **must** be given an explicit
  `KernelBuildOptLevel::O3` (legacy `ComputeConfigDescriptor` resolves to `O3`; Metal 2.0 defaults to
  `O2`). The four DM specs need nothing — their legacy `O2` already matches. This is the
  absent-line failure mode: there is nothing in the ported code to read and object to.
- **`hw_config` — Style A compute config, with a dropped field.** The op resolves a TTNN
  `ComputeKernelConfig` via `get_compute_kernel_config_args`
  (`minimal_matmul_program_descriptor.cpp:221`), so use
  `to_compute_hardware_config(device->arch(), config)` — **but it resolves five fields and sets
  only three** (`:737-740`: `math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode`).

  | resolved at `:221` | set on the descriptor | port action |
  |---|---|---|
  | `math_fidelity` | yes | helper carries it |
  | `math_approx_mode` | yes | helper carries it (`bool` → `Precision`) |
  | `fp32_dest_acc_en` | yes | helper carries it (→ `enable_32_bit_dest`) |
  | `packer_l1_acc` | **no** | no Metal 2.0 counterpart — no action |
  | `dst_full_sync_en` | **no** | ⚠ **dropped field** — the descriptor's own default is `false` (`program_descriptors.hpp:104`), so the op always ran `dst_full_sync_en = false` whatever the caller asked for. Assign **`double_buffer_dest = true`** by hand on the returned config; do *not* let the helper translate the caller's value, or a caller that sets the knob silently changes behaviour |

  This is preserved behaviour, not a bug to fix — a silently-ignored parameter. It is already
  recorded for the ops team.

  **`unpack_modes` needs a newly-required entry.** The intermediate DFB's format is the classic
  `fp32_dest_acc_en ? Float32 : Float16_b` idiom (`:225`), so whenever that flag is on, `c_3` is a
  **Float32 DFB consumed by the compute kernel with `enable_32_bit_dest = true`** — which Metal 2.0
  *requires* an explicit entry for and legacy did not. The op sets no `unpack_to_dest_mode` vector at
  all, so the legacy value is `Default` → **`UnpackMode::UnpackToSrc`**. Gate the entry on the same
  condition as the binding. `bfp8_pack_precise` is unset (default `false`) → leave
  `bfp_pack_precision_mode` at its default; no action.

- **All four DM configs are custom — the helpers do not fit.** Resolved from `:289-304`, with
  `preferred_noc_for_dram_read = NOC_0` and `preferred_noc_for_dram_write = NOC_1` on every arch
  (`kernel_types.hpp:140-152`, Blackhole falls to `default`), and `noc_mode` left at the descriptor
  default `DM_DEDICATED_NOC` (`program_descriptors.hpp:97`):

  | | `transpose_core_grid == false` (M ≤ N) | `transpose_core_grid == true` (M > N) |
  |---|---|---|
  | in0 sender / receiver | `RISCV_1`, `NOC_1`, `DM_DEDICATED_NOC` | `RISCV_0`, `NOC_0`, `DM_DEDICATED_NOC` |
  | in1 sender / receiver | `RISCV_0`, `NOC_0`, `DM_DEDICATED_NOC` | `RISCV_1`, `NOC_1`, `DM_DEDICATED_NOC` |

  Compare against the recipe's defaults — reader is `RISCV_1`/`NOC_0`, writer is `RISCV_0`/`NOC_1`.
  **None of the four triples matches either default, in either orientation**: each pairs a reader's
  RISC with a writer's NOC or vice versa. So **do not** use `create_reader_datamovement_config` /
  `create_writer_datamovement_config` — replicate exactly with
  `DataMovementGen1Config{.processor = …, .noc = …, .noc_mode = …}`, keeping the same
  `transpose_core_grid` computation. (The node invariants hold either way: distinct RISCs, agreed
  `noc_mode`, distinct NOCs under dedicated mode — so a mistake here is silent, not a `TT_FATAL`.)
