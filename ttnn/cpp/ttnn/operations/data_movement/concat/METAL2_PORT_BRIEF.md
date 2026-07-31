# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/concat` *(2 of 6 factories)*

> **Scoped brief — read this box first.** The audit is **RED at the op level**. It cleared every gate
> for two of concat's six program factories, and the gate that fires is factory-scoped, so this brief
> is issued for the clean subset per the audit recipe's config-scoped-GATE rule.
>
> **In scope — port these two, and only these two:**
> - `ConcatS2SRMProgramFactory` (`device/concat_s2s_rm_program_factory.cpp` / `.hpp`)
> - `ConcatS2STiledProgramFactory` (`device/concat_s2s_tiled_program_factory.cpp` / `.hpp`)
>
> **Out of scope — blocked, do not touch:** `ConcatProgramFactory`, `ConcatS2IProgramFactory`,
> `ConcatS2SMultiProgramFactory`, `ConcatBlockShardedProgramFactory`. They stay on `create_descriptor`.
> The op's `program_factory_t` will hold a **mix** of concepts when you are done; that is expected and
> supported — `AllFactoriesValid` requires each variant alternative to satisfy exactly *one* concept,
> not the same one (`ttnn/api/ttnn/operation_concepts.hpp:174-189`), and the framework adapter
> dispatches per-alternative through `std::visit` (`ttnn/api/ttnn/device_operation.hpp:227-243`).
>
> The full record, including why the other four are blocked, is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared (for the two in-scope factories):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `74e2d788513 2026-07-31 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both in-scope factories
port to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — each factory is a `static ProgramDescriptor create_descriptor(...)`
  (`device/concat_s2s_rm_program_factory.hpp:14`, `device/concat_s2s_tiled_program_factory.hpp:14`).
  Both become `create_program_artifacts`.
- **Op-owned tensors:** none. `ConcatDeviceOperation::create_output_tensors` allocates only the single
  output (`device/concat_device_operation.cpp:182-186`).
- **Target concept:** `ProgramSpecFactoryConcept`, no op-owned tensors.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash ·
  `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported
  replacement) · pybind `create_descriptor` — all four gate conjuncts — plus other migration-risky
  pybind, which would have surfaced as a `safe` warning. All `no` on this op: `concat_nanobind.cpp`
  binds no device-op or factory internals, and `ConcatDeviceOperation`'s full static surface
  (`device/concat_device_operation.hpp:37-49`) carries none of those hooks. **So there are no
  device-operation-class edits to make** — the three sanctioned exceptions in `ttnn_factory.md` all
  come up empty here.

**The property that makes these two portable — preserve it.** Both factories are selected only when
`input_tensors.size() == 2` (`device/concat_device_operation.cpp:46-59`). That is what fixes their
argument and resource sets at spec-construction time, and it is exactly what the other four factories
lack. So declare **exactly two input `TensorParameter`s plus one output** — do not carry the
`for (input_id < num_input_tensors)` loop shape into the spec, even though the legacy factories write
their CB creation that way (`device/concat_s2s_rm_program_factory.cpp:53`,
`device/concat_s2s_tiled_program_factory.cpp:91`). Neither factory asserts the two-input property today
(they just index `input_tensors[1]`), so the loop is legacy habit, not a real N.

## Construct — to do

### Tensor bindings

**All bindings on both factories are `clean`** — there is no Case 1 and no Case 2 anywhere in this
subset. Neither factory sets a single runtime arg, and neither kernel constructs a `TensorAccessor`.
Every tensor reaches its kernels as a **borrowed-memory DFB**: the legacy `CBDescriptor::buffer` field
is set to the tensor's buffer, and the kernels read/write that memory by raw pointer off the DFB. The
causal-link gate applies — the DFB *is* the tensor access.

Port each as a `TensorParameter` with a `DataflowBufferSpec::borrowed_from` pointing at it:

| Factory | Binding | Legacy borrow site | Port |
|---|---|---|---|
| `ConcatS2SRMProgramFactory` | `input_0` (CB index 0) | `device/concat_s2s_rm_program_factory.cpp:69` | `TensorParameter` + DFB `borrowed_from` |
| `ConcatS2SRMProgramFactory` | `input_1` (CB index 1) | `device/concat_s2s_rm_program_factory.cpp:69` | same |
| `ConcatS2SRMProgramFactory` | `output` (CB index 16) | `device/concat_s2s_rm_program_factory.cpp:86` | same |
| `ConcatS2STiledProgramFactory` | `input_0` (CB index 0) | `device/concat_s2s_tiled_program_factory.cpp:102` | same |
| `ConcatS2STiledProgramFactory` | `input_1` (CB index 1) | `device/concat_s2s_tiled_program_factory.cpp:102` | same |
| `ConcatS2STiledProgramFactory` | `output` (CB index 2) | `device/concat_s2s_tiled_program_factory.cpp:116` | same |

`ConcatS2STiledProgramFactory` additionally allocates **four local (non-borrowed) DFBs** — indices 3
`input0_transpose`, 4 `input1_transpose`, 5 `concat`, 6 `output_transpose`
(`device/concat_s2s_tiled_program_factory.cpp:128-172`). Ordinary `DataflowBufferSpec`s, no tensor
behind them.

**`.address_offset` is nowhere set in this op** (default 0 at every borrow site), so every borrow is a
plain base — no offset to preserve.

**TensorParameter relaxation:** none. Keep strict tensor-arg matching; there is no custom hash and the
readiness sheet lists `none` on both rows.

**TensorAccessor 3rd arg:** none — neither factory's kernels construct a `TensorAccessor` at all.

### Compile-time and runtime args

- **No runtime args exist in this subset.** Both factories pass everything through
  `KernelDescriptor::compile_time_args` and set no `runtime_args` / `common_runtime_args`
  (`device/concat_s2s_rm_program_factory.cpp:104-164`,
  `device/concat_s2s_tiled_program_factory.cpp:190-205`). So there is no `runtime_arg_schema` and no
  `ProgramRunArgs::kernel_run_args` to build beyond the tensor arguments.
- **Every CTA becomes a named compile-time arg.** Each of the 14 entries in each list is read once as a
  distinct field, so name them all — no varargs anywhere in this subset.
- `ConcatS2SRMProgramFactory` builds **four** CTA lists, not one
  (`device/concat_s2s_rm_program_factory.cpp:104-164`): a reader/writer pair for the full-height cores
  and a second pair for the remainder core, selected at `:190-200`. Two of the 14 values are pure
  CB indices (`cb_dst_id`, `cb_ids[0]`, `cb_ids[1]` — entries 0, 12, 13) and **disappear** into the DFB
  bindings.
- `ConcatS2STiledProgramFactory` shares **one** 14-entry CTA list across all three of its kernels
  (`device/concat_s2s_tiled_program_factory.cpp:190-205`), so each kernel currently receives entries it
  never reads — the reader ignores entries 5 and 6, the writer ignores 0 through 4. Under named args,
  each `KernelSpec` declares only what its own kernel reads; drop the rest per kernel. Entries 0-6 are
  all CB indices and disappear into the DFB bindings either way.
- **Compute kernel `opt_level`:** `ConcatS2STiledProgramFactory`'s `ComputeConfigDescriptor`
  (`device/concat_s2s_tiled_program_factory.cpp:241-246`) sets `math_fidelity`, `fp32_dest_acc_en`, and
  `math_approx_mode` but no optimization level. Set one explicitly on the ported compute `KernelSpec`
  rather than inheriting a default.

### CB endpoints

**`ConcatS2SRMProgramFactory` — assign 1P+1C on all three DFBs.** This factory is the dual-instance
work-split: one kernel source
(`device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors.cpp`) pushed into **two**
`KernelDescriptor`s that differ only by `ReaderConfigDescriptor` / `WriterConfigDescriptor` and their
work-split CTAs, both over the same core range (`device/concat_s2s_rm_program_factory.cpp:166-188`).
Both instances hit every node, so every DFB has **two touchers per node** — and the kernel contains
**no FIFO operations at all**, so both touchers are role-free:

| DFB | Both instances do | Assign |
|---|---|---|
| `input_0` (idx 0) | raw read `input_dfb_0.get_read_ptr() + input_start_0` @ kernel `:45` | one instance PRODUCER, the other CONSUMER |
| `input_1` (idx 1) | raw read `input_dfb_1.get_read_ptr() + input_start_1` @ kernel `:70` | same |
| `output` (idx 16) | raw write `output_dfb.get_write_ptr() + output_stick_offset` @ kernel `:42` | same |

The role labels are cosmetic on Gen1 — they satisfy the spec validator's ≥1-producer/≥1-consumer rule
and drive no FIFO machinery these kernels ever invoke. **Do not reach for the multi-binding advanced
option**: two role-free touchers fit 1P+1C, and no third kernel touches any of these DFBs. Keep the
same assignment across both the full-height and remainder-core kernel pairs.

**`ConcatS2STiledProgramFactory` — six legal 1:1 DFBs plus one self-loop.**

| DFB | Producer | Consumer | Disposition |
|---|---|---|---|
| `input_0` (idx 0, borrowed) | reader `push_back` @ reader `:53` | compute `wait_front`/`pop_front` @ compute `:68,71` | legal 1:1 |
| `input_1` (idx 1, borrowed) | reader `push_back` @ reader `:54` | compute `wait_front`/`pop_front` @ compute `:75,78` | legal 1:1 |
| `output` (idx 2, borrowed) | writer `reserve_back`/`push_back` @ writer `:43,61` | *none* | **self-loop** — bind the writer PRODUCER **and** CONSUMER |
| `input0_transpose` (idx 3) | compute `reserve_back`/`push_back` | reader `wait_front`/`pop_front` @ reader `:59,101` | legal 1:1 |
| `input1_transpose` (idx 4) | compute `reserve_back`/`push_back` | reader `wait_front`/`pop_front` @ reader `:103,145` | legal 1:1 |
| `concat` (idx 5) | reader `reserve_back`/`push_back` @ reader `:57,147` | compute `wait_front`/`pop_front` @ compute `:85,88` | legal 1:1 |
| `output_transpose` (idx 6) | compute `reserve_back`/`push_back` | writer `wait_front`/`pop_front` @ writer `:44,60` | legal 1:1 |

The compute kernel's FIFO calls on DFBs 3/4/5/6 are inside its `transpose` helper
(`device/kernels/compute/height_sharded_width_concat_two_tensors.cpp:12-31`), which takes the DFBs by
reference — the census is not visible from `kernel_main` alone.

**No dead DFB, and nothing needs the multi-binding flag,** in either factory.

## Watch for

- **CB endpoints (multi-binding):** none. But `ConcatS2SRMProgramFactory` is precisely the shape that
  invites the wrong call — one source, two instances, both touching all three DFBs. Every touch there
  is sync-free, so it is a 1P+1C assignment, not a flag. The hidden-second-writer hunt came up
  **negative** for both factories: no kernel raw-writes a DFB it does not FIFO-produce, and the op
  declares **no semaphores at all**, so there is no sync pair for a hidden co-fill to hang off.
- **The tiled compute kernel constructs a DFB handle it never uses.**
  `device/kernels/compute/height_sharded_width_concat_two_tensors.cpp:57` builds
  `DataflowBuffer output_dfb(output_dfb_id)` and then never touches it. Metal 2.0 generates a `dfb::`
  handle only where the host declared a binding, so this line cannot survive unchanged. **Drop the dead
  construction** — it has no side effects, so removing it is behavior-neutral. Do *not* add a binding
  for it just to keep the line compiling: that would put a spurious third toucher on a DFB whose census
  is otherwise a clean self-loop.
- **Cross-op / shared kernels:** none. All four kernel sources in scope are concat-owned and live in
  this directory, and a filename census across `ttnn/cpp/ttnn/operations/` finds **no other op binding
  any of them** — nothing borrowed, nothing lent, no `_metal2` fork to reuse or create. One intra-op
  note: `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors.cpp` is bound **twice by
  `ConcatS2SRMProgramFactory` itself** (as both its reader and its writer instance). That is the
  dual-instance work-split, not a shared-kernel case — both instances convert in this same change, so
  no fork is involved. *(The two borrowed writer kernels the audit records belong to
  `ConcatProgramFactory`, which is out of scope.)*
- **RTA varargs:** none — neither factory sets a runtime arg. Prefer named compile-time args for all 14
  CTAs in each list.
- **Kernel-side offsets that are not offset base pointers.** Both factories' kernels do arithmetic like
  `input_dfb_0.get_read_ptr() + input_start_0` and `output_dfb.get_write_ptr() + output_stick_offset`
  (`reader_height_sharded_width_concat_two_tensors.cpp:42,45,70`). These are kernel-side offsets applied
  to a base the kernel got from its own DFB — **not** host-folded pointers, and not the offset-base-pointer
  gate. They port unchanged; the offsets stay as named compile-time args.
- **Two blocked siblings share nothing with you, but one shares a build file.** The four out-of-scope
  factories keep their `create_descriptor` and their kernels; nothing you write should reach into
  `device/concat_program_factory.*`, `device/concat_s2i_program_factory.*`,
  `device/concat_s2s_multi_program_factory.*`, or `device/concat_block_sharded_program_factory.*`.
