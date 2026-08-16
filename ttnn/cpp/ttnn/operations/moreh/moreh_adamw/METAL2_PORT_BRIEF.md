# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_adamw`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `086a669ff5e 2026-08-15 docs(metal_2.0): two porter-facing gaps a blind cold read turned up` *(carry this line into the port report's Provenance section)*

> **Before you start — confirm the hold has lifted.** The readiness sheet's `Is able to port?` cell for
> `moreh/moreh_adamw` reads `no`. That is **not** an op defect: it is a deliberate family-wide hold on ops targeting
> `CustomProgramSpecFactoryConcept`, whose recipe support is newly added and still being tested. This audit was run
> with the gate treated as `yes` on the recipe maintainer's explicit instruction. **Check with the maintainer that the
> hold is lifted before porting.** Everything below is a verdict on the code, and the code clears every gate.

## Scope

One device operation, one factory. `create_descriptor` and `override_runtime_arguments` are both in the factory file.

- `device/multi_core_program_factory.cpp` — `create_descriptor` @ `:58`, `override_runtime_arguments` @ `:353`
- `device/moreh_adamw_device_operation.hpp` — declarations, and the backdoor hash @ `:35-40`
- `device/kernels/reader_moreh_adamw.cpp`, `device/kernels/writer_moreh_adamw.cpp`,
  `device/kernels/moreh_adamw.cpp` (compute)

The compute kernel is instantiated **twice** — `compute_desc_1` on `core_group_1`, `compute_desc_2` on `core_group_2`
(`:247-264`), each carrying its group's tile count as CTA 0. The two core ranges are **disjoint**, so every node sees
exactly one compute instance. This is the per-group-CTA shape, not a dual-instance work-split: no 1P+1C assignment
question arises anywhere in this op.

**Read the kernels before planning — they are already half-ported.** See *Watch for → kernel starting state*.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`CustomProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `create_descriptor()` returning a `ProgramDescriptor` @
  `device/multi_core_program_factory.cpp:58`
- **Op-owned tensors:** none. The optional outputs built by `create_output_tensors`
  (`device/moreh_adamw_device_operation.cpp:99-133`) are ordinary op outputs.
- **Target concept:** `CustomProgramSpecFactoryConcept` — selected by `Override runtime args method? == yes`.
  `MorehAdamWDeviceOperation::override_runtime_arguments` @ `device/multi_core_program_factory.cpp:353-428` (declared
  `device/moreh_adamw_device_operation.hpp:76-81`) is **translated** into a method returning a `ProgramRunArgs`, not
  deleted.
- **Backdoor custom hash — present, and load-bearing. Leave it exactly as it is.**
  `attribute_names` / `attribute_values` @ `device/moreh_adamw_device_operation.hpp:35-40` deliberately exclude `lr`
  and `step` from the program hash (the comment @ `:31-34` explains: they change every optimizer step, so hashing them
  would recompile every call). That exclusion is safe **only because** the override re-applies them on every cache
  hit — `lr` at reader index 5, `step` at reader index 12 and compute index 0, plus the two β-exponents at reader
  indices 10 and 11, which are *derived from* `step` on the host. If your translated override stops writing any of
  those, the hash exclusion becomes a silent numerical bug that only shows on cache hits. Verify all five slots
  survive translation.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a non-`none` `TensorParameter relaxation` ·
  `get_dynamic_runtime_args` (deprecated hook). A custom hash, an `override_runtime_arguments`, and a pybound
  `create_descriptor` are **not** in this list: none of them gate, and any may be present on a cleared op. Here: no
  `compute_program_hash`, no pybound `create_descriptor`, and the `override_runtime_arguments` above.

## Construct — to do

**Tensor bindings — nine, all Case 1** (fed into a `TensorAccessor`, all memory access through it). Express each as a
`TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::<name>)`. No Case 2 anywhere — no
kernel does raw address arithmetic on a base, so no `get_bank_base_address` bridge is needed.

| Binding | Kernel | Miss-path site | Hit-path site | Kernel accessor |
|---|---|---|---|---|
| `param_in` | reader | `:307` | `:375` | `reader_moreh_adamw.cpp:51` |
| `grad` | reader | `:308` | `:376` | `reader_moreh_adamw.cpp:52` |
| `exp_avg_in` | reader | `:309` | `:377` | `reader_moreh_adamw.cpp:53` |
| `exp_avg_sq_in` | reader | `:310` | `:378` | `reader_moreh_adamw.cpp:54` |
| `max_exp_avg_sq_in` *(amsgrad only)* | reader | `:311` | `:379` | `reader_moreh_adamw.cpp:59` |
| `param_out` | writer | `:326` | `:381` | `writer_moreh_adamw.cpp:28` |
| `exp_avg_out` | writer | `:327` | `:382` | `writer_moreh_adamw.cpp:29` |
| `exp_avg_sq_out` | writer | `:328` | `:383` | `writer_moreh_adamw.cpp:30` |
| `max_exp_avg_sq_out` *(amsgrad only)* | writer | `:329` | `:384` | `writer_moreh_adamw.cpp:35` |

*(Miss/hit sites are in `device/multi_core_program_factory.cpp`.)*

What disappears when you bind them: reader RTA slots 0-4 and writer RTA slots 0-3
(`reader_moreh_adamw.cpp:15-19`, `writer_moreh_adamw.cpp:12-15`); the whole `TensorAccessorArgs(...).append_to(...)`
block @ `:192-207`; the `TensorAccessorArgs<N>` constexpr chains in both kernels
(`reader_moreh_adamw.cpp:46-49, 58`, `writer_moreh_adamw.cpp:24-26, 34`); the `Buffer*` captures @ `:269-281`; and the
`reader_addrs` / `writer_addrs` arrays plus their write loops in the override @ `:374-413`.

**The two optional bindings need care.** When `amsgrad == false` the factory passes `nullptr`, and
`emplace_runtime_args` emits `0u` with no binding (`tt_metal/impl/program/program_descriptors.cpp:243-251`); the
override writes `0u` too, so miss and hit agree today. Under Metal 2.0 these become **conditionally-declared**
`TensorParameter`s on the `amsgrad` path — same condition as the three conditional DFBs below.

**TensorParameter relaxation:** none — the only value that reaches a brief.

**TensorAccessor 3rd arg:** none. All nine accessors are already 2-arg; there is no page-size override to drop.

**CB endpoints — 19 CBs, and one item here is real structure, not translation.**

- **Legal 1:1 (11 CBs, both configs)** — `c_0` `param_in`, `c_1` `grad`, `c_2` `exp_avg_in`, `c_3` `exp_avg_sq_in`,
  `c_5` `scalar_args`, `c_6` `one`, `c_16` `param_out`, `c_17` `exp_avg_out`, `c_18` `exp_avg_sq_out`,
  `c_28` `beta1_exponent`, `c_29` `beta2_exponent`. Ordinary CB→DFB translation, one PRODUCER + one CONSUMER, nothing
  special. (`c_5`, `c_6`, `c_28`, `c_29` are produced by the reader through `fill_cb_with_value`, which does the
  `reserve_back` / `push_back` internally — `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:98-109`.)

- **Self-loop (5 CBs, both configs; plus `c_27` when `amsgrad` is on)** — `c_24` `tmp_param`, `c_25` `tmp_exp_avg`,
  `c_26` `tmp_exp_avg_sq`, `c_30` `tmp1`, `c_31` `tmp2`. Each is compute-kernel scratch that the **same** kernel both
  fills and drains — one toucher. **Bind the compute kernel PRODUCER and CONSUMER.** Legal on Gen1 for a compute
  kernel. No reader or writer kernel references any of these indices.

- **⚠ Dead-CB, config-scoped — the one piece of new structure this port adds.** `c_4` `max_exp_avg_sq_in`
  (allocated @ `:135-140`), `c_19` `max_exp_avg_sq_out` (@ `:182-187`), and `c_27` `tmp_max_exp_avg_sq` (from the
  `c_24..c_31` loop @ `:155-162`) have **zero touchers when `amsgrad == false`**: every kernel reference to them sits
  inside `#ifdef AMSGRAD`, and the define is emitted iff `amsgrad` (`:211-214`).

  A DFB with no producer and no consumer binding is **rejected by the spec validator**, so these cannot be carried
  across unconditionally the way the legacy factory allocates them. **Declare all three DFB specs only on the
  `amsgrad` path** — the same condition that gates the two optional tensor bindings above.

  **Do not read this as "drop them."** `c_4` and `c_19` are legal 1:1 and `c_27` is a live self-loop when `amsgrad` is
  on. The legacy factory has no such branch; adding it is the port work. There is no dead CTA to remove alongside
  them — none of the three indices is threaded to a kernel as a compile-time arg.

**No multi-binding anywhere.** Max touchers on any node for any CB is 2. The op declares no semaphores, and no kernel
takes a raw pointer into any CB (`get_write_ptr` / `get_read_ptr` / `fifo_*_ptr` appear nowhere in the three kernels),
so every endpoint is a FIFO endpoint and the census above is exhaustive.

## Watch for

- **CB endpoints (multi-binding):** none. Nothing to hunt — see the note above on why the census is exhaustive.

- **Cross-op / shared kernels:** none. All three kernels are owned exclusively by this op; no other factory binds any
  of them, no `_metal2` fork exists or is needed, and there is no sunset list. *(One census trap if you re-run it: the
  compute kernel `device/kernels/moreh_adamw.cpp` shares a filename with the host wrapper `moreh_adamw.cpp`, so a
  filename grep hits `moreh/sources.cmake`. That lists the **host** file. Check the bound path, not the filename.)*

- **RTA varargs:** none — name every argument. Both DM kernels read a fixed run via a running `i++` at the top
  (`reader_moreh_adamw.cpp:14-32`, `writer_moreh_adamw.cpp:11-18`), which is legacy positional plumbing, not a loop;
  the compute kernel reads one constant index (`moreh_adamw.cpp:17`). Names come straight from the kernel locals:
  - reader (16) → `param_addr`, `grad_addr`, `exp_avg_addr`, `exp_avg_sq_addr`, `max_exp_avg_sq_addr` *(these five
    become tensor bindings)*, `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `beta1_exponent`, `beta2_exponent`,
    `step`, `amsgrad`, `num_tiles_per_core`, `start_id`
  - writer (6) → `param_addr`, `exp_avg_addr`, `exp_avg_sq_addr`, `max_exp_avg_sq_addr` *(bindings)*,
    `num_tiles_per_core`, `start_id`
  - compute (1) → `step`

  Three of these are dead on arrival — reader `step` and `amsgrad`, and compute `step` (the audit records them as
  team-only anomalies). **Port them as-is anyway.** Removing a dead RTA is a functional change and an ops-team call,
  not port work; and the override's reader guard currently keys on one of them
  (`if (a.size() <= kReaderStepIdx) continue;` @ `:395`, `kReaderStepIdx = 12` @ `:372`), so quietly dropping it would
  break the translated override.

- **Kernel starting state — already half-ported.** All three kernels use `DataflowBuffer` objects, `Noc`, and
  `TensorAccessor` throughout, so the CB→DFB *API* move (`cb_dfb_api_whitelist.md` section A) is **already done**.
  Your work is the binding layer: `DataflowBuffer dfb_param(cb_id_param)` → `DataflowBuffer dfb_param(dfb::param)`.
  Two specifics:
  - **The compute kernel carries both forms for the same buffer.** E.g. `constexpr auto cb_one = tt::CBIndex::c_6`
    *and* `DataflowBuffer dfb_one_obj(cb_one)` (`moreh_adamw.cpp:35-36`), then uses the object for donor helpers
    (`sub_tiles_init_with_dt(dfb_one_obj, …)` @ `:135`) and the raw index for LLK calls
    (`sub_tiles(cb_one, cb_scalar_args, …)` @ `:136`). Both collapse onto the binding — construct the object from
    `dfb::one`, and pass the token directly to the LLK call. Make sure you catch **both** spellings for each of the
    19 buffers; missing the raw-index one leaves a `constexpr` CB index behind that still compiles.
  - **`get_tile_size(cb_id)` here is `const`, not `constexpr`** (`reader_moreh_adamw.cpp:89-94`,
    `writer_moreh_adamw.cpp:46-50`). By `cb_dfb_api_whitelist.md`'s "the legacy declaration is the entire test," these
    take the **member getter** — `dfb_param.get_tile_size()` — not the `get_tile_size(dfb::param)` token form.

- **Donor helpers need no work — but the shape isn't in the audit recipe's table, so here it is.** The reader calls
  `fill_cb_with_value` (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:98`) and the compute kernel calls
  `mul_tiles_to_cb` / `sub_tiles_to_cb` / `add_tiles_to_cb` / `copy_tile_to_cb` / `pack_tile_with_dt` /
  `*_init_with_dt` (`ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`). **Every one takes `DataflowBuffer` by value.**
  `DataflowBuffer` has a non-explicit converting constructor from the binding token —
  `DataflowBuffer(DFBBindingToken)`, `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:72` — so a `dfb::name` token
  converts implicitly at the call site. Pass the token directly, or keep the op's existing local objects and construct
  them from tokens; both compile. **No donor-side change is required, and no `_metal2` fork of either shared header is
  needed.**
