# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/eltwise/binary_ng`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓ · Factory concept available ✓

> **⚠ Provenance — read the audit's provenance note.** Device 2.0 originally **failed** here: the 7 row-major (rm) dataflow kernels used the legacy raw `noc_async_read`/`noc_async_write` idiom. That migration was done **locally on this branch** (`akertesz/port-experiment-eltwise-binary-ng`, 2026-06-11) to unblock this port experiment — it is *not* upstream. The kernels now use `noc.async_read/write` + `CoreLocalMem` and are validated (3224 row-major + 94 tile/sharded bcast tests pass on Wormhole). You are porting on top of that local D2.0 work; treat it as a dependency of this branch.

## Scope reminder — the atomic unit is large here

This op has **one** `ProgramFactory`, but it **runtime-selects its kernel source files** from a ~34-file menu across two roots (`device/kernels` and `device/kernels_ng`) via `get_kernel_file_path` (`binary_ng_utils.cpp:81`). Selection axes: `SubtileBroadcastType` (none/scalar/row/col/row-col-mixed), `is_sfpu`, `is_where_op`, scalar-vs-tensor `b`, and `inputs_row_major` (`binary_ng_program_factory.cpp:556, 687`). **The factory + every source it can select flip to Metal 2.0 together** — there is no partial-factory port. Size the effort against the full selectable set, and map each DFB's producer/consumer role **per selected source path** (e.g. the rm path fills input DFBs differently from the tile path). If this is too large for one pass, a grounded stop after sizing it is a valid deliverable.

## Plan — factory concept

Implement **`ProgramSpecFactoryConcept`** (basic; implemented today). Decision 4 (Advanced) does **not** fire — no op-owned buffers/semaphores beyond CBs and tensors. Caching strategy: confirm against `port_op_to_metal2_ttnn_factory.md` at construction time. **Note the coupling** between the caching strategy and the dynamic-shape decision below: the op's current custom hash deliberately omits tensor shape (it keys on dtype + memory_config + shard_volumes), pairing with the `RuntimeTensorShape` accessors to reuse one program across shapes. Deleting the custom hash (required — see Construct) reverts to the strict default, which keys per shape unless you adopt the `dynamic_tensor_shape` relaxation. Decide these two together.

## Construct — to do

**Tensor bindings** (per binding) — all **clean** (already `TensorAccessor` end-to-end; no Case-2 bridge, no buffer-address-RTA bypass):

- `a` (input) — re-express via `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(ta::a)`. Host plumbing at `binary_ng_program_factory.cpp:855`.
- `b` (input, optional) — re-express via `TensorParameter` / `TensorBinding`. **Conditional binding:** `b` is absent for scalar ops; today the reader binds `*a_buffer` as a placeholder for b's accessor (`binary_ng_program_factory.cpp:858`). Reproduce the *conditional* binding shape (bind `b` only when present), not a placeholder — see the Conditional/optional binding pattern in the catalog.
- `c` (output) — re-express via `TensorParameter` / `TensorBinding`. Host plumbing at `binary_ng_program_factory.cpp:700`.
- All three use `ArgConfig::RuntimeTensorShape` today — see the dynamic-TensorAccessor decision under Watch-for before finalizing.

**Custom hash:** **delete** the custom `compute_program_hash` → default (sanctioned device-op-class edit). Located `binary_ng_device_operation.cpp:487`. It omits tensor shape; a relaxation candidate mined from it is recorded in the audit's Team-only section (fallible; default strict).

**Borrowed-memory DFBs:** the `a`/`b`/`c` CBs (`c_0`/`c_1`/`c_2`) borrow the tensor buffer on the **sharded** path (`.buffer = *_sharded ? *_buffer : nullptr` at `binary_ng_program_factory.cpp:570, 601, 661`). Port these with `DataflowBufferSpec::borrowed_from = <tensor_parameter_name>`, gated on the sharding condition. (Row-major and sharding are mutually exclusive — `binary_ng_program_factory.cpp:667`.)

**DFB specs:** one `DataflowBufferSpec` per legacy `CBDescriptor` (`c_0`–`c_6`; the intermediate/activation CBs `c_3`–`c_6` are conditionally pushed — preserve those conditions). All `format_descriptors` are single-element (no aliasing).

**Compute hw_config:** the op carries fp32-dest-acc logic (`binary_ng_program_factory.cpp:717`) and `unpack_to_dest_mode` — configure `unpack_to_dest_mode` separately from the `to_compute_hardware_config` helper, keyed by DFBs the compute kernel binds (gate any conditionally-bound DFB's entry on the same condition).

## Watch for

- **Dynamic TensorAccessor (`ArgConfig::RuntimeTensorShape`) — DECISION REQUIRED.** All three tensor bindings use the runtime-shape flavor (`binary_ng_program_factory.cpp:700, 855, 858`): the shape is an implicit common runtime arg, so the program is shape-independent today. Metal 2.0 expresses this via `TensorParameterAdvancedOptions::dynamic_tensor_shape = true` (or weaker `match_padded_shape_only`), both documented **UNSAFE** with per-dispatch-caching implications. **This is an explicit user-OK decision, not an automatic step** — and it's coupled to the custom-hash deletion (see Plan). Default is strict; if you don't adopt the relaxation, expect more cache entries (correct, but a caching-behavior change vs. today). Raise this with the invoker before committing.
- **Borrowed-memory DFB** (sharded path) → `borrowed_from`, as in Construct.
- **Runtime kernel-source selection** → the multi-path sizing note at the top; map DFB roles per path.
- **No semaphores, no cross-op kernels, no shared kernel-lib coupling** — nothing to watch there. (All out-of-dir includes are tt_metal HAL; in-op cross-subdir helpers `eltwise_utils*.hpp` / `fill_tile_utils.hpp` only.)
- **Do NOT bundle the unrelated anomaly:** `binary_ng_utils.cpp:125` has a missing-`.cpp` bug in the where+scalar compute path. It's flagged in the audit's Misc anomalies for the op owner — leave it out of the port diff.
