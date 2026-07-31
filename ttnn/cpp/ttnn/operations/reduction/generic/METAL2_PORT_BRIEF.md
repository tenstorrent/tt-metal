# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/reduction/generic/`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `7e91046b794 2026-07-31 docs(metal_2.0): add the op-porting recipe set` *(carry this line into the port report's Provenance section)*

**Scope of the porting unit.** Two `DeviceOperation`s share this directory and share code (`common.hpp`/`common.cpp`,
`reduce_op_utils::get_defines`, two dataflow readers, one borrowed writer), so they were audited — and should be
ported — together:

- `ReduceDeviceOperation` → `ReduceSingleCoreHwProgramFactory`, `ReduceMultiCoreHProgramFactory`, `ReduceMultiCoreWProgramFactory`
- `WelfordReduceDeviceOperation` → `WelfordReduceProgramFactory`

14 own kernels + 2 borrowed writers. Each Reduce factory has several config branches inside one
`create_descriptor` (interleaved-tiled · fused-negate · width-sharded · dense-RM), and the Welford factory has
three (`reduce_dim` = W / H / HW). **The work is volume, not difficulty** — plan the spec branch-by-branch.

> **`ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/` is out of bounds.** A whole-op quasar copy
> of this exact op exists there, with duplicated kernels that look like a solved version of your problem. It is a
> deliberately-shortcut pre-port carrying idioms the whitelist forbids. Do not read it, do not copy its names,
> do not count its `_metal2`-ish files as forks. It binds only its own copies, so it constrains nothing here.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — all four factories declare
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (all four factories), no op-owned tensors.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash ·
  `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) ·
  pybind `create_descriptor` — all gate conjuncts — plus other migration-risky pybind, which would have surfaced
  as a `safe` warning. All `no` on this op.

## Construct — to do

**Tensor bindings** (per binding, per factory). Every tensor address already arrives through the framework's
`MeshTensor`-binding overload of `emplace_runtime_args` — never through a `->address()` RTA — so there is no
correctness hazard to unwind, only the typed-binding rewrite:

- **`ReduceMultiCoreH` — interleaved-tiled config** — input, output: **Case 1** → express each as
  `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`. The
  `TensorAccessorArgs(a).append_to(reader_compile_time_args)` / `TensorAccessorArgs(output).append_to(...)`
  plumbing (`…multi_core_h…:396,433`) and the tensor entries in `emplace_runtime_args` (`:602-611`) both go away.
- **`ReduceMultiCoreH` — width-sharded config** — input, output: **clean**. Both are borrowed-memory CBs today
  (`.tensor = &a` @ `…multi_core_h…:195`, `.tensor = &output` @ `:243`) → port via
  `DataflowBufferSpec::borrowed_from`. There is no address RTA on this path at all; do not invent one.
  ⚠ **The same `TensorParameter` is `clean` here and `Case 1` in this factory's other three configs** — keep the
  per-config split; don't flatten it.
- **`ReduceMultiCoreH` — dense-RM config** — input, output: **Case 1** (`…multi_core_h…:540-553`).
- **`ReduceMultiCoreW`** (both tiled and dense-RM) — input, output: **Case 1** (`…multi_core_w…:366-379, 383-397`).
- **`ReduceSingleCoreHw`** — input, output: **Case 1** (`…single_core_hw…:203, 213`).
- **`WelfordReduce`** (W / H / HW) — input, output: **Case 1** (`…welford…:513,516 · 547-559 · 579-584`).

**No Case 2 anywhere.** No kernel in this op does hand-rolled address arithmetic on a tensor base, so you will
not need the `TensorAccessor::get_bank_base_address` bridge.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — all five `TensorAccessor(...)` sites in the op (and the borrowed writer's) are
two-argument. Nothing to drop.

**CB endpoints.** Every CB is either a plain 1:1 or a **self-loop**. **Set no multi-binding advanced option, and
drop no CB** — there is no ≥3-toucher CB and no dead CB in this op. Self-loops to bind PRODUCER *and* CONSUMER
on the one toucher:

| Factory (config) | CB | One toucher | Why |
|---|---|---|---|
| `MultiCoreH` / `MultiCoreW` / `SingleCoreHw`, fused-negate | `c_4` (acc), `c_5` (ineg/inv) | compute | compute packs into them and unpacks back out (`reduce_h_neg.cpp` / `reduce_w_neg.cpp` / `reduce_hw_neg.cpp`) |
| `MultiCoreH` / `MultiCoreW`, dense-RM | `c_0` (cb_tile_in) | compute | `compute_kernel_lib::tilize<…, cb_rm, cb_tile_in>` writes it, `reduce` drains it — both inside `reduce_rm.cpp` |
| `MultiCoreH` / `MultiCoreW`, dense-RM | `c_5` (cb_acc) | compute | `compute_kernel_lib::Accumulate::at(cb_acc, chunk_idx)` round-trips through it |
| `MultiCoreH` / `MultiCoreW`, dense-RM | `c_4` (clear_value) | reader | reader fills via `get_write_ptr()`, `push_back`s, then re-reads via `get_read_ptr()` as a NoC source (`reader_unary_reduce_rm.cpp:77-79`) |
| `MultiCoreH`, width-sharded | `c_1` (src1, borrowed from input) | reader | `reserve_back(num_tiles)` + `get_write_ptr()` self-read (`reader_..._sharded.cpp:50-51`) — and it is *also* a borrowed-memory DFB |
| `WelfordReduce`, W-reduce | `c_19` (cb_var) | compute | packs the variance tile, then `wait_front`/`transpose_tile`/`pop_front`s it back |
| `WelfordReduce`, **all three** dims | `c_2` (scalar) | reader | the reader fills it (`prepare_reduce_scaler`) and **no Welford compute kernel reads it** — see Watch for |

Everything else is 1:1: `c_0`/`c_2`/`c_3` on the Reduce paths, `c_24`→`c_0` on dense-RM, and `c_0`/`c_16`/`c_21`/`c_22`
on Welford.

**Metadata reads must stay `constexpr`.** Whitelist rule 7 moves these onto the DFB object, and several sites in
this op are `constexpr` initializers whose result becomes a **template argument** — so bind them through a
`constexpr` `DataflowBuffer`, not a runtime one. `DataflowBuffer`'s getters *are* `constexpr`
(`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:167-267`), so this works; the Device 2.0 `CircularBuffer`
wrapper's are not, which is why the legacy code uses the free-function form. Sites:

- `constexpr DataFormat reduce_format = get_dataformat(dfb_id_in0);` →
  `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp:35` and
  `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp:38`, both feeding
  `is_sfpu_reduce_path<…, reduce_format, …>()`.
- `constexpr DataFormat reduce_format = static_cast<DataFormat>(unpack_src_format[…]);` →
  `compute/reduce.cpp:41`, `compute/reduce_h_neg.cpp:38`, `compute/reduce_w_neg.cpp:40`, same use.
- Plain runtime sites (no constexpr constraint): `get_tile_size(cb_id)` ×7 and
  `get_local_cb_interface(cb_id).fifo_page_size` ×1 (`reader_unary_reduce_rm.cpp:82`), plus the borrowed writer's.

**In-op helper signature.** `reduce_rm_dataflow_common.hpp:108` declares
`rm_fill_page_with_clear_template(Noc&, experimental::CB&, …)`. `experimental::CB` is a `using` alias for the
Device 2.0 `CircularBuffer` pulled in from `pool/device/kernels/experimental_device_api.hpp`. The header is the
op's own (no external consumer), so the parameter type is yours to change to the DFB handle. The other symbol
the op takes from that pool header, `experimental::local_addr(uint32_t addr, …)`, takes a raw L1 address and
needs nothing — feed it the DFB read pointer unchanged.

## Watch for

- **CB endpoints (multi-binding):** none. The hidden-second-writer, multiple-reader, and dual-instance
  work-split hunts were all run during the audit and all came back negative — the op has no semaphores at all,
  every raw `get_*_ptr()` is a peek by the kernel that already holds that CB's FIFO role, and the only
  same-source kernel pairs (`compute_desc_g1` / `compute_desc_g2`) cover **disjoint** core groups, so each node
  sees one compute instance. You do not need to re-run these.

- **Cross-op / shared kernels:** two borrowed writers, **no `_metal2` fork exists for either** — this port
  creates the first one beside each original (rung 2), plus the pointer comment in each legacy file:
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
    — used by `ReduceMultiCoreH` (interleaved-tiled), `ReduceMultiCoreW` (tiled), `ReduceSingleCoreHw`, and
    `WelfordReduce` (W and H). Other binding ops: **~34 factories tree-wide** — a **sunset list, not
    authorization to convert the kernel in place**. This is one of the most widely-bound kernels in the tree, so
    the names you give the fork are inherited by every later port: take them from the kernel's own vocabulary
    (`dfb::out`, `tensor::dst`), not from reduce's locals.
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp`
    — used by `ReduceMultiCoreH` width-sharded only. Other binding ops: **~11 factories tree-wide** — again a
    sunset list, not authorization. This one is trivial (a `wait_front` / `pop_front` readiness handshake on an
    in-place sharded output).

  All of the op's **own** kernels are bound only by its own four factories — nothing here is lent, so those you
  convert in place.

- **RTA varargs:** none — every kernel reads a fixed set of runtime args at constant indices. Name them all.
