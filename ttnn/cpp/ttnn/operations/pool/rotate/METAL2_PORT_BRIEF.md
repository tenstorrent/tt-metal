# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/pool/rotate`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `dc266b472bd 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports` *(carry this line into the port report's Provenance section)*

**Port unit:** one DeviceOperation, `RotateDeviceOperation`, with two factories — `NearestProgramFactory` (`device/rotate_nearest_program_factory.cpp`) and `BilinearProgramFactory` (`device/rotate_bilinear_program_factory.cpp`). Both clear; port them together.

**Kernels in scope** (follow `kernel_source`, not directory boundaries):

| Kernel | Owner | Used by |
|---|---|---|
| `device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp` | rotate | Nearest |
| `device/kernels/dataflow/writer_rotate_nearest_interleaved.cpp` | rotate | Nearest |
| `device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp` | rotate | Bilinear |
| `pool/generic/device/kernels/compute/compute_pool_2d.cpp` | **borrowed** — pool/generic | Bilinear |
| `pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp` | **borrowed** — pool/grid_sample | Bilinear, **interleaved config only** |

Both factories branch on sharding, and the branch changes the kernel set and the CB census — so every item below is tagged by config where it differs. The three sharding-relevant configs are: **Nearest interleaved**, **Nearest sharded**, **Bilinear interleaved**, **Bilinear sharded**.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — both factories, confirmed against the readiness sheet and the code.
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) · pybind `create_descriptor` — all gate conjuncts — plus other migration-risky pybind, which would have surfaced as a `safe` warning. All `no` on both factory rows.

## Construct — to do

### Tensor bindings

Four bindings, **all Case 1**, plus two borrowed-memory CBs. There is **no Case 2 anywhere** — no kernel does hand-rolled address arithmetic on a tensor base, so you will not need the `get_bank_base_address` bridge.

- **`input` (Nearest)** — **Case 1** → express as `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::input)`. Today: `Buffer*` at reader RTA 0, consumed at [reader_rotate_nearest_interleaved.cpp:33-34](device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp#L33-L34).
- **`output` (Nearest)** — **Case 1** → same. Today: `Buffer*` at writer RTA 0, consumed at [writer_rotate_nearest_interleaved.cpp:21-22](device/kernels/dataflow/writer_rotate_nearest_interleaved.cpp#L21-L22).
- **`input` (Bilinear)** — **Case 1** → same. Today: `Buffer*` at reader RTA 0, consumed at [reader_rotate_bilinear_interleaved.cpp:42-43](device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp#L42-L43). The accessor is then passed by reference into the shared donor `read_four_corner_inputs_with_fill` — a Shape-1 `TensorAccessor<DSpec>` parameter, which crosses cleanly.
- **`output` (Bilinear, interleaved only)** — **Case 1** → same. Today: `Buffer*` at writer RTA 0, consumed at [writer_grid_sample_interleaved.cpp:19-21](../grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp#L19-L21). **This is a borrowed kernel — see the fork rule below.**
- **`output` (Bilinear, sharded)** — **clean**, borrowed-memory DFB. No writer kernel exists in this config; the output CB is backed by `output_tensor.buffer()` at [rotate_bilinear_program_factory.cpp:214](device/rotate_bilinear_program_factory.cpp#L214), so the DFB *is* the tensor access. Port via `DataflowBufferSpec::borrowed_from` — do not force it into a Case.
- **`output` (Nearest, sharded)** — **both at once**: the output CB is borrowed from `output_tensor.buffer()` at [rotate_nearest_program_factory.cpp:187](device/rotate_nearest_program_factory.cpp#L187) *and* the writer separately accesses the tensor through its Case-1 accessor. You need both the `borrowed_from` DFB and the `TensorParameter`. (The audit flags this pair as a likely self-copy in the op's existing behavior — Misc anomaly 1. **Preserve it byte-for-byte**; it is the ops team's to change, not yours.)

In every Case-1 site the legacy `Buffer*` RTA **and** its `TensorAccessorArgs<N>()` compile-time plumbing both disappear. Note that rotate never calls `->address()` — the factory pushes the `Buffer*` object itself into `emplace_runtime_args`, which the framework registers as a `BufferBinding`. That is correct-on-cache-hit today, so this is routine work, not a correctness rescue.

### TensorParameter relaxation

**None.** The sheet reads `none` on both factory rows, consistent with the absent custom hash.

### TensorAccessor 3rd arg

**None.** Every `TensorAccessor` construction in the reachable code is the 2-argument form. Nothing to drop.

### CB endpoints

Full per-`(CB, config)` census is in the audit. Dispositions:

| CB | Factory / config | Disposition |
|---|---|---|
| `fill_cb` (`c_0`) | Nearest — both configs | **self-loop** (one toucher, sync-free: raw peek + `zero_out_page` + local NoC read source) |
| `fill_cb` (`c_0`) | Bilinear — both configs | **self-loop** (same shape) |
| `output_cb` (`c_5`) | Bilinear — **sharded** | **self-loop** (compute produces; no writer kernel exists, nothing drains it) |
| `output_cb` (`c_1` / `c_2`) | Nearest — both configs | legal 1:1 — no action |
| `input_cb` (`c_1`), `scalar_cb` (`c_3`) | Bilinear — both configs | legal 1:1 — no action |
| `output_cb` (`c_5`) | Bilinear — **interleaved** | legal 1:1 — no action |
| **`input_cb` (`c_1`)** | **Nearest — sharded only** | **dead-CB drop** |

**Dead-CB drop — `input_cb`, Nearest factory, sharded config.** Delete the `CBDescriptor` at [rotate_nearest_program_factory.cpp:161-174](device/rotate_nearest_program_factory.cpp#L161-L174) and the `input_cb_index` variable that feeds it. The index is **never handed to any kernel** — the reader's compile-time args ([:194-206](device/rotate_nearest_program_factory.cpp#L194-L206)) carry only `output_cb_index` and `fill_cb_index`, the writer's ([:212-217](device/rotate_nearest_program_factory.cpp#L212-L217)) only `output_cb_index` — so a zero-endpoint DFB cannot be expressed in Metal 2.0 at all and the drop is required. **No dead CTA accompanies it**; the index never reached a kernel, so there is nothing else to remove. Record the drop with `file:line` in your port report. (A dead CB has no behavior, so removing it changes none — this is not a general licence to change behavior elsewhere.)

**No CB anywhere needs the multi-binding advanced option.** All three hidden-toucher faces were hunted and came back negative.

### Named runtime args

Every RTA is nameable — **no varargs**. Each kernel reads each arg exactly once at a distinct literal index, with no counted loop and no data-selected index:

- Nearest reader (RTAs 0-7): `input` buffer, `num_sticks`, `start_stick_id`, `cos_angle`, `sin_angle`, `center_x`, `center_y`, `fill_value_bf16`.
- Nearest writer (RTAs 0-2): `output` buffer, `num_sticks`, `start_stick_id`.
- Bilinear reader (RTAs 0-7): same shape as the nearest reader, with `fill_value_bits`.
- Bilinear writer (RTAs 0-2): `output` buffer, `num_sticks`, `start_stick_id`.
- Compute: rotate sets **no** runtime args on it. Its lone `get_arg_val<uint32_t>(0)` ([compute_pool_2d.cpp:129](../generic/device/kernels/compute/compute_pool_2d.cpp#L129)) sits on the dead side of a constant-folded ternary, because rotate always supplies a non-zero `max_out_sticks_per_core` CTA.

## Watch for

- **`DUMMY_CB_ID = 32` in the borrowed compute kernel — decide this before you write the fork.** [rotate_bilinear_program_factory.cpp:34](device/rotate_bilinear_program_factory.cpp#L34) defines `DUMMY_CB_ID = 32` and feeds it to **eleven** of `compute_pool_2d.cpp`'s CB-index CTAs ([:256-287](device/rotate_bilinear_program_factory.cpp#L256-L287)) for the pool features rotate doesn't use. The donor then **unconditionally constructs `DataflowBuffer` objects on that index** — `in_dfb_1`, `in_scalar_dfb_1`, `pre_tilize_dfb`, `fast_tilize_dfb` at [compute_pool_2d.cpp:105-110](../generic/device/kernels/compute/compute_pool_2d.cpp#L105-L110) — but index 32 is outside the `c_0`…`c_31` space and no such CB is allocated.

  Harmless on Gen1 today: with `split_reader == 0` and `is_output_tiled == 0`, every *use* of those four objects is compile-time dead, so only the constructors survive and they touch nothing. But the constructions are not guarded, and Metal 2.0 has no `dfb::name` token to bind a nonexistent buffer to. Expect to either guard the constructions behind the same `if constexpr` conditions that already gate their uses, or give the fork a way to express "this operand is unused." This is the one genuine design question on the port — settle it first, since it shapes the fork.

- **Cross-op / shared kernels — two borrowed kernel files, and this port creates the first `_metal2` fork of each.** A repo-wide search finds **no `_metal2` kernel files at all** outside `experimental/quasar/**` (which don't count). So for both: create the fork beside the original, per the shared-kernel caution in `port_patterns.md`.

  | Borrowed kernel | Owner | Other instantiating ops — **sunset list** | Fork |
  |---|---|---|---|
  | `pool/generic/device/kernels/compute/compute_pool_2d.cpp` | pool/generic | `pool/generic` (`pool_multi_core_program_factory.cpp`), `pool/grid_sample` (`grid_sample_bilinear_program_factory.cpp`) | none yet — **you create it** |
  | `pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp` | pool/grid_sample | `pool/grid_sample` (`grid_sample_bilinear_program_factory.cpp`) | none yet — **you create it** |

  Those consumer lists are a **sunset list, not authorization to convert either kernel in place.** Both files are live for their other consumers; the legacy copy goes away only when the last one migrates. Rotate binds its fork; everyone else keeps the original untouched.

- **`experimental/quasar/` is out of bounds.** If a copy of either borrowed kernel turns up there, it is a deliberately hacky pre-port copy — not a precedent, not a naming source, and not evidence that a construct ports. Don't read it and don't bind it.

- **A transitive `circular_buffer.h` include reaches all three rotate kernels — it is inherited, not yours to clean up.** `pool/device/kernels/experimental_device_api.hpp` opens with `#include "api/dataflow/circular_buffer.h"` ([:11](../device/kernels/experimental_device_api.hpp#L11)) and aliases `using CB = CircularBuffer` ([:24](../device/kernels/experimental_device_api.hpp#L24)). It arrives directly in `writer_rotate_nearest_interleaved.cpp` ([:8](device/kernels/dataflow/writer_rotate_nearest_interleaved.cpp#L8)) and transitively in both readers via `pool_kernels_common.hpp` / `grid_sample_reader_common.hpp`. Rotate's own kernels use **none** of it — they are already on `DataflowBuffer` — but the include comes through shared headers rotate doesn't own, so you can't simply delete it. Recognize it and leave it.

- **Two compute `KernelDescriptor`s over disjoint core sets — keep them as two `KernelSpec`s.** [rotate_bilinear_program_factory.cpp:306-315](device/rotate_bilinear_program_factory.cpp#L306-L315) instantiates `compute_pool_2d.cpp` twice, differing only in `core_ranges` (`core_group_1` vs `core_group_2`) and the `total_interpolations` CTA. Each node sees exactly one instance, so this is the ordinary per-group split — **not** a dual-instance work-split, and no CB gains a second toucher from it. Do **not** collapse the pair into one `KernelSpec` by demoting `total_interpolations` to a runtime arg; that is the demoting-per-group-CTA anti-pattern in `port_patterns.md`.

- **CB endpoints (multi-binding):** none. No CB in this op needs the flag.

- **RTA varargs:** none — name every runtime arg (list above).

- **Don't "fix" what the audit flagged as anomalies.** The audit records six latent issues (dead CTAs, burned CB indices, unreachable sharding branches, a redundant NoC barrier, the nearest-sharded self-copy, a factory-to-factory sharding-support mismatch). These route to the ops team and are **not** port work — the confirmed dead CB above is the only one you touch. Preserve everything else byte-for-byte, including the code that looks redundant.

## Verification

Existing test coverage for the op:

- `tests/ttnn/unit_tests/operations/pool/test_rotate.py`
- `tests/ttnn/nightly/unit_tests/operations/pool/test_rotate.py`

Both nearest and bilinear paths, and both interleaved and sharded configs, need to be exercised — the CB census and the kernel set differ across all four combinations.
