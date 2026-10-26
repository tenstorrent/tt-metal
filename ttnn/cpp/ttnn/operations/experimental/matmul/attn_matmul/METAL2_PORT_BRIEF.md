# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `akertesz/op-porting-recipe` @ `4bd4bf42bfe` *(carry this line into the port report's Provenance section; the audit ran on a verbatim local import of the recipe branch's `docs/…/metal_2.0/` tree, which this branch does not carry)*

**Code audited:** `origin/main` @ `fd25f0f4aac` (2026-10-26); all `file:line` references below are against it.
One factory, three kernels (reader + compute owned; writer borrowed, fork exists), six CBs, no semaphores.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to
`ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (PR #45074). The factory lives in the `program_factory_t` variant
  (`device/attn_matmul_device_operation.hpp:23`) — no direct-descriptor conversion needed; the port is a method
  swap inside `AttnMatmulProgramFactory` (`device/attn_matmul_program_factory.hpp:13-21`).
- **Op-owned tensors:** none.
- **Target concept:** **`ProgramSpecFactoryConcept`** — no `override_runtime_arguments` exists; the framework
  refreshes the three tensor bindings on a cache hit.
- **Custom `compute_program_hash`:** none (default reflection hash; the former custom hash was removed by
  PR #47232 — see the stale-comment note under *Watch for*).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that
  is neither `none` nor an analysis pointer (cell reads `none`) · `get_dynamic_runtime_args`. A pybound
  `create_descriptor` is also absent (`attn_matmul_nanobind.cpp` binds only the two public wrappers; nothing to
  delete).
- **Two public entry points, one factory:** `attn_matmul` and `attn_matmul_from_cache` (`attn_matmul.cpp:13`,
  `:34`) differ only in the `num_tokens` / `transpose_hw` attributes. Nothing outside the factory body changes.

## Construct — to do

**Tensor bindings** (per binding; all three arrive in the `Buffer*`-binding form of PR #49134):

- `input_tensor_a` — **Case 1** (via `TensorAccessor`, `device/kernels/dataflow/reader_transformer_attn_matmul.cpp:48`)
  → `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::<in0>)`. Drops: reader RTA slot 0
  (`device/attn_matmul_program_factory.cpp:237`) and the first `TensorAccessorArgs` CTA tail (`:175`, kernel
  `TensorAccessorArgs<2>()` at `reader:46`).
- `input_tensor_b` — **Case 1** (`reader:49`) → `TensorParameter` / `TensorBinding`. Drops: reader RTA slot 1
  (`:238`) and the second CTA tail (`:176`, `reader:47`).
- `output` — **Case 1** (legacy writer `TensorAccessor(dst_args, dst_addr)`,
  `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:42`) → `TensorParameter` /
  `TensorBinding` under the fork's name **`tensor::dst`**. Drops: writer RTA slot 0 (`:260`), the writer's CB-index
  CTA (`:178`) and its CTA tail (`:179`).

No borrowed-memory CB anywhere (no `CBDescriptor` sets `.buffer`), so no `borrowed_from` and no "clean" binding.

**TensorParameter relaxation:** `none`. Keep strict.

**TensorAccessor 3rd arg:** none — no accessor in the op passes one (`reader:48-49`, legacy writer `:42`, fork `:48`).

**CB endpoints** (one configuration — `transpose_hw` / `fp32_dest_acc_en` change CTAs and formats, never the
census; all six CBs are on the full device grid, `attn_matmul_program_factory.cpp:75-76`):

- `in0` (CB 0, `:103-112`, capacity `Kt` tiles) — legal 1:1: reader PRODUCER (`reserve_back(Kt)` `reader:82`,
  `push_back(1)` × `Kt` `:87`), compute CONSUMER (`wait_front(kt + 1)` `compute:54`, `pop_front(Kt)` `:82`).
- `in1` (CB 1, `:114-124`) — legal 1:1: reader PRODUCER (`reader:96-99`), compute CONSUMER (`compute:56`, `:60`).
- `intermed0` (CB 2, `:126-135`) — **self-loop on the compute kernel** (bind it PRODUCER and CONSUMER): the kernel
  packs into it (`compute:64-68`) and `compute_kernel_lib::untilize<1, c_2, c_3, …>(1)` (`:71-77`) drains it.
- `intermed1` (CB 3, `:137-146`) — legal 1:1: compute PRODUCER (the `untilize` helper's output side), reader
  CONSUMER (`wait_front(1)` `reader:109`, `pop_front(1)` `:118`); the reader's `get_read_ptr()` peek (`:58`) is on
  its own binding.
- `intermed2` (CB 4, `:148-157`) — legal 1:1: reader PRODUCER (`reserve_back(1)` `reader:92`, raw NoC write through
  the `get_write_ptr()` peek `:59` / `:110-116`, `push_back(1)` `:124`), compute CONSUMER
  (`compute_kernel_lib::tilize<1, c_4, c_5>(1)` `compute:86`).
- `out` (CB 5, `:159-169`) — legal 1:1: compute PRODUCER (the `tilize` helper's output side), writer CONSUMER
  (fork `writer_unary_interleaved_start_id_metal2.cpp:57`, `:60`) under the fork's name **`dfb::out`**.

No multi-binding flag; no dead CB; no conditional DFB.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:**
  `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` →
  **`_metal2` fork already exists** at
  `…/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` — bind it, don't re-fork.
  Its vocabulary is your constraint: `dfb::out` bound CONSUMER, `tensor::dst`, named args `num_pages` and
  `start_id` (`:23-24`). The legacy factory sets **no defines** (`attn_matmul_program_factory.cpp:190-196`), so
  none of `OUT_SHARDED`, `BACKWARDS` or `ENTRIES_PER_PAGE` / `PAGE_BYTES` is set — supply none (one page per DFB entry, `page_bytes = dfb.get_entry_size()`, fork `:36-38`, matches the legacy `onepage = 1`). Writer run-args today: `num_pages =
  num_output_blocks_per_core * MtNt`, `start_id = num_blocks_written * MtNt` (`:261-262`). Other legacy binders
  (**sunset list, not authorization to convert in place**; issue #52228):
  `data_movement/reshape_on_device/device/reshape_tile_program_factory.cpp`,
  `eltwise/unary_backward/device/unary_backward_program_factory.cpp`,
  `examples/example/device/multi_core_program_factory.cpp`,
  `examples/example/device/single_core_program_factory.cpp`.
  (`data_movement/slice` and `matmul` bind same-named private copies of their own — not this file.)
- **RTA varargs:** none — name every arg. Reader (`reader:15-25`): `Mt`, `Kt`, `Nt`, `MtKt`, `in1_KtNt_skip`,
  `in1_KtNt_mul_32`, `blocks`, `itileA_start`, `itileB_start` (slots 0-1 become the two tensor bindings).
  Compute (`compute:22-25`): `batch`, `Mt`, `Kt`, `Nt`. Writer: the fork's `num_pages`, `start_id`. CTAs are
  fixed too: reader `transpose_hw`, `fp32_acc_en` (`reader:27-28`; the accessor tails go away), compute
  `transpose_hw` (`compute:21`).
- **Full-grid kernels with zero-arg idle cores (`attn_matmul_program_factory.cpp:214-265`).** All three kernels
  and all six CBs are placed on the whole device compute grid; the run-arg loop walks every device core
  (`:220-221`, mapping fixed by PR #59789). Cores in `core_group_1` / `core_group_2` get real values; every other
  core gets an all-zero vector (`:228-230`) so its kernels start and fall straight through (`blocks = 0`,
  `batch = 0`, `num_pages = 0`). Reproduce it: `core_ranges` = full grid on all three `KernelSpec`s, per-core
  named run-args with zeros on idle cores. The `0` sitting in the `Buffer*` slot on idle cores (`:228`, `:230`)
  is an address placeholder and simply vanishes — bindings are per-kernel.
  `tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py::test_attn_matmul_compute_grid_shorter_than_device`
  (`:979`) pins the shape.
- **Loopback L1→L1 row copy in the reader (`reader:65-67`, `:110-117`) — keep as is.** Source: a
  `UnicastEndpoint` addressed with `my_x[noc.get_noc_id()]` / `my_y[…]` and the `intermed1` read pointer plus a
  row offset; destination: `CoreLocalMem<uint32_t>` at the `intermed2` write pointer plus the same offset;
  64 or 128 bytes per row (`:62`). This is the Device 2.0 migration guide's *"Avoid (Device 2.0 API, but
  hand-built loopback endpoint)"* form (guide `:526-551`); the preferred local-handle form is a transfer
  rewrite and out of port scope. Only the two peeks that feed it change spelling (`CircularBuffer` →
  `DataflowBuffer`). Precedent: the checked-in fork
  `data_movement/sharded/device/kernels/dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp:133-137`
  keeps the same hand-built loopback.
- **Cursor peeks taken once, before any FIFO op (`reader:58-59`).** `get_read_ptr()` on `intermed1` and
  `get_write_ptr()` on `intermed2` are captured at kernel entry and reused every iteration; correct only because
  both CBs are exactly one page (`:139`, `:150`). They are public peeks (`cb_dfb_api_whitelist.md` §C) — same
  method names on `DataflowBuffer`; keep the one-page capacities (the port changes no sizes).
- **Compute `hw_config` — Style A with dropped fields.** The op resolves the TTNN config
  (`get_compute_kernel_config_args`, `:32-33`) but copies only `math_fidelity` and `fp32_dest_acc_en` onto the
  `ComputeConfigDescriptor` (`:208-211`); `math_approx_mode` and `dst_full_sync_en` are resolved and dropped
  (descriptor defaults `false` / `false`), `packer_l1_acc` has no counterpart. So: translate with
  `to_compute_hardware_config(device->arch(), config)`, then on the returned `ComputeGen1Config` force
  `sfpu_precision_mode = Precision::Precise` and `double_buffer_dest = true` regardless of the caller's values
  (that is the legacy behaviour; a silently-ignored knob is preserved, not repaired). `bfp_pack_precision_mode`
  stays default. **`unpack_modes`:** the legacy descriptor sets no `unpack_to_dest_mode` (`Default` →
  `UnpackMode::UnpackToSrc`); the validator demands an explicit entry for every Float32 DFB the compute kernel
  consumes while `enable_32_bit_dest` is on. The compute kernel consumes `in0`, `in1`, `intermed0`, `intermed2`.
  `in0` / `in1` are Float32 when the input dtypes are (`:35-36`; validation then forces `fp32_dest_acc_en`,
  `:46-49`); `intermed0` / `intermed2` are Float32 exactly when `fp32_dest_acc_en && in0 == Float32`
  (`:36-39`). Add a `{<dfb>, UnpackMode::UnpackToSrc}` entry for each of those four, gated on that DFB's
  format being `Float32` — a key naming a non-consumed or non-Float32 DFB is rejected / unnecessary.
  `test_attn_matmul_fp32` (`test_attn_matmul.py:93`, `in_dtype = float32`) exercises the branch.
- **`opt_level`:** none set anywhere (`grep -n opt_level` on the factory returns nothing) → the compute spec
  resolves to **O3** and must say `KernelBuildOptLevel::O3` explicitly; both DM specs resolve to O2 (defaults).
  DM configs are `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` (`:188`, `:196`) → the arch-agnostic
  reader / writer helpers.
- **`constexpr` metadata:** none from CB metadata. `reader:43-44` `const uint32_t in0_tile_bytes =
  get_tile_size(cb_id_in0)` / `in1_tile_bytes` are runtime `const` → DFB getters. `bfloat16_row_bytes`
  (`reader:62`) is CTA-derived, untouched.
- **LLK and kernel-lib calls take `dfb::name` directly** (*Pattern: Pass DFB handles directly to LLKs and
  kernel-lib helpers*): `compute_kernel_hw_startup<SrcOrder::Reverse>(c_0, c_1, c_2)` (`compute:43`),
  `matmul_init` / `matmul_block_init` (`:44`, `:80`, `:90`), `matmul_tiles` (`:58`), `pack_tile` (`:66`),
  `reconfig_data_format_srca` (`:79`), `pack_reconfig_data_format` (`:88`), `reconfig_data_format` (`:89`) —
  all `uint32_t` cb ids. The id-only kernel-lib forms `untilize<1, c_2, c_3, InitAndUninit, WaitBlock,
  UnpackReconfigure>(1)` (`:71-77`) and `tilize<1, c_4, c_5>(1)` (`:86`) take the ids as non-type template
  parameters and build their own `DataflowBuffer`s inside (`untilize_helpers.hpp:175-177`,
  `tilize_helpers.hpp:186-189`); the constexpr token conversion covers that position. No `.id` extraction, no
  temporary wrappers.
- **Stale hash comments — do not carry them.** `attn_matmul_program_factory.hpp:14-18` and
  `attn_matmul_program_factory.cpp:99-102` claim `padded_shape` is folded into `compute_program_hash()`; that
  custom hash was removed by PR #47232. The `in0` CB capacity (`Kt` tiles, `:105`) is still keyed per shape —
  by the default reflection hash over the tensor specs — so the behaviour stands; only the comment is dangling.
- **Tests (no-regression baseline, confirm with the invoker):**
  `tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py` — `test_attn_matmul` (`:36`),
  `test_attn_matmul_fp32` (`:93`), `test_attn_matmul_with_program_cache` (`:159`),
  `test_attn_matmul_with_program_cache_exhaustive` (`:801`),
  `test_attn_matmul_compute_grid_shorter_than_device` (`:979`) — the `test_group_attn_matmul*` cases in the same
  file are a different op; `models/demos/falcon7b_common/tests/unit_tests/test_falcon_attn_matmul.py` (covers
  `attn_matmul_from_cache` with both `transpose_hw` values); Galaxy mesh:
  `tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_attn_matmul` (`:413`). Production callers:
  `models/demos/falcon7b_common/tt/falcon_attention.py:688`, `:797`;
  `models/demos/ttnn_falcon7b/tt/falcon_attention.py:159`, `:222`.
