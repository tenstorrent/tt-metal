# Metal 2.0 Port Report — `experimental/transformer/rotary_embedding_llama_fused_qk`

## Outcome

**`PORTED`** — the single `RotaryEmbeddingLlamaFusedQKProgramFactory` (the op's only factory) is
converted to `MetalV2FactoryConcept` (`create_program_artifacts`), and its tiled kernel path passes
the confirmed no-regression baseline (16/16, PCC ≈ 0.99999 vs the 0.9997 gate, including the
program-cache test). One caveat, not a regression: the factory's **row-major** kernel source
(`row_major_QK=true`) is host-compile-clean but is **not JIT/runtime-exercised by any available test**
— a pre-existing coverage gap (see [Open items](#open-items-for-downstream)). Both kernel sources
received the identical mechanical CB→DFB / named-arg transformation.

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

- **Concept realized:** `MetalV2FactoryConcept`. `RotaryEmbeddingLlamaFusedQKProgramFactory::create_descriptor`
  (returned `ProgramDescriptor`) → `::create_program_artifacts` (returns `ttnn::device_operation::ProgramArtifacts`),
  same `(const Params&, const Inputs&, Result&)` signature. `.hpp` include swapped
  `program_descriptors.hpp` → `ttnn/metal_v2_artifacts.hpp`.
- **Custom `compute_program_hash` deletion:** none — the op had no custom hash (default reflection-based).
- **Pybind entry points removed:** none — the op is bound via `ttnn::bind_function` (nanobind); there was
  no pybound `create_descriptor` to remove.
- **Factory-parameter drop:** none.
- **Device-op-class edits:** none. Single-variant `program_factory_t` gets the framework adapter's
  default `select_program_factory` (`mesh_device_operation_adapter.hpp:201-214`), so no
  `select_program_factory` was added. `validate_on_program_cache_miss` / `compute_output_specs` /
  `create_output_tensors` untouched.
- **Open items:** the two Metal-only compute fields (`bfp_pack_precision_mode`, `unpack_modes`) are left
  at defaults — correct for this op (all-bfloat16, no Float32 DFB, so no `enable_32_bit_dest` unpack-mode
  requirement fires).

## Handoff points

none — the port stayed entirely within the op's own directory. No boundary-rule assumption violations
(no `sem::`/`tensor::` handle crossed an out-of-op call site — the op has no semaphores and constructs no
`TensorAccessor`); no kernel-lib gaps; no framework gaps; no cross-op kernel edits (the op owns both
compute sources). No capitulation.

## Successes

- **Compute `hw_config` — the "read the *resolved* settings, diff before/after" discipline
  ([Hardware configuration — Compute kernels](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md)) caught a real silent-regression trap.**
  The op *resolves* a full TTNN `ComputeKernelConfig` (`get_compute_kernel_config_args`,
  `program_factory.cpp:61`) but the legacy factory only threaded `math_fidelity` + `fp32_dest_acc_en`
  into its Metal `ComputeConfigDescriptor`, leaving `math_approx_mode` / `dst_full_sync_en` /
  `bfp8_pack_precise` at the **descriptor defaults** (all `false`). The obvious Style-A port
  (`to_compute_hardware_config(arch, resolved_config)`) would have imported the *resolved*
  `math_approx_mode` (which is `true` for the default `compute_kernel_config=None` — the doc's
  "helper defaults toward high performance" hazard) and silently flipped `sfpu_precision_mode`
  Precise→Approximate. Style B (build `ComputeGen1Config` directly, set only the two threaded knobs)
  reproduces the legacy values exactly, because the remaining `ComputeGen1Config` defaults
  (`sfpu_precision_mode=Precise`, `double_buffer_dest=true`, `bfp_pack_precision_mode=Approximate`)
  coincide with the legacy `ComputeConfigDescriptor` defaults. The port-recipe's insistence on the
  value-diff over the helper is exactly what surfaced this.
- **Self-loop pattern ([Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)) resolved all 10 DFBs cleanly.**
  Single-kernel program → every DFB is a one-toucher; PRODUCER+CONSUMER on the compute kernel (shared
  accessor name) satisfies the validator's ≥1P/≥1C rule with zero functional change, including the
  read-only `cos`/`sin`/`trans_mat` DFBs that have no FIFO ops at all.
- **`DFBAccessor::operator uint32_t()` bridged the runtime CB selection** (`in_cb`/`out_cb` chosen by
  `is_q`) and every compute-LLK call site (`matmul_tiles`, `mul_tiles_bcast`, `add_tiles`, `pack_tile`)
  without a single `.id` extraction or temp wrapper.

## Friction

- **Confusion (minor) — runtime CB selection isn't a named catalog pattern.** The kernel selects between
  two bound DFBs at runtime (`in_cb = dfb::q_in_cb; if (!is_q) in_cb = dfb::k_in_cb;`), which is neither
  [Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)
  (that's a *compile-time* `constexpr` alias) nor [Conditional bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)
  (that's `#ifdef` compile-time). The resolution: seed a runtime `uint32_t` from the `dfb::` handle via
  the constexpr `operator uint32_t()`, then construct the `DataflowBuffer` from that runtime id (the
  low-level `DataflowBuffer(uint16_t)` ctor, whose header comment says "prefer the DFBAccessor overload
  for new kernel code"). This preserves the legacy runtime-selection shape, kept deliberately to avoid
  duplicating the loop body (the kernel's own comments cite a TRISC2 code-size limit). A short catalog
  note — "runtime selection among several *bound* DFBs → runtime `uint32_t` from `dfb::` + low-level
  ctor, not an `.id` extraction" — would have removed a moment's doubt about whether this trips the
  `.id`-extraction anti-pattern (it does not).
- **Gap (minor) — compute-only + all-borrowed-DFB shape isn't a worked example.** The recipe's Example 1
  is reader/writer/DFB; this op has *no* dataflow kernels and *all* tensor I/O is borrowed-memory DFBs on
  a lone compute kernel. Everything needed was present (the `borrowed_from` field, the self-loop pattern,
  the audit's recipe-note acknowledging compute-only Device-2.0), but a one-paragraph worked example of
  "compute-only op, borrowed input+output DFBs, no DM kernels" would shorten the next such port.
- **Confusion (minor) — `hw_config` variant-of-variants.** `KernelSpec::hw_config` is
  `variant<DataMovementHardwareConfig, ComputeHardwareConfig>` and `ComputeHardwareConfig` is itself
  `variant<ComputeGen1Config, ComputeGen2Config>`. Assigning a bare `ComputeGen1Config` relies on a
  double implicit conversion; existing factories assign the inner `ComputeHardwareConfig` instead, so I
  matched that (`ComputeHardwareConfig compute_hw = ComputeGen1Config{...}`). Worth a one-line note in the
  Hardware-configuration section that Style B should wrap in `ComputeHardwareConfig`.

## Open items for downstream

- **Row-major kernel path is JIT/runtime-unverified (pre-existing coverage gap).** The factory selects
  `rotary_embedding_llama_sharded_row_major.cpp` at runtime when `row_major_QK` is true. The confirmed
  baseline (`test_rotary_embedding_llama_fused_qk.py`) is **tiled-only** (`TILE_LAYOUT` inputs), and the
  row-major fused-qk runner (`run_test_row_major_rotary_embedding_llama`,
  `tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py:296`) is
  **defined but never called** by any test, and is written for a multi-device (8×4) mesh
  (`ConcatMesh2dToTensor(mesh_shape=(8,4))`) so it does not run on a single card. Device kernels are
  JIT-compiled at op-execution time, not by `build_metal.sh`, so the row-major kernel's named-arg /
  `dfb::`-token JIT correctness is **not** exercised by any available test. It is host-compile-clean and
  its CB→DFB / named-arg transformation is byte-identical in shape to the tiled kernel (which passes
  fully). Recommend adding a single-device row-major fused-qk test (or wiring up the existing runner) so
  both halves of this atomic factory are runtime-covered. This gap predates the port.
- **Cross-op kernel touches:** none — both kernel sources live in the op's own directory.
- **`unused_cores` RTA completeness (not a defect, a behavior-preserving detail).** Legacy placed the
  kernel/CBs on `all_cores_bb` (the cos/sin grid's bounding box) but set the `is_q` RTA only on
  `q_cores ∪ k_cores`, leaving any bounding-box remainder at the zero default (⇒ k branch). Metal 2.0
  requires an RTA on every target node, so the port sets `is_q=0` on `unused_cores` explicitly — same
  effective values. In the tested configs `unused_cores` is empty (rectangular fully-tiled grid), so
  this is inert there; it is a guard for configs where the cos/sin grid is non-rectangular.
