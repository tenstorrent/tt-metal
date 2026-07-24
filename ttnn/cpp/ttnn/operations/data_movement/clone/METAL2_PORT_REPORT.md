# Metal 2.0 Port Report — `data_movement/clone`

## Outcome

**PORTED.** The single `CloneOperation` factory converted to `MetalV2FactoryConcept` (`CloneProgramFactory::create_program_artifacts`), covering all four config-paths ({tilized, row-major} × {interleaved, sharded}) plus the optional dtype-conversion compute kernel. All nine kernels ported. Verification: `./build_metal.sh --build-tests` clean (exit 0); `tests/ttnn/unit_tests/operations/data_movement/test_clone.py` = **118 passed, 24 skipped** (all skips are pre-existing conditional skips: int32 dtype rules, odd-last-dim row-major, non-tilized dtype conversion — none port-related). Anti-pattern self-audit clean (no `buffer()->address()`, no magic CB indices, no `TensorAccessorArgs<N>()`, no `.id` extraction, no varargs, no CTA→RTA demotion, no multi-binding flag; all CTAs named; `CircularBuffer`/`CBDescriptor` absent from code).

## Provenance

- **Recipe docs (this port):** `4af0db51e3c 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `4af0db51e3c 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — `CloneProgramFactory::create_program_artifacts` (`clone_device_operation.hpp`, `clone_program_factory.cpp`). Single-program; no op-owned tensors; tensor matching strict (default). Matches the audit's decision.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op never had one).
- Pybind entry points removed: none (`clone_nanobind.cpp` binds only the `clone` free function; no `create_descriptor`/device-op-class binding).
- Concept wiring (forced by the port): `CloneOperation` moved off `HasDirectDescriptor` (`create_descriptor` directly on the op) to `using program_factory_t = std::variant<CloneProgramFactory>;`. The `create_descriptor` declaration and the `<tt-metalium/program_descriptors.hpp>` include were removed from `clone_device_operation.hpp`; `ttnn/metal_v2_artifacts.hpp` added. No other device-op-class member touched (`validate_inputs`, `compute_output_specs`, `create_output_tensors`, perf model, `ttnn::prim::clone` unchanged).

### Open items
- None on the factory layer beyond the CRTA-candidate note below.

## Handoff points
None. The port is fully within the op directory: no `sem::`/`tensor::` boundary violations, no kernel-lib gaps, no cross-op kernel touches (clone owns all nine kernels), no framework gaps, no removed pybind surface.

## Successes

- **Multi-config single factory framing.** The recipe's [Multi-variant factories](../shared/port_patterns.md#pattern-multi-variant-factories) pattern and the atomic-unit note made the "one `create_descriptor` → four kernel-source pairs + optional compute" shape straightforward to convert as one unit — no ambiguity about porting sub-paths independently.
- **[Demoting per-group CTA to RTA](../shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) fired correctly.** The obvious "collapse compute to one KernelSpec with `num_tiles` as an RTA" simplification was flagged by the catalog; kept the two-`KernelSpec` / two-`WorkUnitSpec` multiplicity with `num_tiles` a CTA (`clone_program_factory.cpp`, `make_compute`).
- **Case 2 bridge (`get_bank_base_address`).** The brief's explicit Case-2 guidance ("pull the base via `get_bank_base_address`, leave the raw local-L1 walk unchanged") mapped cleanly onto the four sharded kernels — one line each, walk untouched.
- **Dead RM-interleaved CTA resolved for free.** The audit's misc-anomaly (dead `input_unit_size`/`output_unit_size` CTA at index 1 on `read_kernel_rm.cpp`/`write_kernel_rm.cpp`) needed no special handling: the whole positional CTA list is replaced by named bindings, and the dead value had no kernel consumer to name, so it simply doesn't reappear. The live RTA copy (`stick_size`) survives and drives the kernel.

## Friction

### Gaps
- **DFB producer/consumer WorkUnitSpec-membership rule under-specified across docs.** The migration guide's troubleshooting phrasing ("a local DFB's producer and consumer kernels must share identical `WorkUnitSpec` membership") reads stricter than the actual validator, which (`program_spec.cpp:1337-1342`) enforces a purely **per-node** census ("counts actual node occupancy … in node terms rather than `WorkUnitSpec` terms"). Had to read the validator to confirm reader/writer need not be replicated into per-group WUs for the census — though I kept the anti-pattern doc's replicate-into-each-compute-WU shape anyway since it is the documented multi-group pattern and keeps compute-group WUs node-disjoint.

### Confusion
- **Single-header nested-types op vs. the factory-struct wiring.** Every worked `MetalV2FactoryConcept` example (s2i, reduction) has standalone `operation_attributes_t`/`tensor_args_t` in a `*_device_operation_types.hpp`, so the factory `.hpp` includes the types and the device-op `.hpp` includes the factory. Clone nests its types inside `CloneOperation`, so that layering inverts (factory needs the op's types; op's `program_factory_t` needs the factory). Resolved by forward-declaring `CloneProgramFactory` before `CloneOperation` and defining it after, in the same header — avoids a new types header / larger device-op refactor. Docs give no guidance for the nested-types shape; a one-line note in `ttnn_factory.md` would help.

## Open items for downstream

- **`start_id` / `stick_size` RTAs are node-invariant → CRTA candidates.** `stick_size` (= `input_unit_size`/`output_unit_size`) is identical on every node; `start_id` on the interleaved paths is a genuine per-node prefix sum. The recipe forbids RTA→CRTA conversion during a port (dispatch-semantics change). Flagging `stick_size` as a later CRTA cleanup for the ops team.
- **Relaxation candidates:** none identified. Tensor matching left strict (default).
