# Metal 2.0 Port Report — `data_movement/fill_rm`

## Outcome

`PORTED` — the op's single factory (`FillRMProgramFactory`) converted to `MetalV2FactoryConcept`. Build clean (`./build_metal.sh --build-tests`, 0 errors). Tests **passed** on `wormhole_b0`:
- `tests/tt_eager/python_api_testing/unit_testing/misc/test_fill_rm.py::test_fill_rm` — correctness assert (bit-exact `torch.equal`), `fill_ones_rm`, shape 1×1×32×32.
- `tests/ttnn/docs_examples/test_data_movement_examples.py::test_fill_rm` — `fill_rm` **proper** (explicit `val_hi`/`val_lo`), larger shape 2×3×64×96 (exercises the NC loop and multi-tile W=96); smoke (logs shape, no correctness assert).
- `tests/ttnn/docs_examples/test_data_movement_examples.py::test_fill_ones_rm` — `fill_ones_rm`, same larger shape; smoke.

Runtime log confirms the Metal 2.0 DFB path is live (`Finalize dfb: ... dfb size: 32`).

## Provenance

- **Recipe docs (this port):** `7ca84865be5 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `7ca84865be5 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — `create_descriptor()` (returned `ProgramDescriptor`) replaced by `create_program_artifacts()` (returns `ProgramArtifacts`), same 3-arg signature `(FillRmParams, FillRmInputs, Tensor&)`. Matches the audit's decision; no re-decision.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op had none — default reflection hash).
- Pybind entry points removed: none (`fill_rm_nanobind.cpp` binds `&ttnn::fill_rm` / `&ttnn::fill_ones_rm` plain functions, not a factory entry point).

### Open items
- No relaxation candidates: the sole `TensorParameter` (`out`) uses strict spec matching; `W` is hashed (default reflection hash), so no dynamic-shape relaxation is warranted. Confirmed no `ArgConfig::Runtime*` in the kernel.
- No custom-hash / pybind / concept-fit friction — this is the short "success case" the doc anticipates.

## Handoff points

None. The port stayed entirely within the op's own directory; no kernel-lib / LLK / shared-kernel changes, no `sem::`/`tensor::` boundary-crossing call sites, no removed pybind surface. No capitulation.

## Successes

- **Self-loop DFB pattern (catalog) fit the two single-toucher CBs cleanly.** [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md). Re-deriving the census from the kernel body confirmed the brief: each of `in0`/`in1` is touched by exactly one kernel (the reader FIFO-produces and uses the DFB as the NoC write source, no distinct consumer), so binding the reader PRODUCER+CONSUMER on each (shared accessor name → single `dfb::in0`/`dfb::in1` handle) satisfied the ≥1P/≥1C validator with no multi-binding flag. `device/fill_rm_program_factory.cpp:76-83`.
- **TensorAccessor collapse worked as documented.** The legacy `TensorAccessorArgs<0>()` + buffer-address RTA + explicit page-size 3rd arg all collapsed to one line `TensorAccessor(tensor::out)` (`device/kernels/dataflow/fill_rm_interleaved.cpp:28`); the Class-2 page-size drop the audit called out needed no `dynamic_tensor_shape`.

## Friction

### Gaps
- **Whitelist rule 7 (`get_tile_size(cb_id)` → `dfb.get_tile_size()`) has no data-movement-kernel form.** The DFB metadata member getters in `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h` are gated behind `#ifdef DFB_DESCRIPTORS_DEFINED`, which is only set `#if __has_include("chlkc_descriptors.h")` — a **compute/TRISC** JIT artifact absent from a data-movement kernel TU. So `dfb.get_tile_size()` does **not** compile in a DM kernel, even though the legacy free function `get_tile_size(cb_id)` did (Device 2.0 keeps that free fn for DM). In `fill_rm` the sole `get_tile_size(cb_id_in0)` call fed a **dead** local (`num_bytes_per_tile`, audit misc-anomaly), so removing the magic CB id left the line un-rewritable *and* unnecessary — I dropped the single dead line (see [Open items for downstream](#open-items-for-downstream)). **Doc suggestion:** rule 7 / the CB→DFB whitelist §A should note that the object-getter mapping is compute-kernel-only; a *live* `get_tile_size(cb_id)` in a DM kernel would have no mechanical translation and would be a genuine blocker, not a one-liner.

### Confusion
- **`assert.hpp` include path.** `TT_FATAL` lives at `<tt_stl/assert.hpp>`, not `<tt-metalium/assert.hpp>` (my initial guess, a build failure). Minor / self-inflicted; noting only so a future porter greps the sibling ops (`data_movement/transpose/**`) for the canonical include rather than guessing.

## Open items for downstream
- **Correctness-assert coverage is thin (smoke coverage is broader).** The only test that *asserts numerical correctness* is `test_fill_rm.py::test_fill_rm` — one tiny shape (1×1×32×32, `fillH=fillW=31`) via `fill_ones_rm`. The `docs_examples` tests exercise `fill_rm` proper (explicit `val_hi`/`val_lo`) and the larger 2×3×64×96 shape, but they only *smoke* the op (log the output shape, no `torch.equal`/pcc check). Legacy tt_lib sweep configs (`pytorch_fill_rm_test.yaml`, `pytorch_fill_ones_rm_test.yaml`, driven by `sweep_tests/{tt_lib_ops,pytorch_ops,op_map}.py`) add more shapes but run under the sweep harness, not plain pytest. So no *asserting* test covers `fill_rm` with `val_hi≠1`/`val_lo≠0` or multi-tile W. A follow-up adding a correctness assert to (or alongside) the `docs_examples` case would close that gap — noted, not acted on (out of port scope). Model consumers exist (`models/experimental/bert*/fused_ops/layernorm.py`, `ttnn/tt_lib/fused_ops/layernorm.py`) but aren't part of this op's no-regression baseline. `test_deepseek_mla_ops.py` does **not** call the op (the apparent match was a substring of `test_prefill_rmsnorms`).
- **Dead kernel locals (ops-team cleanup).** `fill_rm_interleaved.cpp` declares locals that are computed but never read (`num_bytes_per_tile`, `num_bytes_per_tile_row`, `Wt`, `replicate_dest_addr`, `start_dram_addr_offset_for_tensor_row`). The audit flagged these as misc anomalies, out of port scope. The port left them in place — with one forced exception: `num_bytes_per_tile = get_tile_size(cb_id_in0)` referenced a magic CB id the port removes, so its RHS was mechanically rewritten to `dfb_in0.get_tile_size()` (whitelist rule 7). The value is still dead; removing the whole line is the ops-team's call.
