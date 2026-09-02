# QUASAR_UPLIFT_REPORT — transformer/sdpa_decode

**Op:** `ttnn::transformer::paged_scaled_dot_product_attention_decode` (and the sibling `scaled_dot_product_attention_decode` entry points — all four nanobind bindings route through the single `SdpaDecodeDeviceOperation`).
**Op directory:** `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/`
**Driving test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_scaled_dot_product_attention_decode.py` (captured op: `ttnn.transformer.paged_scaled_dot_product_attention_decode`; 1 signature — paged decode, HEIGHT_SHARDED bf16 Q [1,1,32,64], bf8_b paged KV [128,8,32,64], cur_pos + page_table tensors, SDPAProgramConfig 8×8 grid).
**Date:** 2026-09-01. Uncommitted, for review; delete before merge.

---

## Status: RED — Not Metal 2.0 on Gen1 yet

This is the first RED-stop condition in `quasar_porting.md` ("factory still `create_descriptor`/`ProgramDescriptor`"). Per the recipe, a RED result is a success of the audit: it stops a bad uplift. **The Quasar uplift was NOT performed and no source file was changed.**

### Gate evidence (mainline op, in place)

Host factory — legacy `ProgramDescriptor` API, not Metal 2.0:
- `device/sdpa_decode_device_operation.hpp:95` / `device/sdpa_decode_program_factory.cpp:29` — `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`. No `create_program_artifacts`, no `ProgramArtifacts`, no `program_factory_t` variant anywhere in the op.
- CBs are magic indices (`CBIndex::c_0` … `c_31`, `sdpa_decode_program_factory.cpp:540-613`), not named `DataflowBufferSpec`/`DFBBinding`.
- Tensor addressing is legacy positional CTAs + `TensorAccessorArgs(...).append_to(...)` (`sdpa_decode_program_factory.cpp:686-730`), not `TensorParameter`/`TensorBinding`.
- Semaphores are raw `SemaphoreDescriptor`s with hand-assigned ids (`sdpa_decode_program_factory.cpp:630-638`), not `SemaphoreBinding`s. (All three are zero-init, so `quasar_audit.md` check 2 would pass — noted for the future audit.)

Kernels — legacy CB/positional-arg device idiom (Device-2.0 include paths, but not the Metal 2.0 kernel API):
- `device/kernels/compute/sdpa_flash_decode.cpp`, `device/kernels/dataflow/reader_decode_all.cpp`, `device/kernels/dataflow/writer_decode_all.cpp`, `device/kernels/dataflow/dataflow_common.hpp` all use `cb_wait_front`/`cb_reserve_back`/`cb_push_back`/`cb_pop_front`, `get_arg_val<...>(i)`, `get_read_ptr`/`get_write_ptr` (15–28 hits per file). Zero uses of `dfb::` / `args::` / `tensor::` tokens, `experimental/kernel_args.h`, or `DataflowBuffer` objects.

### About PR #54249 "[Metal 2.0] Port `transformer/sdpa_decode`"

Commit `cafa17411f3` (in this branch's history) carries that title, but **every op-source file it adds lands under `ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/`** (19 files: its own factory, kernels, nanobind), plus repointing the `llama32_1b_quasar` model/tests at `ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode`. It does not touch this directory: `git diff cafa17411f3 HEAD -- ttnn/cpp/ttnn/operations/transformer/sdpa_decode/` is empty, and this directory is legacy at both ends.

Both `quasar_porting.md` (hard rule 1) and `metal2_audit.md` are explicit that `experimental/quasar/` copies are deliberately hacky, non-production bring-up forks that must not be cited, copied, or treated as the port. So for the purposes of this recipe the **mainline op is not Metal 2.0**, and the existence of the fork does not change the gate verdict. (Negative pointer only — nothing from that tree was read for or used in this report.)

---

## Files changed

**None.** Only this report was added. The op's directory, namespace, and every source file are untouched.

## §7–§8 gotchas applied / considered

- **Applied: none** — the gate failed before the uplift audit, and §7–§8 fixes are reactive (no device run in this session anyway).
- **Considered / pre-noted for the eventual uplift:**
  - §8.5 **intra-tensix DFB tile-counter aliasing** explicitly names SDPA as a candidate. The fix (remapping all DFBs/tile counters, 2 indices each) is **runtime-owned** — nothing at op level. Recorded as a deferred runtime item, not an op edit.
  - `quasar_audit.md` check 2 (non-zero-init semaphores): the current descriptor's three semaphores are all `initial_value = 0` — would pass.
  - The op's tree-reduction/mcast paths use both NoCs and reducer-core mcast; §11 (single forward NoC, top-left mcast, degenerate-grid clamp) will need attention at uplift time on small emulator grids.

## Deferred / follow-up items

1. **Mainline Metal 2.0 port of `transformer/sdpa_decode` is the prerequisite** — run `ai/audit/metal2_audit.md` → `METAL2_PREPORT_AUDIT.md`/`METAL2_PORT_BRIEF.md`, then `ai/port/metal2_port.md`, in place in this directory. The `experimental/quasar` fork from #54249 may inform the *owners'* reconciliation plan, but per the recipe it cannot be copied or cited as the port.
2. **Test routing:** the generated graph-ops test's `_OP` currently targets the `experimental/quasar` fork, not the mainline op. Once the mainline op is M2 + Quasar-uplifted, the model tests should point back at `ttnn.transformer.paged_scaled_dot_product_attention_decode`. Out of scope here (file lives outside the op directory).
3. **Runtime team:** §8.5 SDPA intra-tensix DFB tile-counter remap (see above) — verify it is in place before the eventual Quasar bring-up of this op.
4. Shared-kernel coupling: the compute kernel includes `ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp` and `dataflow_common.hpp` includes `../../../../sdpa/device/kernels/dataflow/dataflow_common.hpp` — the eventual port shares device headers with `transformer/sdpa` (owned by a separate workstream); coordinate before converting.

## Parity claim (WH/BH)

Trivially preserved: the diff to op source is **zero** (this report is the only new file, and it is documentation). WH/BH take exactly the code path they took before this session.

## Verification commands (user-run; no builds/tests were run in this session)

BH/WH parity (mainline op):
```
pytest tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py
pytest tests/ttnn/unit_tests/operations/sdpa/test_paged_sdpa_decode_flexible_geometry.py
pytest tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_decode.py
```

Quasar (emulator; per the craqsim runbook env — currently exercises the forbidden `experimental/quasar` fork, see deferred item 2):
```
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_paged_scaled_dot_product_attention_decode.py
```
