# QUASAR_UPLIFT_REPORT — transformer/sdpa (SDPA prefill)

**Status: RED — Not Metal 2.0 on Gen1 yet.**

The Quasar uplift did not proceed. Per `docs/source/ttnn/ttnn/ai/quasar_porting.md` §1 and the
RED-stop conditions, an op whose factory is still `create_descriptor` → `ProgramDescriptor` and
whose kernels are on the legacy device API is not eligible for a Quasar uplift; the Metal 2.0
port (`ai/port/metal2_port.md`) must land first. A RED result is a success of the audit — it
stops a bad port. **No source files were changed.**

## Op / test mapping

- Test: `models/experimental/llama32_1b_quasar/tests/graph_ops/test_scaled_dot_product_attention.py`
- Op: `ttnn.transformer.scaled_dot_product_attention` — causal prefill, interleaved DRAM,
  Q `[1,32,1024,64]` bf16, K/V `[1,8,1024,64]` bfp8_b, `SDPAProgramConfig` (8×8 grid,
  q/k_chunk 64), `is_causal=True`, no attn_mask, no page table, no sliding window.
- Code path exercised:
  - Host: `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp` →
    `device/sdpa_device_operation.cpp` → `device/sdpa_program_factory.cpp`
    (`SDPAOperation::SDPAProgramFactory`, the only factory in
    `program_factory_t = std::variant<SDPAProgramFactory>`).
  - Kernels: `device/kernels/dataflow/reader_interleaved.cpp`,
    `device/kernels/dataflow/writer_interleaved.cpp`, `device/kernels/compute/sdpa.cpp`.

## Gate evidence (why RED)

Metal 2.0 markers required by quasar_porting.md §1 are absent on every leg of the exercised path:

- **Factory is legacy descriptor-era.** `device/sdpa_program_factory.cpp:250` —
  `ProgramDescriptor SDPAOperation::SDPAProgramFactory::create_descriptor(...)`; builds a
  `ProgramDescriptor` (line 768) with `KernelDescriptor` + CB descriptors. There is **no**
  `create_program_artifacts` / `ProgramArtifacts` anywhere under
  `ttnn/cpp/ttnn/operations/transformer/sdpa/` (repo-wide grep of the op directory: zero hits).
- **Kernels are on the legacy device API.** All three kernels include
  `api/dataflow/dataflow_api.h` / `api/dataflow/circular_buffer.h` (reader/writer) and use
  `cb_reserve_back`/`cb_push_back`/`cb_wait_front` sync, positional `get_arg_val<uint32_t>(i)`
  RTAs, and address-RTA tensor addressing (`q_addr`/`k_addr`/`v_addr`/`out_addr`). There is no
  `experimental/kernel_args.h`, no `dfb::`/`args::`/`tensor::`/`scratch::` binding token, no
  `DataflowBuffer` object usage in any kernel of this op.

Per the recipe: "A kernel still on the legacy device API — even if the factory looks M2 — is not
ported; stop and run the Metal 2.0 port first." Here neither the factory nor the kernels are
ported.

## Files changed

None. (The op's directory and namespace were not touched; nothing tempted a move or rename.)

## §7–§8 gotchas: applied vs. considered

- **Applied: none.** All §7–§8 fixes presuppose an already-Metal-2.0 op; none is applicable to a
  descriptor-era op, and the recipe forbids applying them pre-emptively (they are reactive, and
  no device run was performed in this session).
- **Considered and recorded for the future port/uplift:**
  - **§8.5 intra-tensix DFB tile-counter aliasing — SDPA is a named candidate.** The compute
    kernel is heavy on compute-local intermediate CBs (`cb_qkt_im`, statistics CBs, etc.), which
    become intra-tensix DFBs (counter indices 16–31) after the M2 port. The fix (tile-counter
    remap, each DFB taking 2 indices) is **runtime-owned — nothing at the op level**; recorded
    here as a deferred/runtime item, not an op edit.
  - **§7 non-zero-init semaphores:** the exercised causal path creates no semaphores; the
    non-causal KV-chain-forwarding path creates three (`sender`/`receiver`/`valid`,
    `sdpa_program_factory.cpp` ~652–670). Their init values must be checked (zero-init required)
    during the eventual uplift audit — out of scope for the causal path this test exercises.
  - **§7 Int32-only / no uint16-uint32 device formats, §4 `unpack_modes` for FP32 DFBs,
    §5 `get_entry_size()` vs `fifo_page_size`, §11 NoC/multicast normalization:** all noted as
    checks for the post-M2 uplift; not statically actionable on the legacy code.

## Deferred / follow-up items

1. **Prerequisite: in-place Metal 2.0 port of `SDPAProgramFactory` + its three kernels**
   (`ai/audit/metal2_audit.md` → `METAL2_PREPORT_AUDIT.md`/`METAL2_PORT_BRIEF.md`, then
   `ai/port/metal2_port.md`). Only after that does the Quasar-uplift audit
   (`ai/audit/quasar_audit.md`) apply.
2. **Shared-kernel blast radius for the future port:** `device/kernels/dataflow/dataflow_common.hpp`
   and `device/kernels/compute/compute_common.hpp` are shared with the other SDPA variants in this
   directory (joint / ring / sparse / windowed), and `dataflow_common.hpp` is also included by
   kernels in the sibling `transformer/sdpa_decode` op (owned by a different workstream — not
   touched here). The M2 port will need the shared-kernel `_metal2`-fork rule from
   `metal2_port.md`, coordinated with the sdpa_decode owner.
3. **Runtime-owned:** §8.5 intra-tensix DFB tile-counter aliasing remap (SDPA is a listed
   candidate) — track with the runtime team; no op-level action.

## WH/BH parity claim

Trivially holds: the working-tree diff for this op is **zero source changes** (this report file
only), so WH/BH behavior is unchanged by construction. No device tests were run in this session
(recipe §9: the user runs all builds/tests).

## Test commands (for the human to run)

BH/WH parity baseline (current mainline behavior — also the control for any future port):

```
pytest tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py
```

Model-graph case this report is scoped to (same command on BH/WH for parity, and on the Quasar
emulator after the future M2 port + uplift; force JIT if kernels have changed):

```
TT_METAL_FORCE_JIT_COMPILE=1 pytest "models/experimental/llama32_1b_quasar/tests/graph_ops/test_scaled_dot_product_attention.py::test_scaled_dot_product_attention[00_32x1024x64_bf16_int-dram]"
```

---
*Session note: audit-only run per the uplift recipe; the recipe branch's session constraints
prohibited performing the Metal 2.0 port here. Delete this report before merge.*
