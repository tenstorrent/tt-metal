# QUASAR_UPLIFT_REPORT — `ttnn.experimental.minimal_matmul`

**Status: RED — Not Metal 2.0 on Gen1 yet. Uplift stopped at the §1 gate; no changes made.**

- **Op:** `ttnn.experimental.minimal_matmul` (exercised by
  `models/experimental/llama32_1b_quasar/tests/graph_ops/test_minimal_matmul.py`; the test's two
  captured cases route through the single-variant `program_factory_t = std::variant<MinimalMatmulProgramFactory>`
  in `device/minimal_matmul_device_operation.hpp:24`).
- **Op directory:** `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/` (unchanged; nothing moved or renamed).
- **Recipe:** `docs/source/ttnn/ttnn/ai/quasar_porting.md` §1 workflow step 1 (the Metal-2.0 gate) +
  RED-stop condition "Not Metal 2.0 on Gen1 yet"; gate criteria cross-checked against
  `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/metal2_audit.md` and
  `ai/audit/quasar_audit.md`.

## Why RED (gate evidence, file:line)

The Quasar uplift assumes an already-Metal-2.0 op: factory on `create_program_artifacts` →
`ProgramArtifacts` with `dfb::`/`args::`/`tensor::`/`scratch::` bindings, kernels consuming those
named bindings. This op fails the gate on both sides:

**Host factory — legacy imperative builder (pre-`ProgramDescriptor`, two migrations behind Metal 2.0):**
- `device/minimal_matmul_program_factory.cpp` uses `tt::tt_metal::CreateSemaphore` (:337–342, :452),
  `CreateKernel` (:549, :594, :633, :672, :702), `SetRuntimeArgs` (:849–:928), and a
  `MinimalMatmulProgramFactory::cached_program_t` create/override pair (:975, :1006).
- Zero occurrences of `create_program_artifacts`, `ProgramArtifacts`, `KernelSpec`,
  `DataflowBufferSpec`, or even `create_descriptor`/`ProgramDescriptor` anywhere under the op
  directory. Per `metal2_audit.md`, an imperative-builder op fails the TTNN-factory-concept
  prerequisite gate; the `ProgramDescriptor` migration is a separate workstream that precedes the
  Metal 2.0 port, which in turn precedes this uplift.
- The sibling `device/minimal_matmul_fabric_bound_program_factory.cpp` is the same legacy shape
  (:334–339 `CreateSemaphore`, etc.), so no clean already-M2 factory subset exists either.

**Kernels — Device 2.0 compliant, but pre-Metal-2.0 idioms throughout:**
- CB-index-keyed buffers, not named DFBs: `tt::CBIndex::c_0/c_2/c_4/c_5/c_6`
  (`device/kernels/dm_in0_sender.cpp:108–151`, and equivalents in `dm_in1_sender_out.cpp`,
  `compute.cpp`, `matmul_dataflow_common.hpp` — `CircularBuffer cb(cb_id)` at
  `matmul_dataflow_common.hpp:104,165,258–259`).
- Positional args, not `get_arg(args::…)`: `get_compile_time_arg_val(0..20)` and a long
  `get_arg_val<uint32_t>(argidx++)` block (`dm_in0_sender.cpp:18–61,164–218`).
- Buffer-address RTAs + offset-keyed `TensorAccessorArgs<22>()` /
  `TensorAccessorArgs<…next_compile_time_args_offset()>` instead of bound `tensor::` parameters
  (`dm_in0_sender.cpp:42–44,67–79,114–118,189–195`).
- Semaphore ids passed as CTAs into `Semaphore<>` wrappers (`dm_in0_sender.cpp:32–34`) rather than
  Metal 2.0 semaphore bindings.

Note the kernels *are* Device 2.0 (device-2.0 `api/dataflow/*`, `api/compute/*` includes; `Noc`
objects with `noc.async_read/write` methods; `CircularBuffer` wrapper objects — no bare
`dataflow_api.h`, `cb_*` free functions, or `get_local_cb_interface`). So the Device 2.0
prerequisite of the eventual Metal 2.0 port looks already met; only the host-side
`ProgramDescriptor` → Metal 2.0 chain is missing. That chain is out of scope for this uplift
session by the recipe's own ordering.

## Files changed

**None.** No source, header, kernel, or build file was modified. The only file added is this
report (to be deleted before any merge, per the recipe).

## §7–§8 gotchas: applied / considered

- **Applied: none.** The RED gate fires before the uplift begins; the recipe forbids performing the
  Metal 2.0 port under this workflow, and §7–§8 fixes are reactive (no device run happened here).
- **Considered and recorded for the future uplift** (informational only — none acted on):
  - **Non-zero-init semaphores (quasar_audit.md check 2 — will be an uplift blocker):**
    `CreateSemaphore(program, core_grid, VALID)` with `VALID == 1`
    (`hostdevcommon/common_values.hpp:14`) at `device/minimal_matmul_program_factory.cpp:339,342`
    (`in0_valid_semaphore`, `in1_valid_semaphore`) and
    `device/minimal_matmul_fabric_bound_program_factory.cpp:336,339`. These do not port to Quasar;
    removing the dependency is an op-owner change to plan into the Metal 2.0/uplift work.
  - **Sender/receiver mcast-style semaphore signaling and per-core NoC coordinates as RTAs** — will
    interact with §11 (single-direction NoC, top-left-only mcast, degenerate-grid clamps) once the
    op reaches Quasar bring-up.
  - **`compute_kernel_hw_startup` (§7)** — `compute.cpp` includes
    `api/compute/compute_kernel_hw_startup.h`; the once-at-`main()`-start rule must be verified
    during the port.
  - Not applicable / not evaluated further (no M2 port exists to audit): `get_entry_size()` vs
    `fifo_page_size` (§5/§8.3), DM self-loop DFBs (§6), `unpack_modes` (§4), implicit-sync rules
    (§7) — these are properties of the Metal 2.0 artifacts this op does not yet have.

## Deferred / follow-up items

1. **Run the Metal 2.0 pre-port audit** (`ai/audit/metal2_audit.md` → `METAL2_PREPORT_AUDIT.md`) and,
   if GREEN, the in-place Metal 2.0 port (`ai/port/metal2_port.md`) — noting the op is still on the
   legacy imperative builder, so the `ProgramDescriptor` migration prerequisite likely gates even
   that audit. This is the entire blocker for the Quasar uplift.
2. **Non-zero-init (`VALID`) semaphores** (both factories, lines above) — op-owner change required
   before any Quasar uplift can go GREEN.
3. **Out-of-directory kernel coupling** to record for the M2 audit: the DM kernels include
   `ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp`
   (`dm_in0_sender.cpp:14`, `dm_in1_sender_out.cpp:14`) — a shared/donor dependency the Metal 2.0
   port must handle per the shared-kernel rules (out of scope for this op-directory-only session).
4. The `fabric_bound` factory and the `minimal_matmul_split` operation in the same directory share
   this status and these kernels; scope them together in the M2 audit.

## WH/BH parity claim (structural — no device run this session)

The working-tree diff for this op directory is **zero** (this report file is the only addition, and
markdown is not compiled or shipped). A zero diff cannot change WH/BH behavior; parity holds
trivially. Nothing was copied into or cited from `ttnn/cpp/ttnn/operations/experimental/quasar/`,
and no namespace/directory changes were made.

## Test commands (user-run; nothing was built or executed in this session)

BH / WH parity (run on each Gen1 target):

```bash
# Op unit tests (nightly suite)
pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py

# The model graph-op cases this uplift was scoped to
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_minimal_matmul.py
```

Quasar (emulator; run only after a future Metal 2.0 port + uplift — expected to fail/RED today):

```bash
TT_METAL_FORCE_JIT_COMPILE=1 \
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_minimal_matmul.py
```

(Per recipe §9: run Quasar both with `TT_METAL_LLK_ASSERTS` on and off once an uplift exists;
irrelevant until the M2 port lands.)
