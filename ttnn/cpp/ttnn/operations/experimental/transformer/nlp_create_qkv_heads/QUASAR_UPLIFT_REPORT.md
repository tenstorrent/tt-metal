# QUASAR_UPLIFT_REPORT — `ttnn.experimental.nlp_create_qkv_heads`

**Op directory:** `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/`
**Driving test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_create_qkv_heads.py`
(1 captured signature, 32 calls: input `[1,1,1024,3072]` bf16 TILE INTERLEAVED/DRAM, `num_heads=32`,
`num_kv_heads=8`, `transpose_k_heads=False`, interleaved DRAM outputs → selects the **Interleaved**
program factory, fused-QKV single-input branch, no transpose compute kernel.)
**Date:** 2026-09-01 (branch `vsuresh/quasar-porting-recipe`, based on `origin/main`)

---

## Status: RED — Not Metal 2.0 on Gen1 yet

This is the first RED-stop condition in `quasar_porting.md` ("factory still
`create_descriptor`/`ProgramDescriptor`"). Per the recipe, the uplift stops here: the op needs a
base Metal 2.0 port (`ai/port/metal2_port.md`) first, and this session was explicitly instructed
**not** to perform that port. A RED result is a success of the audit — it stops a bad port.

### Gate evidence (§1 of `quasar_porting.md`)

Host factory — `device/nlp_create_qkv_heads_program_factory.cpp` is legacy descriptor-based on
**both** paths; there is no `create_program_artifacts`/`ProgramArtifacts` anywhere in the op:

- `Interleaved::create_descriptor(...)` → `ProgramDescriptor desc;` (lines 84, 158) — the path the
  driving test exercises.
- `Sharded::create_descriptor(...)` → `ProgramDescriptor desc;` (lines 503, 510).
- Runtime args are positional address-derived slots (buffer base addresses + per-core tile ids),
  the legacy RTA idiom; `grep create_program_artifacts|ProgramArtifacts|kernel_args.h` over the op
  directory returns nothing.

Device kernels — all three are on the legacy device API (the recipe's explicit "not ported" marker,
even though the include paths carry the `api/dataflow/` prefix):

- `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp`:
  `#include "api/dataflow/circular_buffer.h"`, `CircularBuffer cb_qv(cb_id_qv)` with raw CB indices
  (`constexpr uint32_t cb_id_qv = 1;`), positional `get_arg_val<uint32_t>(0..)`, address-RTA +
  `constexpr auto in0_args = TensorAccessorArgs<2>();`, `get_tile_size(cb_id)`.
- `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp`: same idiom
  (positional `get_arg_val`, `TensorAccessorArgs<5>()`, raw CB ids).
- `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp`: same family
  (sharded path, not exercised by the driving test).

No `dfb::` / `args::` / `tensor::` / `scratch::` bindings, no `DataflowBuffer` objects, no
`get_arg(args::…)`, no `get_entry_size()` — the op has not been through the Metal 2.0 port at all.

## Files changed

**None.** No source file was modified; this report is the only artifact. (The op's directory and
namespace were never in question — nothing tempted a move or rename.)

## §7–§8 gotchas applied / considered

**Applied: none** — the workflow gates on the Metal 2.0 status *before* the Quasar-uplift audit,
and the gate failed, so no §7–§8 fix is applicable (they presuppose an M2 op, and most are reactive
to device symptoms; no build or device run was performed per session constraints).

Observations recorded in passing, for whoever runs the future Metal 2.0 port + uplift (not acted on):

- **No semaphores at all** in either factory → `quasar_audit.md` check 2 (non-zero-init semaphores)
  will trivially pass.
- **No DM self-loop candidates visible**: the interleaved path is reader → CB → writer with two
  distinct kernels per CB (a normal FIFO); classification must be redone post-port on the actual
  `dfb_bindings` (`quasar_audit.md` check 1 / `cb_dfb_quasar_audit_helper.md`).
- **Sharded path CBs are globally allocated on tensor shards** (q/k/v CBs overlay output shards) —
  the future M2 port will express these as `borrowed_from` DFBs; capacity checks then apply (§6).
- **Data formats are bf16/tile on the model path** — no uint16/uint32 device-format branch in the
  kernels, so the §7 "Quasar has Int32, no uint16/uint32" quirk has no visible foothold here.
- **`transpose_k_heads=true`** pulls in the shared compute kernel
  `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` (out-of-op-dir) — see Deferred.

## Deferred / follow-up items

1. **Base Metal 2.0 port of this op** (`ai/port/metal2_port.md`): both factories
   (`Interleaved`, `Sharded`) and all three dataflow kernels. This is the prerequisite for any
   Quasar uplift and is a separate, deliberate task — not started here by instruction.
2. **Shared compute kernel** `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`
   (the `transpose_k_heads=true` branch of the Interleaved factory): shared with other ops and
   outside this op's directory. The future M2 port must either find/create a `_metal2` fork beside
   the original per `metal2_port.md` "Porting a shared kernel", or confirm a ported fork already
   exists. Out of scope for this session (no edits outside the op dir).
3. **Sibling op** `nlp_create_qkv_heads_decode` is owned by a different workstream — nothing here
   touches it, and no coordination need was discovered.
4. Re-run the Quasar-uplift audit (`ai/audit/quasar_audit.md` + `quasar_porting.md` §7–§12) *after*
   the M2 port lands green on WH/BH.

## WH/BH parity claim

**Trivially holds: the diff against `origin/main` for this op is empty** (zero source changes;
the only new file is this uncommitted report, which is not compiled or shipped). WH and BH take
exactly the code path they took before this session. No device run was performed (per session
constraints), and none is needed to support a zero-diff parity claim.

### Test commands for the human (verification / future baseline)

BH / WH parity baseline (mainline op behavior; also the pre-port baseline the future M2 port
must capture **before** its first kernel edit):

```bash
# Model-path geometry (the Quasar-capture case), on WH or BH:
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_create_qkv_heads.py

# Op's own suites (interleaved + sharded coverage, program-cache):
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py
pytest tests/ttnn/unit_tests/operations/experimental/transformer/test_nlp_create_qkv_heads_program_cache.py
```

Quasar (emulator, once an M2 port + uplift exist — not runnable today, op is pre-M2):

```bash
TT_METAL_FORCE_JIT_COMPILE=1 \
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_nlp_create_qkv_heads.py
```

(Per §9: run Quasar both with `TT_METAL_LLK_ASSERTS` on and off; purge
`~/.cache/tt-metal-cache` between baseline-era and post-port runs.)

## RED-stop conditions checked

| Condition | Result |
|---|---|
| Not Metal 2.0 on Gen1 yet (`create_descriptor`/`ProgramDescriptor`) | **HIT — this is the RED** |
| Required capability missing from sanctioned Quasar API | n/a (not reached; none observed) |
| Construct needing an owner decision (non-zero semaphore, DM self-loop, open HW bug) | n/a (not reached; no semaphores, no self-loop candidates seen) |
| Only fix changes WH/BH un-guarded / would need `experimental/quasar/` copy | n/a (no fix attempted) |
| LLK stub/unported on the op's path | n/a (dataflow-only on the test path; not evaluated) |

---

*This report is intentionally left uncommitted for review and should be deleted before any merge,
per the recipe's deliverable rules. Nothing was copied from or modeled on
`ttnn/cpp/ttnn/operations/experimental/quasar/`.*
