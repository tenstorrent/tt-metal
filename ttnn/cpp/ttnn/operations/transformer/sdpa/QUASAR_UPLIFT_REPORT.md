# QUASAR_UPLIFT_REPORT — transformer/sdpa (SDPA prefill)

*Run on the post-#54468-integration state of branch `vsureshTT/llama_quasar_uplift`
(`7aec097fe18 [Metal 2.0] Port transformer/sdpa (SDPA + JointSDPA factories) (#54468)`).*

This audit covers **two distinct code locations** with two distinct verdicts. Read both.

---

## (A) MAINLINE `ttnn/cpp/ttnn/operations/transformer/sdpa/` — **RED, still not Metal 2.0 on Gen1.**

**#54468 did NOT port mainline SDPA in place.** The changes landed as a **Quasar FORK** under
`ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa/` (see (B)); the production op at
`transformer/sdpa/` is untouched and remains descriptor-era. This is the same pattern as
sdpa_decode (#54249).

Gate evidence (mainline is still legacy on every leg):
- **Factory is descriptor-era.** `device/sdpa_program_factory.cpp`:
  `create_program_artifacts` count **0**, `create_descriptor` count **3**
  (`ProgramDescriptor SDPAOperation::SDPAProgramFactory::create_descriptor(...)` at line 250,
  builds a `ProgramDescriptor desc;` at line 768). No `ProgramArtifacts` anywhere in the mainline
  op directory.
- **Kernels are on the legacy device API.** `device/kernels/dataflow/reader_interleaved.cpp`,
  `writer_interleaved.cpp`, `device/kernels/compute/sdpa.cpp` (and the shared
  `dataflow_common.hpp` / `compute_common.hpp`) include the legacy `dataflow_api.h` /
  `circular_buffer.h`, use `cb_reserve_back`/`cb_wait_front`/`cb_push_back` sync, positional
  `get_arg_val<uint32_t>(i)` RTAs, and address-RTA tensor addressing. Zero Metal 2.0 markers
  (`experimental/kernel_args.h`, `dfb::`/`args::`/`tensor::`, `DataflowBuffer`) in any mainline kernel.

**Per the recipe (§1 RED-stop "Not Metal 2.0 on Gen1 yet"), an in-place mainline uplift is what a
GREEN would require, and it cannot start here.** The prerequisite is the in-place Metal 2.0 port of
mainline `SDPAProgramFactory` + its kernels (`ai/audit/metal2_audit.md` → `ai/port/metal2_port.md`),
*not* the experimental fork. **No mainline source was changed** (recipe forbids porting mainline in
this session and forbids copying from / into `experimental/quasar/`). A RED result is a success of
the audit — it stops a bad port.

---

## (B) FORK `ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa/` — what the test runs.

The graph_ops test now binds `_OP = ttnn.experimental.quasar.transformer.scaled_dot_product_attention`
(`sdpa_nanobind.cpp:151`), i.e. it exercises the **fork**, not mainline. Per the recipe the fork is a
"deliberately hacky, non-production copy" whose *structure* must never be copied into a mainline port;
this section audits it only because it is the test's actual target.

### Test path scoped

- Op: causal prefill, interleaved DRAM. Q `[1,32,1024,64]` bf16, K/V `[1,8,1024,64]` bfp8_b,
  `SDPAProgramConfig` (8×8 grid, q/k_chunk 64, `exp_approx_mode=False`), `is_causal=True`,
  `sliding_window=None`, `scale=0.125`. (`test_scaled_dot_product_attention.py`, case
  `00_32x1024x64_bf16_int-dram`.)
- Factory reached: **`SDPAProgramFactory`** (`device/sdpa_program_factory.cpp`). The **JointSDPA**
  factory (`device/joint_sdpa_program_factory.cpp`) is also M2-ported in the fork but is a *different
  op* (`joint_scaled_dot_product_attention`) and is **not** reached by this test.
- Kernels reached: `device/kernels/dataflow/{reader_interleaved,writer_interleaved}.cpp`,
  `device/kernels/compute/sdpa.cpp`, and the shared `dataflow_common.hpp`, `compute_common.hpp`,
  `compute_streaming.hpp`.

### Metal 2.0 verdict: **genuinely Metal 2.0** on the exercised path.

- **Factory:** `create_program_artifacts` → `ProgramArtifacts` (count 1, `create_descriptor` 0;
  signature at `sdpa_program_factory.cpp:191`, returns `ProgramArtifacts{.spec, .run_params}` at
  1848). JointSDPA factory likewise (`create_program_artifacts` 1 / `create_descriptor` 0).
- **Kernels:** device-2.0 API only — `api/dataflow/*` (`noc.h`, `dataflow_buffer.h`,
  `noc_semaphore.h`), `api/compute/*` (`compute_kernel_api.h`, `compute_kernel_hw_startup.h`),
  `experimental/kernel_args.h`; bindings `dfb::` / `args::` / `tensor::` and `get_arg(args::…)`;
  `DataflowBuffer` + the device-2.0 `CircularBuffer` wrapper (`api/dataflow/circular_buffer.h`).
  **No** `cb_wait_front`/`cb_reserve_back`/`get_arg_val` in the exercised reader/writer/compute.
  (The `dataflow_api.h`/`circular_buffer.h` include hits are the `api/dataflow/` device-2.0 headers,
  not the legacy top-level ones.)
- Every DFB carries a valid `data_format_metadata` (`sdpa_program_factory.cpp:632…`); no
  `DataFormat::Invalid` (§4). No `create_mesh_workload`/`MeshWorkload` framework block.

### Quasar-readiness verdict: **largely ready, but NOT clean-GREEN — one op-level construct on the test path is a Quasar concern, plus two runtime/owner-deferred items.**

Already Quasar-aware in the fork (no action):
- **No `disable_dfb_implicit_sync_for_all` / `disable_implicit_sync_for`** anywhere (§7 / §10 — relies
  on the Gen2 implicit-sync default). **No `evil_set_*`** hand-rolled Gen1 rewind at the DFB level.
- `reader_interleaved.cpp:60` — `#ifdef ARCH_QUASAR noc.write_zeros_l1_barrier()` for the Quasar iDMA
  zero-fill ack path (the §11 `async_write_zeros` normalization; WH/BH keep the read-barrier path).
- `compute_streaming.hpp:35` — `reduce_trigger_supported=false` under `#ifdef ARCH_QUASAR` (the
  packer→unpacker early-reduce handshake doesn't exist on Quasar; falls back to normal CB sync).

### §7–§8 gotchas: applied vs considered

- **Applied: none.** No device run was performed this session, so §7–§8 fixes (which are reactive) are
  not applied; and no *statically-determinable* `ARCH_QUASAR`-guarded fix is genuinely required that
  I could apply safely inside the audited directory (see the fifo_page_size item below for why the
  one real concern is **not** an op-level guard). **No files were changed** (mainline or fork).
- **Considered — genuine concern on the test path (§5 / §7 / §8.3): hand-rolled ring-rewind reading
  stale `fifo_page_size`.** `compute_streaming.hpp:97` `cb_push_back_hold_wr_ptr(cb_id, num_tiles)`
  does `get_local_cb_interface(cb_id).fifo_wr_ptr -= num_tiles * intf.fifo_page_size` (lines 100–101)
  to rewind the write pointer after `push_back`, and it **is on the exercised causal streaming path**
  (`compute/sdpa.cpp:142` "Streaming SDPA v2: direct cb_qkt_im writes"; call sites at
  `compute_streaming.hpp:1476` etc.). It is **not** `ARCH_QUASAR`-guarded. Two recipe rules bite:
  (a) §5/§8.3 — `fifo_page_size` read from `get_local_cb_interface` is **stale on Quasar** (a top
  cause of value-inflation / wrong-output), and (b) §7/§8.3 — a hand-rolled `g_dfb_interface`
  ring-rewind is exactly the construct a **mainline in-place port may not use**: the sanctioned Quasar
  DFB rewind API is absent (`evil_set_*` is `#ifndef ARCH_QUASAR`), so the correct action is to **flag
  the missing Quasar DFB rewind primitive for the runtime team**, not to hand-roll it. On the fork
  this is a latent Quasar-correctness risk that can only be confirmed with a device run; I did **not**
  edit the fork's hot compute path blindly (no safe static ARCH_QUASAR guard exists — the fix is a
  missing runtime primitive, and a wrong edit here would corrupt the streaming accumulator). Recorded
  as deferred/runtime-facing.
- **Considered — not on the test path:**
  - **§7 non-zero-init semaphore.** `sdpa_program_factory.cpp:814` sets
    `valid_sem.advanced_options.initial_value = VALID` (non-zero). It is created **only under
    `if (!is_causal)`** (KV-chain forwarding, lines 809–815); the test is `is_causal=True`, so no
    semaphore is created on the exercised path. The fork's own comment already flags it as WH/BH-only
    deprecated. Blocks the **non-causal** path's Quasar uplift until the op-owner removes the
    non-zero-init dependency (`quasar_audit.md` check 2) — deferred, not on this test's path.
  - §4 `unpack_modes` for FP32 DFBs, §11 NoC/multicast reverse-rectangle normalization: not triggered
    by this geometry (bf16/bfp8_b inputs; top-left-forward interleaved reads).

### Deferred / follow-up items

1. **Runtime-owned — §8.5 intra-tensix DFB tile-counter aliasing.** SDPA is a **named candidate**
   (§8.5). The compute kernel is heavy on intra-tensix intermediate DFBs (`cb_qkt_im`, stats/max/sum
   CBs, `im` CBs — `sdpa_program_factory.cpp:651…671`) which use tile-counter indices 16–31. The remap
   fix (all DFBs remapped, each taking 2 indices) is **implemented in the runtime, nothing at the op
   level** (DataflowBuffer.md §C3/§C7). Track with the runtime team; no op edit.
2. **Runtime-owned — missing Quasar DFB ring-rewind primitive.** The fork's `cb_push_back_hold_wr_ptr`
   substitutes for the Gen1-only `evil_set_read/write_ptr`. A production mainline port cannot copy it
   (§7). Flag the missing whitelisted Quasar DFB rewind API for the runtime team so both arches share
   one mechanism.
3. **Op-owner — non-causal non-zero-init `valid` semaphore** (item above) must be removed before the
   non-causal path is uplifted on Quasar.
4. **Prerequisite for GREEN — in-place Metal 2.0 port of MAINLINE `transformer/sdpa`.** The fork is not
   the sanctioned location; the recipe requires the uplift to land in `transformer/sdpa/` in the
   production namespace. That port must also handle the shared-kernel blast radius:
   `device/kernels/dataflow/dataflow_common.hpp` / `compute/compute_common.hpp` are shared with the
   other SDPA variants (joint / ring / sparse / windowed) in that directory, so the `_metal2`-fork
   rule from `metal2_port.md` applies.

### WH/BH structural parity claim

**Zero source files were changed** (mainline or fork) — this report is the only working-tree addition
for this op — so WH/BH behavior is unchanged by construction. Within the fork (informational, not this
session's diff), every Quasar-specific divergence already present is `#ifdef ARCH_QUASAR`-guarded
(`reader_interleaved.cpp:60`, `compute_streaming.hpp:35`), so WH/BH take the original path there too.
No device tests were run (recipe §9: the human runs all builds/tests).

---

## Files changed

**None.** (Neither the mainline op directory/namespace nor the fork was edited; nothing tempted a move
or rename. This report file only.)

## Test commands (for the human to run)

Order: BH → WH → Quasar. Force JIT when kernels change (`TT_METAL_FORCE_JIT_COMPILE=1`); run LLK
asserts on first (`unset TT_METAL_LLK_ASSERTS` only for DPRINT).

BH/WH parity baseline for **mainline** SDPA prefill (the control that a future in-place port is a
no-op refactor):

```
pytest tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py
```

The graph-op case this report is scoped to — exercises the **fork** op. Run on BH/WH for parity, then
on the Quasar emulator:

```
TT_METAL_FORCE_JIT_COMPILE=1 pytest "models/experimental/llama32_1b_quasar/tests/graph_ops/test_scaled_dot_product_attention.py::test_scaled_dot_product_attention[00_32x1024x64_bf16_int-dram]"
```

---
*Session note: audit-only run per the uplift recipe. Mainline is RED (not ported in place by #54468);
the fork is genuinely M2 and largely Quasar-ready for the causal interleaved path, with the
fifo_page_size ring-rewind and the §8.5 remap as the tracked concerns. Delete this report before merge.*
