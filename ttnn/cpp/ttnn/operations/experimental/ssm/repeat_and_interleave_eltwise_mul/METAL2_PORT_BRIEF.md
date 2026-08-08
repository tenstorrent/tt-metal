# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels` *(carry this line into the port report's Provenance section)*

**Shape of the op:** one DeviceOperation, one factory (`RepeatAndInterleaveEltwiseMulProgramFactory`, `device/repeat_and_interleave_eltwise_mul_program_factory.cpp`), three kernels — reader / writer / compute, all op-owned. Interleaved + tiled only.

**Read this before anything else — the op has three kernel-source configurations, and several items below are scoped to them.** One factory, three `defines` variants selected per cache miss by input width (`..._program_factory.cpp:100-106`):

| Label | Defines | Trigger (`a` width × `b` width) |
|---|---|---|
| **Config A** | `REPEAT_IN0` + `REPEAT_INTERLEAVE_IN1` | `a[-1] == 32`, `b[-1] == 5120` |
| **Config B** | `REPEAT_INTERLEAVE_IN1` only | `a[-1] == 32·5120`, `b[-1] == 5120` |
| **Config C** | `REPEAT_IN0` only | `a[-1] == 32`, `b[-1] == 32·5120` |

The fourth combination is unreachable (`TT_FATAL(ashape[3] != bshape[3])`, `..._device_operation.cpp:72`). All three are live CI paths — `tests/ttnn/nightly/unit_tests/operations/ssm/test_ssm_repeat_and_interleave_eltwise_mul.py:81-87`, with `:95-107` asserting three program-cache entries. Verify against all three.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `static ProgramDescriptor create_descriptor(...)` (`..._program_factory.hpp:15-16`)
- **Op-owned tensors:** none (no `CBDescriptor` sets `.buffer`; no borrowed-memory CBs anywhere in the op)
- **Target concept:** `ProgramSpecFactoryConcept`
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · `get_dynamic_runtime_args` (deprecated hook) · `override_runtime_arguments` (not-yet-supported replacement) · pybind `create_descriptor` — all gate conjuncts — plus other migration-risky pybind, which would have surfaced as a `safe` warning. All `no` on the readiness sheet, all confirmed against the code.

## Construct — to do

**Tensor bindings** (all three tensors are fed to a `TensorAccessor` — no raw-pointer case, no borrowed-DFB case):

- `a` — **Case 1** → express as `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::a)`. Legacy: `Buffer*` RTA at `..._program_factory.cpp:224` → `src0_addr` (`reader:15`) → `TensorAccessor s0` (`reader:29`).
- `b` — **Case 1** → same. Legacy: `Buffer*` RTA at `..._program_factory.cpp:224` → `src1_addr` (`reader:16`) → `TensorAccessor s1` (`reader:32`).
- `output` — **Case 1** → same. Legacy: `Buffer*` RTA at `..._program_factory.cpp:242` → `dst_addr` (`writer:13`) → `TensorAccessor s` (`writer:25`).

The `TensorAccessorArgs(...).append_to(...)` CTA plumbing (`..._program_factory.cpp:84-85, :89`) and the kernel-side `TensorAccessorArgs<N>()` declarations (`reader:28, :31`, `writer:24`) go away with the bindings.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no site passes a page size.

**CB endpoints** (7 CBs, all over the same `all_cores` range; dispositions are per `(CB, config)`):

- **Legal 1:1, all configs — nothing to do:** `c_0` (`src0`: reader P → compute C) · `c_1` (`src1`: reader P → compute C) · `c_16` (`output`: compute P → writer C).
- **Self-loop (Configs A, B):** `c_24` (`in0_transposed`) and `c_27` (`out_transposed`) — compute is the only toucher, producing and consuming both. Bind compute PRODUCER **and** CONSUMER.
- **1:1 (Configs A, B):** `c_26` (`in1_bcast_row`) — reader produces (`reader:83,114,122,152`), compute consumes (`ssm_eltwise_mul.cpp:126,144`).
- **Multi-binding advanced option (Configs A, B):** `c_25` (`in1_transposed`). Census is two touchers but **two locked consumers**: compute produces (`ssm_eltwise_mul.cpp:94,99`) **and pops** (`:165`), while the reader also consumes (`reader:77` wait, `reader:156` pop, tile read via `get_read_ptr()` at `reader:78`). No relabelling fits 1P+1C → set the flag. Do not "fix" the double pop; it is a behavior change and belongs to the ops team (raised with them in the audit).
- **Config C — four CBs have zero endpoints:** `c_24`, `c_25`, `c_26`, `c_27`. Every access to them is inside `#ifdef REPEAT_INTERLEAVE_IN1` (`reader:76-158`, `ssm_eltwise_mul.cpp:47-65, :88-166`), but the factory allocates all seven CBs unconditionally (`..._program_factory.cpp:145-175`) and the kernels still *name* them — `CircularBuffer` wrapper constructions at `reader:36-37` and `ssm_eltwise_mul.cpp:37-40`, plus `pack_reconfig_data_format(cb_in0_transposed, cb_id_out)` at `ssm_eltwise_mul.cpp:78`. A bindingless DFB is rejected by the spec validator, so the Config-C spec cannot be emitted unchanged.
  **Recommended:** **self-loop each from the kernel that constructs its wrapper** (compute for `c_24`, `c_25`, `c_27`; reader or compute for `c_26`) — role labels are cosmetic on Gen1 for a kernel that runs no FIFO ops, so this needs **no kernel edits** and leaves L1 footprint and runtime behavior identical.
  The alternative — dropping the four DFBs from the Config-C spec, per the recipe's literal dead-CB rule — is also correct but forces `#ifdef` guards on those kernel-side constructions and the `:78` metadata reference. The audit asked the user to confirm the self-loop route (`METAL2_PREPORT_AUDIT.md` → *Questions*, item 1); check for an answer before you build the Config-C spec.

**Runtime args:** all nameable — reader 7 (`reader:15-21`), writer 5 (`writer:13-17`), compute 2 (`ssm_eltwise_mul.cpp:13-14`), each read once at a fixed constant index. Name every one; nothing here needs the vararg mechanism.

## Watch for

- **CB endpoints (multi-binding):** `c_25` only, and the extra endpoint is *not* a hidden raw writer — it is a **producer-side `pop_front`** (`ssm_eltwise_mul.cpp:165`) sitting alongside the reader's consumer pop (`reader:156`). Confirm both sites before setting the flag; the rest of the CBs need no flag.
- **Config-scoped census:** dispositions flip between configs (see above). Don't carry Config A's census into Config C.
- **Cross-op / shared kernels:** none — all three kernels are op-owned, no `_metal2` fork exists beside any of them, and no other op or test instantiates them. This port creates no fork and carries no sunset list.
- **RTA varargs:** none — prefer named RTAs throughout (see *Runtime args* above).
- **Perf:** the readiness sheet flags this op with `Pointer patching perf issue? = suspect perf regression (+ fixed latent bug)`. The op is classified `PD Op (pointer-patching)`, which the Metal 2.0 typed binding supersedes, so the port plausibly *is* the fix — but measure rather than assume, and record numbers in the port report.
- **Two known-inert oddities in the reader, deliberately not port work** (they are in the audit's *Misc anomalies* for the ops team; leave them exactly as they are so the diff stays a pure port): the hardcoded `5120` at `reader:130` where the sibling loop uses the `in0_num_blocks_w` RTA (`reader:91`), and reader RTA index 6 being dead in Configs A and C.
