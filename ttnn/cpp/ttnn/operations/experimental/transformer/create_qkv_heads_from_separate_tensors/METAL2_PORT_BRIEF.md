# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/create_qkv_heads_from_separate_tensors`

> Audit cleared all gates (one by user waiver; see below). This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (user waiver) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section. It comes from the `Port_Recipe` checkout; the provenance command prints nothing in `Metal_Ports`.)*

**Waiver.** The TTNN factory concept gate was RED because the readiness sheet is out of date for this op, and the user waived it on 2026-10-06. The live sheet still shows the op as it was before PD batch #57409 (`f5093e705ae`, 2026-09-25):

- `Concept` = `legacy device-op`
- `Factory (variant)` = `CreateQKVHeadsSeparateTensorsProgramFactory`, which #57409 deleted
- `Is able to port?` = `yes (with PD step)`

The code is already `descriptor`, so the PD step has landed. The rest of the primary cross-check is clean, and nothing in the op's code blocks the port. Record the waiver in the port report's Provenance section. The sheet refresh stays with the readiness-sheet owner (Diego); it is not port work.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`, in the **direct-descriptor shape**: `create_descriptor` is a static member of `CreateQKVHeadsSeparateTensorsDeviceOperation`, with no `program_factory_t` (`device/create_qkv_heads_from_separate_tensors_device_operation.hpp:26-29`). See Watch for.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer, and `get_dynamic_runtime_args` (deprecated hook).
- **Non-gating facts:**
  - custom hash: none (the default hash is used)
  - `override_runtime_arguments`: none. The pre-PD one was replaced by CB `.buffer` re-pegging (`program_factory.cpp:108-111`). The framework refreshes tensor bindings on a cache hit.
  - pybound `create_descriptor`: none. The only binding is the user entry point via `bind_function` (`create_qkv_heads_from_separate_tensors_nanobind.cpp:21`), so there is no user-visible API removal.

## Construct — to do

**Tensor bindings** (per binding). All are **clean** (borrowed-memory DFB): the op has **no runtime args at all**, and every tensor reaches the kernels through a CB backed by its shard. Express each as a `TensorParameter` plus a `DataflowBufferSpec` with `borrowed_from` that tensor. There is no `TensorAccessor` and no `get_bank_base_address` bridge.

| Tensor | Legacy CB | Factory site | Producer / touchers |
|---|---|---|---|
| `input_tensor` (Q in) | `c_0` | `program_factory.cpp:114-123` | reader, raw read-ptr peek |
| `input_tensor_kv` | `c_1` | `:125-134` | reader, raw read-ptr peek |
| `output_q` | `c_16` | `:137-146` | reader produces |
| `output_k` | `c_17` | `:148-157` | reader produces if `!transpose_k`; compute produces if `transpose_k` |
| `output_v` | `c_18` | `:159-168` | reader produces |

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none. The op constructs no `TensorAccessor`.

**CB endpoints** (node set `all_cores` = the Q shard grid; two configs, keyed on `transpose_k_heads`):

- **Self-loop** (one toucher, bound PRODUCER + CONSUMER):
  - `c_0`, `c_1`, `c_16`, `c_18`: the reader, in both configs.
  - `c_17`: the **reader** when `transpose_k = false`, and the **compute** kernel when `transpose_k = true`. The producer kernel flips with config.
- **Plain 1:1** on `c_24` (the K transpose intermediate, not borrowed), only when `transpose_k = true`: reader is PRODUCER (`reader :93,130`), compute is CONSUMER (`transpose_wh_sharded.cpp:28,34`). Its DFB spec is **conditional** on `transpose_k`, mirroring the existing `if (transpose_k)` at `program_factory.cpp:170-180`. That is a direct translation, not new structure.
- No dead CBs. No multi-binding.

## Watch for

- **CB endpoints (multi-binding):** none. No hidden second writer (no kernel raw-writes a CB another produces, and there are no semaphores).
- **Cross-op / shared kernels:** compute `experimental/transformer/split_query_key_value_and_split_heads/device/kernels/compute/transpose_wh_sharded.cpp` (borrowed, in-family). Read the shared-kernel caution in `port_patterns.md` before touching it.
  - **No `_metal2` fork exists beside it yet.** This port takes rung 2: create `transpose_wh_sharded_metal2.cpp` in *that* directory, and add the pointer comment to the original.
  - **Do not bind `data_movement/transpose/device/kernels/compute/transpose_wh_sharded_metal2.cpp`.** It shares the stem but forks a *different* kernel (RTA-driven `NHtWt`/`Ht`/`Wt`, bulk `wait_front`, CB indices from CTAs).
  - **Ignore anything under `experimental/quasar/`** (out of bounds).
  - The original hardcodes `c_24` (in) / `c_17` (out) at `:14-18`. Name the fork's bindings for the kernel's role (e.g. `dfb::in`, `dfb::out`). Its CTA 0 (`num_tiles`) gets a role name too.
  - Other binders (**sunset list, not authorization to convert the kernel in place**):
    - `experimental/transformer/split_query_key_value_and_split_heads` (`split_query_key_value_and_split_heads_sharded_program_factory.cpp:114-116`)
    - `experimental/transformer/create_qkv_heads` (`create_qkv_heads_program_factory.cpp:167-169`)
- **RTA varargs:** none. There are no RTAs or CRTAs. CTAs are fixed-index (reader `0..6`, compute `0`), so name each.
- **Direct-descriptor shape → introduce a factory struct.** Follow `ttnn_factory.md` §"3. Give a direct-descriptor op a conventional program factory":
  - nest a factory struct (e.g. `CreateQKVHeadsSeparateTensorsProgramFactory`, the pre-#57409 name) with `create_program_artifacts`
  - add `using program_factory_t = std::variant<...>;`
  - remove the device-op-level `create_descriptor`

  Keep the body in `device/create_qkv_heads_from_separate_tensors_program_factory.cpp`, and record the change under Handoff points. Check first whether TTNN has added a `program_factory_t` since the audit.
- **`TRANSPOSE_K_HEADS` picks the reader's K-destination CB** (`reader :27-31`): `c_24` under transpose, `c_17` otherwise. Keep the `#ifdef`, and give each branch its own `dfb::` token. The reader's binding set therefore differs by config: under transpose it binds the intermediate DFB and not `output_k`'s DFB. Keep supplying the define (`program_factory.cpp:79-81`).
- **`get_tile_size(cb_inq)` sits in a `constexpr`** (`reader :41`), and four further `constexpr`s derive from it (`:63-64,89-91`). The port moves it onto the DFB object (whitelist rule 7). Confirm the DFB getter is usable in a constant expression. If it isn't, demote those locals to `const`, and don't restructure the arithmetic.
- **Stale include.** `api/dataflow/circular_buffer.h` appears in the reader (`:8`) and in the compute fork's copy (`:9`). It goes with the `CircularBuffer` → `DataflowBuffer` swap.
- **Raw L1 walks over borrowed DFBs stay unchanged.** The reader reads its own core over the NoC. The source addresses are `get_read_ptr() + seq_tile_offset + head_offset`, and the address is formed via `UnicastEndpoint` + `{.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr}` (`:54-59,72-77,103-108,117-122,143-148`). The output `get_write_ptr()` is walked linearly inside one bulk `reserve_back` / `push_back` per output. Keep all of it as is; only the CB objects and tokens change.
- **Tests to exercise both configs:**
  - `tests/tt_eager/python_api_testing/unit_testing/misc/test_create_qkv_heads.py`
  - `tests/ttnn/unit_tests/operations/transformers/test_head_count_zero.py`

  `transpose_k_heads` defaults to `true` in Python (`create_qkv_heads_from_separate_tensors_nanobind.cpp:34`), so make sure the `false` path runs too.
