# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads`

> Audit cleared all gates. One gate was cleared by user waiver; see below. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ *(user waiver: readiness-sheet row is stale, see below)* · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

**TTNN gate waiver (record in the port report).** The live readiness sheet (fetched 2026-10-07) still describes this op as it was before PR #57409 (`f5093e705ae`, 2026-09-25):
- both rows (`SplitFusedQKVAndSplitHeadsProgramFactory`, `SplitFusedQKVAndSplitHeadsShardedProgramFactory`) show `Concept` = `legacy device-op`
- their `Factory definition path` cells name `device/split_query_key_value_and_split_heads_program_factory.hpp` and `device/split_query_key_value_and_split_heads_sharded_program_factory.hpp`, which #57409 deleted (the factories survive, declared in `device/split_query_key_value_and_split_heads_device_operation.hpp:22,31`)

That is a primary-column conflict, which the audit recipe classes as "spreadsheet-broken". In the code, both factories are `descriptor` (`create_descriptor`, `device_operation.hpp:23,32`). Every other cross-checked column matches the code, and the sheet's own verdict is `yes (with PD step)`, with the PD step already landed. On 2026-10-07 the user waived this gate ("Waive"). The sheet refresh is owed to the readiness-sheet owner and does not block the port.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (both factories)
- **Op-owned tensors:** none
- **Target concept:** `ProgramSpecFactoryConcept` (both factories)
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args` (deprecated hook). A custom hash, an `override_runtime_arguments`, and a pybound `create_descriptor` are **not** in this list: none of them gate, and any may be present on a cleared op. On this op, none of the three is present:
  - default hash
  - no `override_runtime_arguments` on either factory
  - `split_query_key_value_and_split_heads_nanobind.cpp:21` binds only the function

## Construct — to do

Two factories, selected by `input_tensor.is_sharded()` (`device_operation.cpp:12-18`). Each has a reader, a writer, and a compute kernel over one core range.

**Tensor bindings** (per binding):

- *Interleaved factory* (`device/split_query_key_value_and_split_heads_program_factory.cpp`):
  - `input` — **Case 1** (via `TensorAccessor`) → `TensorParameter` / `TensorBinding`. Today it is the `Buffer*` reader RTA 0 (`:187`) plus `TensorAccessorArgs` at reader CTA 2 (`:99`). The reader builds `TensorAccessor(in0_args, in0_tensor_addr)` (`reader_tm_tile_layout_create_qkv_heads.cpp:28`), which becomes `TensorAccessor(tensor::input)`.
  - `q`, `k`, `v` — **Case 1** → `TensorParameter` / `TensorBinding` each. Today they are `Buffer*` writer RTAs 0–2 (`:193-195`) plus chained `TensorAccessorArgs` from writer CTA 7 (`:111-113`). The writer builds `TensorAccessor(q_args/k_args/v_args, …)` (`writer_tm_tile_layout_create_qkv_heads.cpp:39-41`).
- *Sharded factory* (`device/split_query_key_value_and_split_heads_sharded_program_factory.cpp`): `input` (`c_0`, `:133`), `q` (`c_16`, `:154`), `k` (`c_17`, `:165`), `v` (`c_18`, `:176`) — all **clean**. They are borrowed-memory CBs (`.buffer =`), so they become `DataflowBufferSpec::borrowed_from` the matching `TensorParameter`. No accessor, no address args.

**TensorParameter relaxation:** `none`

**TensorAccessor 3rd arg:** none — no accessor passes a 3rd arg

**CB endpoints:**

| Factory | CB | Touchers | Disposition |
|---|---|---|---|
| Interleaved | `c_0` (K pre-transpose, `:150`) | reader produces, compute consumes | legal 1:1 |
| Interleaved | `c_1` (Q and V tiles, `:160`) | reader produces, writer consumes | legal 1:1 |
| Interleaved | `c_16` (K transposed, `:170`) | compute produces, writer consumes | legal 1:1 |
| Sharded | `c_0` (input shard, borrowed) | sharded reader and sharded writer, both raw `get_read_ptr` only | **assign 1P+1C** (bind one PRODUCER, the other CONSUMER; cosmetic on Gen1). Not multi-binding |
| Sharded | `c_24` (K intermediate, `:136`) | sharded writer produces, compute consumes | legal 1:1 |
| Sharded | `c_16` (Q shard, borrowed) | sharded reader only | **self-loop** (DM) |
| Sharded | `c_18` (V shard, borrowed) | sharded reader only | **self-loop** (DM) |
| Sharded | `c_17` (K shard, borrowed) | compute only | **self-loop** (compute). Same shape as `create_qkv_heads_from_separate_tensors`' `OUT_K` binding on this fork |

No dead CBs. The unused `constexpr` CB indices in the sharded kernels are declarations only.

## Watch for

- **CB endpoints (multi-binding):** none
- **Cross-op / shared kernels:**
  - Interleaved compute binds `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` (shared pool). **A `_metal2` fork already exists at `ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp` — bind it, don't re-fork, don't edit it.** Its interface is `dfb::in` CONSUMER (← `c_0`), `dfb::out` PRODUCER (→ `c_16`), and a named CTA `NHtWt` (today positional CTA 0 = `num_tiles_per_tensor`, `:138`). Mirror `nlp_create_qkv_heads_program_factory.cpp:335-356`, including `opt_level = KernelBuildOptLevel::O3`. Other legacy binders — **sunset list, not authorization to convert the kernel in place**: `nlp_create_qkv_heads_boltz`, `nlp_create_qkv_heads_vit`.
  - Sharded compute binds this op's own `kernels/compute/transpose_wh_sharded.cpp`, which `experimental/transformer/create_qkv_heads` also borrows. **A `_metal2` fork already exists beside it: `kernels/compute/transpose_wh_sharded_metal2.cpp` — bind it, don't re-fork.** Its interface is `dfb::in` CONSUMER (← `c_24`), `dfb::out` PRODUCER + CONSUMER (self-loop on `c_17`), and a named CTA `num_tiles` (today positional CTA 0 = `num_tiles_per_tensor`, `:119`). Mirror `create_qkv_heads_from_separate_tensors_program_factory.cpp:194-221`, including O3. Legacy binder left after this port (sunset list): `create_qkv_heads`. Don't confuse it with `data_movement/transpose/.../transpose_wh_sharded_metal2.cpp`, a different file.
  - The four dataflow kernels are used by this op only, so no co-borrower constrains them.
  - Do not read or reuse anything under `experimental/quasar/`.
- **RTA varargs:** none. Name every arg:
  - interleaved reader: `in0_tensor_addr` → binding, `in0_tensor_tile_id`
  - interleaved writer: `q/k/v_tensor_addr` → bindings, `out_tensor_tile_id`, `out_tensor_tile_id_with_transpose`
  - the sharded factory has no RTAs
- **Anything else the porter needs:**
  - **Runtime CB selection in the interleaved reader:** it builds `CircularBuffer cb_cur(cb_id)` from a runtime `cb_id` that switches between `c_1` and `c_0` per output tensor (`reader_tm_tile_layout_create_qkv_heads.cpp:33-46`). With named tokens, bind both DFBs to the reader as PRODUCER and pick between two `DataflowBuffer` objects built from `dfb::` tokens (e.g. `DataflowBuffer cb_cur = (out_tensor == 1) ? dfb_k : dfb_qv;`). Keep the loop and the linear `l1_write_addr` walk unchanged.
  - **Interleaved writer CB counters (don't "fix"):** the writer waits with cumulative counts and pops once at the end (`writer_tm_tile_layout_create_qkv_heads.cpp:48-49`, `:139-140`). Because the counters start at `block_size` (1), the final `pop_front` pops N+1 pages after N were pushed. Carry it over byte-for-byte; it is recorded for the ops team in the audit's Misc anomalies.
  - **Sharded writer order quirk:** `cb_im0_obj.get_write_ptr()` is read before `reserve_back` (`writer_tm_tile_layout_create_qkv_heads_sharded.cpp:48-49`). Preserve it.
  - **Sharded raw L1 walks stay:** the sharded reader and writer address `c_0` through `get_read_ptr()` + offsets and read it via a `UnicastEndpoint` self-loopback built from `my_x[noc_id]` / `my_y[noc_id]` (reader `:36-38`, writer `:35-37`). Keep that arithmetic and the endpoint verbatim.
  - `get_tile_size(cb_id)` (reader `:27`, writer `:38`, sharded reader/writer `:30`) moves onto the DFB object (kernel-side whitelist rule 7).
  - The kernels are already Device 2.0 (`Noc`, `CircularBuffer`, `TensorAccessor`, `UnicastEndpoint`, `CoreLocalMem`), so the kernel change is a binding-layer swap, not an idiom rewrite.
  - Known latent issues, **not yours to fix**: hardcoded `out_w = 64` (`program_factory.cpp:48`) and `num_heads_per_tensor = 2` (`sharded_program_factory.cpp:43`), both ignoring `num_heads`. Preserve them; they are recorded in the audit's Misc anomalies.
  - **Verification on this 8×8 Wormhole:** `tests/tt_eager/python_api_testing/unit_testing/misc/test_split_query_key_value_and_split_heads.py` (interleaved, batch 8, `seq_len = 32*min(12, grid.x)`) should run here. Find a sharded-path test early: the BERT-large sharded configs historically need a ≥12-column grid and may skip on this card. Ask the user how to verify if they do.
