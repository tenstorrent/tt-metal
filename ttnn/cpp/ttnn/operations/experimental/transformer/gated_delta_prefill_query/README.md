# `ttnn.experimental.gated_delta_prefill_query` — WIP

Experimental TTNN op for Qwen3.5/3.6 **Gated DeltaNet**, intended to replace the
hand-written GDN kernel path exercised by
`models/demos/blackhole/qwen36/tests/test_gdn_tp.py`.

## Goal (final semantics)

Starting from a per-V-head recurrent **state** matrix, run the gated delta-rule
recurrence over a K/V **prefill sequence** using a per-head, per-token decay `g_t`
and write-strength `β_t`, then apply a single query **Q** to the final state to emit
the **first decode output token**. Returns the updated state and that first token.

Per-head recurrence (h = S, for t = 0…S−1; `g_t`, `β_t` per token):

```
h      = h * exp(g_t)        # decay (g log-space, per token)
v_read = Σ_k (h * k_t)       # read
Δ      = (v_t − v_read) * β_t # delta (β per token)
h      = h + outer(k_t, Δ)   # write
# after the scan:
o = h @ q                    # single query → first output token
S' = h                       # updated state
```

L2-normalize q/k per the GDN contract; scale q by 1/√d; GVA-expand Q/K from Nk→Nv
heads inside the op.

## Interface (final; general in Nk/Nv/S/d)

Dims for Qwen3.6-27B: `Nk=16`, `Nv=48`, `d=128` (GVA ratio 3). The op infers
Nk/Nv/S/d from tensor shapes.

| tensor | shape | layout | dtype |
|---|---|---|---|
| `q` | `[1, 1,  Nk, d]` | ROW_MAJOR | bf16 |
| `k` | `[1, Nk, S,  d]` | TILE | bf16 |
| `v` | `[1, Nv, S,  d]` | TILE | bf16 |
| `gate` (β, write strength) | `[1, Nv, S, 1]` | TILE | fp32 |
| `decay` (g, log-space) | `[1, Nv, S, 1]` | TILE | fp32 |
| `state` | `[1, Nv, d, d]` | TILE | fp32 |
| → `O` (first token) | `[1, 1,  Nv, d]` | TILE | bf16 |
| → `state'` (updated) | `[1, Nv, d, d]` | TILE | fp32 |

`gate`/`decay` are one scalar per V-head per K/V token (β_t and g_t vary along the
prefill sequence).

## Current status — MULTI-CORE SKELETON + K READ (recurrence not implemented)

The op builds, is registered as `ttnn.experimental.gated_delta_prefill_query`,
dispatches across the full compute grid, validates all shapes/layouts/dtypes, and
returns correctly-shaped/typed outputs. **The recurrence is NOT implemented yet**, so
`O` / `state'` values are not yet meaningful (the outputs are allocated but not
written). What *is* implemented:

- **Work distribution (factory):** one V-head's recurrence per core. Each K-head is
  replicated across its `gva_ratio = Nv/Nk` V-heads, so a `v_head` core reads K-head
  `v_head / gva_ratio` (blocked GVA, matching `repeat_interleave(rf)` in the torch
  reference). All grid cores are used: the `Nv` V-heads are spread balanced-greedily
  over the grid, and cores beyond `Nv` split a V-head's **sequence** (never the hidden
  dim) into more sections. Cores sharing a V-head are the group a later step will
  tree-reduce (their `v_head_id`/`section_id`/`num_sections` are already passed as
  runtime args).
- **K read (reader kernel):** each core streams its seq-tile range into `cb_k`, one
  block at a time. A block is `block_height` seq-tiles tall × the **full hidden dim**
  (`out_block_size` tiles), where `out_block_size` is seeded from `num_out_blocks` and
  rounded down to a multiple of `d_tiles` (`TT_FATAL` asserts the alignment). `cb_k` is
  sized to exactly one block.
- **Compute:** placeholder — just drains `cb_k` so the reader can't deadlock.
- `q`/`v`/`gate`/`decay`/`state` are validated but not yet read.

Structure follows the modern descriptor-based experimental pattern
(`dit_layernorm_pre_all_gather`): `ttnn::device_operation` + a `ProgramDescriptor`
program factory (`create_descriptor`), prim in `ttnn::experimental::prim`, public
API in `ttnn::experimental`, bound under `ttnn.experimental.`.

### Files

```
ttnn/cpp/ttnn/operations/experimental/transformer/gated_delta_prefill_query/
├── gated_delta_prefill_query.hpp / .cpp                     # public API
├── gated_delta_prefill_query_nanobind.hpp / .cpp            # python binding
├── README.md                                                # this file
└── device/
    ├── gated_delta_prefill_query_device_operation_types.hpp # Params + Inputs
    ├── gated_delta_prefill_query_device_operation.hpp / .cpp# DeviceOperation + prim launch
    ├── gated_delta_prefill_query_program_factory.cpp        # multi-core factory (V-head split + K read)
    └── kernels/{dataflow/reader, compute}/...cpp            # K reader + placeholder drain
```

Wiring: `experimental/transformer/sources.cmake` (SRCS + NANOBIND_SRCS),
`experimental/transformer/CMakeLists.txt` (kernels glob),
`experimental/experimental_nanobind.cpp` (include + `bind_gated_delta_prefill_query`).

Test: `tests/ttnn/unit_tests/operations/experimental/transformer/test_gated_delta_prefill_query.py`
(pins registration / shapes / layouts / dtypes at small and Qwen3.6 shapes; value
checks return with the recurrence).

## Build & run

```bash
./build_metal.sh                 # or: ninja -C build_Release _ttnncpp.so _ttnn.so
source python_env/bin/activate
pytest tests/ttnn/unit_tests/operations/experimental/transformer/test_gated_delta_prefill_query.py -v
```

### Device note (this machine only)

This box is a **P300 board presenting only 1 chip** (2nd chip down), which the
runtime classifies as a `CUSTOM` cluster and then demands a fabric mesh-graph
descriptor. It is **not** related to this op (the op uses no fabric); it fires
during device bring-up in the test fixture. Two options:

1. **Single-device view (no reset):** point the fabric at a 1×1 Blackhole mesh:
   ```bash
   export TT_METAL_HOME=/localdev/vsuresh/tt-metal   # must match the built checkout
   export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_p300_mesh_graph_descriptor.textproto
   ```
   (`single_p300_mesh_graph_descriptor.textproto` is included in this branch.)
2. **Dual-device view:** `tt-smi -r` to recover the 2nd chip so it comes up as a
   standard P300 (num_chips=2). On a healthy 2-chip P300 neither env var is needed.

With the workaround the interface test passes (2/2).

## Next steps (implement the recurrence)

1. Read `v`/`gate`/`decay`/`state` for this core's V-head (and `q` for the query),
   adding their CBs alongside `cb_k`.
2. Compute: L2-norm q/k, scale q; per-block delta-rule scan over the core's seq
   section accumulating into the V-head state (matmul + eltwise per step, or chunked).
3. Per-V-head **tree reduction** across the section cores (metadata already passed:
   `v_head_id`/`section_id`/`num_sections`; add the peer-core topology).
4. Final `o = h @ q` per V-head → pack O `[1,1,Nv,d]`; write O and `state'`
   (re-introduce the writer).
5. Enable `UnpackToDestFp32` for fp32 state math (see `dit_layernorm_pre_all_gather`
   welford factory for the pattern).
6. Validate against the torch reference in `test_gdn_tp.py` / the FLA
   `recurrent_gated_delta_rule` in
   `models/experimental/gated_attention_gated_deltanet/torch_functional/`.
