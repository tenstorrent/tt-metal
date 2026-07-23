# `ttnn.experimental.gated_delta_prefill_query` — WIP

Experimental TTNN op for Qwen3.5/3.6 **Gated DeltaNet**, intended to replace the
hand-written GDN kernel path exercised by
`models/demos/blackhole/qwen36/tests/test_gdn_tp.py`.

## Goal (final semantics)

Starting from a per-V-head recurrent **state** matrix, run the gated delta-rule
recurrence over a K/V **prefill sequence** using a per-head constant decay `g` and
write-strength `β`, then apply a single query **Q** to the final state to emit the
**first decode output token**. Returns the updated state and that first token.

Per-head recurrence (h = S, for t = 0…S−1; `g`, `β` constant per head):

```
h      = h * exp(g)          # decay (g log-space, scalar per head)
v_read = Σ_k (h * k_t)       # read
Δ      = (v_t − v_read) * β   # delta (β scalar per head)
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
| `gate` (β, write strength) | `[1, Nv, 1, 1]` | TILE | fp32 |
| `decay` (g, log-space) | `[1, Nv, 1, 1]` | TILE | fp32 |
| `state` | `[1, Nv, d, d]` | TILE | fp32 |
| → `O` (first token) | `[1, 1,  Nv, d]` | TILE | bf16 |
| → `state'` (updated) | `[1, Nv, d, d]` | TILE | fp32 |

`gate`/`decay` are one scalar per V-head, constant across the whole sequence.

## Current status — SCAFFOLDING ONLY

The op builds, is registered as `ttnn.experimental.gated_delta_prefill_query`,
dispatches on device, validates all shapes/layouts/dtypes, and returns
correctly-shaped/typed outputs. **The recurrence is NOT implemented yet.** The
placeholder single-core kernels:

- `state'` = passthrough copy of `state` (close, not bit-exact — `copy_tile` routes
  fp32 through TF32 in the unpacker unless `UnpackToDestFp32` is enabled).
- `O` = placeholder values (copied state tiles).
- `q`/`k`/`v`/`gate`/`decay` are validated but unused by the placeholder compute.

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
    ├── gated_delta_prefill_query_program_factory.cpp        # ProgramDescriptor factory (basic)
    └── kernels/{dataflow/reader,writer, compute}/...cpp      # placeholder copy kernels
```

Wiring: `experimental/transformer/sources.cmake` (SRCS + NANOBIND_SRCS),
`experimental/transformer/CMakeLists.txt` (kernels glob),
`experimental/experimental_nanobind.cpp` (include + `bind_gated_delta_prefill_query`).

Test: `tests/ttnn/unit_tests/operations/experimental/transformer/test_gated_delta_prefill_query.py`
(pins interface + asserts the state passthrough at small and Qwen3.6 shapes).

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

With the workaround the scaffold test passes (2/2).

## Next steps (implement the recurrence)

1. Read q/k/v/gate/decay in the reader; add per-head CBs.
2. Compute: L2-norm q/k, scale q, GVA-expand; sequential delta-rule scan over S
   accumulating into the state (matmul + eltwise per step, or chunked).
3. Final `o = h @ q` per head → pack O `[1,1,Nv,d]`.
4. Enable `UnpackToDestFp32` for fp32 state math (see `dit_layernorm_pre_all_gather`
   welford factory for the pattern).
5. Multi-core parallelization (one core per V-head is the natural first split).
6. Validate against the torch reference in `test_gdn_tp.py` / the FLA
   `recurrent_gated_delta_rule` in
   `models/experimental/gated_attention_gated_deltanet/torch_functional/`.
