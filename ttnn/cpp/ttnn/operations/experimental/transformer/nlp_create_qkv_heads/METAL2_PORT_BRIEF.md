# Metal 2.0 Port Brief — `experimental/transformer/nlp_create_qkv_heads`

> Audit cleared all gates (2nd pass). This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry into the port report's Provenance section)*

**Prerequisite already landed:** the `Sharded` offset-base-pointer fold + `get_dynamic_runtime_args` hook were refactored out in commit `86872e0a06a` (built, 111 unit tests pass on WH). You are porting the **post-refactor** code — the `Sharded` factory now passes clean input-buffer bases as `Buffer*` bindings (slots 6/15) + scalar region offsets (slots 7/16).

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (both `Interleaved` and `Sharded` factories)
- **Op-owned tensors:** none
- **Target concept:** `MetalV2FactoryConcept`
- **Gate-cleared, confirmed absent:** custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · runtime-args-update hook (removed by the prerequisite refactor). All `no`.

## Construct — to do

**Tensor bindings** (per binding):

- **Interleaved** — `in0` (q input), `in1` (kv input, optional), and outputs `q`/`k`/`v`: **Case 1** (via `TensorAccessor`). Express each as `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)` and addresses by `page_id`. The legacy `Buffer*` RTAs + `TensorAccessorArgs` CTAs both disappear.
- **Sharded** — `input_tensor_q` (slot 6) and `input_tensor_kv` (slot 15): **Case 2** (raw NoC arithmetic on the base). Bind as `TensorParameter`, pull the base via `get_bank_base_address`, and keep the existing raw walk unchanged. **Note:** slot 15 binds the *kv* tensor when present, else the *q* tensor (fused path) — mirror that selection in the binding.
- **Sharded outputs** — `c_16` / `c_17` / `c_18` are borrowed-memory CBs (`.buffer = output.buffer()`): **clean** (borrowed-DFB). Port via `DataflowBufferSpec::borrowed_from` on the respective output `TensorParameter`.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none (no site passes one).

**CB endpoints:**

- **Interleaved** — `cb1` reader→writer FIFO, legal `(1,1)`. With `transpose_k_heads`: `cb0` (reader→compute) and `cb16` (compute→writer), each legal `(1,1)`. No special disposition.
- **Sharded `c_16`** (Q output) — **assign 1P+1C**: the reader-config and writer-config instances of the same kernel source both raw-`get_write_ptr` into `c_16` (risc0 vs risc1 Q heads, disjoint offsets), no FIFO ops → two role-free touchers. Bind one instance PRODUCER, the other CONSUMER (cosmetic on Gen1). **This is the dual-instance work-split, not multi-binding — do not set the multi-binding flag** (no third toucher).
- **Sharded `c_17`** (K, reader FIFO-produces) and **`c_18`** (V, writer FIFO-produces) — single locked producer each → **self-loop** (bind the one kernel both PRODUCER and CONSUMER).

## Watch for

- **CB endpoints (dual-instance work-split):** `Sharded` instantiates one kernel source (`reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp`) **twice** over `q_cores` — reader-config + writer-config, differing only by CTAs (K→c_17 vs V→c_18, Q offset) and per-instance args. Both hit every node, so `c_16` genuinely has two touchers per node → 1P+1C. Don't mistake it for the demoting-per-group-CTA pattern (which covers disjoint node sets).
- **Cross-op / shared kernels:** `transpose_wh.cpp` is file-path-instantiated from the shared pool `ttnn/cpp/ttnn/kernel/compute/` (Interleaved, `transpose_k_heads` only). Port that shared kernel as one unit with its co-borrowers — its CB→DFB rewrite is a single change across every op that instantiates it.
- **RTA varargs:** the sharded kernel reads the per-remote-core NoC coordinate arrays `in0_mcast_noc_x` / `in0_mcast_noc_y` via `get_arg_addr(19)` / `get_arg_addr(19 + num_x)` — a variable-length region (count = `num_cores_x`/`num_cores_y`, varies per instantiation), indexed at runtime by `q_x`/`kv_x`. Port as RTA **varargs** (kernel-side vararg mechanism), not named args. Slots 0–18 are fixed named scalars.
