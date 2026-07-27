# Metal 2.0 Audit Findings — `experimental/transformer/nlp_create_qkv_heads`

- **`NlpCreateHeadsDeviceOperation`** (single device operation in this directory)
  - `Interleaved` — `create_descriptor` (`device/nlp_create_qkv_heads_program_factory.cpp:19`)
  - `Sharded` — `create_descriptor` (`device/nlp_create_qkv_heads_program_factory.cpp:430`)

Kernels referenced:
- Interleaved: `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp`, `.../writer_tm_tile_layout_nlp_create_qkv_heads.cpp`, and (only when `transpose_k_heads`) the shared compute donor `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`.
- Sharded: `device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp` — instantiated **twice** (reader-config + writer-config, both over `q_cores`).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

**Audited at commit:** `86872e0a06a` (working tree clean for the op's `device/` files). *Provenance note:* this commit split the `Sharded` factory's former host-folded `base + head_offset` addresses into clean buffer bases + scalar region offsets and removed the `get_dynamic_runtime_args` hook — the two conditions that would otherwise gate this op (see Offset base pointers and TTNN factory concept below). It is a prerequisite refactor, validated on Wormhole (111 unit-test cases pass, incl. both program-cache tests).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `NlpCreateHeadsDeviceOperation` → `Interleaved`, `Sharded` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes (GREEN — all kernels Device 2.0, incl. donor `transpose_wh.cpp`) |
| *Prereqs* — Cross-op escapes | Ok (one file-path donor: `kernel/compute/transpose_wh.cpp`, shared pool) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (all CTAs at constexpr offsets) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factory rows; cross-check consistent) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD | N/A |
| *TTNN Readiness* — Is safe to port? | `yes` (sheet); no smuggled pointer in code |
| *TTNN Readiness* — Custom hash | No (no `compute_program_hash`) |
| *TTNN Readiness* — Runtime-args update | No (no `get_dynamic_runtime_args`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (clean bases bound as `Buffer*`) |
| *Port work* — Tensor bindings (per binding) | Interleaved: Case 1 · Sharded inputs: Case 2 · outputs: clean (borrowed-DFB) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no site) |
| *Port work* — CB endpoints | Interleaved: legal · Sharded: self-loop (c_17, c_18) + 1P+1C (c_16) |

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`). All five gates clear:

- **Device 2.0** ✓ — every kernel the op uses is Device 2.0 compliant.
- **Feature compatibility** ✓ — no Appendix A feature in use.
- **TTNN factory concept** ✓ — `Is able to port? = yes` for both factories; sheet ↔ code cross-check consistent.
- **Offset base pointers** ✓ — no address RTA carries a host-folded offset; input bases are clean `Buffer*` bindings.
- **TensorAccessor 3rd argument** ✓ — no accessor passes a 3rd arg.

No portable-subset scoping needed — the whole op clears.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Fresh readiness-sheet fetch (this run) rows for `NlpCreateHeadsDeviceOperation` / {`Interleaved`, `Sharded`}: `Concept=descriptor`, `Custom hash=no`, `Runtime-args update=no`, `Pybind descriptor=no`, `Is safe to port?=yes`, `Is able to port?=yes`, `TensorParameter relaxation=none`, no op-owned tensors. Cross-check against the committed code:
  - `Concept=descriptor` ✓ — both factories define `create_descriptor()` returning a `ProgramDescriptor`.
  - `Custom hash=no` ✓ — no `compute_program_hash` / `compute_descriptor_program_hash`.
  - `Runtime-args update=no` ✓ — no `get_dynamic_runtime_args` in code (only in explanatory comments at program_factory.cpp:280,514).
  - `Pybind descriptor=no` ✓ — `nlp_create_qkv_heads_nanobind.cpp` binds only `ttnn::experimental::nlp_create_qkv_heads`.
  - Cross-column invariants OK (no runtime-args-update on the descriptor concept; no op-owned tensors).
- **Device 2.0 (every kernel used):** GREEN. `Noc` / `CircularBuffer` / `UnicastEndpoint` / `CoreLocalMem` idioms; `get_tile_size(cb_id)` is the sanctioned free function. No raw `noc_async_read`, no legacy `InterleavedAddrGen`/`ShardedAddrGen`, no CB-index pointer holdovers.
  - `reader_tm_tile_layout_nlp_create_qkv_heads.cpp` ✓ · `writer_tm_tile_layout_nlp_create_qkv_heads.cpp` ✓ · `reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp` ✓ (its raw NoC reads use the modern `Noc`/`UnicastEndpoint` API) · `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` ✓
- **Feature compatibility:** all Appendix A entries absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no GCB type / `remote_cb` / `.remote_index(` / `.global_circular_buffer` field |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset` / `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | every `get_compile_time_arg_val` at a constexpr offset; fixed input count (`std::vector<std::optional<Tensor>>` is the 3-output container, not a variable input list) |

- **Offset base pointers:** GREEN. No address RTA folds a host-side offset into its base. The `Sharded` factory passes the clean Q and K/V input-buffer bases as `Buffer*` bindings (`emplace_runtime_args`, program_factory.cpp:523–531, slots 6/15) and the per-region byte offsets as plain scalars (`q_region_offset`/`k_region_offset`/`v_region_offset`, program_factory.cpp:330–332, slots 7/16); the sharded kernel reconstructs `src = base + region_offset + head_idx·head_size` (`reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp:42,43,89,90`). The Interleaved factory binds `in0`/`in1`/`q`/`k`/`v` as `Buffer*` and addresses tiles by `page_id`. *Reconcile with the dated triage `analyses/2026-07-19_offset_base_pointers.md`:* it lists this op as Type-1, but **no fold is present in the audited code → the doc is stale for this op** (the recipe's "no fold, op in tables → doc stale → GREEN" outcome; the fold was split out in `86872e0a06a`).
- **TensorAccessor 3rd argument:** GREEN. Every `TensorAccessor` construction is 2-arg (interleaved reader lines 36, 40; interleaved writer lines 42–44); the sharded kernel constructs none. Op is (correctly) absent from `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - *Interleaved* — `in0` (q input), `in1` (kv input, optional), `q`/`k`/`v` outputs: **Case 1** (`Buffer*`→`TensorAccessor` by `page_id`). Express as `TensorParameter`; kernel builds `TensorAccessor(tensor::name)`.
  - *Sharded* — `input_tensor_q` (slot 6) and `input_tensor_kv` (slot 15; the q tensor in the fused path, the kv tensor when present): **Case 2** (raw NoC arithmetic on the base). Bind as `TensorParameter`, pull the base via `get_bank_base_address`, keep the raw walk.
  - *Sharded outputs* — `c_16`/`c_17`/`c_18` are borrowed-memory CBs (`.buffer = output.buffer()`, program_factory.cpp:447/463/479): **clean** (borrowed-DFB via `DataflowBufferSpec::borrowed_from`).
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:**
  - *Interleaved* — `cb1` (reader→writer FIFO) legal `(1,1)`; with `transpose_k_heads`: `cb0` (reader→compute), `cb16` (compute→writer) each legal `(1,1)`.
  - *Sharded* — `c_16` (Q output): **1P+1C** — the reader-config and writer-config instances both raw-`get_write_ptr` into it (risc0 vs risc1 Q heads), no FIFO ops → two role-free touchers; bind one PRODUCER + one CONSUMER (cosmetic on Gen1). `c_17` (K, reader FIFO-produces) and `c_18` (V, writer FIFO-produces): single locked producer each → **self-loop**.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (dual-instance work-split):** `Sharded` `c_16` has two touchers per node because one kernel source is instantiated twice over `q_cores`. It is **1P+1C, not multi-binding** (no third toucher, both sync-free) — do not set the multi-binding flag.
- **Cross-op / shared kernels:** `transpose_wh.cpp` is file-path-instantiated from the shared pool `ttnn/cpp/ttnn/kernel/compute/` (Interleaved, `transpose_k_heads` only). Port the shared kernel as one unit with its co-borrowers.
- **RTA varargs:** the sharded kernel reads the per-remote-core NoC coordinate arrays `in0_mcast_noc_x` / `in0_mcast_noc_y` via `get_arg_addr(19)` / `get_arg_addr(19 + num_x)` (count = `num_cores_x`/`num_cores_y`, varies per instantiation; indexed at runtime). Port as RTA **varargs**, not named args. Slots 0–18 are fixed named scalars.

## Team-only

- **Out-of-directory coupling & donor shape:** function-call escapes — none (kernels include only `api/*` HAL headers). File-path instantiation — `transpose_wh.cpp` (shared compute pool). Roll-up `✓ clean` on call-escapes; one shared-pool borrow to sequence as a port-together unit.
- **TTNN factory analysis:** `descriptor` concept, no op-owned tensors, no custom hash, no pybound `create_descriptor`, no custom `override_runtime_arguments`, no runtime-args-update hook. Target concept `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Interleaved reader dead placeholder RTA (non-KV path):** slot 1 = `uint32_t{0}` pushed when no KV tensor (`program_factory.cpp:245`), read into `in1_tensor_addr` but used only under `#ifdef READ_FROM_INPUT_TENSOR_KV`. Harmless dead arg ("keep offsets stable").
- **Sharded slot 7 (`q_region_offset`) is a constant 0** (Q begins at shard offset 0). Kept as an explicit scalar for symmetry with the K/V region offset (slot 16) and to preserve the arg-index layout; the kernel adds it (not dead, not a fold).

## Recipe notes

- **Dated offset-base triage now over-lists this op.** `analyses/2026-07-19_offset_base_pointers.md` still lists `nlp_create_qkv_heads` (and its sibling `nlp_create_qkv_heads_boltz`) as Type-1; after commit `86872e0a06a` the fold is gone here. This is the recipe's anticipated "no fold, op in tables → doc stale" case — flagging for the triage-doc owner to drop the `nlp_create_qkv_heads` row (and `_boltz` once it gets the same fix).
