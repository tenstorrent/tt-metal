# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `2cd0286fa17 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — both factories (`RotaryEmbeddingHfMultiCore`, `RotaryEmbeddingHfMultiCoreSharded`), each with two internal descriptor shapes (single-tile vs multi-tile) selected on `padded_shape()[-1] / TILE_WIDTH == 1`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none` `TensorParameter relaxation` · `get_dynamic_runtime_args` (deprecated hook). Also absent (none of these gate, but this op happens to carry none of them either): custom `compute_program_hash`, `override_runtime_arguments`, pybound `create_descriptor`.

## Construct — to do

**Tensor bindings** (per binding, per factory/config):

- `RotaryEmbeddingHfMultiCore`, **multi-tile prefill**:
  - `input` — **Case 1** (reader RTA arg0 is the `Buffer*`, fed to `TensorAccessor s0` at `reader_rotary_embedding_hf_interleaved.cpp:33,39`) → `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::<name>)`. When `in_sharded`, the same tensor **also** backs `c_0` via `CBDescriptor::buffer` (factory `:406`) → that DFB is `borrowed_from` the same `TensorParameter`; both constructs coexist (see Watch for: self-aliasing read).
  - `cos` — **Case 1** (reader `:34,42`). `sin` — **Case 1** (reader `:35,45`).
  - `output` — **Case 1** (writer `writer_rotary_embedding_hf_interleaved.cpp:20,23`); under `out_sharded` also the borrowed backing of `c_16` (factory `:499`), and the accessor path is `#ifdef`-ed out by `OUT_SHARDED`.
- `RotaryEmbeddingHfMultiCore`, **single-tile prefill**:
  - interleaved variant: `input`/`cos`/`sin` — **Case 1** (`reader_..._single_tile_interleaved_start_id.cpp:87-99`); `output` — **Case 1** (writer, shared with multi-tile).
  - `in_sharded` variant: `input` — **clean** (borrowed-DFB only; the reader `..._start_id_sharded.cpp` takes no src accessor, just cursor-advances `c_0` at `:95-96`); `cos`/`sin` — **Case 1** (`:85-86,99,102`); `output` as above.
- `RotaryEmbeddingHfMultiCoreSharded` (both decode shapes): `input`/`cos`/`sin`/`output` — **clean** (borrowed-memory DFBs at sharded factory `:87,99,111,168` (single-tile) / `:278,290,302,360` (multi-tile); the factory sets **no** runtime args at all).
- Note: all address delivery today is the descriptor-API `Buffer*`-in-RTA `BufferBinding` form (`emplace_runtime_args` at multi_core factory `:298-312`, `:608-618`) — correct-on-cache-hit today; the typed bindings supersede it and the address RTAs disappear.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** drop the redundant page-size arg (all Class 2, pure no-op) at:
`reader_rotary_embedding_hf_interleaved.cpp:39,42,45` · `writer_rotary_embedding_hf_interleaved.cpp:23` · `reader_..._single_tile_interleaved_start_id.cpp:93,96,99` · `reader_..._single_tile_interleaved_start_id_sharded.cpp:99,102`. No `dynamic_tensor_shape` needed anywhere.

**CB endpoints:**

- **Self-loop** (bind the one toucher PRODUCER **and** CONSUMER — all are compute-kernel self-loops, legal on Gen1):
  - `c_24`/`c_25`/`c_26` (interm) — every factory, every config;
  - `c_0` (input), `c_1` cos, `c_2` sin, `c_16` (output) — **sharded factory, both decode configs** (compute reserve/push/wait/pops the borrowed inputs itself; output stays resident, nothing drains).
- **Legal 1:1** everywhere else — bind the FIFO producer/consumer as the ops dictate: reader→compute on `c_0`/`c_1`/`c_2`/`c_3`/`c_4` (per config; includes the trans_mat and scalar CBs where the consumer only ever `wait_front`s — still the CONSUMER binding), compute→writer on `c_16` in the MultiCore factory.
- **No** multi-binding flag, **no** dead-CB drop, **no** conditional DFB.

## Watch for

- **CB endpoints (multi-binding):** none — no CB exceeds two touchers on any node in any config.
- **Cross-op / shared kernels:** `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/rotary_embedding_single_tile.cpp` (bound by the single-tile-prefill path, multi_core factory `:257-259`, `:273-275`) → caution per `port_patterns.md` "Caution: Porting a shared kernel". **No `_metal2` fork exists yet** — by rung 2 this port creates `rotary_embedding_single_tile_metal2.cpp` **beside the original** (in the `rotary_embedding` op's directory — the sanctioned two-edit carve-out) *unless* the sibling `rotary_embedding` port (running in the same effort) has created it first — re-run the rung-1 locational check at port time and reuse its fork + binding names if present. Other binding ops: {`rotary_embedding`} — **sunset list, not authorization to convert the kernel in place**. Fork note: the kernel has a `DECODE_MODE` define-path (CTAs 9–14) that only `rotary_embedding` exercises; this op supplies no defines and only CTAs 0–8.
- **RTA varargs:** none — every arg in every kernel is a fixed-index scalar; name them all (prefer named RTAs/CTAs throughout).
- **Per-core-group compute duplicate:** the MultiCore factory pushes the **same compute source** twice over **disjoint** core groups, differing only in the `num_rows_per_core` CTA (single-tile: CTA[8], factory `:271`; multi-tile: CTA[9], factory `:581`). Keep it as two KernelSpecs with per-group CTAs — do **not** demote the per-group CTA to an RTA (anti-pattern in `port_patterns.md`). Each node sees one instance, so bindings are ordinary 1:1.
- **`OUT_SHARDED` define** (multi_core factory `:219-221`, `:528-531`): gates the writer between drain-to-DRAM and wait-only behavior; carry it via the KernelSpec's defines, keeping the conditional emission (`out_sharded` only).
- **Self-aliasing read in in-sharded multi-tile prefill** (`reader_rotary_embedding_hf_interleaved.cpp:89-95` with borrowed `c_0`): the reader NoC-reads src tiles into the very region borrowed from `input.buffer()`. Intentional legacy behavior — port byte-for-byte; do not elide the copy even though it looks redundant.
- **Dead CTA in the sharded multi-tile compute kernel** (`rotary_embedding_hf_sharded.cpp:25,29` — `Ht` read then `(void)`-discarded, fed `n_heads_t` by factory `:373`): an anomaly recorded for the ops team; the port makes no functional change around it — translate the arg list faithfully per the recipe's naming rules.
- **Scalar/trans_mat CBs are consumed by `wait_front` without a final `pop_front`** in some kernels (e.g. multi-tile prefill compute `rotary_embedding_hf.cpp:60`; trans_mat consumers). That is still a locked CONSUMER binding — don't misread it as a role-free peek.
