# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** version cannot be pinned — the metal_2.0 docs directory is untracked in this checkout (`git log` prints nothing for its path). *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (`create_descriptor` at `device/rotary_embedding_llama_fused_qk_program_factory.cpp:18`; single factory in `program_factory_t`)
- **Op-owned tensors:** none
- **Target concept:** `ProgramSpecFactoryConcept`
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none` `TensorParameter relaxation` · `get_dynamic_runtime_args` (deprecated hook). Also confirmed absent (none gate, but each shapes the port and this op has **none of them**): custom `compute_program_hash`, `override_runtime_arguments`, pybound `create_descriptor` — nothing to preserve, translate, or delete on those fronts.

**Op shape in one line:** one factory → **one compute kernel** (source selected by `operation_attributes.row_major_QK`, factory:237-242; tiled → `device/kernels/compute/rotary_embedding_llama_sharded.cpp`, row-major → `..._row_major.cpp`), **zero dataflow kernels**, 10 CBs (7 borrowed-memory over sharded tensors + 3 local interms), one named per-core RTA.

## Construct — to do

**Tensor bindings** (per binding): all seven are **clean** — borrowed-memory DFBs, no Case 1 / Case 2 anywhere (the op has no address RTAs, no `TensorAccessor`s):

- `q_input` (c_0, factory:100-110), `k_input` (c_1, :112-122), `cos` (c_2, :124-134), `sin` (c_3, :136-146), `trans_mat` (c_4, :148-160), `q_output` (c_16, :196-206), `k_output` (c_17, :207-217) — each is a `CBDescriptor` with `.buffer = <tensor buffer>` → express as `TensorParameter` + `DataflowBufferSpec::borrowed_from`.
- The three interms — `rotated_input_interm` (c_24, :163-172), `cos_interm` (c_25, :174-183), `sin_interm` (c_26, :185-194) — are plain (non-borrowed) DFBs.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — the op constructs no `TensorAccessor`.

**CB endpoints:** **self-loop all 10 CBs, in both configs** (the census is identical for the tiled and row-major kernel variants) — the single compute kernel is the only toucher of every CB; bind it PRODUCER **and** CONSUMER on each. Details per CB in the audit's endpoint table. No multi-binding flag anywhere, no dead-CB drops, no conditional DFBs.

**RTAs / CTAs:** one named RTA — `is_q` (`uint32_t`, 1 on q cores / 0 on k cores; factory:258-268, kernels:29). CTAs 0-12 (factory:220-236): the ten CB indices dissolve into `dfb::` tokens; `q_Ht`, `k_Ht`, `Wt` become named CTAs. No varargs of any kind.

## Watch for

- **CB endpoints (multi-binding):** none — every CB is single-toucher; no hidden-2nd-writer or multi-reader shape exists (no `get_write_ptr`/`get_read_ptr`/`fifo_*_ptr` call anywhere in either kernel).
- **Cross-op / shared kernels:** none — both kernel sources are op-owned and this factory is their **sole binder** (filename census run; the same-named file in the sibling `rotary_embedding_llama` op is that op's own private copy). No `_metal2` fork exists or is needed — convert both sources **in place** (this is the ordinary sole-binder case, not the shared-kernel fork convention). One caution: `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama_sharded.cpp` (the *sibling's*) is already a landed Metal 2.0 kernel — same filename, different directory; don't mix them up. Its binding vocabulary (`dfb::input`/`dfb::cos`/`dfb::sin`/`dfb::trans_mat`/`dfb::rotated_interm`/`dfb::cos_interm`/`dfb::sin_interm`, lines 27-35) is the family's established naming — extend it for the q/k split (e.g. `dfb::q_input`, `dfb::k_input`, `dfb::q_out`, `dfb::k_out`) rather than inventing a new scheme.
- **RTA varargs:** none — the single `is_q` RTA is a fixed named field; prefer named RTAs throughout.
- **q/k runtime mux (the one non-mechanical kernel spot):** the kernel picks its in/out CBs and `Ht` at *runtime* from `is_q` into non-constexpr `uint32_t` locals (kernels:40-47), then builds `CircularBuffer` objects from the runtime-selected index (sharded.cpp:59-60). `dfb::name`'s constexpr `uint32_t` cast covers the LLK-call positions (assign tokens into the runtime locals); but confirm the sanctioned form for the *object* side — constructing a `DataflowBuffer` from a runtime-selected index may not be supported, in which case build both objects (q and k) and select by branch/reference. Do not compile-time-split the kernel per q/k (that would change kernel instantiation shape — the legacy design is one instance with a runtime flag).
- **Kernel vs CB core ranges — deliberate asymmetry, keep it:** the compute kernel is on `work_cores = q_cores ∪ k_cores` (factory:76), *not* the CBs' `all_cores_bb` (factory:69) — the comment at factory:71-76 documents a watcher SIGABRT (out-of-bounds `get_arg_val(0)`) if the kernel lands on bounding-box hole cores that get no RTAs. Keep the KernelSpec on `work_cores`. Choose DFB spec core ranges deliberately (borrowed DFBs plausibly follow their tensor's shard grid); legacy configures q-backed CBs on k cores and vice versa, harmlessly (never dereferenced there).
- **TRISC2 code-size cliff:** both kernels are within ~4 bytes of the TRISC2 code-size limit with the profiler enabled (kernels:24-28, factory:255-257 — the `has_work` early-return is commented out for exactly this reason). Any kernel-side growth or optimization-level change can tip it — set the compute kernel's opt level explicitly on the `KernelSpec` (prior ports regressed silently on the default) and re-check a profiler-enabled build.
- **Dead locals in the tiled kernel:** `cos_cb_obj`/`sin_cb_obj`/`trans_mat_cb_obj` (sharded.cpp:61-63) are constructed and never used; cos/sin/trans_mat are consumed only via LLK index calls (no FIFO ops, no wrapper needed — the row-major variant already omits the objects). The port still binds those DFBs (the kernel names their indices), but need not recreate the unused objects.
