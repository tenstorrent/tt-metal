# Port Report — rotary_embedding_llama_fused_qk

Metal 2.0 port of `rotary_embedding_llama_fused_qk` (single, compute-only factory).
**Verified on Blackhole p150b: 12 passed, 4 failed** (`test_rotary_embedding_llama_fused_qk.py`, run with the blanket `skip_for_blackhole` temporarily lifted for verification). The 4 failures are all the `decode_8` (batch=8) cases; the **legacy `create_descriptor` version fails the identical 4 cases on BH** (legacy PCC ≈0.52, this port ≈0.71), so batch-8 is a **pre-existing Blackhole limitation** (the reason for `skip_for_blackhole` #12349), not a port regression. All 12 non-batch-8 cases pass at PCC ≈0.99999.

### Port-time fix (worth noting)
The initial port set the compute WorkUnit's `target_nodes` to `all_cores.bounding_box()` (mirroring the legacy kernel placement) but provided the per-core `is_q` RTA only for `q_cores ∪ k_cores`. Metal 2.0 requires the named RTA on **every** placed node, so configs with a non-rectangular working set hit `TT_FATAL ... nodes_with_named_params.contains(node)`. Fix: target the actual working set `all_cores = q_cores.merge(k_cores)` (placement == args-coverage by construction). The legacy "place on bounding box, let unused/garbage cores run with unread output" idiom does not map to Metal 2.0's stricter per-node-args + borrowed-DFB-residency model — the union placement is the correct Metal 2.0 expression.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept` for `RotaryEmbeddingLlamaFusedQKProgramFactory`. Single-program, compute-only, no op-owned device resources. The variant has a single factory (no `select_program_factory`); the framework auto-selects. The factory selects the compute `.source` on `operation_attributes.row_major_QK` (both sources have identical CTA layout and DFB bindings — not a multi-variant fan-out).

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op used the default reflection hash).
- Pybind entry points removed: none. The nanobind file binds only the user-facing op entry (`ttnn::experimental::rotary_embedding_llama_fused_qk`); `create_descriptor` was never pybound.

### Open items
- Strict tensor matching kept (default). No relaxation candidates noted.

## Handoff points

- **None.** No out-of-op kernel edits, no `sem::`/`ta::` boundary violations (the kernels build no TensorAccessor and use no semaphores), no kernel-lib gaps. All LLK call sites (`mm_init`, `matmul_tiles`, `mul_tiles`, `mul_tiles_bcast`, `add_tiles`, `pack_tile`, `binary_op_init_common`, …) accept the `dfb::`/`uint32_t` CB ids unchanged via the `DFBAccessor::operator uint32_t()` implicit conversion.

## Successes

- **Borrowed-memory DFBs (patterns catalog: Borrowed-memory DFB).** All 7 resident sharded CBs (`q_in` c_0, `k_in` c_1, `cos` c_2, `sin` c_3, `trans_mat` c_4, `q_out` c_16, `k_out` c_17) ported cleanly from legacy `CBDescriptor::buffer` to `DataflowBufferSpec::borrowed_from = <TensorParameter>`. The 7 backing tensors are declared as `TensorParameter`s and supplied as `TensorArgument`s, but **none is a `TensorBinding`** — the compute kernel never builds a TensorAccessor and never reads a base address, so there is no Case-1/Case-2 binding at all. This is a cleaner shape than the nlp reference (which needed a Case-2 `get_bank_base_address` bridge); here the borrowed DFB is the only address mechanism the kernel needs.
- **Runtime-dynamic CB selection (recipe kernel-side whitelist).** The kernel picks `in_cb`/`out_cb` at runtime from the per-core RTA `is_q`. Ported to `uint32_t in_cb = is_q ? dfb::q_in : dfb::k_in;` (and `out_cb`) using the `DFBAccessor`→`uint32_t` implicit conversion — no `.id` extraction, no temp wrappers. `DataflowBuffer in_cb_obj(in_cb)` uses the `uint16_t` ctor. All four candidate DFBs (q_in/k_in/q_out/k_out) are bound on the single compute kernel, so both runtime branches reference bound names.
- **Compute self-loop with INTRA connectivity (patterns catalog: Self-loop DFB binding; advanced_options).** The 3 real intermediate FIFOs (`rotated_in_interm` c_24, `cos_interm` c_25, `sin_interm` c_26) are genuine producer==consumer self-loops on the compute kernel — exactly the accumulation-reference shape. Setting `dfb_self_loop_connectivities[DFB] = INTRA` for every self-looped DFB was the load-bearing extra step (the accumulation reference predates that field).

## Friction

- **Gap — compute-only op = ALL CBs are fake CBs.** The patterns catalog's "Fake CB → self-loop" entry frames fake CBs as the exception (a few address-source CBs alongside real reader/writer-driven FIFOs). Here the op has **no dataflow kernels at all**, so *every* borrowed CB is one-ended from the validator's view and *all 7* become fake-CB self-loops on the single compute kernel. The catalog's single-reading-kernel self-loop recipe applies, but the "the audit flags these FYI" framing undersells how pervasive it is for a pure-compute op — worth a sub-note that a compute-only op with resident I/O turns every borrowed CB into a self-loop.
- **Confusion — `dfb_self_loop_connectivities` not in the accumulation reference.** The accumulation reference (the canonical compute-self-loop example) does the ACC PRODUCER+CONSUMER self-loop but does **not** set `dfb_self_loop_connectivities`. The task brief (and `advanced_options.hpp:120-137`) require it for compute-kernel self-loops. Following the header comment, I set INTRA for every self-looped DFB. If the validator in fact tolerates an absent entry (defaulting to INTRA), the reference and this port disagree — flagging so the doc/reference can be reconciled.

## Open items for downstream

- **Fake-CB self-loop bindings (interim hack — flag prominently).** All 7 borrowed DFBs are bound as self-loops (PRODUCER+CONSUMER on the single compute kernel) purely to satisfy the validator's producer-and-consumer rule; none is a real FIFO. Site: `device/rotary_embedding_llama_fused_qk_program_factory.cpp` — the `compute_spec.dfb_bindings` block and the `dfb_self_loop_connectivities` table. Classification:
  - **Tensor-local-view (read-only address source)** → forthcoming local-`TensorAccessor`: `q_in` (`(c_0, read)`), `k_in` (`(c_1, read)`), `cos` (`(c_2, read)`), `sin` (`(c_3, read)`), `trans_mat` (`(c_4, read)`).
  - **Tensor-local-view (write-only address sink)** → forthcoming local-`TensorAccessor`/output mechanism: `q_out` (`(c_16, write)`), `k_out` (`(c_17, write)`).
  The 3 interm DFBs (`rotated_in_interm`, `cos_interm`, `sin_interm`) are **real** self-loops (genuine intra-kernel FIFO scratch), not fake CBs — they do not need replacement.
- **Cross-op kernel touches:** none. Both compute kernels are op-owned.
- **Test coverage:** the op's dedicated test is `tests/tt_eager/python_api_testing/unit_testing/misc/test_rotary_embedding_llama_fused_qk.py` (`test_rotary_embedding_llama_fused_qk` + `_with_program_cache`), which is **blanket `@skip_for_blackhole` (#12349)** — so on this BH box it has no default coverage. Verified by temporarily lifting the skip: 12 passed / 4 failed, with the 4 batch-8 (`decode_8`) failures confirmed **pre-existing** (legacy fails them identically, PCC ≈0.52). The nightly `test_rotary_embedding_llama.py` does not exercise the fused_qk op in its active (single-chip) tests (`fuse_qk` is never parametrized True and `run_test_row_major_*` has no caller). Recommend the owner revisit the `skip_for_blackhole` once the batch-8 BH numerics issue (#12349) is resolved.
- **Doc-evolution suggestion:** a "compute-only op (resident sharded I/O, no dataflow kernels)" worked example would be a useful addition — every CB self-loops on the compute kernel, the only RTA is a per-core work-selector, and there are zero TensorBindings despite seven tensors.
