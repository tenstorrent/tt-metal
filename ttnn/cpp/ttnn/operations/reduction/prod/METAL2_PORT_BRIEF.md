# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/reduction/prod`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `37f03926088 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

**Porting unit:** two bundled device operations sharing this directory and a donor kernel — port them together:
- `ProdAllDeviceOperation` / `ProdAllProgramFactory` (`device/prod_all_program_factory.cpp`)
- `ProdNcDeviceOperation` / `ProdNcProgramFactory` (`device/prod_nc_program_factory.cpp`)

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both ops port to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (both factories — vanilla single-program `ProgramDescriptor`).
- **Op-owned tensors:** none (both).
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · smuggled pointer. All `no` on the readiness sheet and confirmed in code.

## Construct — to do

**Tensor bindings** (per binding — all four are the mechanical Case 1 case):

- **prod_all** `input` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter`/`TensorBinding`; the donor reader builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(src_args, src_addr)`. Drops reader RTA arg 0 (`prod_all_program_factory.cpp:103`) and its `TensorAccessorArgs` CTAs.
- **prod_all** `output` — **Case 1** → bind; donor writer builds `TensorAccessor(tensor::name)`. Drops writer RTA arg 0 (`prod_all_program_factory.cpp:104`) and its `TensorAccessorArgs` CTAs.
- **prod_nc** `input` — **Case 1** → bind; `reader_prod_nc.cpp:29` builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(dram_input_addrg_args, input_addr)`. Drops reader RTA arg 0 (`prod_nc_program_factory.cpp:189`) and its `TensorAccessorArgs` CTAs. **Keep** the tile-navigation scalars (`num_reduce_input_tile`, `num_tiles_per_core`, `input_tile_offset`, `tile_offset`, `HtWt`, `CHtWt`) — they are page indices, not addresses.
- **prod_nc** `output` — **Case 1** → bind; donor writer builds `TensorAccessor(tensor::name)`. Drops writer RTA arg 0 (`prod_nc_program_factory.cpp:200`) and its `TensorAccessorArgs` CTAs.

No Case 2 (raw-pointer) bindings — no `get_bank_base_address` bridge needed anywhere.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every `TensorAccessor` is already 2-arg.

**CB endpoints:** all legal. Both factories have exactly `c_0` (input) and `c_3` (output), each a plain 1-producer + 1-consumer FIFO on every node. No self-loop, no 1P+1C assignment, no multi-binding flag, no dead-CB drop.
- prod_all: `c_0` = donor-reader (P) + compute (C); `c_3` = compute (P) + donor-writer (C).
- prod_nc: `c_0` = own-reader (P) + compute (C); `c_3` = compute (P) + donor-writer (C). Compute is split across disjoint core groups (`compute_desc_1`/`compute_desc_2`) — each node sees one instance, so still 1:1.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** you are porting two **broadly-shared** `eltwise/unary` donor kernels — `writer_unary_interleaved_start_id.cpp` (both factories; ~29 co-borrowers) and `reader_unary_interleaved_start_id.cpp` (prod_all only; ~12 co-borrowers). Their Metal 2.0 rewrite is a **single shared change** that every co-borrower must adopt in the same migration — coordinate with the shared-kernel owners; do not migrate them for prod in isolation, or the co-borrowers break the instant one op moves.
- **RTA varargs:** none — name each runtime arg directly.
- **(Not port work, but be aware):** the audit flagged three dead args in prod_nc (a dead reader RTA `dim`, a dead writer RTA `is_dram`, a dead compute CTA) — routed to the ops team, **not** part of the port diff. Do not "clean them up" during the port; leave behavior unchanged.
