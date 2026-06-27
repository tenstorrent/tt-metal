# matmul — changelog

## Phase 0 — 2026-06-27 — fused 2D dual-multicast matmul (initial implementation)

**What was done.** Implemented `C = A @ B` as fused on-device kernels with the
2D dual-orthogonal-multicast topology from `op_design.md`:

- **Reader (`matmul_reader.cpp`, NCRISC):** per output block, per K-block, the
  first column (X==0) reads its activation row-block from DRAM and multicasts it
  ACROSS the grid-row; the first row (Y==0) reads its weight column-block and
  multicasts it DOWN the grid-column. Uses `dataflow_kernel_lib::SenderPipe` /
  `ReceiverPipe` (`mcast_pipe.hpp`) for the NoC-multicast + semaphore handshake,
  and raw `TensorAccessor` + `Noc::async_read` for the DRAM tile gather (no
  kernel-lib helper covers interleaved tile addressing). in0 phase fully precedes
  in1 phase each K-block (acyclic — in0 confined to rows, in1 to columns).
- **Compute (`matmul_compute.cpp`, TRISC):** one `compute_kernel_lib::matmul_block`
  per output block (all defaults: `last_block_target=Out`, `SubblockMajor`,
  `init_mode=Short`, `WaitAndPopPerKBlock`, `reconfig=INPUT_AND_OUTPUT`). The
  helper loops the K-blocks internally and spills/reloads via `cb_interm`. Boot:
  `compute_kernel_hw_startup` + `mm_block_init` once.
- **Writer (`matmul_writer.cpp`, BRISC):** drains one SubblockMajor output block
  from `cb_out` and scatters tiles to interleaved DRAM, skipping out-of-range
  (ragged/phantom) tiles.
- **L1 bounded for arbitrary M/N/K:** K streamed in K-blocks of
  `in0_block_w = find_max_divisor(Kt, 4)`; M/N output blocks bounded by an L1
  budget with an outer per-core block loop run lock-step across the grid.

**Accuracy (measured).** fp32 activation + fp32 weight, HiFi4, `fp32_dest_acc_en=True`:
- PCC = 1.000000 on 256x256, 512x512, and batched (1,2,128,256)@(256,128).
- relative RMS ≈ 0.0013–0.0017.
- Acceptance suite tolerance is PCC ≥ 0.999; golden tolerance (float32, acc=True)
  is (PCC 0.999, RMS 0.02). Comfortably inside both.

**Tests.**
- Acceptance (`tests/.../matmul/test_matmul.py`): **16/16 pass** in BOTH `--dev`
  (watcher clean) and non-dev (production timing — no race in the mcast handshake).
- Golden (`eval/golden_tests/matmul/test_golden.py`): **20 passed, 1 skipped,
  646 xfailed**. Every Phase 0 SUPPORTED cell passes; all out-of-contract cells
  xfail-strict; NO XPASS-strict (SUPPORTED/EXCLUSIONS consistent with behavior).

**SUPPORTED (Phase 0).** dtype=float32, weight_dtype=float32, layout=TILE,
fp32_dest_acc_en=True, alignment=tile_aligned, weight_batch=single.
**EXCLUSIONS:** `{dtype=float32, fp32_dest_acc_en=False}` (maxed input demands
maxed accumulator).

**Issues encountered.** None — passed first device run.

**Advisory deviations (from `op_design.md`).**
- The design's RISC-split table suggests the reader as "one source, 4
  KernelDescriptors keyed by per-group CT role flags `(is_in0_sender,
  is_in1_sender)`." Implemented instead as a SINGLE reader KernelDescriptor with
  per-core RUNTIME role flags (derived from `grid_row`/`grid_col`) and per-K-block
  pipe construction. This realizes the IDENTICAL inter-core contract (same 4
  semaphores, same sender/receiver roles, same in0→in1 ordering) with simpler host
  wiring. Per-call pipe construction is the proven-safe pattern from
  `pipe_rotating.cpp` (the ReceiverPipe ctor's INVALID always precedes the sender's
  VALID because the sender's mcast is gated behind the receiver's ack). Verified
  correct under both `--dev` and non-dev timing.
- `cb_out` is double-buffered (×2) for writer pipelining; `cb_interm` is ×1. The
  design listed the out-CB ×2 as optional and interm as ×1 — within latitude.

**Follow-up axes (TARGET, not yet implemented — all currently xfail-strict).**
mixed/low precision (`dtype`/`weight_dtype` ∈ {bfloat16, bfloat8_b}),
`fp32_dest_acc_en=False`, non-tile-aligned M/K/N (`alignment` masking paths),
batched weight (`weight_batch=batched`).
