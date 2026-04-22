# neighbor_pad_conv3d (fused NeighborPad + Conv3d)

Single-dispatch 2D halo exchange fused with Conv3d.  Replaces the two-dispatch
(`neighbor_pad_async` then `conv3d`) pattern with a single program that runs
halo-write, halo-read, and conv3d compute on the same device program.

## Halo exchange pipeline

```
  ┌──────────────────┐   fabric writes   ┌──────────────────┐
  │  H-writer cores  │──────────────────▶│  H-top / H-bot   │  (halo buffer
  │  (each device)   │                   │  halo sections   │   on DRAM)
  └──────────────────┘                   └──────────────────┘
           │ barrier_sem (once per dispatch, end-of-kernel)
           ▼
  ┌──────────────────┐   fabric writes   ┌──────────────────┐
  │  W-reader cores  │──────────────────▶│  W-left / W-right│
  │  + paired W-wr   │                   │  halo sections   │
  └──────────────────┘                   └──────────────────┘
           │ progress_sem
           ▼
  ┌──────────────────┐  reads input+halo  ┌──────────────────┐
  │  Conv3d reader   │───────────────────▶│  output tensor   │
  └──────────────────┘                    └──────────────────┘
```

- **H halo** is written *during* the H-writer's main loop; the final barrier to
  the W-fabric cores fires once at end-of-kernel.
- **W halo** is written *during* the W-reader's main loop; the paired W-writer
  fabric-sends each stick to the neighbor's halo buffer.
- **Conv3d reader** waits on `progress_sem` (GlobalSemaphore pinned to L1 of
  every conv3d reader core), then runs the normal vol2col + matmul conv3d.

## Progress-sem contract

Progress-sem signalling is always-on in the fused op; `conv_config.input_progress_t_batch_size`
controls the per-T-batch granularity (0 = one signal at end of NP, N = one per N input T frames).
- Host resets the sem via `reset_global_semaphore_value(progress_sem, 0)` before each dispatch.
  The sem is allocated on the full compute grid (`ccl_cores` spans the whole grid),
  so conv3d reader cores *are* reset.
- **Receiver-side W-reader only** signals per batch (sender has no incoming-data wait,
  its signal would fire before the remote fabric write for that batch has landed).
  Receiver waits on `w_neighbor_sem ≥ (batch + 1) * sticks_per_batch` before each
  increment — fabric in-order delivery guarantees the batch's data is in DRAM by then.
- The conv3d reader does a per-T-block `noc_semaphore_wait_min(progress_sem, threshold)`
  where `threshold = desired_batches × signal_count`, capped by `max_progress_signals`
  (the total signals for this dispatch, passed as compile-time arg).
- End-of-kernel defensive reset on the conv3d reader (kept against host/dispatch ordering drift).

Per-batch signalling unlocks true T-pipelining: conv3d cores can process early t_out blocks
while later NP batches are still in flight. Empirically the overlap only pays off on shapes
where `T_out > t_out_parallel` (i.e. each conv3d core handles multiple t_out values). For
small-T shapes the production code in `vae_wan2_1.py` routes to standalone NP+Conv3d
instead — see `MIN_T_FOR_FUSED`.

## RTA refresh contract

**Every per-dispatch address must be refreshed in
`NpConv3dMeshWorkloadFactory::override_runtime_arguments`.** Things that change
between dispatches include:

| RTA | Kernel | Index | Source |
|---|---|---|---|
| `input_tensor_address` | NP H-reader (common) | 0 | current input buffer |
| `halo_buffer_addr` | NP H-reader (common) | 1 | ping-pong halo buffer |
| `input_tensor_address` | NP H-writer (common) | 0 | current input buffer |
| `halo_buffer_addr` | NP H-writer (common) | 1 | ping-pong halo buffer |
| `progress_sem_l1_addr` | NP H-writer (common) | 4 | ping-pong progress sem |
| `halo_buffer_addr` | NP W-reader (common) | 0 | ping-pong halo buffer |
| `progress_sem_l1_addr` | NP W-reader (common) | 3 | ping-pong progress sem |
| `input_buffer->address()` | NP W-reader **per-core** | **10** | current input buffer |
| `halo_buffer_addr` | NP W-writer (common) | 0, 1 | ping-pong halo buffer |
| `input_addr` | Conv3d reader (per-core) | 0 | current input buffer |
| `progress_sem_l1_addr` | Conv3d reader (per-core) | 11 | ping-pong progress sem |
| `halo_buffer_addr` | Conv3d reader (per-core) | 13 | ping-pong halo buffer |
| `output_addr` | Conv3d writer (per-core) | 0 | current output buffer |
| `weight_addr` | Conv3d writer (per-core) | 1 | weight buffer |
| `bias_addr` | Conv3d writer (per-core) | 2 | bias buffer |

The W-reader per-core RTA[10] in particular is easy to miss because the W-reader
only otherwise uses common runtime args. Missing the refresh causes the
W-reader to pull its local data from a stale DRAM address on every dispatch
after the first, fabric-writing garbage into the neighbor's halo buffer, and
producing a bell-curve seam at the D/D+1 W boundary in the receiver's output
(see commit `d8a939bf1e`).

## Known issues / future work

- **T-pipelining across the halo/compute boundary** is currently one-shot (all
  halo ready → conv3d starts). Per-T-batch signalling in the W-reader plus an
  in-T-loop cumulative wait in the conv3d reader is sketched in
  `PLAN_FUSED_CLEANUP.md` §Phase 3 with the correct threshold formula.
- **2x4 mesh validation** is pending CI on a real Blackhole LoudBox.
  bh-lb-09's ETH link topology supports only 2x2.
