# Single-Device Profiling Op for Ring Joint SDPA

## Goal

Create a profiling op (`ring_joint_sdpa_profile`) that isolates and measures the compute portion of `ring_joint_sdpa` when running in **causal + balanced** mode, without multi-device communication overhead.

## Key Insight: Zigzag/Balanced Distribution

When `is_causal=True` and `is_balanced=True`, the sequence is distributed using a **zigzag pattern**:

- For `ring_size=2` with 4 chunks: `balanced_order = [0, 3, 1, 2]`
  - Device 0 gets chunks **[0, 3]**
  - Device 1 gets chunks **[1, 2]**

- For `ring_size=4` with 8 chunks: `balanced_order = [0, 7, 1, 6, 2, 5, 3, 4]`
  - Device 0: [0, 7], Device 1: [1, 6], Device 2: [2, 5], Device 3: [3, 4]

This balances causal attention workload: early Q chunks have less work, late Q chunks have more work.

---

## 1. Helper Functions

```python
def create_balanced_chunk_order(ring_size: int) -> List[int]:
    """
    Create balanced chunk order for sequence reordering.
    For ring_size=2: [0, 3, 1, 2]
    For ring_size=4: [0, 7, 1, 6, 2, 5, 3, 4]
    """
    num_chunks = 2 * ring_size
    balanced_order = []
    left, right = 0, num_chunks - 1
    for i in range(num_chunks):
        if i % 2 == 0:
            balanced_order.append(left)
            left += 1
        else:
            balanced_order.append(right)
            right -= 1
    return balanced_order


def get_device_chunk_indices(ring_index: int, ring_size: int) -> List[int]:
    """
    Get original chunk indices assigned to a device.
    Device 0 in ring_size=2: [0, 3]
    Device 1 in ring_size=2: [1, 2]
    """
    chunk_order = create_balanced_chunk_order(ring_size)
    chunks_per_device = 2
    start_pos = ring_index * chunks_per_device
    return chunk_order[start_pos : start_pos + chunks_per_device]


def get_ring_id_arrival_order(ring_index: int, ring_size: int) -> List[int]:
    """
    Compute ring_id arrival order for Linear topology.
    Simulates fused_op_receiver bidirectional alternating logic.

    ring_size=2:
      Device 0: [0, 1]
      Device 1: [1, 0]

    ring_size=4:
      Device 0: [0, 1, 2, 3]
      Device 1: [1, 2, 0, 3]
      Device 2: [2, 3, 1, 0]
      Device 3: [3, 2, 1, 0]
    """
    from_forward = ring_size - 1 - ring_index
    from_backward = ring_index
    expected_inputs = (from_forward, from_backward)

    received_inputs = [0, 0]
    curr_dir = 0
    arrival_order = []

    for transfer_idx in range(ring_size):
        if transfer_idx == 0:
            sender_ring_id = ring_index
        else:
            received_inputs[curr_dir] += 1
            if curr_dir == 1:
                sender_ring_id = (ring_index - received_inputs[curr_dir] + ring_size) % ring_size
            else:
                sender_ring_id = (ring_index + received_inputs[curr_dir]) % ring_size

        arrival_order.append(sender_ring_id)

        if transfer_idx == 0:
            if expected_inputs[curr_dir] == 0:
                curr_dir = 1 - curr_dir
        else:
            next_dir = 1 - curr_dir
            if received_inputs[next_dir] < expected_inputs[next_dir]:
                curr_dir = next_dir

    return arrival_order


def get_kv_arrival_chunk_indices(ring_index: int, ring_size: int) -> List[Tuple[int, List[int]]]:
    """
    Get (ring_id, chunk_indices) pairs in arrival order.
    For device 0, ring_size=2: [(0, [0,3]), (1, [1,2])]
    """
    ring_id_order = get_ring_id_arrival_order(ring_index, ring_size)
    return [(rid, get_device_chunk_indices(rid, ring_size)) for rid in ring_id_order]
```

---

## 2. Data Preparation Functions

```python
def extract_chunks(tensor: torch.Tensor, chunk_indices: List[int], chunk_size: int, seq_dim: int = 2) -> torch.Tensor:
    """Extract and concatenate specified chunks from tensor."""
    chunks = []
    for idx in chunk_indices:
        start = idx * chunk_size
        end = start + chunk_size
        if seq_dim == 2:
            chunks.append(tensor[:, :, start:end, :])
    return torch.cat(chunks, dim=seq_dim)


def build_gathered_kv_buffer(
    kv_full: torch.Tensor,
    ring_index: int,
    ring_size: int,
    chunk_size: int
) -> torch.Tensor:
    """
    Build gathered KV buffer in ring arrival order.
    Layout: [local_kv | ring_iter_1_kv | ring_iter_2_kv | ...]

    Includes local KV at the start (Option A) for simpler kernel indexing.
    """
    kv_arrival = get_kv_arrival_chunk_indices(ring_index, ring_size)
    chunks = []
    for ring_id, chunk_indices in kv_arrival:
        chunks.append(extract_chunks(kv_full, chunk_indices, chunk_size))
    return torch.cat(chunks, dim=2)
```

---

## 3. PyTorch Reference Implementation

```python
def compute_causal_balanced_reference(
    Q_full: torch.Tensor,
    K_full: torch.Tensor,
    V_full: torch.Tensor,
    ring_index: int,
    ring_size: int
) -> torch.Tensor:
    """
    Compute expected output for a device in causal+balanced ring attention.

    1. Compute full causal attention in original order
    2. Extract output for this device's Q chunk positions
    3. Return in device's local order (matching profiling op output layout)
    """
    # Full causal attention
    full_output = torch.nn.functional.scaled_dot_product_attention(
        Q_full, K_full, V_full, is_causal=True
    )

    # Get this device's chunk indices
    seq_len = Q_full.shape[2]
    chunk_size = seq_len // (2 * ring_size)
    device_chunks = get_device_chunk_indices(ring_index, ring_size)

    # Extract output for this device's Q positions (in device's local order)
    return extract_chunks(full_output, device_chunks, chunk_size)
```

---

## 4. Op Signature

```python
def ring_joint_sdpa_profile(
    input_tensor_q: ttnn.Tensor,      # [b, nh, local_seq_len, d] - this device's Q
    input_tensor_k: ttnn.Tensor,      # [b, nh, local_seq_len, d] - this device's local K
    input_tensor_v: ttnn.Tensor,      # [b, nh, local_seq_len, d] - this device's local V
    gathered_k: ttnn.Tensor,          # [b, nh, full_seq_len, d] - all KV in arrival order
    gathered_v: ttnn.Tensor,          # [b, nh, full_seq_len, d] - all KV in arrival order
    *,
    ring_size: int,
    ring_index: int,
    is_causal: bool = True,
    is_balanced: bool = True,
    logical_n: int,
    joint_tensor_q: Optional[ttnn.Tensor] = None,
    joint_tensor_k: Optional[ttnn.Tensor] = None,
    joint_tensor_v: Optional[ttnn.Tensor] = None,
    joint_strategy: Optional[str] = None,
    program_config: ttnn.SDPAProgramConfig,
    scale: Optional[float] = None,
    compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor], ttnn.Tensor]:
    """
    Single-device profiling op for ring_joint_sdpa (causal+balanced).

    Simulates what one device in a ring would compute, but reads "gathered" KV
    from DRAM instead of via ring all-gather. This isolates compute performance
    from inter-device communication.

    Returns:
        output: [b, nh, local_seq_len, d] - attention output for this device's Q chunks
        joint_output: [b, nh, joint_seq_len, d] - joint output (if joint tensors provided)
        lse_output: [b, nh, local_seq_len + joint_seq_len, 1] - log-sum-exp
    """
```

---

## 5. Minimal Test Case

```python
def test_ring_joint_sdpa_profile_device0_ring2():
    """Minimal test: ring_size=2, ring_index=0, causal+balanced, no joint."""

    # Config
    ring_size = 2
    ring_index = 0
    b, nh, seq_len, d = 1, 2, 256, 64
    q_chunk_size, k_chunk_size = 64, 64
    chunk_size = seq_len // (2 * ring_size)  # 64

    # Create full tensors
    Q_full = fa_rand(b, nh, seq_len, d)
    K_full = fa_rand(b, nh, seq_len, d)
    V_full = fa_rand(b, nh, seq_len, d)

    # Prepare local Q (this device's chunks: [0, 3])
    device_chunks = get_device_chunk_indices(ring_index, ring_size)
    Q_local = extract_chunks(Q_full, device_chunks, chunk_size)
    K_local = extract_chunks(K_full, device_chunks, chunk_size)
    V_local = extract_chunks(V_full, device_chunks, chunk_size)

    # Prepare gathered KV (all KV in arrival order)
    K_gathered = build_gathered_kv_buffer(K_full, ring_index, ring_size, chunk_size)
    V_gathered = build_gathered_kv_buffer(V_full, ring_index, ring_size, chunk_size)

    # Expected output
    expected = compute_causal_balanced_reference(Q_full, K_full, V_full, ring_index, ring_size)

    # Convert to ttnn tensors
    tt_Q_local = ttnn.from_torch(Q_local, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_K_local = ttnn.from_torch(K_local, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_V_local = ttnn.from_torch(V_local, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_K_gathered = ttnn.from_torch(K_gathered, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_V_gathered = ttnn.from_torch(V_gathered, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 7),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )

    # Run profiling op
    tt_output, _, tt_lse = ttnn.transformer.ring_joint_sdpa_profile(
        tt_Q_local,
        tt_K_local,
        tt_V_local,
        gathered_k=tt_K_gathered,
        gathered_v=tt_V_gathered,
        ring_size=ring_size,
        ring_index=ring_index,
        is_causal=True,
        is_balanced=True,
        logical_n=seq_len,
        program_config=program_config,
    )

    # Compare
    tt_output_torch = ttnn.to_torch(tt_output)
    assert_close(tt_output_torch, expected, rtol=1e-2, atol=1e-2)
```

---

## 6. Implementation Plan (Kernel Side)

### Files to Create/Modify

1. **New op files** in `ttnn/cpp/ttnn/operations/transformer/sdpa/device/`:
   - `ring_joint_sdpa_profile_device_operation.hpp`
   - `ring_joint_sdpa_profile_device_operation.cpp`
   - `ring_joint_sdpa_profile_device_operation_types.hpp`
   - `ring_joint_sdpa_profile_program_factory.hpp`
   - `ring_joint_sdpa_profile_program_factory.cpp`

2. **New kernels** in `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/`:
   - `dataflow/ring_joint_profile_reader.cpp` - copy of `ring_joint_reader.cpp` with gather removed
   - `dataflow/ring_joint_profile_writer.cpp` - copy of `ring_joint_writer.cpp` with gather sync removed
   - `compute/ring_joint_sdpa.cpp` - **reuse exactly as-is** (no copy needed)

3. **Python bindings** in `ttnn/cpp/ttnn/operations/transformer/sdpa/`:
   - Add to `sdpa_nanobind.cpp`
   - Add to `sdpa.hpp` / `sdpa.cpp`

### Kernel Changes

**Reader kernel (`ring_joint_profile_reader.cpp`):**
- Remove `#include "fused_op_receiver.hpp"`
- Remove `RingSDPAOpReceiver` initialization and calls
- Remove semaphore waits (`noc_semaphore_wait`, `noc_semaphore_inc`)
- Remove S&F chain forwarding logic
- For ring_iter > 0: read from `gathered_k/v` at offset `ring_id * local_padded_Nt`
- `ring_id` order computed at compile time or passed as runtime args

**Writer kernel (`ring_joint_profile_writer.cpp`):**
- Remove gather-related synchronization
- Direct DRAM writes only

**Compute kernel:**
- No changes - reuse `ring_joint_sdpa.cpp` exactly

---

## 7. Summary Tables

### Chunk Assignment (Balanced Order)

| ring_size | balanced_order | Device 0 | Device 1 | Device 2 | Device 3 |
|-----------|----------------|----------|----------|----------|----------|
| 2 | [0,3,1,2] | [0,3] | [1,2] | - | - |
| 4 | [0,7,1,6,2,5,3,4] | [0,7] | [1,6] | [2,5] | [3,4] |

### KV Arrival Order (Linear Topology)

| ring_size=2 | Device 0 | Device 1 |
|-------------|----------|----------|
| ring_id order | [0, 1] | [1, 0] |
| KV chunks | [0,3]→[1,2] | [1,2]→[0,3] |

| ring_size=4 | Device 0 | Device 1 | Device 2 | Device 3 |
|-------------|----------|----------|----------|----------|
| ring_id order | [0,1,2,3] | [1,2,0,3] | [2,3,1,0] | [3,2,1,0] |

---

## 8. What This Measures

- Pure compute time for one device's workload
- Memory bandwidth (DRAM reads/writes)
- **Without**: inter-device latency, gather synchronization overhead, fabric contention
