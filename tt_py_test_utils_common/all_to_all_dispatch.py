# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import random
import ttnn


def tt_to_torch_dtype(tt_dtype):
    if tt_dtype == ttnn.bfloat16:
        return torch.bfloat16
    elif tt_dtype == ttnn.bfloat8_b:
        return torch.bfloat16
    elif tt_dtype == ttnn.float32:
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype: {tt_dtype}")


def gen_tokens(batch, hidden_size, seq_len, mesh_shape, devices, scheme="random", dtype=torch.bfloat16):
    tokens = []
    factor = 1
    for _ in range(batch):
        for _ in range(seq_len):
            if scheme == "sequential":
                tokens.append(torch.ones(1, 1, 1, hidden_size, dtype=dtype) * factor)
                factor += 1
            else:
                tokens.append(torch.rand(1, 1, 1, hidden_size, dtype=dtype))
    res = torch.cat(tokens, dim=0)
    return res.reshape(batch, 1, seq_len, hidden_size)


def gen_expert_mapping(experts, devices, scheme="random"):
    assert experts % devices == 0

    expert_mapping = torch.zeros(1, 1, experts, devices, dtype=torch.int16)
    device_id = 0
    experts_per_devices = experts // devices
    device_expert_count = {d: 0 for d in range(devices)}
    for i in range(experts):
        if scheme == "random":
            device_id = random.choice(
                [d for d, _ in filter(lambda kv: kv[1] < experts_per_devices, device_expert_count.items())]
            )
            expert_mapping[0, 0, i, device_id] = 1
            device_expert_count[device_id] += 1
        else:
            if i > 0 and i % experts_per_devices == 0:
                device_id += 1
            expert_mapping[0, 0, i, device_id] = 1

    # identical across all devices
    return expert_mapping


def get_metadata_tensor(expert_indices, expert_mapping, mesh_shape):
    # metadata tensor is expert_indices duplicated for each device
    # [D, B, 1, K]
    batch = expert_indices.shape[0]
    seq_len = expert_indices.shape[2]
    devices = mesh_shape[0] * mesh_shape[1]
    selected_experts_k = expert_indices.shape[3]
    metadata_tensor = torch.reshape(expert_indices, (1, batch, seq_len, selected_experts_k))
    return metadata_tensor.repeat(devices, 1, 1, 1)


def get_expert_indices(batch, experts, selected_experts_k, seq_len, mesh_shape, scheme="random"):
    expert_indices = torch.ones(batch, 1, seq_len, selected_experts_k, dtype=torch.int16) * -1
    current_expert = 0

    # For avg_perf scheme, track how many tokens are assigned to each expert
    if scheme == "avg_perf":
        tokens = batch * seq_len
        # Use ceiling division to ensure we have enough capacity, with minimum of 1
        max_tokens_per_expert = max(1, (tokens * selected_experts_k + experts - 1) // experts)
        expert_token_count = {e: 0 for e in range(experts)}

    token = 0
    for b in range(batch):
        for s in range(seq_len):
            token += 1
            for k in range(selected_experts_k):
                if scheme == "sequential":
                    expert_indices[b, 0, s, k] = current_expert % experts
                    current_expert += 1 + (k % 2)
                elif scheme == "random" or scheme == "random_sequential_experts":
                    # need to ensure a set of unique indices
                    current_indices = expert_indices[b, 0, s, :].tolist()
                    expert_indices[b, 0, s, k] = random.choice(
                        list(filter(lambda e: e not in current_indices, range(experts)))
                    )
                elif scheme == "avg_perf":
                    # Random selection but each expert is capped at max_tokens_per_expert
                    # to simulate average performance case
                    current_indices = expert_indices[b, 0, s, :].tolist()
                    # First, try to find experts under the cap
                    available_experts = [
                        e
                        for e in range(experts)
                        if e not in current_indices and expert_token_count[e] < max_tokens_per_expert
                    ]
                    # Fallback: if all capped experts are used, pick any expert not in current token
                    if not available_experts:
                        available_experts = [e for e in range(experts) if e not in current_indices]
                    chosen_expert = random.choice(available_experts)
                    expert_indices[b, 0, s, k] = chosen_expert
                    expert_token_count[chosen_expert] += 1
                elif scheme == "worst_perf":  # worst perf is when the expert index is always on the last device
                    expert_indices[b, 0, s, k] = (
                        experts - 1
                    )  # technically each expert index should be different, but we're sending to the same device regardless
                elif scheme == "worst_congestion":
                    # Worst case link congestion: each token selects experts on consecutive
                    # devices in one direction, maximizing traffic through intermediate links.
                    # From device D, select experts on devices D+1, D+2, ..., D+k
                    # This creates a cascade where link i→i+1 carries traffic from all
                    # devices 0..i to all devices i+1..n, maximizing link utilization.
                    devices = mesh_shape[0] * mesh_shape[1]
                    experts_per_device = experts // devices

                    # Determine source device for this token (batch is sharded in chunks across devices)
                    # Device 0 gets batch 0 to batch/devices-1, device 1 gets next chunk, etc.
                    batches_per_device = batch // devices
                    src_device = b // batches_per_device

                    # Target device is src_device + k + 1 (wrapping around)
                    target_device = (src_device + 1 + k) % devices

                    # Expert offset: if we wrap around and visit a device multiple times,
                    # pick a different expert on that device each time
                    expert_offset = k // devices
                    expert_id = target_device * experts_per_device + (expert_offset % experts_per_device)

                    expert_indices[b, 0, s, k] = expert_id
                elif scheme == "best_congestion":
                    # Best case for congestion: tokens prefer local experts first, then nearest
                    # neighbors, minimizing average hop distance and spreading traffic.
                    # Algorithm:
                    # 1. Fill local device first (up to experts_per_device)
                    # 2. Alternate between CCW and CW neighbors, picking 1 expert at a time
                    # 3. When a device is full, move to the next device in that direction
                    #
                    # Example with k=8, experts_per_device=2:
                    # - 2 local
                    # - 2 from CCW neighbor (1-hop)
                    # - 2 from CW neighbor (1-hop)
                    # - 1 from CW 2-hop neighbor
                    # - 1 from CCW 2-hop neighbor
                    devices = mesh_shape[0] * mesh_shape[1]
                    experts_per_device = experts // devices
                    # Batch is sharded in chunks across devices (not round-robin)
                    # Device 0 gets batch 0 to batch/devices-1, device 1 gets next chunk, etc.
                    batches_per_device = batch // devices
                    src_device = b // batches_per_device

                    # Build list of (device, local_expert_idx) for all k experts
                    if k == 0:
                        # First expert slot - start building the selection list
                        # We need to compute all k experts at once for this token
                        picked = []
                        remaining = selected_experts_k

                        # Step 1: Fill local device
                        local_count = min(remaining, experts_per_device)
                        for i in range(local_count):
                            picked.append((src_device, i))
                        remaining -= local_count

                        # Step 2: Alternate CCW and CW
                        ccw_hop = 1
                        cw_hop = 1
                        ccw_count = 0  # experts picked from current CCW device
                        cw_count = 0  # experts picked from current CW device
                        use_ccw = True  # alternate starting with CCW

                        while remaining > 0:
                            if use_ccw:
                                ccw_device = (src_device - ccw_hop) % devices
                                if ccw_count < experts_per_device:
                                    picked.append((ccw_device, ccw_count))
                                    ccw_count += 1
                                    remaining -= 1
                                else:
                                    # Move to next CCW device
                                    ccw_hop += 1
                                    ccw_count = 0
                                    continue  # Don't switch direction yet, retry with new device
                            else:
                                cw_device = (src_device + cw_hop) % devices
                                if cw_count < experts_per_device:
                                    picked.append((cw_device, cw_count))
                                    cw_count += 1
                                    remaining -= 1
                                else:
                                    # Move to next CW device
                                    cw_hop += 1
                                    cw_count = 0
                                    continue  # Don't switch direction yet, retry with new device
                            use_ccw = not use_ccw  # Alternate direction

                        # Store the selection in a way we can access for subsequent k values
                        # We use a simple approach: just compute all k at once and store in tensor
                        for idx, (device, local_idx) in enumerate(picked):
                            expert_id = device * experts_per_device + local_idx
                            expert_indices[b, 0, s, idx] = expert_id
                    # For k > 0, the values were already set when k == 0
                elif scheme == "worst_congestion_descending":
                    # Worst case for sparse multicast: send to furthest device first (antipode),
                    # then progressively closer devices. This maximizes the hop distance for
                    # each packet, as opposed to worst_congestion (ascending) which starts
                    # from the nearest neighbor.
                    #
                    # For a 16-device ring from src_device:
                    # - k=0: hop 8 (antipode, maximum distance)
                    # - k=1: hop 7
                    # - k=2: hop 6
                    # - ...
                    # - k=7: hop 1 (nearest neighbor)
                    # - k=8: hop 0 (local, if k > devices/2)
                    #
                    # This is the worst case for sparse multicast because:
                    # 1. Each packet travels maximum distance before hitting any destination
                    # 2. No benefit from early delivery along the path
                    # 3. Maximum total hop-distance across all packets
                    devices = mesh_shape[0] * mesh_shape[1]
                    experts_per_device = experts // devices

                    # Batch is sharded in chunks across devices
                    batches_per_device = batch // devices
                    src_device = b // batches_per_device

                    # Start from antipode (devices // 2 hops away) and decrement
                    antipode_hop = devices // 2
                    hop_distance = max(0, antipode_hop - k)  # Clamp to 0 for local when k > antipode
                    target_device = (src_device + hop_distance) % devices

                    # Expert offset: if we wrap around and visit a device multiple times,
                    # pick a different expert on that device each time
                    expert_offset = k // devices
                    expert_id = target_device * experts_per_device + (expert_offset % experts_per_device)

                    expert_indices[b, 0, s, k] = expert_id
                else:
                    raise ValueError(f"Invalid scheme: {scheme}")
    return expert_indices


def get_output_tensor(input_tokens, expert_indices, expert_mapping, seq_len, mesh_shape, dtype=torch.bfloat16):
    # output tensor is [devices, batch, seq_len, hidden_size]
    # depending on the expert indices, the input tokens are scattered to different experts
    # these experts are sent to one ore more device based on the expert mapping
    batch = input_tokens.shape[0]
    mesh_columns = mesh_shape[1]
    mesh_rows = mesh_shape[0]
    devices = mesh_columns * mesh_rows
    hidden_size = input_tokens.shape[3]
    selected_experts_k = expert_indices.shape[3]

    # initialize the output tensor with random values so we can test the non-populated rows too
    output_tensor = torch.rand(devices, batch, seq_len, hidden_size, dtype=dtype)

    idx = expert_indices[:, 0, :, :].long()  # [batch, seq_len, k]
    mapping = expert_mapping[0, 0] == 1  # [experts, devices] bool
    present = mapping[idx].any(dim=2)  # [batch, seq_len, devices] bool
    mask = present.permute(2, 0, 1)  # [devices, batch, seq_len] bool
    tokens = input_tokens[:, 0, :, :].unsqueeze(0).expand(devices, -1, -1, -1)  # [devices, batch, seq_len, hidden]
    output_tensor[mask] = tokens[mask]

    return output_tensor


def get_mesh_mapper(mesh_device, mesh_shape, cluster_axis, shard_dim):
    if cluster_axis is None:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim)
    elif cluster_axis == 1:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, shard_dim), mesh_shape=mesh_shape)
    elif cluster_axis == 0:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(shard_dim, None), mesh_shape=mesh_shape)
    else:
        raise ValueError(f"Invalid cluster_axis: {cluster_axis}")
    return mesh_mapper
