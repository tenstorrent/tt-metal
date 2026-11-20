# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import ttnn
from tt_py_test_utils_common.ccl_semaphores import create_global_semaphores
from tt_py_test_utils_common.comparison_funcs import comp_equal, comp_pcc


def run_reduce_scatter_impl(
    mesh_device,
    num_devices,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    rs_topology,
    num_iters=1,
    enable_trace=True,
    ones_tensor=False,
    small_random_ints=False,
    mem_config_intermediate=None,
    cluster_axis=None,
    use_barrier=False,
    use_persistent_buffers=True,
    chunks_per_sync=None,
    num_workers_per_link=None,
    num_buffers_per_channel=None,
    verify_output=True,
    use_new=False,
    use_strided=False,
    verify_output_shape=True,
    verify_output_pcc=True,
    mm_cores_y=None,
    mm_block_ht=None,
    mm_block_wt=None,
    mm_N_full_block_wt=None,
    chunk_width_in_mm_blocks=None,
):
    torch.manual_seed(0)

    ##### Fabric setup #####
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [create_global_semaphores(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

    barrier_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    ### Create persistent output buffers
    logger.debug("Creating persistent buffers")
    intermediate_shape = rs_input_shape[:]
    if rs_topology == ttnn.Topology.Linear:
        # Line RS requires double-sized input for forward/backward
        intermediate_shape.insert(0, 2)
    else:
        # Ring RS intermediate holds only one batch element at a time
        intermediate_shape[0] = 1
    if use_persistent_buffers:
        persistent_intermediate_buffers = [
            ttnn.from_torch(
                torch.zeros(intermediate_shape),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=rs_input_dtype,
                memory_config=mem_config_intermediate,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for _ in range(num_iters)
        ]
    rs_output_shape = rs_input_shape[:]
    rs_output_shape[dim] //= num_devices
    if use_persistent_buffers:
        persistent_output_buffers = [
            ttnn.from_torch(
                torch.zeros(rs_output_shape),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=rs_input_dtype,
                memory_config=mem_config_rs,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for _ in range(num_iters)
        ]

    logger.debug("Done creating persistent buffers")

    ##### Reduce scatter input setup #####
    logger.debug(f"Reduce scatter shape: {rs_input_shape}")
    logger.debug(f"Reduce scatter dim: {dim}")
    logger.debug(f"input mem config: {mem_config_input}")
    logger.debug(f"Reduce input mem config: {mem_config_rs}")
    logger.debug(f"intermediate mem config: {mem_config_intermediate}")
    logger.debug(f"topology: {rs_topology}")

    tt_input_tensor_mesh_list = []
    torch_input_tensor_list = []

    for i in range(num_iters):
        rs_global_input_shape = rs_input_shape[:]
        rs_global_input_shape[dim] *= num_devices
        if ones_tensor:
            rs_input_tensor = torch.ones(rs_global_input_shape).bfloat16()
        elif small_random_ints:
            # Random integers from {0, 1, 2, 3} for easier debugging
            rs_input_tensor = torch.randint(0, 4, rs_global_input_shape).bfloat16()
        else:
            rs_input_tensor = torch.rand(rs_global_input_shape).bfloat16()
        input_tensors = torch.chunk(rs_input_tensor, num_devices, dim)
        torch_input_tensor_list.append(input_tensors)

        input_tensor_mesh = ttnn.from_torch(
            rs_input_tensor,
            device=mesh_device,
            layout=layout,
            dtype=rs_input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(dim)], ttnn.MeshShape(1, num_devices)
                ),
            ),
        )
        tt_input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Verify Correct Reference Output #####
    torch_reduce_scatter_output_list = []
    for i in range(num_iters):
        reduce_output = torch.sum(torch.stack(torch_input_tensor_list[i]), dim=0)
        scatter_output = torch.chunk(reduce_output, num_devices, dim)
        torch_reduce_scatter_output_list.append(scatter_output)

    ##### Perform the TT ops #####
    tt_reduce_scatter_output_list = []

    def run_op(i):
        if use_strided:
            logger.debug(f"Using strided reduce scatter")
            tt_reduce_scatter_output_tensor = ttnn.experimental.strided_reduce_scatter_async(
                tt_input_tensor_mesh_list[i],
                persistent_output_buffers=(
                    [persistent_intermediate_buffers[i], persistent_output_buffers[i]]
                    if use_persistent_buffers
                    else None
                ),
                dim=dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                barrier_semaphore=barrier_semaphore_handles[i] if use_barrier else None,
                num_links=num_links,
                memory_config=mem_config_rs,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
                mm_cores_y=mm_cores_y,
                mm_block_ht=mm_block_ht,
                mm_block_wt=mm_block_wt,
                mm_N_full_block_wt=mm_N_full_block_wt,
                chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
            )
        elif use_new:
            logger.debug(f"Using new reduce scatter")
            tt_reduce_scatter_output_tensor = ttnn.reduce_scatter(
                tt_input_tensor_mesh_list[i],
                dim=dim,
                num_links=num_links,
                memory_config=mem_config_rs,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
            )
        else:
            logger.debug(f"Using experimental reduce scatter")
            tt_reduce_scatter_output_tensor = ttnn.experimental.reduce_scatter_minimal_async(
                tt_input_tensor_mesh_list[i],
                persistent_output_buffers=(
                    [persistent_intermediate_buffers[i], persistent_output_buffers[i]]
                    if use_persistent_buffers
                    else None
                ),
                dim=dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                barrier_semaphore=barrier_semaphore_handles[i] if use_barrier else None,
                num_links=num_links,
                memory_config=mem_config_rs,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
            )
        return tt_reduce_scatter_output_tensor

    if enable_trace:
        for i in range(num_iters):
            run_op(i)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.debug("Done compiling ops")

        logger.debug(f"Capturing trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_reduce_scatter_output_trace_list = []
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
            tt_reduce_scatter_output_trace_list.append(tt_reduce_scatter_output_tensor)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.debug(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        logger.debug(f"Done executing trace")

        # Synchronize the devices
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        for tt_tensor in tt_reduce_scatter_output_trace_list:
            tt_rs_out = ttnn.from_device(tt_tensor)
            tt_rs_out = ttnn.to_torch(tt_rs_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))
            tt_tensor.deallocate(True)
            tt_reduce_scatter_output_list.append(tt_rs_out)
    else:
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
            tt_rs_out = ttnn.from_device(tt_reduce_scatter_output_tensor)
            tt_rs_out = ttnn.to_torch(tt_rs_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))
            tt_reduce_scatter_output_tensor.deallocate(True)
            tt_reduce_scatter_output_list.append(tt_rs_out)

            logger.debug(f"Waiting for op")
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.debug(f"Done op")

            logger.debug(f"Done iteration {i}")

    if verify_output:
        implementation_name = "strided" if use_strided else ("new" if use_new else "experimental")
        logger.debug(f"Verifying {implementation_name} reduce scatter output")

        for i in range(num_iters):
            tt_rs_out = tt_reduce_scatter_output_list[i]
            torch_rs_out_tensor = torch_reduce_scatter_output_list[i]

            # Split the concatenated output back into per-device chunks
            tt_output_chunks = torch.chunk(tt_rs_out, num_devices, dim=dim)
            for device_id in range(num_devices):
                tt_output_tensor = tt_output_chunks[device_id]
                torch_output_tensor = torch_rs_out_tensor[device_id]

                if small_random_ints:
                    eq, output = comp_equal(tt_output_tensor, torch_output_tensor)
                else:
                    eq, output = comp_pcc(tt_output_tensor, torch_output_tensor)
                logger.debug(f"{output}, device {device_id}, iteration {i}")

                if verify_output_shape:
                    assert (
                        tt_output_tensor.shape == torch_output_tensor.shape
                    ), f"Shape mismatch: {tt_output_tensor.shape} != {torch_output_tensor.shape}"
                if verify_output_pcc:
                    assert eq, f"{i} FAILED reduce scatter: {output}"

    mesh_device.reset_sub_device_stall_group()
    logger.debug("Done")
