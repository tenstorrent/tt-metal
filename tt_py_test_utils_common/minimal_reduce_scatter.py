# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import ttnn
from tt_py_test_utils_common.comparison_funcs import comp_equal, comp_pcc


def create_global_semaphores(mesh_device, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(3)]
    return ccl_semaphore_handles


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
    mem_config_intermediate=None,
    cluster_axis=None,
    use_barrier=False,
    use_persistent_buffers=True,
    chunks_per_sync=None,
    num_workers_per_link=None,
    num_buffers_per_channel=None,
    verify_output=True,
    use_new=False,
    contiguous_staging=None,
):
    use_sub_devices = False
    torch.manual_seed(0)

    tile = (32, 32)

    ##### Fabric setup #####
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    if use_sub_devices:
        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [create_global_semaphores(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

    barrier_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    rs_output_shape = rs_input_shape[:]
    rs_output_shape[dim] //= num_devices
    intermediate_shape = rs_input_shape[:]
    if rs_topology == ttnn.Topology.Linear:
        # Line RS requires double-sized input for forward/backward
        intermediate_shape.insert(0, 2)
    # Ring / scatter-dim != 0 can use the contiguous fast path: the intermediate is a chunk-paged staging
    # buffer (not input-shaped) and the 2nd-last ring iteration stages one direction's contribution into
    # a second penult intermediate instead of scatter-writing into the tiled output. Both must be
    # allocated via the op's own sizing helper; other configs use an input-shaped tiled intermediate and
    # have no penult intermediate.
    #
    # The op picks the path from the intermediate it is handed, so on the ring path a caller may opt into
    # the tiled layout by passing contiguous_staging=False. Everywhere else the tiled layout is the only
    # option, and asking for the contiguous one is unsupported.
    if contiguous_staging is None:
        contiguous_staging = rs_topology == ttnn.Topology.Ring and dim != 0
    use_contiguous_staging = contiguous_staging
    assert not (
        use_contiguous_staging and not (rs_topology == ttnn.Topology.Ring and dim != 0)
    ), "contiguous staging requires Ring topology and scatter dim != 0"

    ##### All gather input setup #####
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

    ### Create persistent output buffers
    logger.debug("Creating persistent buffers")
    persistent_intermediate_buffers = []
    persistent_output_buffers = []
    persistent_penult_intermediate_buffers = []
    if use_persistent_buffers:
        for i in range(num_iters):
            if use_contiguous_staging:
                (
                    interm_buf,
                    penult_interm_buf,
                ) = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                    tt_input_tensor_mesh_list[i], dim=dim, topology=rs_topology, cluster_axis=cluster_axis
                )
            else:
                interm_buf = ttnn.from_torch(
                    torch.zeros(intermediate_shape),
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=rs_input_dtype,
                    memory_config=mem_config_intermediate,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                penult_interm_buf = None
            persistent_intermediate_buffers.append(interm_buf)
            persistent_penult_intermediate_buffers.append(penult_interm_buf)
            persistent_output_buffers.append(
                ttnn.from_torch(
                    torch.zeros(rs_output_shape),
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=rs_input_dtype,
                    memory_config=mem_config_rs,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            )
    logger.debug("Done creating persistent buffers")

    ##### Perform torch ops #####
    torch_reduce_scatter_output_list = []
    for i in range(num_iters):
        reduce_output = torch.sum(torch.stack(torch_input_tensor_list[i]), dim=0)
        scatter_output = torch.chunk(reduce_output, num_devices, dim)
        torch_reduce_scatter_output_list.append(scatter_output)

    ##### Perform the TT ops #####
    tt_reduce_scatter_output_list = []

    def run_op(i):
        if use_new:
            logger.debug(f"Using new reduce scatter")
            tt_reduce_scatter_output_tensor = ttnn.reduce_scatter(
                tt_input_tensor_mesh_list[i],
                dim=dim,
                num_links=num_links,
                memory_config=mem_config_rs,
                intermediate_memory_config=mem_config_intermediate,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
            )
        else:
            logger.debug(f"Using experimental reduce scatter")
            if use_persistent_buffers:
                buffers = [persistent_intermediate_buffers[i], persistent_output_buffers[i]]
                if persistent_penult_intermediate_buffers[i] is not None:
                    buffers.append(persistent_penult_intermediate_buffers[i])
            else:
                buffers = None
            tt_reduce_scatter_output_tensor = ttnn.experimental.reduce_scatter_minimal_async(
                tt_input_tensor_mesh_list[i],
                persistent_output_buffers=buffers,
                dim=dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                barrier_semaphore=barrier_semaphore_handles[i] if use_barrier else None,
                num_links=num_links,
                memory_config=mem_config_rs,
                intermediate_memory_config=mem_config_intermediate,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
            )

        return tt_reduce_scatter_output_tensor

    if enable_trace:
        # Compile the op
        tt_reduce_scatter_output_trace_list = []
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
        logger.debug(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
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
        for i in range(num_iters):
            tt_rs_out = tt_reduce_scatter_output_list[i]
            torch_rs_out_tensor = torch_reduce_scatter_output_list[i]

            torch_rs_out = torch.cat(torch_rs_out_tensor, dim)

            if ones_tensor:
                eq, output = comp_equal(tt_rs_out, torch_rs_out)
            else:
                eq, output = comp_pcc(tt_rs_out, torch_rs_out)

            logger.debug(f"{output}, iteration {i}")
            assert eq, f"{i} FAILED ag: {output}"

    mesh_device.reset_sub_device_stall_group()
    if use_sub_devices:
        mesh_device.clear_loaded_sub_device_manager()
