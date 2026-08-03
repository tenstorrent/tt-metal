import ttnn
import numpy as np


def create_program_descriptor(input_a: ttnn.Tensor, input_b: ttnn.Tensor, input_c: ttnn.Tensor, output: ttnn.Tensor):
    assert input_a.shape() == input_b.shape()
    assert input_a.shape() == input_c.shape()
    assert input_a.shape() == output.shape()
    assert input_a.memory_config() == input_b.memory_config()
    assert input_a.memory_config() == input_c.memory_config()


    tile_size = ttnn.TILE_SIZE * ttnn.TILE_SIZE
    num_tiles = int(input_a.shape[-1] * input_a.shape[-2] / tile_size)

    device = input_a.device()
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    print(f"grid size {(grid.x, grid.y)}")
    if num_tiles % (grid.x * grid.y) != 0:
        exit(1)
    num_tiles_per_core = num_tiles // (grid.x * grid.y)




        
    if input_a.is_sharded():
        cb_a = ttnn.cb_descriptor_from_sharded_tensor(0, input_a)
        cb_b = ttnn.cb_descriptor_from_sharded_tensor(1, input_b)
        cb_c = ttnn.cb_descriptor_from_sharded_tensor(2, input_c)
    else:
        cb_a = ttnn.CBDescriptor(
            total_size=2 * input_a.buffer_page_size(),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=input_a.dtype, page_size=input_a.buffer_page_size())
            ],
        )
        cb_b = ttnn.CBDescriptor(
            total_size=2 * input_b.buffer_page_size(),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=1, data_format=input_b.dtype, page_size=input_b.buffer_page_size())
            ],
        )
        cb_c = ttnn.CBDescriptor(
            total_size=2 * input_c.buffer_page_size(),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=2, data_format=input_c.dtype, page_size=input_c.buffer_page_size())
            ],
        )

    if output.is_sharded():
        cb_out = ttnn.cb_descriptor_from_sharded_tensor(16, output)
    else: 
        cb_out = ttnn.CBDescriptor(
            total_size=2 * output.buffer_page_size(),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=16, data_format=output.dtype, page_size=output.buffer_page_size())
            ],
        ),
    cbs = [
        cb_a,
        cb_b,
        cb_c,
        ttnn.CBDescriptor(
            total_size=2 * input_a.buffer_page_size(),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=8, data_format=input_a.dtype, page_size=input_a.buffer_page_size())
            ],
        ),
        
        cb_out
    ]

    # --- Reader ---
    reader_ct_args = []
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_a).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_b).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_c).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    for x, y in np.ndindex((grid.x, grid.y)):
        reader_rt_args[x][y] = [
            input_a.buffer_address(),
            input_b.buffer_address(),
            input_c.buffer_address(),
            (y * grid.x + x) * num_tiles_per_core,
            num_tiles_per_core,
        ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/ttnn/operations/muladd_test/device/reader.cpp",
        core_ranges=cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer ---
    writer_ct_args = []
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    for x, y in np.ndindex((grid.x, grid.y)):
        writer_rt_args[x][y] = [
            output.buffer_address(),
            (y * grid.x + x) * num_tiles_per_core,
            num_tiles_per_core,
        ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/ttnn/operations/muladd_test/device/writer.cpp",
        core_ranges=cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute ---

    compute_rt_args = ttnn.RuntimeArgs()
    for x, y in np.ndindex((grid.x, grid.y)):
        compute_rt_args[x][y] = [
            (y * grid.x + x) * num_tiles_per_core,
            num_tiles_per_core,
        ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/ttnn/operations/muladd_test/device/compute.cpp",
        core_ranges=cores,
        compile_time_args=[],
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return ttnn.ProgramDescriptor(kernels=[reader_kernel, compute_kernel], semaphores=[], cbs=cbs)
