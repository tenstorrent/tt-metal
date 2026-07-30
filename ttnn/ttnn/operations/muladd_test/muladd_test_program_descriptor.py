import ttnn
import numpy as np


def create_program_descriptor(input_a: ttnn.Tensor, input_b: ttnn.Tensor, input_c: ttnn.Tensor, output: ttnn.Tensor):
    tile_size = ttnn.TILE_SIZE * ttnn.TILE_SIZE
    num_tiles = int(input_a.shape[-1] * input_a.shape[-2] / tile_size)

    device = input_a.device()
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    print(f"grid size {(grid.x, grid.y)}")
    if num_tiles % (grid.x * grid.y) != 0:
        exit(1)
    num_tiles_per_core = num_tiles // (grid.x * grid.y)

    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * input_a.buffer_page_size(),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=input_a.dtype, page_size=input_a.buffer_page_size())
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * input_b.buffer_page_size(),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=1, data_format=input_b.dtype, page_size=input_b.buffer_page_size())
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * input_c.buffer_page_size(),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=2, data_format=input_c.dtype, page_size=input_c.buffer_page_size())
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * input_a.buffer_page_size(),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=8, data_format=input_a.dtype, page_size=input_a.buffer_page_size())
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * output.buffer_page_size(),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=16, data_format=output.dtype, page_size=output.buffer_page_size())
            ],
        ),
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
            (x * grid.y + y) * num_tiles_per_core,
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
            (x * grid.y + y) * num_tiles_per_core,
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
            (x * grid.y + y) * num_tiles_per_core,
            num_tiles_per_core,
        ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/ttnn/operations/muladd_test/device/compute.cpp",
        core_ranges=cores,
        compile_time_args=[],
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel, compute_kernel], semaphores=[], cbs=cbs)
