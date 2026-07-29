import ttnn


def create_program_descriptor(input_a: ttnn.Tensor, input_b: ttnn.Tensor, input_c: ttnn.Tensor, output: ttnn.Tensor):
    tile_size = ttnn.TILE_SIZE * ttnn.TILE_SIZE
    num_tiles = int(input_a.shape[-1] * input_a.shape[-2] / tile_size)

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=input_a.dtype, page_size=input_a.buffer_page_size())
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=1, data_format=input_b.dtype, page_size=input_a.buffer_page_size())
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=2, data_format=input_c.dtype, page_size=input_a.buffer_page_size())
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=8, data_format=input_a.dtype, page_size=input_a.buffer_page_size())
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=16, data_format=output.dtype, page_size=input_a.buffer_page_size())
            ],
        ),
    ]

    # --- Reader ---
    reader_ct_args = []
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_a).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_b).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_c).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_a.buffer_address(),
        input_b.buffer_address(),
        input_c.buffer_address(),
        num_tiles,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/ttnn/operations/muladd_test/device/reader.cpp",
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer ---
    writer_ct_args = []
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [output.buffer_address(), num_tiles]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/ttnn/operations/muladd_test/device/writer.cpp",
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute ---

    compute_rt_args = ttnn.RuntimeArgs()
    compute_rt_args[core.x][core.y] = [num_tiles]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/ttnn/operations/muladd_test/device/compute.cpp",
        core_ranges=core_grid,
        compile_time_args=[],
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel, compute_kernel], semaphores=[], cbs=cbs)
