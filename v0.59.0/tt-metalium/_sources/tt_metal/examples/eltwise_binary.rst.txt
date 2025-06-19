.. _Eltwise binary example:

Eltwise binary
==============

We now build a program that performs elementwise binary operations on a pair of equal-sized tensors using the FPU (matrix engine). The FPU can perform a variety of complex operations efficiently, including matrix multiplication and reduction. It can also perform common elementwise operations like addition. This example will add two tensors together using the FPU.

Although looking trivial, this example introduces essential concepts that are used in more complicated programs. Namely the compute kernel and the use of circular buffers. Tenstorrent devices are designed with explicit data movement in mind. Leading to a separate compute kernel. And for circular buffers is the primary way to move data between kernels.

We'll go through any new code section by section. This builds on top of
previous examples. Note that we have this exact, full example program is located under
``tt_metal/programming_examples/eltwise_binary``, so you can follow along.

To build and execute, you may use the following commands. Note that we include
the necessary environment variables here, but you may possibly need more
depending on the most up-to-date installation methods.

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples
    # To run the example
    ./build/programming_examples/eltwise_binary

Program setup
-------------

Initializing Metalium is almost the same as before. To recap, we need to

1. Open a device (again we are using device 0)
2. Obtain the command queue to issue down/uploads and program execution
3. Create a program object to hold our kernels
4. Define the core we are going to use (in this case, only one at {0, 0})
5. Some basic constants for tile and total buffer sizes

.. code-block:: cpp

    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    CommandQueue& cq = device->command_queue();

    Program program = CreateProgram();

    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t n_tiles = 64;
    constexpr uint32_t elements_per_tile = TILE_WIDTH * TILE_HEIGHT;
    constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

This time, instead of 2 DRAM buffers to copy data out and back, we will create 3 buffers. 2 as data sources and 1 as the output. Page size is set to one tile. This is the most common setting for buffers in Metalium as the compute engines expect to operate on tiles of data.

.. code-block:: cpp

    InterleavedBufferConfig config{
        .device = device,                       // The device to create the buffer on
        .size = n_tiles * tile_size_bytes,      // The size of the buffer in bytes
        .page_size = tile_size_bytes,           // The page size of the buffer in bytes. In this case, will be
                                                // one tile per page.
        .buffer_type = BufferType::DRAM};       // This is a DRAM buffer.
    auto src0_dram_buffer = CreateBuffer(config);
    auto src1_dram_buffer = CreateBuffer(config);
    auto dst_dram_buffer = CreateBuffer(config);

Data preparation and upload is the same as before. For this example, buffer 0 will be filled with random values, and buffer 1 will be filled with a constant of -1 just for demonstration purposes. The data is then uploaded to the device asynchronously like before.

.. code-block:: cpp

    constexpr float val_to_add = -1.0f;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::vector<bfloat16> src0_data(elements_per_tile * n_tiles);
    std::vector<bfloat16> src1_data(elements_per_tile * n_tiles, bfloat16(val_to_add));
    for(auto& val : a_data) {
        val = bfloat16(distribution(rng));
    }
    // Upload the data from host to the device.
    EnqueueWriteBuffer(cq, src0_dram_buffer, a_data, false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b_data, false);

Circular buffers
----------------

Here we introduce a new concept: circular buffers. They are communication channels between the different kernel on a Tensix. Conceptually they act as pipes between different kernels. There are in total 16 circular buffers supported on a Tensix. To utilize them, the host program must allocate the circular buffers and utilize the appropriate circular buffer index in the kernel.



.. code-block:: cpp

    constexpr uint32_t tiles_per_cb = 2;
    tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
    CircularBufferConfig c0_cfg = CircularBufferConfig(
        /*total_size=*/tiles_per_cb * tile_size_bytes,
        /*data_format_spec=*/{{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, tile_size_bytes);
    CBHandle cb_src0 = CreateCircularBuffer(program, core, c0_cfg);

    tt::CBIndex src1_cb_index = tt::CBIndex::c_1;
    CBHandle cb_src1 = CreateCircularBuffer(program, core, CircularBufferConfig(
        /*total_size=*/tiles_per_cb * tile_size_bytes,
        /*data_format_spec=*/{{src1_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src1_cb_index, tile_size_bytes));
    tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
    CBHandle cb_dst = CreateCircularBuffer(program, core, CircularBufferConfig(
        /*total_size=*/tiles_per_cb * tile_size_bytes,
        /*data_format_spec=*/{{dst_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(dst_cb_index, tile_size_bytes));

The API to create a circular buffer is more complicated than for a buffer, providing finer grained control for advanced use cases. At the core, there are a few critical parameters:

* The index of the circular buffer
* The total size of the circular buffer in bytes (and thus the number of pages within)
* The data format of the circular buffer (bfloat16, block float4, etc..)
* How large each pages is
  * For most cases, this should be the same as the size of a tile of the underlying data format

For instance, to create a circular buffer of 2 tiles of bfloat16 data, we need to set the total size to ``2 * tile_size_bytes``. The page size set to ``tile_size_bytes`` and the data format is set to bfloat16.

Data movement and compute kernels
---------------------------------

In the previous example (DRAM loopback), we used a single kernel to perform the entire operation; to read data from DRAM and write it back out. The Tensix core in fact contains 5 RISC-V cores. 2 of them are the data movement cores, which connects to the NoC and can issue commands to access other on chip resources (including DRAM). The other 3 are compute cores, which operates cooperatively and runs a single compute kernel. They have access to the matrix and vector engines, which performs the majority of the compute work on a Tensix.

.. note::
    Unlike traditional multi core processors. Where a problem is broken down into subtasks as assigned to the cores while each core runs the same code (SPMD, single program multiple data). The compute cores on a Tensix are designed to run different code. The compute kernel is compiled 3 times. Once for each of the 3 compute cores and generating 3 different binaries.  They work collaboratively to perform a single task. The 3 compute cores are the Unpack, Math and Pack cores. They are responsible for moving data from L1 into the matrix or vector engines, issue commands for computation and moving the results back out to L1. Which can be done at the same time for high throughput.

.. code-block:: cpp

    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_binary/kernels/dataflow/read_tiles.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    auto writer = CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_binary/kernels/dataflow/write_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/eltwise_binary/kernels/compute/tiles_add.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

Notice the ``ComputeConfig`` object. This indicates to the framework that a compute kernel should be created. There's a plethora of settings that can be set here. The most important one is the ``math_fidelity`` setting. This controls how accurate certain floating point operations are on the FPU specifically. Other operations (like the ones in the vector engine) are _not_ affected by this setting.

Now let's look at the kernels. First the reader kernel. This kernel reads both input buffers from DRAM and pushes them into the circular buffers (will discuss in the following section) for the compute kernel to consume. For now, consider the circular buffers as a pipe. Data can be pushed into the pipe and read out of it. And both ends must ensure there is space to do so.

To do so, the reader creates 2 interleaved address generators. Unlike on most processors, the Tensix doesn't just see raw bytes. The address generator contains metadata about the data it is associated to enable the data be read out of the DRAM in the correct format. Then in a loop, the program waits for space to be available in the circular buffers. Once there is, it reads a tile from DRAM and pushes it into the circular buffer. The ``noc_async_read_barrier()`` call ensures waits for the read to finish before committing the data to the circular buffer. This is important as the read is asynchronous and data is not guaranteed to be there when the program continues.

.. code-block:: cpp

    // tt_metal/programming_examples/eltwise_binary/kernels/dataflow/read_tiles.cpp
    void kernel_main() {
        uint32_t in0_addr = get_arg_val<uint32_t>(0);
        uint32_t in1_addr = get_arg_val<uint32_t>(1);
        uint32_t n_tiles = get_arg_val<uint32_t>(2);

        // The circular buffers to read the tiles into (same as the ones in the host program)
        constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
        constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

        const uint32_t tile_size_bytes = get_tile_size(cb_in0);

        const InterleavedAddrGenFast<true> in0 = {
            .bank_base_address = in0_addr,
            .page_size = tile_size_bytes,
            .data_format = DataFormat::Float16_b,
        };
        const InterleavedAddrGenFast<true> in1 = {
            .bank_base_address = in1_addr,
            .page_size = tile_size_bytes,
            .data_format = DataFormat::Float16_b,
        };

        for (uint32_t i = 0; i < n_tiles; i++) {
            cb_reserve_back(cb_in0, 1);
            cb_reserve_back(cb_in1, 1);

            noc_async_read_tile(i, in0, get_write_ptr(cb_in0));
            noc_async_read_tile(i, in1, get_write_ptr(cb_in1));

            // Wait until tile reads are done
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
            cb_push_back(cb_in1, 1);
        }
    }

The compute kernel is a bit more complicated. It is responsible for performing the actual computation. After the initialization calls to configure the matrix engine to perform addition on the input tiles, it enters a loop then waits for the destination registers to be available. Please refer to the Programming Model section for more information on this. In short, the destination registers are a set of 16 registers that are used send data in and out of the computation engines. 8 of them can be used at a time. Once the destination registers are available, it waits for the reader kernel to make data available in the circular buffers. Once there is data, it adds the first tile from each buffer together and writes the result to destination register 0. Wait for space on the output circular buffer, and pushes the computed tile into it. Finally, it marks the input as consumed, output as produced and computation done.

.. code-block:: cpp

    // tt_metal/programming_examples/eltwise_binary/kernels/compute/tiles_add.cpp
    namespace NAMESPACE {
    void MAIN {
        uint32_t n_tiles = get_arg_val<uint32_t>(0);

        constexpr auto cb_in0 = tt::CBIndex::c_0;
        constexpr auto cb_in1 = tt::CBIndex::c_1;
        constexpr auto cb_out0 = tt::CBIndex::c_16;

        constexpr uint32_t dst_reg_id = 0;

        binary_op_init_common(cb_in0, cb_in1, cb_out0);
        add_tiles_init(cb_in0, cb_in1);

        // Loop over all the tiles and perform the computation
        for (uint32_t i = 0; i < n_tiles; i++) {
            acquire_dst();
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            add_tiles(cb_in0, cb_in1, /*offset_0*/0, /*offset_1*/0, dst_reg_id);

            cb_reserve_back(cb_out0, 1);
            pack_tile(dst_reg, cb_out0); // copy result to out0
            cb_push_back(cb_out0, 1);
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
            release_dst();
        }
    }
    }

The writer kernel looks similar to the reader kernel. Instead of reading, it writes data back into DRAM and uses the appropriate API to do so.


.. code-block:: cpp

    // tt_metal/programming_examples/eltwise_binary/kernels/dataflow/write_tile.cpp
    void kernel_main() {
        uint32_t out_addr = get_arg_val<uint32_t>(0);
        uint32_t n_tiles = get_arg_val<uint32_t>(1);

        // same as the one in the host program
        constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

        const uint32_t tile_size_bytes = get_tile_size(cb_out0);

        const InterleavedAddrGenFast<true> out = {
            .bank_base_address = out_addr,
            .page_size = tile_size_bytes,
            .data_format = DataFormat::Float16_b,
        };

        for (uint32_t i = 0; i < n_tiles; i++) {
            cb_wait_front(cb_out0, 1);
            noc_async_write_tile(i, out, get_read_ptr(cb_out0));
            noc_async_write_barrier();
            cb_pop_front(cb_out0, 1);
        }


Then the host program sets the kernel arguments and launches the program. There is no difference in setting the arguments for the compute kernel.

.. note::
    Unlike OpenCL/CUDA. Each kernel (reader, compute and writer) can have it's own set of arguments. Furthermore, on a multi cored program (i.e. using more then 1 Tensix core), kernels within each core can have different arguments. This enables Metalium to exploit the grid like nature of the Tenstorrent processors to achieve high performance.

.. code-block:: cpp

    SetRuntimeArgs(program, reader, core, {src0_dram_buffer->address(), src1_dram_buffer->address(), n_tiles});
    SetRuntimeArgs(program, writer, core, {dst_dram_buffer->address(), n_tiles});
    SetRuntimeArgs(program, compute, core, {n_tiles});

    EnqueueProgram(cq, program, false);
    Finish(cq);

Download the result and verify output
-------------------------------------

Finally, we download the result and verify the output. Again we read the data into a vector and wait for the transfer to finish. The result is then compared to the expected output. Note that we use a loose tolerance for the comparison because of the nature of bfloat16.

.. code-block:: cpp

    std::vector<bfloat16> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, /*blocking=*/true);

    constexpr float eps = 1e-2f; // loose tolerance because of the nature of bfloat16
    TT_FATAL(result_vec.size() == a_data.size(), "Result vector size mismatch");
    for (size_t i = 0; i < result_vec.size(); ++i) {
        const float expected = a_data[i].to_float() + val_to_add;
        const float actual = result_vec[i].to_float();

        if (std::abs(expected - actual) > eps) {
            pass = false;
            tt::log_error(tt::LogTest, "Result mismatch at index {}: expected {}, got {}", i, expected, actual);
        }
    }

Validation and teardown
-----------------------

.. code-block:: cpp

   pass &= CloseDevice(device);

We now use ``CloseDevice`` to teardown our device. This releases resources associated with the device.

Next we will explore the use of the vector engine (SFPU) to perform the same operation.
:ref:`Eltwise sfpu example<Eltwise sfpu example>`.
