.. _DRAM Loopback Example:

DRAM Loopback
=============

This is the simplest example of using the TT-Metal API. A data movement core in the Tensix copies data from DRAM into its L1(SRAM) buffer and back out to DRAM. Hence "loopback".


We'll go through this code section by section. The full source code for this example is available under the ``tt_metal/programming_examples/loopback`` directory.

Building the example can be done by adding a ``--build-programming-examples`` flag to the build script or adding the ``-DBUILD_PROGRAMMING_EXAMPLES=ON`` flag to the cmake command and results in the ``loopback`` executable in the ``build/programming_examples`` directory. For example:

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples
    # To run the example
    ./build/programming_examples/loopback

Device initialization
---------------------

.. code-block:: cpp

   constexpr int device_id = 0;
   auto device = CreateDevice(device_id);

First we open a device. This is our gateway to all operations on the accelerator. The device ID is simply an index into the list of available devices (from 0 to N-1). Thus device 0 is the first device and always available if one is installed.

Program setup
-------------

.. code-block:: cpp

   CommandQueue& cq = device->command_queue();
   Program program = CreateProgram();

Operations in Metalium are almost always capable to be run asynchronously and the ordering of operations is managed by a command queue. The command queue, like the name suggests, is a FIFO queue of commands that are executed in order. And commands are operations that are run on the device, including but not limited to upload/download of data and program execution.

Next, we create a ``Program`` object that we will fill in later. A program is a set of kernels that are executed on the device. If you are familiar with OpenCL, the program in Metalium is different from OpenCL in that All Tensix cores need to to run the exact same kernel at the same time. However in this example, we are only going to use one of all the cores in the device.

Create buffers in DRAM and L1 (SRAM)
------------------------------------

Next, we need to declare buffers that will hold the actual data and an intermediate buffer on chip,

There's in total 3 buffers to be created:
* An L1 (SRAM) buffer within the core itself that will act as temporary storage
* A DRAM buffer that will house input data
* A DRAM buffer that will be written to with output data

Note that almost all operations on the Tensix are aligned with tiles. And a tile is a 32x32 grid of values. The data type used in this example is bfloat16 as it is what the math engine uses internally (though we won't touch the math engine in this example). Making each tile 32 x 32 x 2 bytes = 2048 bytes. And we wish to allocate 50 tiles in each buffer.

There are two types of buffers in the Tensix: L1 and DRAM. L1 is a misnomer as it can be mistaken as similar to L1 cache in a CPU. In fact, the L1 is a SRAM scratchpad on the Tensix. Each generation of Tenstorrent processors has a different amount of L1 memory per Tensix. Grayskull had 1MB and Wormhole/Blackhole has 1.5MB.

Note the ``page_size`` argument in the buffer config and the ``Interleaved`` in the buffer type. Both L1 and DRAM are split into banks. Each bank is a physical memory unit that can be accessed independently. However, managing banks separately is tricky and not scalable. Interleaved buffers simply round-robin the data across all banks every ``page_size`` bytes. This allows the programmer to treat the buffer as a single unit, while taking advantage of the parallelism of the banks for higher bandwidth. Setting page size equal to the buffer size means that the entire buffer will live on a single bank. This is not recommended for performance and in most cases, page size is set to the size of a tile. However, this configuration allows easy illustration of NoC operations. However, these are implementation details and the programmer should not be overly concerned with them.

.. code-block:: cpp

  constexpr uint32_t tile_size = TILE_WIDTH * TILE_HEIGHT;
  constexpr uint32_t single_tile_size = sizeof(bfloat16) * tile_size;
  constexpr uint32_t num_tiles = 50;
  constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;
  tt_metal::InterleavedBufferConfig l1_config{
                                        .device=device,
                                        .size = dram_buffer_size,
                                        .page_size = dram_buffer_size,
                                        .buffer_type = tt_metal::BufferType::L1
                                        };

  Buffer l1_buffer = CreateBuffer(l1_config);

The only difference between the L1 and DRAM buffer is the ``BufferType``. The L1 buffer is created with a ``BufferType::L1`` and the DRAM buffer is created with a ``BufferType::DRAM``.

.. code-block:: cpp

  tt_metal::InterleavedBufferConfig dram_config{
                                        .device=device,
                                        .size = dram_buffer_size,
                                        .page_size = dram_buffer_size,
                                        .buffer_type = tt_metal::BufferType::DRAM
                                        };

  Buffer input_dram_buffer = CreateBuffer(dram_config);
  const uint32_t input_dram_buffer_addr = input_dram_buffer.address();

  Buffer output_dram_buffer = CreateBuffer(dram_config);
  const uint32_t output_dram_buffer_addr = output_dram_buffer.address();

  const uint32_t input_bank_id = 0;
  const uint32_t output_bank_id = 0;

Sending real data into DRAM
---------------------------

.. code-block:: cpp

  std::vector<bfloat16> input_vec(num_tiles * tile_size);
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
  for (auto& val : input_vec) {
      val = bfloat16(distribution(rng));
  }
  EnqueueWriteBuffer(cq, input_dram_buffer, input_vec, false);

Send in a randomly-generated BFP16 (Brain 16bit floating point) vector that will act as our input data tensor.

Note the final ``false`` argument. This indicates to tt-Metalium that the upload is non-blocking. The function may return as soon as possible while data transfer is still in progress. This is useful for performance, but the program is responsible for ensuring that the the source buffer is not freed before the transfer is complete. In this case, there are future blocking calls/calls to ``Finish`` that will ensure commands are completed before the program exits, which is also when the source buffer is freed.

Creating a data movement kernel
-------------------------------

Create a kernel that will copy data from DRAM to L1 and back. Since we are only using one Tensix core, ``{0, 0}`` is the only core (core on the most top left) we use. And as we are moving data from DRAM to L1, This is a data movement kernel using the movement processor 0, and the default NoC interface.

.. code-block:: cpp

    constexpr CoreCoord core = {0, 0};

    KernelHandle dram_copy_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );


The kernel itself is simple. It takes the address and bank indices we just created. Copies data from the input DRAM buffer to the L1 buffer and then back out to the output DRAM buffer. You might notice that the kernel is using ``uint32_t`` instead of pointers for addresses. This is intended design as the DRAM is not directly addressable by the kernels. Instead, access requests are sent to the NoC (Network on Chip) and be brought to the L1 before the kernel can access it in a meaningful way. However, letting the RISC-V core directly access the L1 is not the most efficient way to move data around. Thus the L1 address is also an integer.

.. code-block:: cpp

    // tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp
    void kernel_main() {
        std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
        std::uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(1);
        std::uint32_t dram_buffer_src_bank  = get_arg_val<uint32_t>(2);
        std::uint32_t dram_buffer_dst_addr  = get_arg_val<uint32_t>(3);
        std::uint32_t dram_buffer_dst_bank  = get_arg_val<uint32_t>(4);
        std::uint32_t dram_buffer_size      = get_arg_val<uint32_t>(5);

        std::uint64_t dram_buffer_src_noc_addr =
            get_noc_addr_from_bank_id</*dram=*/true>(dram_buffer_src_bank, dram_buffer_src_addr);
        // Read data into L1 buffer
        noc_async_read(dram_buffer_src_noc_addr, l1_buffer_addr, dram_buffer_size);
        noc_async_read_barrier(); // wait for transfer to complete

        std::uint64_t dram_buffer_dst_noc_addr =
            get_noc_addr_from_bank_id</*dram=*/true>(dram_buffer_dst_bank, dram_buffer_dst_addr);
        // write data from L1 back into DRAM
        noc_async_write(l1_buffer_addr, dram_buffer_dst_noc_addr, dram_buffer_size);
        noc_async_write_barrier(); // wait for transfer to complete
    }

.. note::
    Accessing DRAM using an address and bank pair is complicated and not scalable. Most kernels uses interleaved buffers and ``InterleavedAddrGenFast`` instead (introduced in the next example). This acts much more like a pointer and is much easier to use. The interleaved buffer is simply a buffer with page size equal to the tile size.


Setting runtime arguments for the data movement kernel
------------------------------------------------------

.. code-block:: cpp

  const std::vector<uint32_t> runtime_args = {
      l1_buffer.address(),
      input_dram_buffer.address(),
      input_bank_id,
      output_dram_buffer.address(),
      output_bank_id,
      l1_buffer.size()
  };

  SetRuntimeArgs(
      program,
      dram_copy_kernel_id,
      core,
      runtime_args
  );

We now set runtime arguments for our data movement kernel. The kernel can then access these arguments at runtime. For this specific kernel, we need to pass in the following arguments:

* Where the L1 buffer starts (memory address)
* Where the input DRAM buffer starts (memory address)
* The channel index of the input DRAM buffer
* Where the output DRAM buffer starts (memory address)
* The channel index of the output DRAM buffer
* The size of the copy
  * Which happens to be the same as the size of the L1 buffer

Running the program
-------------------

.. code-block:: cpp

    EnqueueProgram(cq, program, false);
    Finish(cq);
    // Equivalently, we could have done:
    // EnqueueProgram(cq, program, true);


Finally, we launch our program. The ``Finish`` call waits for the the host program only continues execution after everything in the command queue has been completed. The final argument in ``EnqueueProgram`` indicates that the program is non-blocking. Setting it to ``true`` would cause the program to block until the program is finished. Efficiently, this is the same as calling ``Finish`` after the program is enqueued.

Download the result and verify output
-------------------------------------

Then we can finally read back the data from the output buffer and assert that
it matches what we sent. Again the final ``true`` argument causes the data transfer to be blocking. Thus we know that the data is fully avaliable when the function returns.

.. code-block:: cpp

  std::vector<bfloat16> result_vec;
  EnqueueReadBuffer(cq,output_dram_buffer, result_vec, true);

  for (int i = 0; i < input_vec.size(); i++) {
    if (input_vec[i] != result_vec[i]) {
        pass = false;
        break;
    }
  }

Validation and teardown
-----------------------

.. code-block:: cpp

   pass &= CloseDevice(device);

We now use ``CloseDevice`` to teardown our device. This releases resources associated with the device.

Now we can start adding some compute to our program. Please refer to the :ref:`Eltwise binary example<Eltwise binary example>`.
