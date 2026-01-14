.. _Custom_SFPI_Smoothstep:

Smoothstep using SFPI
=====================

This document details the implementation of a custom SFPI kernel for the ``smoothstep`` function. It is intended for developers familiar with parallel programming concepts who are new to the Tenstorrent platform.

This example builds upon the :ref:`Vector addition using SFPI<Custom_SFPI_Add>` example, and introduces the following advanced SFPI concepts:

*   **Parameter Passing:** Passing scalar arguments to an SFPI kernel.
*   **Vector Predicates:** Performing element-wise conditional operations.

The ``smoothstep`` function is a non-linear interpolation function commonly used in graphics (see `GLSL smoothstep documentation <https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/smoothstep.xhtml>`_ for reference). It is defined as:

.. math::

    \operatorname{smoothstep}(e_0, e_1, x) =
    \begin{cases}
    0, & x \leq e_0, \\
    1, & x \geq e_1, \\
    \left( \dfrac{x - e_0}{e_1 - e_0} \right)^2 \bigl(3 - 2 \tfrac{x - e_0}{e_1 - e_0}\bigr),
    & e_0 < x < e_1 .
    \end{cases}

Although ``smoothstep`` is conceptually simple, its implementation is complex enough to demonstrate several advanced features of SFPI, such as parameter passing and vector predicates.

The full source code is available in the ``tt_metal/programming_examples/custom_sfpi_smoothstep`` directory.

Building and Running
--------------------

Build the example using the ``--build-programming-examples`` flag:

.. code-block:: bash

    export TT_METAL_HOME=</path/to/tt-metal>
    ./build_metal.sh --build-programming-examples

Run the example:

.. code-block:: bash

    ./build/programming_examples/metal_example_custom_sfpi_smoothstep

.. warning::

    Tenstorrent does not guarantee backward compatibility for user-implemented SFPI functions.

Program setup
-------------

The host-side setup for this custom SFPI example follows the standard pattern: mesh device initialization, DRAM buffer creation, circular buffer allocation for kernel communication, and kernel creation. Unlike binary operations that require two input buffers, smoothstep is a unary operation requiring only a single input buffer (``src0_dram_buffer``) plus the output buffer (``dst_dram_buffer``). Correspondingly, we need only one input circular buffer (``cb_in0``) and one output buffer (``cb_out0``).

.. code-block:: cpp

    // Create a 1x1 mesh on device 0 (same API scales to multi-device meshes)
    constexpr int device_id = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    // Submit work via the mesh command queue: uploads/downloads and program execution
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    Program program = CreateProgram();

    // A MeshWorkload is a collection of programs that will be executed on the mesh
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // Configure mesh buffers with single-tile page size
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig dram_buffer_config{
        .size = dram_buffer_size};

    // DRAM buffers: single input + one output for unary operation
    auto src0_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(dram_buffer_config, dram_config, mesh_device.get());

    // Circular buffers for kernel communication
    CreateCircularBuffer(program, core, /* cb_in0 config */);
    CreateCircularBuffer(program, core, /* cb_out0 config */);

    // Kernels: reader, writer, and custom SFPU compute
    auto reader = CreateKernel(program, "..../read_tiles.cpp", core, DataMovementConfig{...});
    auto writer = CreateKernel(program, "..../write_tile.cpp", core, DataMovementConfig{...});
    auto compute = CreateKernel(program, "..../tiles_smoothstep.cpp", core, ComputeConfig{});

The Kernels
-----------

Data Movement Kernels
~~~~~~~~~~~~~~~~~~~~~

The reader kernel reads tiles from a single source DRAM buffer and pushes them into the input circular buffer. Since smoothstep is a unary operation, we only need to read from one source buffer.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_smoothstep/kernels/dataflow/read_tiles.cpp
    void kernel_main() {
        uint32_t in0_addr = get_arg_val<uint32_t>(0);
        uint32_t n_tiles = get_arg_val<uint32_t>(1);
        ...
        for (uint32_t i = 0; i < n_tiles; i++) {
            cb_reserve_back(cb_in0, 1);
            uint32_t cb_in0_addr = get_write_ptr(cb_in0);
            noc_async_read_tile(i, in0, cb_in0_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
        }
    }

The writer kernel is straightforward: it reads result tiles from the output circular buffer and writes them to the destination DRAM buffer.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_smoothstep/kernels/dataflow/write_tile.cpp
    void kernel_main() {
        uint32_t c_addr = get_arg_val<uint32_t>(0);
        uint32_t n_tiles = get_arg_val<uint32_t>(1);
        ...
        for (uint32_t i = 0; i < n_tiles; i++) {
            cb_wait_front(cb_out0, 1);
            uint32_t cb_out0_addr = get_read_ptr(cb_out0);
            noc_async_write_tile(i, out0, cb_out0_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_out0, 1);
        }
    }

SFPI Compute Kernel
~~~~~~~~~~~~~~~~~~~

The compute kernel is where the custom SFPI logic resides. It waits for tiles from the input CB, performs the smoothstep operation using the SFPI, and pushes the result to the output CB.

The overall flow follows the standard pattern for unary compute kernels:

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_smoothstep/kernels/compute/tiles_smoothstep.cpp
    namespace NAMESPACE {
    void MAIN {
        uint32_t n_tiles = get_arg_val<uint32_t>(0);

        constexpr auto cb_in0 = tt::CBIndex::c_0;
        constexpr auto cb_out0 = tt::CBIndex::c_16;

        constexpr float edge0 = 0.0f;
        constexpr float edge1 = 1.0f;
        // pre-calculate inverse as it is used multiple times and slow (the Baby RISC-V cores)
        // uses software floating-point. Constexpr making this evaulation compile-time
        constexpr float inv_delta = 1.0f / (edge1 - edge0);

        init_sfpu(cb_in0, cb_out0);

        for (uint32_t i = 0; i < n_tiles; i++) {
            cb_wait_front(cb_in0, 1);
            tile_regs_acquire();
            copy_tile(cb_in0, 0, 0); // input x
            my_smoothstep_tiles(0, edge0, edge1, inv_delta);  // <-- Custom SFPI smoothstep
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out0, 1);
            pack_tile(0, cb_out0);
            cb_push_back(cb_out0, 1);
            cb_pop_front(cb_in0, 1);
            tile_regs_release();
        }
    }

Custom SFPI Implementation of Smoothstep
----------------------------------------

The ``my_smoothstep_tiles`` function uses the layered abstraction pattern shown in previous examples. This section focuses on the new concepts introduced in this kernel.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_smoothstep/kernels/compute/tiles_smoothstep.cpp

    #ifdef TRISC_MATH

    // Low-level function operating on a tile face
    void my_smoothstep_tile_face(float edge0, float edge1, float inv_delta) {
        constexpr size_t vectors_per_face = 8;
        for (size_t i = 0; i < vectors_per_face; i++) {
            vFloat x = dst_reg[i];
            vFloat t = (x - edge0) * inv_delta;
            v_if(t < sfpi::vConst0) { t = sfpi::vConst0; }
            v_elseif(t > sfpi::vConst1) { t = sfpi::vConst1; }
            v_endif;
            vFloat result = t * t * (3.0f - 2.0f * t);
            dst_reg[i] = result;
        }
    }
    #endif // TRISC_MATH

    // High-level API function
    // Accepts `edge0`, `edge1` and `inv_delta` as parameters
    inline void my_smoothstep_tile(uint32_t idx_dst0, float edge0, float edge1, float inv_delta) {
        MATH(_llk_math_eltwise_unary_sfpu_params_<false>(
            smoothstep_tile_face,
            idx_dst0,
            VectorMode::RC, // Apply on all 4 faces of the tile
            edge0,
            edge1,
            inv_delta));
    }

Parameter Passing
~~~~~~~~~~~~~~~~~

The `smoothstep` function needs two scalar parameters: ``edge0`` and ``edge1``. These are passed to the SFPI kernel using the ``_llk_math_eltwise_unary_sfpu_params_`` helper function.

.. code-block:: cpp

    // Passes edge0 and edge1 as arguments to the SFPI kernel
    my_smoothstep_tile(uint32_t idx_dst0, float edge0, float edge1, float inv_delta);
    // ↓
    // Use the parameters for all elements in the tile face
    my_smoothstep_tile_face(float edge0, float edge1, float inv_delta);

The helper function is a template that takes the low-level face function as its first argument, followed by the destination register index, vector mode, and any scalar parameters required by the face function. This approach makes it easy to pass constants or runtime values into the SFPI kernel.

Vector Predicates
~~~~~~~~~~~~~~~~~

The clamping of the intermediate value ``t`` to the [0, 1] range is implemented using vector predicates.

.. code-block:: cpp

    v_if(t < sfpi::vConst0) { t = sfpi::vConst0; }
    v_elseif(t > sfpi::vConst1) { t = sfpi::vConst1; }
    v_endif;

The ``v_if`` and ``v_elseif`` instructions perform element-wise conditional assignments on the ``vFloat`` vector ``t``. Each lane of the SIMD vector is evaluated independently. A ``v_endif`` is required to terminate the conditional block.

The SFPI constants ``sfpi::vConst0`` and ``sfpi::vConst1`` are vectors with all 32 lanes set to 0.0f and 1.0f, respectively. These constants are hardware-defined, readily available for SFPI programs, and do not require manual initialization. Using these pre-defined constants is more efficient than using literal values because the SFPU operates on vectors. Literal values would require broadcasting to a vector, which adds instructions and overhead.

This is analogous to conditional execution in other parallel programming models, where a mask is used to control which processing elements are active.

Runtime Arguments and Execution
-------------------------------

Back on the host, we set the runtime arguments for the kernels. Since this is a unary operation, the reader and writer kernels need only a single DRAM buffer address each, and all three kernels need to know the number of tiles to process.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_smoothstep/custom_sfpi_smoothstep.cpp
    SetRuntimeArgs(program, reader, core, {
        src0_dram_buffer->address(),
        n_tiles
    });

    SetRuntimeArgs(program, writer, core, {
        dst_dram_buffer->address(),
        n_tiles
    });

    SetRuntimeArgs(program, compute, core, {
        n_tiles
    });

Finally, we add the program to the mesh workload and enqueue it for execution, then read back the results from the destination DRAM buffer to verify correctness against the expected smoothstep function output.

.. code-block:: cpp

    // tt_metal/programming_examples/custom_sfpi_smoothstep/custom_sfpi_smoothstep.cpp
    // Add the program to the workload for the mesh
    workload.add_program(device_range, std::move(program));

    // Enqueue the workload for execution on the mesh (non-blocking) and wait for completion
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    std::vector<bfloat16> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking*/ true);

    // Validation against golden smoothstep output
    for (size_t i = 0; i < result_vec.size(); ++i) {
        // CPU version of the same smoothstep function for validation
        auto smoothstep = [](float edge0, float edge1, float x) {
            x = (x - edge0) / (edge1 - edge0);
            x = std::clamp(x, 0.0f, 1.0f);
            return x * x * (3 - 2 * x);
        };
        const float expected = smoothstep(0.0f, 1.0f, a_data[i].to_float());
        const float actual = result_vec[i].to_float();
        // Check for match within tolerance...
    }

Conclusion
----------

This example demonstrates the implementation of a custom SFPI kernel with parameter passing and conditional logic. Key takeaways are:

*   **Parameter Passing:** The ``_llk_math_eltwise_*_sfpu_params_`` family of functions is used to pass scalar arguments to a custom SFPI kernel.
*   **Vector Predicates:** The ``v_if``, ``v_elseif``, and ``v_endif`` instructions provide a mechanism for element-wise conditional logic within an SFPI kernel.
*   **Unary Operations:** Unary SFPI kernels can be implemented efficiently by performing the computation in-place in the destination registers.
