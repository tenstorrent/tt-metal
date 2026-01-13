// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/core_coord.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;
using namespace ttnn;

// Define prefix path for kernel files to be an empty string if not set in the makefile.
// This prefix enables overriding the default kernel path with a custom path, so that the
// example works in both development environment and when installed.
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// clang-format off
/**
 * Reference implementation of element-wise addition.
 * This is intended to be used for testing the Metalium implementation.
 *
 * | Argument | Description                         |
 * |----------|-------------------------------------|
 * | a        | Input vector A                      |
 * | b        | Input vector B                      |
 * | output   | Output vector (will be overwritten) |
 */
// clang-format on
void reference_add(const std::vector<bfloat16>& a, const std::vector<bfloat16>& b, std::vector<bfloat16>& output) {
    TT_FATAL(a.size() == b.size(), "Input vectors must have the same size");
    TT_FATAL(output.size() == a.size(), "Output vector must have the same size as input vectors");
    for (size_t i = 0; i < a.size(); i++) {
        output[i] = static_cast<bfloat16>(static_cast<float>(a[i]) + static_cast<float>(b[i]));
    }
}

// clang-format off
/**
 * Structure to hold program-related state including device, program, workload, and execution context.
 *
 * | Member        | Description                                               |
 * |---------------|-----------------------------------------------------------|
 * | mesh_device   | Shared pointer to the mesh device                         |
 * | program       | Program object containing kernels, circular buffers, etc. |
 * | core          | Core coordinate where the program executes                |
 * | workload      | Workload object that bundles programs for execution       |
 * | device_range  | Range of devices where the program should execute         |
 * | cq            | Command queue for ordering operations on the mesh         |
 */
// clang-format on
struct ProgramState {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    Program program;
    tt::tt_metal::CoreCoord core;
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range;
    distributed::MeshCommandQueue& cq;

    ProgramState(
        std::shared_ptr<distributed::MeshDevice> mesh_device,
        Program program,
        tt::tt_metal::CoreCoord core,
        distributed::MeshWorkload workload,
        distributed::MeshCoordinateRange device_range,
        distributed::MeshCommandQueue& cq) :
        mesh_device(std::move(mesh_device)),
        program(std::move(program)),
        core(core),
        workload(std::move(workload)),
        device_range(std::move(device_range)),
        cq(cq) {}
};

// clang-format off
/**
 * Initialize program state for single-core execution.
 * Creates a unit mesh device, sets up command queue, workload, device range, and program.
 * This program uses only a single Tensix core at [0, 0].
 *
 * Return value: ProgramState
 */
// clang-format on
ProgramState init_program() {
    // Open device
    constexpr int device_id = 0;
    // In TT-Metal, all operations use a mesh abstraction - even a single device is represented as a 1x1 mesh.
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    // Ordering of operations in the mesh is managed by a command queue.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // MeshWorkload represents a collection of programs to be executed across the mesh.
    distributed::MeshWorkload workload;
    // Each program in the workload is associated with a range of devices where it should run.
    // In our case, we have a single program running on our entire (unit) mesh.
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    // This program uses only a single Tensix core at [0, 0].
    tt::tt_metal::CoreCoord core({0, 0});
    // Create a program object. A program is a collection of kernels that are executed on the device.
    // Kernels will be specified later.
    Program program = CreateProgram();

    return ProgramState(
        std::move(mesh_device), std::move(program), core, std::move(workload), std::move(device_range), cq);
}

// clang-format off
/**
 * Helper function to create a circular buffer with the specified number of tiles and CB index.
 * Page size is set to the size of a single tile.
 *
 * | Argument  | Description                                               |
 * |-----------|-----------------------------------------------------------|
 * | program   | The program to which the circular buffer will be added    |
 * | core      | Core coordinate where the circular buffer will be created |
 * | num_tiles | Number of tiles to allocate in the circular buffer        |
 * | cb_index  | Circular buffer index (c_0 to c_31)                       |
 */
// clang-format on
void create_cb(Program& program, const tt::tt_metal::CoreCoord& core, uint32_t num_tiles, tt::CBIndex cb_index) {
    constexpr uint32_t single_tile_bytes = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    // Technically, circular buffers operate on pages, not tiles. However, it is most common to have one tile per page.
    CircularBufferConfig cb_config = CircularBufferConfig(num_tiles * single_tile_bytes, {{cb_index, cb_data_format}})
                                         .set_page_size(cb_index, single_tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_config);
}

// clang-format off
/**
 * Element-wise addition using the Tensix device.
 * Performs element-wise addition of two input vectors and stores the result in the output vector.
 * All vectors must have the same size, which must be divisible by TILE_HEIGHT * TILE_WIDTH.
 *
 * | Argument  | Description                                                         |
 * |-----------|---------------------------------------------------------------------|
 * | a         | Input matrix A in row-major format, size MxN                        |
 * | b         | Input matrix B in row-major format, size MxN                        |
 * | output    | Output matrix (will be overwritten)                                 |
 * | M         | Number of rows in matrices A and B                                  |
 * | N         | Number of columns in matrices A and B                               |
 * | prog_state| Program state containing device, program, and execution context     |
 */
// clang-format on
void eltwise_add_tensix(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    const uint32_t M,
    const uint32_t N,
    ProgramState& prog_state) {
    const uint32_t total_elements = M * N;
    TT_FATAL(a.size() == total_elements, "Input vectors must have size M * N");
    TT_FATAL(a.size() == b.size(), "Input vectors must have the same size");
    TT_FATAL(output.size() == a.size(), "Output vector must have the same size as input vectors");

    constexpr uint32_t elements_per_tile = TILE_HEIGHT * TILE_WIDTH;
    TT_FATAL(total_elements % elements_per_tile == 0, "Total elements must be divisible by elements per tile");
    const uint32_t n_tiles = total_elements / elements_per_tile;

    // Create ttnn::Tensor objects for the input and output data.
    // We use TILE layout as that's what the hardware natively operates on.
    // Tensors are allocated in device DRAM (i.e. DRAM that is directly attached to the Tensix processor,
    // which is distinct from the host DRAM).
    TensorLayout tile_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig(BufferType::DRAM));
    TensorSpec t_spec(Shape({M, N}), tile_layout);

    // Create device tensors from input data.
    // This creates the tensors and queues transfer of data to device in one step.
    Tensor src0_tensor = Tensor::from_vector<bfloat16>(a, t_spec, prog_state.mesh_device.get());
    Tensor src1_tensor = Tensor::from_vector<bfloat16>(b, t_spec, prog_state.mesh_device.get());
    // Create output tensor on device (no initialization needed - kernel will write into it).
    Tensor dst_tensor = create_device_tensor(t_spec, prog_state.mesh_device.get());

    // Create circular buffers for the input and output data.
    // Using 2 tiles per circular buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case, but generally
    // there are diminishing returns observed after several tiles.
    constexpr uint32_t tiles_per_cb_input = 2;
    // There are 32 circular buffers (c_0 - c_31) on the device. We can use any of them, as long as they are not already
    // in use. Kernel code is responsible for using the correct circular buffer for the input and output data (e.g.
    // reader kernel reads data into c_0 and c_1, while the compute kernel reads data from these same buffers).
    create_cb(prog_state.program, prog_state.core, tiles_per_cb_input, CBIndex::c_0);
    create_cb(prog_state.program, prog_state.core, tiles_per_cb_input, CBIndex::c_1);

    constexpr uint32_t tiles_per_cb_output = 2;
    // Compute kernel will write output data to c_16, which will be consumed by the writer kernel.
    // c_16 chosen arbitrarily (e.g. to leave c_2-c_15 free for other potential inputs when code is extended in the
    // future).
    create_cb(prog_state.program, prog_state.core, tiles_per_cb_output, tt::CBIndex::c_16);

    // Get MeshBuffer pointers from tensors. Mesh buffers hold info about how tensor data is distributed
    // across physical DRAM banks (at least for our case when data is stored in DRAM).
    // Programmer doesn't need to understand the internals, but needs to pass this info to the kernels.
    auto src0_mesh_buffer = src0_tensor.mesh_buffer();
    auto src1_mesh_buffer = src1_tensor.mesh_buffer();
    auto dst_mesh_buffer = dst_tensor.mesh_buffer();

    // Create the reader, writer and compute kernels. The kernels do the following:
    // * Reader: Reads data from the DRAM buffer and pushes it into the circular buffer.
    // * Compute: Waits for data to be available in the circular buffer, pops it, adds the two inputs together and
    //            pushes the result into the output circular buffer.
    // * Writer: Waits for data to be available in the output circular buffer, pops it and writes it back into DRAM.
    // These kernels work together to form a pipeline. The reader reads data from the DRAM buffer and makes them
    // available in the compute kernel. The compute kernel does math and pushes the result into the writer kernel. The
    // writer kernel writes the result back to DRAM.
    std::vector<uint32_t> reader_compile_time_args;
    // TensorAccessorArgs just extracts data distribution details from MeshBuffer object into
    // the vector of uint32_t so it can be pushed into compile-time arguments.
    TensorAccessorArgs(*src0_mesh_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_mesh_buffer).append_to(reader_compile_time_args);

    // There are two data movement RISCV processors in each Tensix core:
    // DataMovementProcessor::RISCV_0 and DataMovementProcessor::RISCV_1, corresponding to
    // "RISC-V 0" and "RISC-V 4" in the Tensix core diagram in the documentation.
    // We pick one to read data from DRAM into circular buffers and the other to write result
    // from circular buffer to DRAM. Which one is used for reading vs writing doesn't impact
    // functionality or performance.
    KernelHandle reader_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab_eltwise_binary/kernels/dataflow/read_tiles.cpp",
        prog_state.core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_mesh_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab_eltwise_binary/kernels/dataflow/write_tiles.cpp",
        prog_state.core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});

    // Compile time arguments for the compute kernel.
    // Note that these are evaluated at the kernel's compile time, which is JIT compile done at program creation time.
    // Having arguments at compile time generally allows the compiler to optimize the kernel for the specific use case
    // (e.g. apply loop unrolling, constant folding, etc.), resulting in a more efficient kernel.
    // For this simple example it may have little to no impact, but is done to illustrate this possibility.
    std::vector<uint32_t> compute_compile_time_args = {n_tiles};

    tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab_eltwise_binary/kernels/compute/tiles_add.cpp",
        prog_state.core,
        tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    // Set the runtime arguments for the kernels.
    uint32_t src0_addr = src0_mesh_buffer->address();
    uint32_t src1_addr = src1_mesh_buffer->address();
    uint32_t dst_addr = dst_mesh_buffer->address();
    tt_metal::SetRuntimeArgs(prog_state.program, reader_id, prog_state.core, {src0_addr, src1_addr, n_tiles});
    tt_metal::SetRuntimeArgs(prog_state.program, writer_id, prog_state.core, {dst_addr, n_tiles});

    // NOTE: Observe that we never set the runtime arguments for the compute kernel. This is because everything needed
    // has been set at compile time. The compute kernel does not need any runtime arguments to execute, so we can skip
    // this step.

    // Execute the kernels (data is already on device from Tensor::from_vector)
    prog_state.workload.add_program(prog_state.device_range, std::move(prog_state.program));
    // Last argument is set to true to wait for the workload to complete (blocking call).
    tt_metal::distributed::EnqueueMeshWorkload(prog_state.cq, prog_state.workload, true);

    // Read the result back from device using Tensor::to_vector
    output = dst_tensor.to_vector<bfloat16>();
}

///////////////////////////////////////

// clang-format off
/**
 * Main function that demonstrates single-core element-wise addition using ttnn::Tensor API.
 * Creates test data, runs a reference implementation on host CPU, executes addition on Tensix device,
 * and verifies results.
 *
 * Return value: int (0 on success, non-zero on failure)
 */
// clang-format on
int main() {
    bool pass = true;

    try {
        // Define some constants that will be used throughout the program.
        // We will be adding two matrices of shape MxN
        constexpr uint32_t M = 640;  // user-defined
        constexpr uint32_t N = 640;  // user-defined

        // In C++, the matrices are represented as vectors, to emphasize that memory
        // space is one-dimensional in general.
        // Filled with random values in the range [0, 1) for testing.
        // Use a fixed seed for reproducible results. Change this value to get different random sequences.
        constexpr uint32_t rng_seed = 42;
        std::mt19937 rng(rng_seed);
        std::uniform_real_distribution<float> rng_dist(0.f, 1.0f);
        std::vector<bfloat16> src0_vec(M * N);
        for (bfloat16& v : src0_vec) {
            v = static_cast<bfloat16>(rng_dist(rng));
        }

        std::vector<bfloat16> src1_vec(M * N);
        for (bfloat16& v : src1_vec) {
            v = static_cast<bfloat16>(rng_dist(rng));
        }

        // Reference addition running on x86 CPU so we can verify Tensix result.
        std::vector<bfloat16> reference_result(M * N);
        reference_add(src0_vec, src1_vec, reference_result);

        // Invoke the element-wise addition on the Tensix device
        std::vector<bfloat16> result_vec(M * N);

        // Initialize program state (includes device creation)
        ProgramState prog_state = init_program();
        eltwise_add_tensix(src0_vec, src1_vec, result_vec, M, N, prog_state);

        log_info(tt::LogAlways, "Output vector of size {}", result_vec.size());

        // Validate results
        TT_FATAL(result_vec.size() == reference_result.size(), "Result vector size mismatch");
        // Compare results with some tolerance (loose tolerance because of limited precision of bfloat16).
        constexpr float RELTOL = 0.04;
        for (size_t i = 0; i < result_vec.size(); ++i) {
            const float expected = static_cast<float>(reference_result[i]);
            const float actual = static_cast<float>(result_vec[i]);

            float relative_error = (expected == 0.0f) ? std::abs(actual) : std::abs(actual - expected) / expected;
            if (relative_error > RELTOL) {
                log_error(tt::LogAlways, "Mismatch at index {}: {} vs expected {}", i, actual, expected);
                log_error(tt::LogAlways, "Expected relative tolerance: {} actual relative error: {}", RELTOL, relative_error);
                pass = false;
            }
        }

        pass &= prog_state.mesh_device->close();

    } catch (const std::exception& e) {
        log_error(tt::LogAlways, "Test failed with exception!");
        log_error(tt::LogAlways, "{}", e.what());

        throw;
    }

    if (pass) {
        log_info(tt::LogAlways, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
