// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 This example demonstrates how to multicast a full tensor from a sender (coordinator) core to multiple
 receiver cores using double-buffering for improved performance. It covers the setup of semaphores and
 multicore addressing on Tenstorrent hardware. In the default configuration, (0,0) is the sender core,
 while (1,0), (2,0), and (3,0) are the receiver cores. The user can modify these coordinates as desired.

 This version uses ttnn::Tensor for cleaner buffer management, abstracting away DRAM buffer internals.
*/

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

using namespace std;
using namespace tt;
using namespace tt::constants;
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
 * Structure to hold program-related state including device, program, workload, and execution context.
 *
 * | Member        | Description                                               |
 * |---------------|-----------------------------------------------------------|
 * | mesh_device   | Shared pointer to the mesh device                         |
 * | program       | Program object containing kernels, circular buffers, etc. |
 * | workload      | Workload object that bundles programs for execution       |
 * | device_range  | Range of devices where the program should execute         |
 * | cq            | Command queue for ordering operations on the mesh         |
 */
// clang-format on
struct ProgramState {
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
    Program program;
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range;
    tt::tt_metal::distributed::MeshCommandQueue& cq;

    ProgramState(
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device,
        Program program,
        tt::tt_metal::distributed::MeshWorkload workload,
        tt::tt_metal::distributed::MeshCoordinateRange device_range,
        tt::tt_metal::distributed::MeshCommandQueue& cq) :
        mesh_device(std::move(mesh_device)),
        program(std::move(program)),
        workload(std::move(workload)),
        device_range(std::move(device_range)),
        cq(cq) {}
};

// clang-format off
/**
 * Initialize program state for multicast execution.
 * Creates a unit mesh device, sets up command queue, workload, device range, and program.
 *
 * Return value: ProgramState
 */
// clang-format on
ProgramState init_program() {
    // Open device
    constexpr int device_id = 0;
    // In TT-Metal, all operations use a mesh abstraction - even a single device is represented as a 1x1 mesh.
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device =
        tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    // Ordering of operations in the mesh is managed by a command queue.
    tt::tt_metal::distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // MeshWorkload represents a collection of programs to be executed across the mesh.
    tt::tt_metal::distributed::MeshWorkload workload;
    // Each program in the workload is associated with a range of devices where it should run.
    // In our case, we have a single program running on our entire (unit) mesh.
    tt::tt_metal::distributed::MeshCoordinateRange device_range =
        tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    // Create a program object. A program is a collection of kernels that are executed on the device.
    // Kernels will be specified later.
    Program program = CreateProgram();

    return ProgramState(std::move(mesh_device), std::move(program), std::move(workload), std::move(device_range), cq);
}

// clang-format off
/**
 * Helper function to create a circular buffer with the specified number of tiles and CB index.
 * Page size is set to the size of a single tile.
 *
 * | Argument  | Description                                               |
 * |-----------|-----------------------------------------------------------|
 * | program   | The program to which the circular buffer will be added    |
 * | cores     | Cores where the circular buffer will be created           |
 * | num_tiles | Number of tiles to allocate in the circular buffer        |
 * | cb_index  | Circular buffer index (c_0 to c_31)                       |
 */
// clang-format on
void create_cb(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& cores,
    uint32_t num_tiles,
    tt::CBIndex cb_index) {
    constexpr uint32_t single_tile_bytes = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    // Technically, circular buffers operate on pages, not tiles. However, it is most common to have one tile per page.
    CircularBufferConfig cb_config = CircularBufferConfig(num_tiles * single_tile_bytes, {{cb_index, cb_data_format}})
                                         .set_page_size(cb_index, single_tile_bytes);
    tt_metal::CreateCircularBuffer(program, cores, cb_config);
}

// clang-format off
/**
 * Verify that all receivers received the correct tensor data.
 * Compares each receiver's output against the reference tensor.
 *
 * | Argument        | Description                                                      |
 * |-----------------|------------------------------------------------------------------|
 * | reference       | The original tensor data that was multicast                      |
 * | received        | Vector containing output from all receivers (num_receivers copies)|
 * | n_tiles         | Number of tiles in the tensor                                    |
 * | num_receivers   | Number of receiver cores                                         |
 *
 * Return value: bool (true if all receivers match reference, false otherwise)
 */
// clang-format on
bool verify_multicast_results(
    const std::vector<bfloat16>& reference,
    const std::vector<bfloat16>& received,
    uint32_t n_tiles,
    uint32_t num_receivers) {
    log_info(tt::LogAlways, "=========== MULTICAST TENSOR VERIFICATION ===========");

    const uint32_t elements_per_tile = TILE_HEIGHT * TILE_WIDTH;
    const uint32_t total_elements = n_tiles * elements_per_tile;

    TT_FATAL(reference.size() == total_elements, "Reference size mismatch");
    TT_FATAL(received.size() == num_receivers * total_elements, "Received size mismatch");

    bool all_pass = true;

    // Check each receiver's copy of the tensor
    for (uint32_t receiver = 0; receiver < num_receivers; receiver++) {
        uint32_t mismatch_count = 0;
        uint32_t first_mismatch_idx = 0;

        // Compare this receiver's output against the reference
        for (uint32_t i = 0; i < total_elements; i++) {
            uint32_t received_idx = receiver * total_elements + i;
            if (received[received_idx] != reference[i]) {
                if (mismatch_count == 0) {
                    first_mismatch_idx = i;
                }
                mismatch_count++;
            }
        }

        if (mismatch_count == 0) {
            log_info(tt::LogAlways, "[PASS] Receiver {} received correct tensor ({} tiles)", receiver + 1, n_tiles);
        } else {
            log_error(
                tt::LogAlways,
                "[FAIL] Receiver {} has {} mismatches (first at index {})",
                receiver + 1,
                mismatch_count,
                first_mismatch_idx);
            all_pass = false;
        }
    }

    if (all_pass) {
        log_info(tt::LogAlways, "[PASS] All {} receivers received correct tensor data", num_receivers);
    } else {
        log_error(tt::LogAlways, "[FAIL] One or more receivers have incorrect data");
    }

    log_info(tt::LogAlways, "=====================================================");

    return all_pass;
}

// clang-format off
/**
 * Perform multicast operation: send a full tensor from coordinator core to multiple receiver cores.
 * Uses double-buffering for improved performance. Each receiver gets a complete copy of the tensor.
 *
 * | Argument        | Description                                               |
 * |-----------------|-----------------------------------------------------------|
 * | input_data      | Input tensor data to multicast                            |
 * | output_data     | Output vector to store received tensors from all receivers|
 * | M               | Number of rows in the tensor                              |
 * | N               | Number of columns in the tensor                           |
 * | num_receivers   | Number of receiver cores                                  |
 * | prog_state      | Program state containing device, program, and context     |
 */
// clang-format on
void multicast_tensor_tensix(
    const std::vector<bfloat16>& input_data,
    std::vector<bfloat16>& output_data,
    const uint32_t M,
    const uint32_t N,
    const uint32_t num_receivers,
    ProgramState& prog_state) {
    const uint32_t total_elements = M * N;
    constexpr uint32_t elements_per_tile = TILE_HEIGHT * TILE_WIDTH;
    TT_FATAL(input_data.size() == total_elements, "Input data size must be M * N");
    TT_FATAL(output_data.size() == num_receivers * total_elements, "Output data size must be num_receivers * M * N");

    TT_FATAL(total_elements % elements_per_tile == 0, "Total elements must be divisible by elements per tile");
    const uint32_t n_tiles = total_elements / elements_per_tile;

    // Create ttnn::Tensor objects for the input and output data.
    // We use TILE layout as that's what the hardware natively operates on.
    // Tensors are allocated in device DRAM (i.e. DRAM that is directly attached to the Tensix processor,
    // which is distinct from the host DRAM).
    TensorLayout tile_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig(BufferType::DRAM));

    // Input tensor: MxN elements
    TensorSpec input_spec(Shape({M, N}), tile_layout);
    // Output tensor: num_receivers copies of the input tensor stacked vertically.
    TensorSpec output_spec(Shape({num_receivers * M, N}), tile_layout);

    // Create device tensors from input data.
    // This creates the tensors and queues transfer of data to device in one step.
    Tensor src_tensor = Tensor::from_vector<bfloat16>(input_data, input_spec, prog_state.mesh_device.get());
    // Create output tensor on device (no initialization needed - kernels will write into it).
    Tensor dst_tensor = create_device_tensor(output_spec, prog_state.mesh_device.get());

    // Get MeshBuffer pointers from tensors. Mesh buffers hold info about how tensor data is distributed
    // across physical DRAM banks.
    auto src_mesh_buffer = src_tensor.mesh_buffer();
    auto dst_mesh_buffer = dst_tensor.mesh_buffer();

    ////////// TENSIX CORE SETUP //////////
    // Define logical sender core and receiver core range (for kernel creation on the host).
    CoreRange all_cores_logical = CoreRange({0, 0}, {3, 0});
    CoreCoord sender_core_logical = {0, 0};
    CoreRange receiver_cores_logical = CoreRange({1, 0}, {3, 0});

    // Convert logical coordinates to device coordinates (necessary for device-side multicasting).
    CoreCoord sender_core_device = prog_state.mesh_device->worker_core_from_logical_core(sender_core_logical);
    CoreRange receiver_cores_device = CoreRange(
        prog_state.mesh_device->worker_core_from_logical_core(receiver_cores_logical.start_coord),
        prog_state.mesh_device->worker_core_from_logical_core(receiver_cores_logical.end_coord));

    // Grab the number of destinations, which will act as our "atomic counter" for semaphores.
    size_t num_dests = receiver_cores_logical.size();
    TT_FATAL(num_dests == num_receivers, "Number of receiver cores must match num_receivers parameter");

    ////////// SEMAPHORE SETUP //////////
    // Semaphores are used for synchronization between the coordinator and receiver cores.
    // receivers_ready_semaphore: receivers signal when they're ready to receive a tile
    // tile_sent_semaphore: coordinator signals when a tile has been multicast
    uint32_t receivers_ready_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, 0);
    uint32_t tile_sent_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, 0);

    ////////// CIRCULAR BUFFER SETUP //////////
    // Create circular buffers with 2 tiles for double-buffering.
    // Double-buffering allows overlapping data movement and computation:
    // while one tile is being processed, the next can be loaded.
    constexpr uint32_t tiles_per_cb = 2;
    create_cb(prog_state.program, all_cores_logical, tiles_per_cb, tt::CBIndex::c_0);
    create_cb(prog_state.program, all_cores_logical, tiles_per_cb, tt::CBIndex::c_16);

    ////////// DATA MOVEMENT CONFIG SETUP //////////
    // Compile-time args for coordinator kernel to read input tiles from DRAM.
    // TensorAccessorArgs extracts data distribution details from MeshBuffer so kernels
    // don't need to deal with low-level details like bank IDs.
    std::vector<uint32_t> coordinator_compile_time_args;
    TensorAccessorArgs(*src_mesh_buffer).append_to(coordinator_compile_time_args);
    DataMovementConfig DataMovementConfigCoordinator = {
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = coordinator_compile_time_args};

    // Receiver cores use the default config for inbound (no compile-time args needed).
    DataMovementConfig DataMovementConfigIn = {
        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default};

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_mesh_buffer).append_to(writer_compile_time_args);
    DataMovementConfig DataMovementConfigOut = {
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile_time_args};

    ////////// COORDINATOR KERNEL SETUP //////////
    // The coordinator kernel reads tiles from DRAM and multicasts them to all receiver cores.
    KernelHandle coordinator_kernel_id = CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab_multicast/kernels/dataflow/coordinator_kernel.cpp",
        sender_core_logical,
        DataMovementConfigCoordinator);

    ////////// DATAFLOW KERNELS SETUP //////////
    // Inbound kernel: receives the multicast tiles on each receiver core.
    KernelHandle inbound_kernel_id = CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab_multicast/kernels/dataflow/inbound_kernel.cpp",
        receiver_cores_logical,
        DataMovementConfigIn);

    // Outbound kernel: writes the received tiles back to DRAM for verification.
    KernelHandle outbound_kernel_id = CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab_multicast/kernels/dataflow/outbound_kernel.cpp",
        receiver_cores_logical,
        DataMovementConfigOut);

    ////////// COMPUTE KERNEL SETUP //////////
    // Void compute kernel - no computation needed for this multicast example.
    // The user can extend this to perform operations on the received tiles.
    vector<uint32_t> compute_kernel_args = {};
    CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab_multicast/kernels/compute/void_compute_kernel.cpp",
        receiver_cores_logical,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args});

    ////////// RUNTIME ARGS SETUP //////////

    // Args for the sender core to multicast tiles.
    // They must have access to coordinates of all receiver cores to execute multicast operation.
    // Observe how SetRuntimeArgs, which is a host-level function, takes in logical coordinates,
    // but the runtime arguments passed to kernels use device coordinates.
    // This is because runtime arguments are used by kernels, which run on the device, and they need
    // to know the device coordinates of the cores.
    SetRuntimeArgs(
        prog_state.program,
        coordinator_kernel_id,
        sender_core_logical,
        {static_cast<uint32_t>(receiver_cores_device.start_coord.x),
         static_cast<uint32_t>(receiver_cores_device.start_coord.y),
         static_cast<uint32_t>(receiver_cores_device.end_coord.x),
         static_cast<uint32_t>(receiver_cores_device.end_coord.y),
         receivers_ready_semaphore,
         tile_sent_semaphore,
         src_mesh_buffer->address(),
         n_tiles,
         num_dests});

    // Args for the receiver cores to receive tiles.
    // They must have access to coordinates of the sender core to listen for multicast operation.
    SetRuntimeArgs(
        prog_state.program,
        inbound_kernel_id,
        receiver_cores_logical,
        {static_cast<uint32_t>(sender_core_device.x),
         static_cast<uint32_t>(sender_core_device.y),
         receivers_ready_semaphore,
         tile_sent_semaphore,
         n_tiles});

    // Args for the receiver cores to send tiles back to DRAM.
    // Each receiver writes to a different section of the output buffer.
    // receiver_idx determines the starting tile offset for each receiver.
    int receiver_idx = 0;
    for (const CoreCoord& core : receiver_cores_logical) {
        SetRuntimeArgs(
            prog_state.program,
            outbound_kernel_id,
            core,
            {dst_mesh_buffer->address(), n_tiles, static_cast<uint32_t>(receiver_idx)});
        receiver_idx++;
    }

    log_info(tt::LogAlways, "Launching multicast of {} tiles to {} receivers", n_tiles, num_receivers);

    prog_state.workload.add_program(prog_state.device_range, std::move(prog_state.program));
    // Last argument is set to true to wait for the workload to complete (blocking call).
    tt_metal::distributed::EnqueueMeshWorkload(prog_state.cq, prog_state.workload, true);

    log_info(tt::LogAlways, "Multicast complete");

    // Read the result back from device using Tensor::to_vector
    output_data = dst_tensor.to_vector<bfloat16>();
}

///////////////////////////////////////

// clang-format off
/**
 * Main function that demonstrates multicast operation using ttnn::Tensor API.
 * Creates a tensor with random data, multicasts it to receiver cores, and verifies results.
 *
 * Return value: int (0 on success, non-zero on failure)
 */
// clang-format on
int main() {
    bool pass = true;

    try {
        // Number of receiver cores (cores 1,0 through 3,0)
        constexpr uint32_t num_receivers = 3;

        // Define tensor dimensions (same as lab_eltwise_binary: 400 tiles)
        constexpr uint32_t M = 640;
        constexpr uint32_t N = 640;
        constexpr uint32_t total_elements = M * N;
        constexpr uint32_t n_tiles = total_elements / (TILE_HEIGHT * TILE_WIDTH);

        log_info(
            tt::LogAlways, "Multicast example: {}x{} tensor ({} tiles) to {} receivers", M, N, n_tiles, num_receivers);

        // Create input tensor filled with random data.
        // Use a fixed seed for reproducible results.
        constexpr uint32_t rng_seed = 42;
        std::mt19937 rng(rng_seed);
        std::uniform_real_distribution<float> rng_dist(0.f, 1.0f);

        std::vector<bfloat16> input_data(total_elements);
        for (bfloat16& v : input_data) {
            v = static_cast<bfloat16>(rng_dist(rng));
        }

        // Output vector to hold received tensors from all receiver cores.
        // Each receiver gets a complete copy of the tensor.
        std::vector<bfloat16> output_data(num_receivers * total_elements);

        // Initialize program state (includes device creation)
        ProgramState prog_state = init_program();

        // Perform the multicast operation
        multicast_tensor_tensix(input_data, output_data, M, N, num_receivers, prog_state);

        log_info(tt::LogAlways, "Output vector size: {} elements", output_data.size());

        // Verify that all receiver cores received the correct tensor
        pass = verify_multicast_results(input_data, output_data, n_tiles, num_receivers);

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
