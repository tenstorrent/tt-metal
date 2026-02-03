// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 This example demonstrates how to create a 32x32 data tile and multicast it from a sender (coordinator) core to multiple
 receiver cores.  It covers the setup of semaphores and multicore addressing on Tenstorrent hardware.  In the original
 configuration, (0,0) is the sender core, while (1,0), (2,0), and (3,0) are the receiver cores.  The user can modify
 these coordinates as desired.

 This version uses ttnn::Tensor for cleaner buffer management, abstracting away DRAM buffer internals.
*/

#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>
#include <optional>
#include <thread>
#include <chrono>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;
using namespace std;
using namespace ttnn;
using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

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
 * | core      | Core specification (CoreCoord, CoreRange, or CoreRangeSet)|
 * | num_tiles | Number of tiles to allocate in the circular buffer        |
 * | cb_index  | Circular buffer index (c_0 to c_31)                       |
 */
// clang-format on
void create_cb(Program& program, const CoreSpec& core, uint32_t num_tiles, tt::CBIndex cb_index) {
    constexpr uint32_t single_tile_bytes = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    // Technically, circular buffers operate on pages, not tiles. However, it is most common to have one tile per page.
    CircularBufferConfig cb_config = CircularBufferConfig(num_tiles * single_tile_bytes, {{cb_index, cb_data_format}})
                                         .set_page_size(cb_index, single_tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_config);
}

// clang-format off
/**
 * Verify that received tiles match the golden (original) tile.
 *
 * | Argument       | Description                                               |
 * |----------------|-----------------------------------------------------------|
 * | golden_tile    | The original tile data that was multicast                 |
 * | received_tiles | Vector containing all received tiles from receiver cores  |
 * | num_tiles      | Number of receiver tiles to verify                        |
 * | verbose_verify | If true, print full tile contents for debugging           |
 *
 * Return value: bool (true if all tiles match, false otherwise)
 */
// clang-format on
bool verify_tiles(
    const std::vector<bfloat16>& golden_tile,
    const std::vector<bfloat16>& received_tiles,
    int num_tiles,
    bool verbose_verify) {
    fmt::print("\n=========== MULTICASTED TILE VERIFICATION ===========\n");

    ////////// LAMBDAS FOR ITERATING AND VERBOSE PRINTING TILES //////////
    auto tile_elem_idx = [](int tile, int i, int j) {
        return (tile * TILE_WIDTH * TILE_HEIGHT) + (i * TILE_WIDTH) + j;
    };

    auto print_tile =
        [&](const std::string& label, const std::vector<bfloat16>& data, std::optional<int> tile_num = std::nullopt) {
            if (tile_num.has_value()) {
                fmt::print("\n[{} TILE {}]\n", label, tile_num.value() + 1);
            } else {
                fmt::print("\n[{} TILE]\n", label);
            }

            for (int i = 0; i < TILE_HEIGHT; i++) {
                for (int j = 0; j < TILE_WIDTH; j++) {
                    int idx = tile_num.has_value() ? tile_elem_idx(tile_num.value(), i, j) : (i * TILE_WIDTH) + j;
                    fmt::print("{} ", static_cast<float>(data[idx]));
                }
                fmt::print("\n");
            }
        };

    ////////// VERIFICATION FLOW //////////
    if (verbose_verify) {
        print_tile("ORIGINAL", golden_tile);
    }

    bool all_match = true;  // verification loop primer.

    // iterate over all received tiles.
    for (int tile = 0; tile < num_tiles; tile++) {
        bool tile_match = true;
        if (verbose_verify) {
            print_tile("RECEIVED", received_tiles, tile);
        }
        // iterate over ith tile's elements, check whether they match with golden's elements.
        for (int i = 0; i < TILE_HEIGHT && tile_match; i++) {
            for (int j = 0; j < TILE_WIDTH; j++) {
                int idx = tile_elem_idx(tile, i, j);
                float received = static_cast<float>(received_tiles[idx]);
                float golden = static_cast<float>(golden_tile[(i * TILE_WIDTH) + j]);
                if (received != golden) {
                    tile_match = false;
                    break;
                }
            }
        }
        if (tile_match) {
            fmt::print("[PASS] Receiver tile {} matches the golden tile.\n", tile + 1);
        } else {
            fmt::print("[FAIL] Receiver tile {} does not match the golden tile.\n", tile + 1);
        }

        all_match &= tile_match;
    }
    if (all_match) {
        fmt::print("[PASS] All {} receiver tiles match the golden tile.\n", num_tiles);
    } else {
        fmt::print("[FAIL] One or more tiles did not match the golden tile.\n");
    }
    fmt::print("=====================================================\n\n");

    return all_match;
}

// clang-format off
/**
 * Perform multicast operation: send a tile from coordinator core to multiple receiver cores.
 * Uses ttnn::Tensor for clean buffer management while demonstrating multicast concepts.
 *
 * | Argument        | Description                                               |
 * |-----------------|-----------------------------------------------------------|
 * | input_tile      | Input tile data to multicast (single tile, TILE_HW elements)|
 * | output_tiles    | Output vector to store received tiles from all receivers  |
 * | num_receivers   | Number of receiver cores                                  |
 * | prog_state      | Program state containing device, program, and context     |
 */
// clang-format on
void multicast_tile_tensix(
    const std::vector<bfloat16>& input_tile,
    std::vector<bfloat16>& output_tiles,
    const uint32_t num_receivers,
    ProgramState& prog_state) {
    constexpr uint32_t num_input_tiles = 1;

    TT_FATAL(input_tile.size() == TILE_HW, "Input tile must have exactly TILE_HW ({}) elements", TILE_HW);
    TT_FATAL(
        output_tiles.size() == num_receivers * TILE_HW,
        "Output vector must have {} elements for {} receivers",
        num_receivers * TILE_HW,
        num_receivers);

    // Create ttnn::Tensor objects for the input and output data.
    // We use TILE layout as that's what the hardware natively operates on.
    // Tensors are allocated in device DRAM.
    TensorLayout tile_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig(BufferType::DRAM));

    // Input tensor: single tile (32x32 elements)
    TensorSpec input_spec(Shape({TILE_HEIGHT, TILE_WIDTH}), tile_layout);
    // Output tensor: num_receivers tiles stacked vertically
    TensorSpec output_spec(Shape({num_receivers * TILE_HEIGHT, TILE_WIDTH}), tile_layout);

    // Create device tensors from input data.
    // This creates the tensors and queues transfer of data to device in one step.
    Tensor src0_tensor = Tensor::from_vector<bfloat16>(input_tile, input_spec, prog_state.mesh_device.get());
    // Create output tensor on device (no initialization needed - kernels will write into it).
    Tensor dst_tensor = create_device_tensor(output_spec, prog_state.mesh_device.get());

    // Get MeshBuffer pointers from tensors. Mesh buffers hold info about how tensor data is distributed
    // across physical DRAM banks.
    auto src0_mesh_buffer = src0_tensor.mesh_buffer();
    auto dst_mesh_buffer = dst_tensor.mesh_buffer();

    ////////// TENSIX CORE SETUP //////////
    // Define logical sender core and receiver core range (for kernel creation on the host).
    CoreRange all_cores_logical = CoreRange({0, 0}, {3, 0});
    CoreCoord sender_core_logical = {0, 0};
    CoreRange receiver_cores_logical = CoreRange({1, 0}, {3, 0});

    // Convert logical coordinates to physical coordinates (necessary for multicasting).
    CoreCoord sender_core_physical = prog_state.mesh_device->worker_core_from_logical_core(sender_core_logical);
    CoreRange receiver_cores_physical = CoreRange(
        prog_state.mesh_device->worker_core_from_logical_core(receiver_cores_logical.start_coord),
        prog_state.mesh_device->worker_core_from_logical_core(receiver_cores_logical.end_coord));

    // Define physical sender core and receiver core range (for runtime arguments on the device).
    CoreCoord sender_core = sender_core_physical;
    CoreCoord receiver_core_start = receiver_cores_physical.start_coord;
    CoreCoord receiver_core_end = receiver_cores_physical.end_coord;

    // Grab the number of destinations, which will act as our "atomic counter" for semaphores.
    size_t num_dests = receiver_cores_logical.size();
    TT_FATAL(num_dests == num_receivers, "Number of receiver cores must match num_receivers parameter");

    ////////// SEMAPHORE SETUP //////////
    // Semaphores are used for synchronization between the coordinator and receiver cores.
    uint32_t sender_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, 0);
    uint32_t receiver_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, 0);

    ////////// CIRCULAR BUFFER SETUP //////////
    // Create circular buffers for input and output on all cores.
    create_cb(prog_state.program, all_cores_logical, num_input_tiles, tt::CBIndex::c_0);
    create_cb(prog_state.program, all_cores_logical, num_input_tiles, tt::CBIndex::c_16);

    ////////// DATA MOVEMENT CONFIG SETUP //////////
    // Compile-time args for coordinator kernel to read input tile from DRAM.
    // TensorAccessorArgs extracts data distribution details from MeshBuffer so kernels
    // don't need to deal with low-level details like bank IDs.
    std::vector<uint32_t> coordinator_compile_time_args;
    TensorAccessorArgs(*src0_mesh_buffer).append_to(coordinator_compile_time_args);
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
    // The coordinator kernel reads the tile from DRAM and multicasts it to all receiver cores.
    KernelHandle coordinator_kernel_id = CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab_multicast/kernels/dataflow/coordinator_kernel.cpp",
        sender_core_logical,
        DataMovementConfigCoordinator);

    ////////// DATAFLOW KERNELS SETUP //////////
    // Inbound kernel: receives the multicast tile on each receiver core.
    KernelHandle inbound_kernel_id = CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab_multicast/kernels/dataflow/inbound_kernel.cpp",
        receiver_cores_logical,
        DataMovementConfigIn);

    // Outbound kernel: writes the received tile back to DRAM for verification.
    KernelHandle outbound_kernel_id = CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab_multicast/kernels/dataflow/outbound_kernel.cpp",
        receiver_cores_logical,
        DataMovementConfigOut);

    ////////// COMPUTE KERNEL SETUP //////////
    // Void compute kernel - no computation needed for this multicast example.
    // The user can extend this to perform operations on the received tile.
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
    // Args for the sender core to multicast tile.
    // They must have access to coordinates of all receiver cores to execute multicast operation.
    // Note: DRAM addressing is handled via TensorAccessorArgs (compile-time args), so no bank ID needed.
    SetRuntimeArgs(
        prog_state.program,
        coordinator_kernel_id,
        sender_core_logical,
        {(uint32_t)(receiver_core_start.x),
         (uint32_t)(receiver_core_start.y),
         (uint32_t)(receiver_core_end.x),
         (uint32_t)(receiver_core_end.y),
         sender_semaphore,
         receiver_semaphore,
         src0_mesh_buffer->address(),
         sizeof(bfloat16) * TILE_HW,
         num_dests});

    // Args for the receiver cores to receive tile.
    // They must have access to coordinates of the sender core to listen for multicast operation.
    SetRuntimeArgs(
        prog_state.program,
        inbound_kernel_id,
        receiver_cores_logical,
        {(uint32_t)(sender_core.x), (uint32_t)(sender_core.y), sender_semaphore, receiver_semaphore});

    // Args for the receiver cores to send tile back to DRAM.
    // Each receiver writes to a different tile offset in the output buffer.
    int tile_index = 0;
    for (const CoreCoord& core : receiver_cores_logical) {
        SetRuntimeArgs(
            prog_state.program,
            outbound_kernel_id,
            core,
            {dst_mesh_buffer->address(), static_cast<uint32_t>(tile_index)});
        tile_index++;
    }

    ////////// PROGRAM LAUNCH //////////
    fmt::print("Launching program\n");
    fmt::print(
        "Hello, Core ({}, {}) on Device {}, please multicast the tile to your neighbors.\n",
        sender_core_logical.x,
        sender_core_logical.y,
        0);

    prog_state.workload.add_program(prog_state.device_range, std::move(prog_state.program));
    // Last argument is set to true to wait for the workload to complete (blocking call).
    tt_metal::distributed::EnqueueMeshWorkload(prog_state.cq, prog_state.workload, true);

    fmt::print(
        "Thank you, Core ({}, {}) on Device {}, for the multicast.\n",
        sender_core_logical.x,
        sender_core_logical.y,
        0);

    // Introduce artificial delay to avoid jumbling host-side tile verification prints with
    // potentially echoing device-side DPRINTs. Delay may be adjusted or removed as needed.
    this_thread::sleep_for(chrono::milliseconds(200));

    // Read the result back from device using Tensor::to_vector
    output_tiles = dst_tensor.to_vector<bfloat16>();
}

///////////////////////////////////////

// clang-format off
/**
 * Main function that demonstrates multicast operation using ttnn::Tensor API.
 * Creates an identity matrix tile, multicasts it to receiver cores, and verifies results.
 *
 * Return value: int (0 on success, non-zero on failure)
 */
// clang-format on
int main() {
    bool pass = true;

    try {
        // Number of receiver cores (cores 1,0 through 3,0)
        constexpr uint32_t num_receivers = 3;

        // Create the identity matrix tile to multicast.
        // This is a 32x32 tile where diagonal elements are 1.0 and off-diagonal elements are 0.0.
        std::vector<bfloat16> identity_tile = create_identity_matrix(TILE_WIDTH, TILE_HEIGHT, TILE_WIDTH);

        // Output vector to hold received tiles from all receiver cores.
        std::vector<bfloat16> received_tiles(num_receivers * TILE_HW);

        // Initialize program state (includes device creation)
        ProgramState prog_state = init_program();

        // Perform the multicast operation
        multicast_tile_tensix(identity_tile, received_tiles, num_receivers, prog_state);

        log_info(tt::LogAlways, "Output vector of size {}", received_tiles.size());

        // Verify that all receiver cores received the correct tile
        bool verbose_verify = false;  // Set to true to print full tile contents
        pass = verify_tiles(identity_tile, received_tiles, num_receivers, verbose_verify);

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
