// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <random>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include "tt-metalium/core_coord.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

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
 * Reference implementation of matrix multiplication.
 * Array A is of size MxK, Array B is of size KxN, and the output C is of size MxN.
 * The implementation is bare bones and does not include optimizations such as tiling or vectorization.
 * This is intended to be used as a golden reference for testing the Metalium implementation.
 *
 * | Argument | Description                                                         |
 * |----------|---------------------------------------------------------------------|
 * | a        | Input matrix A in row-major format, size MxK                        |
 * | b        | Input matrix B in row-major format, size KxN                        |
 * | output   | Output matrix C in row-major format, size MxN (will be overwritten) |
 * | M        | Number of rows in matrix A and output matrix C                      |
 * | N        | Number of columns in matrix B and output matrix C                   |
 * | K        | Number of columns in matrix A and rows in matrix B                  |
 */
// clang-format on
void reference_matmul(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    const uint32_t M,
    const uint32_t N,
    const uint32_t K) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            // Compute C[i * N + j] += A[i * K + k] * B[k * N + j] in every iteration of the "k" loop.
            std::uint32_t idx_c = (i * N) + j;
            std::uint32_t idx_a = i * K;
            std::uint32_t idx_b = j;
            float c_f = 0;
            for (uint32_t k_m = 0; k_m < K; k_m++) {
                c_f += static_cast<float>(a[idx_a]) * static_cast<float>(b[idx_b]);
                idx_a += 1;
                idx_b += N;
            }
            // Convert the result to bfloat16 only at the very end.
            output.at(idx_c) = static_cast<bfloat16>(c_f);
        }
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
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range;
    distributed::MeshCommandQueue& cq;

    ProgramState(
        std::shared_ptr<distributed::MeshDevice> mesh_device,
        Program program,
        distributed::MeshWorkload workload,
        distributed::MeshCoordinateRange device_range,
        distributed::MeshCommandQueue& cq) :
        mesh_device(std::move(mesh_device)),
        program(std::move(program)),
        workload(std::move(workload)),
        device_range(std::move(device_range)),
        cq(cq) {}
};

// clang-format off
/**
 * Initialize program state for multi core execution.
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
 * | Argument  | Description                                                 |
 * |-----------|-------------------------------------------------------------|
 * | program   | The program to which the circular buffer will be added      |
 * | core_spec | One or more cores where the circular buffer will be created |
 * | num_tiles | Number of tiles to allocate in the circular buffer          |
 * | cb_index  | Circular buffer index (c_0 to c_31)                         |
 */
// clang-format on
void create_cb(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    uint32_t num_tiles,
    tt::CBIndex cb_index) {
    constexpr uint32_t single_tile_bytes = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    // Technically, circular buffers operate on pages, not tiles. However, it is most common to have one tile per page.
    CircularBufferConfig cb_config = CircularBufferConfig(num_tiles * single_tile_bytes, {{cb_index, cb_data_format}})
                                         .set_page_size(cb_index, single_tile_bytes);
    tt_metal::CreateCircularBuffer(program, core_spec, cb_config);
}

// clang-format off
/**
 * Matrix multiplication using multiple cores on the Tensix device employing data reuse.
 * The input a is of size MxK, input b is of size KxN, and the output c is of size MxN.
 * This function assumes that TILE_HEIGHT == TILE_WIDTH, and that M, N and K are divisible
 * by TILE_HEIGHT.
 *
 * | Argument              | Description                                                         |
 * |-----------------------|---------------------------------------------------------------------|
 * | a                     | Input matrix A in row-major format, size MxK                        |
 * | b                     | Input matrix B in row-major format, size KxN                        |
 * | output                | Output matrix C in row-major format, size MxN (will be overwritten) |
 * | M                     | Number of rows in matrix A and output matrix C                      |
 * | N                     | Number of columns in matrix B and output matrix C                   |
 * | K                     | Number of columns in matrix A and rows in matrix B                  |
 * | core_grid             | Number of cores to use in the compute grid                          |
 * |                         (e.g. core_grid.x = 10, core_grid.y = 10 means 100 cores)           |
 * | prog_state            | Program state containing device, program, and execution context     |
 */
// clang-format on
void matmul_multi_core(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const CoreCoord core_grid,
    ProgramState& prog_state) {
    // Calculate the number of tiles along each dimension.
    const uint32_t Mt = M / TILE_HEIGHT;
    const uint32_t Kt = K / TILE_WIDTH;
    const uint32_t Nt = N / TILE_WIDTH;

    // Figure out how many cores are available on the device.
    CoreCoord max_core_grid = prog_state.mesh_device.get()->compute_with_storage_grid_size();
    log_info(
        tt::LogAlways,
        "Using core grid size of ({} x {}) out of available core grid size of ({} x {})",
        core_grid.x,
        core_grid.y,
        max_core_grid.x,
        max_core_grid.y);
    TT_FATAL(
        core_grid.x <= max_core_grid.x && core_grid.y <= max_core_grid.y,
        "Core grid size must be less than or equal to available core grid size.");
    TT_FATAL((Mt % core_grid.x == 0) && (Nt % core_grid.y == 0), "Mt and Nt must be divisible by core grid size.");

    // Figure out blocking parameters, per lab writeup.
    const uint32_t M_block_tiles = Mt / core_grid.x;
    const uint32_t N_block_tiles = Nt / core_grid.y;
    // This needs to be chosen so that all the data fits in on-chip SRAM.
    const uint32_t K_block_tiles = 2;
    TT_FATAL(Kt % K_block_tiles == 0, "Kt must be divisible by K_block_tiles.");
    const uint32_t num_k_blocks = Kt / K_block_tiles;

    // The number of tiles in the input A and B slabs.
    const uint32_t A_slab_tiles = M_block_tiles * K_block_tiles;
    const uint32_t B_slab_tiles = K_block_tiles * N_block_tiles;

    // The number of tiles in the output C block.
    const uint32_t C_block_tiles = M_block_tiles * N_block_tiles;

    // Create ttnn::Tensor objects for the input and output data.
    // We use TILE layout as that's what the hardware natively operates on.
    // Tensors are allocated in device DRAM (i.e. DRAM that is directly attached to the Tensix processor,
    // which is distinct from the host DRAM).
    TensorLayout tile_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig(BufferType::DRAM));
    TensorSpec src0_spec(Shape({M, K}), tile_layout);
    TensorSpec src1_spec(Shape({K, N}), tile_layout);
    TensorSpec dst_spec(Shape({M, N}), tile_layout);

    // Create device tensors from input data.
    // This creates the tensors and queues transfer of data to device in one step.
    Tensor src0_tensor = Tensor::from_vector<bfloat16>(a, src0_spec, prog_state.mesh_device.get());
    Tensor src1_tensor = Tensor::from_vector<bfloat16>(b, src1_spec, prog_state.mesh_device.get());
    // Create output tensor on device (no initialization needed - kernel will write into it).
    Tensor dst_tensor = create_device_tensor(dst_spec, prog_state.mesh_device.get());

    // Create a set of all cores in the compute grid.
    CoreCoord top_left_core_logical{0, 0};
    CoreRange all_cores_logical(top_left_core_logical, CoreCoord(core_grid.x - 1, core_grid.y - 1));

    // All cores except the top row and left column are receiving A and B.
    CoreRange ab_receiver_cores_logical(
        {top_left_core_logical.x + 1, top_left_core_logical.y + 1}, CoreCoord(core_grid.x - 1, core_grid.y - 1));

    // First column, excluding the top left core is sending A and receiving B.
    CoreRange a_sender_b_receiver_cores_logical(
        {top_left_core_logical.x, top_left_core_logical.y + 1}, CoreCoord(top_left_core_logical.x, core_grid.y - 1));

    // First row, excluding the top left core is sending B and receiving A.
    CoreRange b_sender_a_receiver_cores_logical(
        {top_left_core_logical.x + 1, top_left_core_logical.y}, CoreCoord(core_grid.x - 1, top_left_core_logical.y));

    // Create circular buffers for the input and output data.
    // Using 2x tiles when double buffering is desired.
    const uint32_t tiles_cb_in0 = M_block_tiles * K_block_tiles * 2;
    const uint32_t tiles_cb_in1 = K_block_tiles * N_block_tiles * 2;
    const uint32_t tiles_cb_out = M_block_tiles * N_block_tiles;
    const uint32_t tiles_cb_interm = M_block_tiles * N_block_tiles;

    // There are 32 circular buffers (c_0 - c_31) on the device. We can use any of them, as long as they are not already
    // in use. Kernel code is responsible for using the correct circular buffer for the input and output data (e.g.
    // reader kernel reads data into c_0 and c_1, while the compute kernel reads data from these same buffers).
    create_cb(prog_state.program, all_cores_logical, tiles_cb_in0, CBIndex::c_0);
    create_cb(prog_state.program, all_cores_logical, tiles_cb_in1, CBIndex::c_1);

    // Compute kernel will write output data to c_16, which will be consumed by the writer kernel.
    // c_16 chosen arbitrarily (e.g. to leave c_2-c_15 free for other potential inputs when code is extended in the
    // future).
    create_cb(prog_state.program, all_cores_logical, tiles_cb_out, tt::CBIndex::c_16);

    // Use c_24 (arbitrarily chosen) for the intermediate buffer.
    create_cb(prog_state.program, all_cores_logical, tiles_cb_interm, tt::CBIndex::c_24);

    ////////// SEMAPHORE SETUP //////////
    // Semaphores are used for synchronization between the coordinator and receiver cores.
    // receivers_ready_semaphore: receivers signal when they're ready to receive a tile
    // tile_sent_semaphore: coordinator signals when a tile has been multicast

    // For simplicity, we create semaphores on all cores, although not all cores
    // will use all the semaphores.
    uint32_t a_receivers_ready_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, 0);
    uint32_t a_tile_sent_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, 0);
    uint32_t a_num_receivers = core_grid.x - 1;

    uint32_t b_receivers_ready_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, 0);
    uint32_t b_tile_sent_semaphore = CreateSemaphore(prog_state.program, all_cores_logical, 0);
    uint32_t b_num_receivers = core_grid.y - 1;

    // Get MeshBuffer pointers from tensors. Mesh buffers hold info about how tensor data is distributed
    // across physical DRAM banks (at least for our case when data is stored in DRAM).
    // Programmer doesn't need to understand the internals, but needs to pass this info to the kernels.
    auto src0_mesh_buffer = src0_tensor.mesh_buffer();
    auto src1_mesh_buffer = src1_tensor.mesh_buffer();
    auto dst_mesh_buffer = dst_tensor.mesh_buffer();

    // Create the reader, writer and compute kernels.
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
    KernelHandle ab_sender_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_matmul_ex3/kernels/dataflow/ab_sender.cpp",
        top_left_core_logical,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    KernelHandle ab_receiver_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_matmul_ex3/kernels/dataflow/ab_receiver.cpp",
        ab_receiver_cores_logical,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    KernelHandle a_sender_b_receiver_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_matmul_ex3/kernels/dataflow/a_sender_b_receiver.cpp",
        a_sender_b_receiver_cores_logical,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    KernelHandle b_sender_a_receiver_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_matmul_ex3/kernels/dataflow/b_sender_a_receiver.cpp",
        b_sender_a_receiver_cores_logical,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});
    std::vector<uint32_t> writer_compile_time_args;

    TensorAccessorArgs(*dst_mesh_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_matmul_ex3/kernels/dataflow/write_tiles.cpp",
        all_cores_logical,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});

    // Compile time arguments for the compute kernel.
    // Note that these are evaluated at the kernel's compile time, which is JIT compile done at program creation time.
    // Having arguments at compile time generally allows the compiler to optimize the kernel for the specific use case
    // (e.g. apply loop unrolling, constant folding, etc.), resulting in a more efficient kernel.
    std::vector<uint32_t> compute_compile_time_args = {
        num_k_blocks, M_block_tiles, N_block_tiles, K_block_tiles, A_slab_tiles, B_slab_tiles, C_block_tiles};

    tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab3_matmul_ex3/kernels/compute/tiles_matmul.cpp",
        all_cores_logical,
        tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    // Set the runtime arguments for the kernels.
    uint32_t src0_addr = src0_mesh_buffer->address();
    uint32_t src1_addr = src1_mesh_buffer->address();
    uint32_t dst_addr = dst_mesh_buffer->address();

    // Remember that core_grid.x corresponds to columns and core_grid.y corresponds to rows.
    for (uint32_t row_block = 0; row_block < core_grid.y; row_block++) {
        for (uint32_t col_block = 0; col_block < core_grid.x; col_block++) {
            // Pass appropriate offsets to the reader and writer kernels.
            uint32_t tile_offset_row = row_block * M_block_tiles;
            uint32_t tile_offset_col = col_block * N_block_tiles;

            CoreCoord core_logical = {col_block, row_block};

            // Aliases for easier reading of conditions below.
            uint32_t x = core_logical.x;
            uint32_t y = core_logical.y;

            // log_info(tt::LogAlways, "Setting runtime arguments for core ({}, {})", x, y);

            CoreCoord second_core_in_the_row_device =
                prog_state.mesh_device->worker_core_from_logical_core(CoreCoord(1, y));
            CoreCoord second_core_in_the_column_device =
                prog_state.mesh_device->worker_core_from_logical_core(CoreCoord(x, 1));
            CoreCoord last_core_in_row_device =
                prog_state.mesh_device->worker_core_from_logical_core(CoreCoord(core_grid.x - 1, y));
            CoreCoord last_core_in_column_device =
                prog_state.mesh_device->worker_core_from_logical_core(CoreCoord(x, core_grid.y - 1));

            CoreCoord leftmost_core_device = prog_state.mesh_device->worker_core_from_logical_core(CoreCoord(0, y));
            CoreCoord topmost_core_device = prog_state.mesh_device->worker_core_from_logical_core(CoreCoord(x, 0));

            // Set runtime arguments for the kernels.
            if (x == 0 && y == 0) {
                // Top left core is sending A and B.
                tt_metal::SetRuntimeArgs(
                    prog_state.program,
                    ab_sender_id,
                    core_logical,
                    // A receivers are in the current row, but one column to the right.
                    {second_core_in_the_row_device.x,
                     second_core_in_the_row_device.y,
                     last_core_in_row_device.x,
                     last_core_in_row_device.y,
                     a_receivers_ready_semaphore,
                     a_tile_sent_semaphore,
                     a_num_receivers,
                     // B receivers are in the current column, but one row below.
                     second_core_in_the_column_device.x,
                     second_core_in_the_column_device.y,
                     last_core_in_column_device.x,
                     last_core_in_column_device.y,
                     b_receivers_ready_semaphore,
                     b_tile_sent_semaphore,
                     b_num_receivers,
                     // Other parameters, similar to baseline implementation from lab 2.
                     src0_addr,
                     src1_addr,
                     Nt,
                     Kt,
                     M_block_tiles,
                     N_block_tiles,
                     K_block_tiles,
                     tile_offset_row,
                     tile_offset_col});
            } else if (x == 0 && y > 0) {
                // First column, excluding the top left core is sending A and receiving B.
                tt_metal::SetRuntimeArgs(
                    prog_state.program,
                    a_sender_b_receiver_id,
                    core_logical,
                    // A receivers are in the current row, but starting one column to the right (x + 1).
                    {second_core_in_the_row_device.x,
                     second_core_in_the_row_device.y,
                     last_core_in_row_device.x,
                     last_core_in_row_device.y,
                     a_receivers_ready_semaphore,
                     a_tile_sent_semaphore,
                     a_num_receivers,
                     // B sender is at the top of the current column.
                     topmost_core_device.x,
                     topmost_core_device.y,
                     b_receivers_ready_semaphore,
                     b_tile_sent_semaphore,
                     // Other parameters, similar to baseline implementation from lab 2.
                     src0_addr,
                     Nt,
                     Kt,
                     M_block_tiles,
                     N_block_tiles,
                     K_block_tiles,
                     tile_offset_row,
                     tile_offset_col});
            } else if (x > 0 && y == 0) {
                // First row, excluding the top left core is sending B and receiving A.
                tt_metal::SetRuntimeArgs(
                    prog_state.program,
                    b_sender_a_receiver_id,
                    core_logical,
                    // A sender is in the leftmost column of the current row.
                    {leftmost_core_device.x,
                     leftmost_core_device.y,
                     a_receivers_ready_semaphore,
                     a_tile_sent_semaphore,
                     // B receivers are in the current column, but starting one row below.
                     second_core_in_the_column_device.x,
                     second_core_in_the_column_device.y,
                     last_core_in_column_device.x,
                     last_core_in_column_device.y,
                     b_receivers_ready_semaphore,
                     b_tile_sent_semaphore,
                     b_num_receivers,
                     // Other parameters, similar to baseline implementation from lab 2.
                     src1_addr,
                     Nt,
                     Kt,
                     M_block_tiles,
                     N_block_tiles,
                     K_block_tiles,
                     tile_offset_row,
                     tile_offset_col});
            } else {
                // All other cores are receiving A and B.
                tt_metal::SetRuntimeArgs(
                    prog_state.program,
                    ab_receiver_id,
                    core_logical,
                    // A sender is in the leftmost column of the current row.
                    {leftmost_core_device.x,
                     leftmost_core_device.y,
                     a_receivers_ready_semaphore,
                     a_tile_sent_semaphore,
                     // B sender is in the topmost row of the current column.
                     topmost_core_device.x,
                     topmost_core_device.y,
                     b_receivers_ready_semaphore,
                     b_tile_sent_semaphore,
                     // Other parameters, similar to baseline implementation from lab 2.
                     Nt,
                     Kt,
                     M_block_tiles,
                     N_block_tiles,
                     K_block_tiles,
                     tile_offset_row,
                     tile_offset_col});
            }

            tt_metal::SetRuntimeArgs(
                prog_state.program,
                writer_id,
                core_logical,
                {dst_addr, Nt, M_block_tiles, N_block_tiles, tile_offset_row, tile_offset_col});
        }
    }

    log_info(tt::LogAlways, "Executing kernels");
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
 * Main function that demonstrates multi core matrix multiplication with data reuse using ttnn::Tensor API.
 * Creates test data, runs a reference implementation on host CPU, executes matmul on Tensix device,
 * and verifies results.
 *
 * Return value: int (0 on success, non-zero on failure)
 */
// clang-format on
int main() {
    bool pass = true;

    try {
        // Define some constants that will be used throughout the program.
        // We will be multiplying two matrices of shape MxK and KxN to get a result of shape MxN
        constexpr uint32_t M = 640;  // user-defined
        constexpr uint32_t N = 640;  // user-defined
        constexpr uint32_t K = 320;  // user-defined

        static_assert(TILE_HEIGHT == TILE_WIDTH, "Tiles must be square.");
        static_assert(M % TILE_HEIGHT == 0, "M must be divisible by tile size");
        static_assert(N % TILE_HEIGHT == 0, "N must be divisible by tile size");
        static_assert(K % TILE_HEIGHT == 0, "K must be divisible by tile size");
        // In C++, the matrices are represented as vectors, to emphasize that memory
        // space is one-dimensional in general.
        // Filled with random values in the range [0, 1) for testing.
        // Use a fixed seed for reproducible results. Change this value to get different random sequences.
        constexpr uint32_t rng_seed = 42;
        std::mt19937 rng(rng_seed);
        std::uniform_real_distribution<float> rng_dist(0.f, 1.0f);
        std::vector<bfloat16> src0_vec(M * K);
        std::vector<bfloat16> src1_vec(K * N);

        for (bfloat16& v : src0_vec) {
            v = static_cast<bfloat16>(rng_dist(rng));
        }
        for (bfloat16& v : src1_vec) {
            v = static_cast<bfloat16>(rng_dist(rng));
        }

        // Reference matmul running on x86 CPU so we can verify Tensix result.
        std::vector<bfloat16> reference_result(M * N);
        reference_matmul(src0_vec, src1_vec, reference_result, M, N, K);

        // Invoke the matrix multiplication on the Tensix device
        std::vector<bfloat16> result_vec(M * N);

        // Initialize program state (includes device creation)
        ProgramState prog_state = init_program();
        matmul_multi_core(src0_vec, src1_vec, result_vec, M, N, K, {10, 10}, prog_state);

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
                log_error(
                    tt::LogAlways, "Expected relative tolerance: {} actual relative error: {}", RELTOL, relative_error);
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
