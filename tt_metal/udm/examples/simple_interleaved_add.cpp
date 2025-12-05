// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Simple UDM Program Example: Interleaved Add
 *
 * Setup:
 * - Mesh: 1×4 (4 devices in a row)
 * - Global tensor: (4, 16) tiles, width-sharded across devices
 * - Per-device tensor: (4, 4) tiles, interleaved
 * - Grid: 1×16 workers per device (flattened from 4×4 device grid)
 * - Block: 1×4 (same as mesh)
 *
 * Operation:
 * - Each worker reads one local row (4 tiles)
 * - Adds 1 to each tile
 * - Writes back
 *
 * Worker assignment:
 * - 4 rows per device → 4 workers per device
 * - Total: 16 gcores (4 per device × 4 devices)
 * - Gcore 0-3: device 0, rows 0-3
 * - Gcore 4-7: device 1, rows 0-3
 * - Gcore 8-11: device 2, rows 0-3
 * - Gcore 12-15: device 3, rows 0-3
 */

#include "tt_metal/udm/tensor_builder.hpp"
#include "tt_metal/udm/program.hpp"
#include "tt_metal/udm/block_kernel.hpp"
#include "tt_metal/udm/block_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/api/ttnn/distributed/api.hpp"
#include "tt_metal/api/tt-metalium/host_api.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::udm;
using namespace ttnn;

/**
 * Create a width-sharded tensor on 1×4 mesh
 * Global shape: (4, 16) tiles
 * Per-device: (4, 4) tiles
 */
Tensor create_width_sharded_tensor(
    distributed::MeshDevice* mesh_device,
    const Shape& global_shape,   // (4, 16) in tiles -> (128, 512) in elements for tile size 32
    const Shape& local_shape) {  // (4, 4) in tiles per device

    auto mesh_shape = mesh_device->shape();  // [1, 4]

    // Create width-shard placement: mesh dim 1 shards tensor dim 1
    ttsl::SmallVector<ttnn::distributed::MeshMapperConfig::Placement> placements;
    placements.push_back(ttnn::distributed::MeshMapperConfig::Replicate{});  // mesh dim 0: replicate
    placements.push_back(ttnn::distributed::MeshMapperConfig::Shard{1});     // mesh dim 1: shard tensor dim 1

    // Create interleaved memory config
    MemoryConfig mem_config(BufferType::L1, TensorMemoryLayout::INTERLEAVED);
    TensorSpec tensor_spec(global_shape, TensorLayout(DataType::UINT16, PageConfig(Layout::TILE), mem_config));

    // Create host data
    uint32_t volume = 1;
    for (size_t i = 0; i < global_shape.rank(); ++i) {
        volume *= global_shape[i];
    }
    std::vector<uint16_t> src_data(volume);
    std::iota(src_data.begin(), src_data.end(), 0);

    auto host_tensor = Tensor::from_vector(src_data, tensor_spec);

    // Distribute tensor
    auto mapper = ttnn::distributed::create_mesh_mapper(
        *mesh_device,
        ttnn::distributed::MeshMapperConfig{.placements = std::move(placements), .mesh_shape_override = mesh_shape});

    return ttnn::distributed::distribute_tensor(host_tensor, *mapper, std::ref(*mesh_device));
}

/**
 * Create UDM program that adds 1 to each tile
 */
BlockProgram create_add_program(const Tensor& input_tensor) {
    // 1. Create TensorBuilder with custom grid shape
    // Grid: {1, 16} (flattened from device's 4×4 compute grid)
    // Block: use default mesh shape (1×4)
    auto tensor_builder = CreateTensorBuilder(input_tensor, Grid{{1, 16}});

    // 2. Create BlockProgram
    auto program = CreateBlockProgram(tensor_builder);

    // 3. Map tensor to gcores using UDM API
    // Partition work on dimension 0 (rows) - each worker processes 1 row
    // Data is width-sharded (dim 1), so each row spans multiple devices
    auto gcores_info = map_tensor_to_gcores(
        tensor_builder,
        input_tensor,
        0  // partition_dim = 0 (rows)
    );

    // 4. Get compile-time args from BlockTensorAccessor
    auto compile_time_args = tensor_builder.tensor_accessor().get_compile_time_args();

    // 5. Create kernel on all mapped gcores
    BlockKernelHandle kernel_id = CreateBlockKernel(
        program,
        "tt_metal/udm/examples/kernels/interleaved_add.cpp",
        gcores_info.gcores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args,
        });

    // 6. Set runtime args for each gcore
    // All information comes from gcores_info - no hardcoding!
    for (uint32_t i = 0; i < gcores_info.num_cores; i++) {
        std::vector<uint32_t> runtime_args = {
            static_cast<uint32_t>(tensor_builder.tensor_accessor().get_buffer_address()),
            gcores_info.gcore_to_block_page_start[i],  // Starting block page ID from mapping
            gcores_info.pages_per_gcore,               // Number of pages to process
            gcores_info.gcore_to_device_id[i],         // Device ID from mapping
        };

        SetBlockKernelRuntimeArgs(program, kernel_id, gcores_info.gcores[i], runtime_args);
    }

    return program;
}

/**
 * Run the UDM program
 */
void run_interleaved_add_example(distributed::MeshDevice* mesh_device) {
    log_info(LogTest, "=== UDM Interleaved Add Example ===");

    // Create width-sharded tensor
    Shape global_shape({128, 512});  // (4, 16) tiles in element count
    Shape local_shape({128, 128});   // (4, 4) tiles per device

    auto input_tensor = create_width_sharded_tensor(mesh_device, global_shape, local_shape);

    log_info(LogTest, "Created width-sharded tensor:");
    log_info(LogTest, "  Global shape: {}", global_shape);
    log_info(LogTest, "  Local shape: {}", local_shape);

    // Create program
    auto program = create_add_program(input_tensor);

    log_info(LogTest, "Created block program");

    // Run program
    const auto& mesh_shape = input_tensor.tensor_topology().distribution_shape();
    auto mesh_coord_range = distributed::MeshCoordinateRange(mesh_shape);

    auto mesh_workload = distributed::MeshWorkload();
    mesh_workload.add_program(mesh_coord_range, std::move(program.program()));

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, true);

    log_info(LogTest, "Program executed successfully");
}
