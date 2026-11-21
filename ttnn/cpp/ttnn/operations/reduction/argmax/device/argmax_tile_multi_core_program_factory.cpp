// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "argmax_device_operation.hpp"
#include "argmax_tile_multi_core_split.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

/*
 * Multi-core argmax over the last dim of a TILE input.
 *
 * The input is a list of tile rows (32 input rows each, over every outer index). Cores form groups
 * of cores_per_group: a group owns a contiguous range of tile rows and each of its cores scans a
 * contiguous range of tile columns of those rows, reading the tile bytes and comparing them in the
 * single-core TILE reader's order (argmax_order_key.hpp). When a tile row holds more than one face of
 * rows, two instances per core split each tile into its top and bottom faces. With one core per group a core writes its
 * indices itself; otherwise the group's first core collects one (value, index) pair per row from each core and merges
 * them in column order. See argmax_tile_multi_core_split.hpp for how the split is chosen.
 */
ttnn::device_operation::ProgramArtifacts ArgMaxTileMultiCoreProgramFactory::create_program_artifacts(
    const ArgmaxParams& operation_attributes, const ArgmaxInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const bool keepdim = operation_attributes.keepdim;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;
    const tt::tt_metal::distributed::MeshDevice& device = output.device();

    // Resource names. Declared function-locally: the argmax factories share a unity-build
    // translation unit, and namespace-scope `const` objects would collide by name there.
    // Each kernel instance has its own buffers and semaphore: index 0 is the top faces (or the
    // whole tile), index 1 the bottom faces.
    const std::array<KernelSpecName, 2> READER{KernelSpecName{"reader0"}, KernelSpecName{"reader1"}};
    const std::array<DFBSpecName, 2> SRC{DFBSpecName{"src0"}, DFBSpecName{"src1"}};
    const std::array<DFBSpecName, 2> OUT{DFBSpecName{"out0"}, DFBSpecName{"out1"}};
    const std::array<DFBSpecName, 2> PARTIALS{DFBSpecName{"partials0"}, DFBSpecName{"partials1"}};
    const std::array<SemaphoreSpecName, 2> DONE{SemaphoreSpecName{"done0"}, SemaphoreSpecName{"done1"}};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    const auto& padded = input.padded_shape();
    const uint32_t rank = padded.size();
    const auto& logical = input.logical_shape();
    const uint32_t logical_rank = logical.size();
    const uint32_t tile_width = input.tensor_spec().tile().get_width();
    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    const uint32_t w_tiles = padded[rank - 1] / tile_width;
    const uint32_t h_tiles = padded[rank - 2] / tile_height;
    const uint32_t w_logical = logical[logical_rank - 1];
    const uint32_t h_logical = logical[logical_rank - 2];
    const uint32_t outer = input.logical_volume() / (h_logical * w_logical);
    const uint32_t tile_rows = outer * h_tiles;

    const auto grid = device.compute_with_storage_grid_size();
    const CoreRangeSet available =
        sub_core_grids.has_value() ? sub_core_grids.value() : CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const auto available_cores = corerange_to_cores(available, std::nullopt, /*row_wise=*/true);

    const auto split = argmax_tile_multi_core_split(tile_rows, w_tiles, available_cores.size());
    const uint32_t cores_per_group = split.cores_per_group;
    const uint32_t num_groups = split.num_groups;
    const uint32_t rows_per_group = split.rows_per_group;
    const uint32_t num_used = cores_per_group * num_groups;

    const std::vector<CoreCoord> used_cores(available_cores.begin(), available_cores.begin() + num_used);
    const CoreRangeSet used = CoreRangeSet(ttsl::Span<const CoreCoord>(used_cores));
    validate_reduce_op_program_grid(
        "Argmax TILE multicore", used, grid, sub_core_grids.has_value() ? &sub_core_grids.value() : nullptr, true, {});

    const tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t src_page_size = input.tensor_spec().compute_page_size_bytes();
    const uint32_t element_size = input.element_size();
    const uint32_t face_height = tile_height / 2;
    const uint32_t face_width = tile_width / 2;
    const uint32_t face_bytes = face_height * face_width * element_size;
    const bool src_is_dram = input.mesh_buffer().device_local_config().buffer_type == BufferType::DRAM;
    const uint32_t src_alignment = src_is_dram ? hal::get_dram_alignment() : hal::get_l1_alignment();

    // One instance per core scans whole tiles. When some tile row holds more than one face of rows,
    // two instances share each tile, one on the top faces and one on the bottom faces, on the two
    // data movement processors and NoCs.
    const uint32_t bottom_rows = h_logical % tile_height == 0 ? tile_height : h_logical % tile_height;
    const bool two_instances = h_tiles > 1 || bottom_rows > face_height;
    const uint32_t num_instances = two_instances ? 2 : 1;
    const uint32_t face_count = two_instances ? 2 : 4;
    const uint32_t rows_per_instance = face_count * face_height / 2;
    const uint32_t slot_bytes = face_count * face_bytes;
    // Two halves of batch_tiles slots each, about 16 KB per half.
    const uint32_t batch_tiles = std::max<uint32_t>(1, (16 * 1024) / slot_bytes);

    const bool dst_is_dram = output.mesh_buffer().device_local_config().buffer_type == BufferType::DRAM;
    const uint32_t dst_alignment = dst_is_dram ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const uint32_t out_stride = keepdim ? dst_alignment : static_cast<uint32_t>(sizeof(uint32_t));
    const uint32_t partials_size =
        cores_per_group > 1 ? 2 * cores_per_group * rows_per_group * rows_per_instance * sizeof(uint32_t) : 32;

    Group<DataflowBufferSpec> dataflow_buffers;
    Group<SemaphoreSpec> semaphores;
    Group<KernelSpec> kernels;
    Group<KernelRunArgs> kernel_run_args;
    uint32_t assigned_tiles = 0;
    for (uint32_t inst = 0; inst < num_instances; inst++) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = SRC[inst],
            .entry_size = 2 * batch_tiles * slot_bytes,
            .num_entries = 1,
            .data_format_metadata = input_data_format,
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT[inst],
            .entry_size = rows_per_instance * out_stride,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt32,
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = PARTIALS[inst],
            .entry_size = partials_size,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt32,
        });
        semaphores.push_back(SemaphoreSpec{.unique_id = DONE[inst], .target_nodes = used});

        // Every DFB is touched by its instance alone, through raw pointers, so each is bound self-loop.
        // The leader's partials are written by other cores through bare NoC addresses, not a binding.
        Group<DFBBinding> dfb_bindings;
        for (const auto& [name, accessor] : std::initializer_list<std::pair<DFBSpecName, const char*>>{
                 {SRC[inst], "src"}, {OUT[inst], "out"}, {PARTIALS[inst], "partials"}}) {
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::PRODUCER});
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::CONSUMER});
        }
        kernels.push_back(KernelSpec{
            .unique_id = READER[inst],
            .source =
                "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_tile_layout_multicore.cpp",
            .dfb_bindings = std::move(dfb_bindings),
            .semaphore_bindings = {SemaphoreBinding{.semaphore_spec_name = DONE[inst], .accessor_name = "done"}},
            .tensor_bindings =
                {
                    TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"},
                    TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
                },
            .compile_time_args =
                {
                    {"src_page_size", src_page_size},
                    {"tile_height", tile_height},
                    {"tile_width", tile_width},
                    {"input_height", h_tiles},
                    {"input_width", w_tiles},
                    {"logical_height", h_logical},
                    {"logical_width", w_logical},
                    {"keepdim", static_cast<uint32_t>(keepdim)},
                    {"batch_tiles", batch_tiles},
                    {"face_begin", inst * 2},
                    {"face_count", face_count},
                    {"face_bytes", face_bytes},
                    {"src_alignment", src_alignment},
                    {"cores_per_group", cores_per_group},
                    {"rows_per_group", rows_per_group},
                    {"out_stride", out_stride},
                },
            .runtime_arg_schema =
                {
                    .runtime_arg_names =
                        {"row_begin", "row_count", "col_begin", "col_count", "core_in_group", "leader_x", "leader_y"},
                },
            // The two instances of a core use different processors and NoCs, as the ROW_MAJOR multi-core
            // reader's primary and secondary do.
            .hw_config =
                DataMovementHardwareConfig{
                    .config_1xx =
                        DataMovementHardwareConfig::DataMovement1XXConfig{
                            .processor = inst == 0 ? tt::tt_metal::DataMovementProcessor::RISCV_1
                                                   : tt::tt_metal::DataMovementProcessor::RISCV_0,
                            .noc = inst == 0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0,
                            .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
                        },
                },
        });

        KernelRunArgs reader_run_args{.kernel = READER[inst]};
        for (uint32_t g = 0; g < num_groups; g++) {
            const auto [row_begin, row_count] = split.rows_of_group(g, tile_rows);
            const auto leader = device.worker_core_from_logical_core(used_cores.at(g * cores_per_group));
            for (uint32_t c = 0; c < cores_per_group; c++) {
                const auto [col_begin, col_count] = split.cols_of_core(c, w_tiles);
                AddRuntimeArgsForNode(
                    reader_run_args.runtime_arg_values,
                    used_cores.at((g * cores_per_group) + c),
                    {{"row_begin", row_begin},
                     {"row_count", row_count},
                     {"col_begin", col_begin},
                     {"col_count", col_count},
                     {"core_in_group", c},
                     {"leader_x", static_cast<uint32_t>(leader.x)},
                     {"leader_y", static_cast<uint32_t>(leader.y)}});
                if (inst == 0) {
                    assigned_tiles += row_count * col_count;
                }
            }
        }
        kernel_run_args.push_back(std::move(reader_run_args));
    }
    // The per-core ranges must cover every tile exactly once.
    TT_FATAL(
        assigned_tiles == tile_rows * w_tiles,
        "Argmax TILE multicore split covers {} of {} tiles",
        assigned_tiles,
        tile_rows * w_tiles);

    WorkUnitSpec work_unit{.name = "main", .target_nodes = used};
    for (uint32_t inst = 0; inst < num_instances; inst++) {
        work_unit.kernels.push_back(READER[inst]);
    }
    ProgramSpec spec{
        .name = "argmax_tile_multi_core",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = std::move(semaphores),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {std::move(work_unit)},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = std::move(kernel_run_args);
    run_args.tensor_args = {
        {INPUT, input},
        {OUTPUT, output},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
