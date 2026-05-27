// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 port — see METAL2_PORT_PLAN.md alongside the tilize op directory.

#include "tilize_multi_core_default_program_factory.hpp"

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::metal2_host_api;

namespace ttnn::prim {

// Named namespace (not anonymous) to avoid Unity-build redefinition.
namespace tilize_multi_core_default_factory {

constexpr const char* PROGRAM_ID = "tilize_multi_core_default";
constexpr const char* READER = "reader";
constexpr const char* WRITER = "writer";
constexpr const char* COMPUTE = "compute";
constexpr const char* SRC_DFB = "src_dfb";
constexpr const char* OUT_DFB = "out_dfb";
constexpr const char* INPUT = "input";
constexpr const char* OUTPUT = "output";
constexpr const char* MAIN_WU = "main";

NodeRangeSet to_node_range_set(const CoreRangeSet& crs) {
    std::vector<NodeRange> ranges;
    ranges.reserve(crs.ranges().size());
    for (const auto& cr : crs.ranges()) {
        ranges.emplace_back(NodeCoord{cr.start_coord.x, cr.start_coord.y}, NodeCoord{cr.end_coord.x, cr.end_coord.y});
    }
    return NodeRangeSet(std::move(ranges));
}

NodeCoord to_node_coord(const CoreCoord& c) { return NodeCoord{c.x, c.y}; }

}  // namespace tilize_multi_core_default_factory

ttnn::device_operation::ProgramArtifacts TilizeMultiCoreDefaultProgramFactory::create_program_spec(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tilize_multi_core_default_factory;
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);
    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32;

    auto* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto logical_shape = a.logical_shape();
    uint32_t logical_width = logical_shape[-1];
    uint32_t ntiles_per_block = tt::div_up(logical_width, TILE_WIDTH);
    uint32_t ntiles = dst_buffer->num_pages();
    uint32_t nblocks = tt::div_up(ntiles, ntiles_per_block);

    auto grid_size = a.device()->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, nblocks);

    uint32_t page_size = a.buffer()->page_size();
    uint32_t aligned_page_size = a.buffer()->aligned_page_size();
    uint32_t num_pages_in_row = 1;
    uint32_t size_of_valid_data_in_last_page_in_row = page_size;
    if (a.is_sharded()) {
        uint32_t shard_width =
            a.shard_spec().has_value() ? a.shard_spec().value().shape[1] : a.nd_shard_spec().value().shard_shape[-1];
        num_pages_in_row = tt::div_up(logical_width, shard_width);
        uint32_t padding_size = (num_pages_in_row * page_size) - (a.logical_shape()[-1] * a.element_size());
        size_of_valid_data_in_last_page_in_row = page_size - padding_size;
    }
    (void)aligned_page_size;  // legacy CTA[0] — unread by kernel, dropped in the metal2 fork.

    const bool has_cliff = !core_range_cliff.empty();

    NodeRangeSet all_nodes = to_node_range_set(all_cores);

    // ---- DFB specs ----
    DataflowBufferSpec src_dfb{
        .unique_id = SRC_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = ntiles_per_block,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = ntiles_per_block,
        .data_format_metadata = output_cb_data_format,
    };

    // ---- Reader / Writer kernels (shared across full and cliff) ----
    KernelSpec reader{
        .unique_id = READER,
        .source = KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                             "reader_unary_stick_layout_split_rows_multicore_metal2.cpp"},
        .dfb_bindings = {{
            .dfb_spec_name = SRC_DFB,
            .local_accessor_name = "input",
            .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {{
            .tensor_parameter_name = INPUT,
            .accessor_name = "input",
        }},
        .compile_time_arg_bindings =
            {
                {"num_pages_in_row", num_pages_in_row},
                {"size_of_valid_data_in_last_page_in_row", size_of_valid_data_in_last_page_in_row},
            },
        .runtime_arguments_schema =
            {.named_runtime_args =
                 {
                     "num_rows",
                     "num_tiles_per_block",
                     "block_width_size",
                     "num_full_blocks_in_row",
                     "start_page_id",
                 }},
        .config_spec =
            DataMovementConfiguration{
                .gen1_data_movement_config =
                    DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                    },
            },
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                             "writer_unary_interleaved_start_id_metal2.cpp"},
        .dfb_bindings = {{
            .dfb_spec_name = OUT_DFB,
            .local_accessor_name = "out",
            .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {{
            .tensor_parameter_name = OUTPUT,
            .accessor_name = "out",
        }},
        .runtime_arguments_schema = {.named_runtime_args = {"num_pages", "start_id"}},
        .config_spec =
            DataMovementConfiguration{
                .gen1_data_movement_config =
                    DataMovementConfiguration::Gen1DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                    },
            },
    };

    // ---- Compute kernel: single KernelSpec with per_core_block_cnt as RTA so
    // full/cliff cores share a single DFB consumer (satisfies the DFB invariant).
    std::vector<ComputeConfiguration::UnpackToDestModeEntry> unpack_to_dest_mode;
    if (fp32_llk_acc) {
        unpack_to_dest_mode.emplace_back(SRC_DFB, UnpackToDestMode::UnpackToDestFp32);
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = KernelSpec::SourceFilePath{"ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp"},
        .dfb_bindings =
            {
                {.dfb_spec_name = SRC_DFB,
                 .local_accessor_name = "src",
                 .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER},
                {.dfb_spec_name = OUT_DFB,
                 .local_accessor_name = "dst",
                 .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER},
            },
        .compile_time_arg_bindings =
            {
                {"per_core_block_tile_cnt", ntiles_per_block},
            },
        .runtime_arguments_schema = {.named_runtime_args = {"per_core_block_cnt"}},
        .config_spec =
            ComputeConfiguration{
                .fp32_dest_acc_en = fp32_llk_acc,
                .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            },
    };

    WorkUnitSpec main_wu{
        .unique_id = MAIN_WU,
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_nodes,
    };

    ProgramSpec spec{
        .program_id = PROGRAM_ID,
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = {std::move(src_dfb), std::move(out_dfb)},
        .tensor_parameters =
            {
                {.unique_id = INPUT, .spec = a.tensor_spec()},
                {.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {std::move(main_wu)},
    };

    // ---- Run params ----
    using KernelRunParams = ProgramRunParams::KernelRunParams;

    // 1D block distribution: ncores_full normal cores, then 1 cliff core (if has_cliff).
    uint32_t ncores_full = ncores - has_cliff;
    uint32_t tile_start_id = 0;
    uint32_t page_start_id = 0;
    const auto& cores = corerange_to_cores(available_grid);

    KernelRunParams reader_rp{.kernel_spec_name = READER};
    KernelRunParams writer_rp{.kernel_spec_name = WRITER};
    KernelRunParams compute_rp{.kernel_spec_name = COMPUTE};

    for (uint32_t i = 0; i < ncores_full; ++i) {
        const CoreCoord& core = cores[i];
        reader_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args =
                {
                    {"num_rows", nblocks_per_core * TILE_HEIGHT},
                    {"num_tiles_per_block", ntiles_per_block},
                    {"block_width_size", page_size},
                    {"num_full_blocks_in_row", std::uint32_t{1}},
                    {"start_page_id", page_start_id},
                },
        });
        writer_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args =
                {
                    {"num_pages", ntiles_per_block * nblocks_per_core},
                    {"start_id", tile_start_id},
                },
        });
        compute_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args = {{"per_core_block_cnt", nblocks_per_core}},
        });
        tile_start_id += ntiles_per_block * nblocks_per_core;
        page_start_id += TILE_HEIGHT * nblocks_per_core * num_pages_in_row;
    }
    if (has_cliff) {
        const CoreCoord& core = cores[ncores_full];
        reader_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args =
                {
                    {"num_rows", nblocks_per_core_cliff * TILE_HEIGHT},
                    {"num_tiles_per_block", ntiles_per_block},
                    {"block_width_size", page_size},
                    {"num_full_blocks_in_row", std::uint32_t{1}},
                    {"start_page_id", page_start_id},
                },
        });
        writer_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args =
                {
                    {"num_pages", ntiles_per_block * nblocks_per_core_cliff},
                    {"start_id", tile_start_id},
                },
        });
        compute_rp.named_runtime_args.push_back({
            .node = to_node_coord(core),
            .args = {{"per_core_block_cnt", nblocks_per_core_cliff}},
        });
    }

    ProgramRunParams run_params;
    run_params.kernel_run_params.push_back(std::move(reader_rp));
    run_params.kernel_run_params.push_back(std::move(writer_rp));
    run_params.kernel_run_params.push_back(std::move(compute_rp));
    run_params.tensor_args = {
        {.tensor_parameter_name = INPUT, .tensor = std::cref(a.mesh_tensor())},
        {.tensor_parameter_name = OUTPUT, .tensor = std::cref(output.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
