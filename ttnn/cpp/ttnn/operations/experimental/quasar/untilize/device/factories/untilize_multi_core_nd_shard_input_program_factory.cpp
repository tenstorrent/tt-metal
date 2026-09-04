// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_nd_shard_input_program_factory.hpp"

#include <algorithm>
#include <filesystem>

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/experimental/quasar/untilize/device/untilize_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

// Metal 2.0 port (nd-sharded input -> row-major output). Normal (allocated) double-buffered DFBs: a
// block reader streams the nd shards in, compute untilizes, and the writer scatters rows out over the
// NoC. The writer binds BOTH tensors (it walks the input shard page map to place output pages). Host
// work-distribution / page-mapping is preserved verbatim.
ttnn::device_operation::ProgramArtifacts UntilizeMultiCoreNDShardInputProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t tensor_width = a.padded_shape()[-1];
    uint32_t output_tensor_width = output.padded_shape()[-1];
    uint32_t output_tensor_height = output.physical_volume() / output_tensor_width;
    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    uint32_t num_tiles_per_input_row = tensor_width / tile_width;

    const auto& nd_shard_spec = a.nd_shard_spec().value();
    uint32_t input_shard_height = nd_shard_spec.shard_shape[-2];
    uint32_t input_shard_width = nd_shard_spec.shard_shape[-1];

    const auto distribution_spec = a.buffer()->buffer_distribution_spec().value();

    uint32_t num_shards = distribution_spec.num_shards();
    const auto page_mapping = distribution_spec.compute_page_mapping();
    const auto& groups = distribution_spec.core_groups();
    const auto& ordered_cores_with_data = get_optimal_worker_cores_for_sharded_tensor(a);
    uint32_t num_compute_cores = ordered_cores_with_data.size();
    const auto compute_core_range = CoreRangeSet(ttsl::Span<const CoreCoord>(ordered_cores_with_data));

    uint32_t num_tiles_per_input_block = input_shard_width / tile_width;
    uint32_t num_blocks_per_shard_plane = input_shard_height / tile_height;
    const auto& shard_shape = nd_shard_spec.shard_shape;
    size_t num_planes_per_shard = 1;
    if (shard_shape.rank() > 2) {
        for (int i = 0; i < static_cast<int>(shard_shape.rank()) - 2; ++i) {
            num_planes_per_shard *= shard_shape[i];
        }
    }
    uint32_t num_blocks_per_shard = num_planes_per_shard * num_blocks_per_shard_plane;
    uint32_t num_input_blocks_per_full_core = groups.num_shards_per_core_in_group_1 * num_blocks_per_shard;

    uint32_t input_cb_num_tiles =
        (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;
    uint32_t output_cb_num_tiles =
        (num_input_blocks_per_full_core == 1) ? num_tiles_per_input_block : num_tiles_per_input_block * 2;

    // Writer geometry.
    uint32_t output_element_size = output.element_size();
    uint32_t output_page_width = output_tensor_width;
    uint32_t output_num_blocks_across_width = 1;
    if (output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
        output.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        if (output.shard_spec().has_value()) {
            output_page_width = output.shard_spec().value().shape[1];
        } else {
            output_page_width = output.nd_shard_spec().value().shard_shape[-1];
        }
        output_num_blocks_across_width = tt::div_up(output_tensor_width, output_page_width);
    }
    uint32_t num_cols_per_input_block = num_tiles_per_input_block * tile_width;
    uint32_t num_cols_per_output_block = output_page_width;

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};    // legacy c_0
    const DFBSpecName OUT_DFB{"out"};  // legacy c_16
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = input_cb_num_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = output_cb_num_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    const std::filesystem::path kdir("ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/kernels/");

    KernelSpec reader{
        .unique_id = READER,
        .source = kdir / "dataflow/reader_unary_nd_sharded_blocks_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"num_tiles_per_input_block", num_tiles_per_input_block},
             {"num_shards", num_shards},
             {"num_cores", num_compute_cores}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_shard_id"}},
        .hw_config =
            ttnn::create_reader_datamovement_config(a.device()->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = kdir / "dataflow/writer_unary_stick_layout_split_rows_multi_core_nd_shard_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
             TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"tile_height", tile_height},
             {"num_tiles_per_input_block", num_tiles_per_input_block},
             {"num_output_blocks_across_width", output_num_blocks_across_width},
             {"output_element_size", output_element_size},
             {"num_cols_per_input_block", num_cols_per_input_block},
             {"num_cols_per_output_block", num_cols_per_output_block},
             {"num_shards", num_shards},
             {"num_cores", num_compute_cores},
             {"num_tiles_per_row", num_tiles_per_input_row},
             {"tile_width", tile_width},
             {"output_tensor_width", output_tensor_width},
             {"output_tensor_height", output_tensor_height}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_shard_id"}},
        .hw_config =
            ttnn::create_writer_datamovement_config(a.device()->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    ttnn::ComputeKernelConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(a.device()->arch(), compute_config);
    if (fp32_dest_acc_en) {
        std::visit(
            [&](auto& c) { c.unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest); }, compute_hw);
    }
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kdir / "compute/untilize_variable_num_blocks_metal2.cpp",
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args = {{"per_core_block_tile_cnt", num_tiles_per_input_block}},
        .runtime_arg_schema = {.runtime_arg_names = {"per_core_block_cnt"}},
        .hw_config = compute_hw,
    };

    Group<KernelSpec> kernels = {reader, writer, compute};
    Group<WorkUnitSpec> work_units = {WorkUnitSpec{
        .name = "untilize_nd_shard_input", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = compute_core_range}};

    // Per-core runtime args. Count non-padding blocks per core from the page mapping.
    const auto& mapped_cores = page_mapping.all_cores;
    KernelRunArgs::RuntimeArgValues reader_node_args;
    KernelRunArgs::RuntimeArgValues writer_node_args;
    KernelRunArgs::RuntimeArgValues compute_node_args;
    uint32_t start_shard_id = 0;
    for (const auto& core : ordered_cores_with_data) {
        auto core_it = std::find(mapped_cores.begin(), mapped_cores.end(), core);
        uint32_t num_input_blocks_to_process = 0;
        if (core_it != mapped_cores.end()) {
            const size_t core_idx = std::distance(mapped_cores.begin(), core_it);
            const auto& host_page_indices = page_mapping.core_host_page_indices[core_idx];
            uint32_t page_offset = 0;
            const uint32_t total_pages = host_page_indices.size();
            while (page_offset < total_pages) {
                if (host_page_indices[page_offset] != UncompressedBufferPageMapping::PADDING) {
                    num_input_blocks_to_process++;
                } else if (page_offset == 0) {
                    break;
                }
                page_offset += num_tiles_per_input_block;
            }
        }

        reader_node_args["start_shard_id"][core] = start_shard_id;
        writer_node_args["start_shard_id"][core] = start_shard_id;
        compute_node_args["per_core_block_cnt"][core] = num_input_blocks_to_process;
        start_shard_id++;
    }

    ProgramSpec spec{
        .name = "untilize_nd_shard_input",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_node_args)},
        KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_node_args)},
        KernelRunArgs{.kernel = COMPUTE, .runtime_arg_values = std::move(compute_node_args)}};
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
