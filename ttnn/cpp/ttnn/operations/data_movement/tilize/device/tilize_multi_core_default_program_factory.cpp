// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_default_program_factory.hpp"

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TilizeMultiCoreDefaultProgramFactory::create_program_artifacts(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);
    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto logical_shape = a.logical_shape();
    uint32_t logical_width = logical_shape[-1];
    uint32_t ntiles_per_block = tt::div_up(logical_width, TILE_WIDTH);
    uint32_t ntiles = dst_buffer->num_pages();
    uint32_t nblocks = tt::div_up(ntiles, ntiles_per_block);
    auto* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, nblocks);

    // Resource names
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};
    const DFBSpecName IN{"in"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    // Dataflow buffers (input c_0, output c_16). One size across all_cores.
    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = input_single_tile_size,
        .num_entries = ntiles_per_block,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = ntiles_per_block,
        .data_format_metadata = output_cb_data_format,
    };

    // reader compile-time args (see legacy: aligned_page_size CTA slot 0 was dead and drops away)
    uint32_t page_size = src0_buffer->page_size();
    uint32_t num_pages_in_row = 1;
    uint32_t size_of_valid_data_in_last_page_in_row = page_size;
    if (a.is_sharded()) {
        uint32_t shard_width =
            a.shard_spec().has_value() ? a.shard_spec().value().shape[1] : a.nd_shard_spec().value().shard_shape[-1];
        num_pages_in_row = tt::div_up(logical_width,
                                      shard_width);  // Compute number of pages in one tensor row.
        uint32_t padding_size =
            (num_pages_in_row * page_size) -
            (a.logical_shape()[-1] * a.element_size());  // Compute padding size for the last page in the row.
        size_of_valid_data_in_last_page_in_row = page_size - padding_size;
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
            "reader_unary_stick_layout_split_rows_multicore.cpp",
        .dfb_bindings = {{.dfb_spec_name = IN, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"num_pages_in_row", num_pages_in_row},
             {"size_of_valid_data_in_last_page_in_row", size_of_valid_data_in_last_page_in_row}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_rows", "num_tiles_per_block", "block_width_size", "num_full_blocks_in_row", "start_page_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // compute hw config: mirrors legacy ComputeConfigDescriptor{ .fp32_dest_acc_en, .unpack_to_dest_mode }.
    // All other ComputeConfigDescriptor defaults coincide with ComputeGen1Config defaults.
    auto make_compute_hw = [&]() {
        ComputeGen1Config cfg{.enable_32_bit_dest = fp32_llk_acc};
        if (fp32_llk_acc) {
            cfg.unpack_modes = {{IN, UnpackMode::UnpackToDest}};  // legacy unpack_to_dest_mode[c_0] = UnpackToDestFp32
        }
        return ComputeHardwareConfig{cfg};
    };

    const char* compute_source = "ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp";

    bool has_full = !core_range.ranges().empty();
    bool has_cliff = !core_range_cliff.empty();

    KernelSpec compute_full{
        .unique_id = COMPUTE_FULL,
        .source = compute_source,
        .dfb_bindings =
            {{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             {.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args = {{"per_core_block_cnt", nblocks_per_core}, {"per_core_block_tile_cnt", ntiles_per_block}},
        .hw_config = make_compute_hw(),
    };
    KernelSpec compute_cliff{
        .unique_id = COMPUTE_CLIFF,
        .source = compute_source,
        .dfb_bindings =
            {{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             {.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"per_core_block_cnt", nblocks_per_core_cliff}, {"per_core_block_tile_cnt", ntiles_per_block}},
        .hw_config = make_compute_hw(),
    };

    // Assemble kernels + work units.
    Group<KernelSpec> kernels;
    kernels.push_back(reader);
    kernels.push_back(writer);
    Group<WorkUnitSpec> work_units;
    if (has_full) {
        kernels.push_back(compute_full);
        work_units.push_back(
            WorkUnitSpec{.name = "full", .kernels = {READER, WRITER, COMPUTE_FULL}, .target_nodes = core_range});
    }
    if (has_cliff) {
        kernels.push_back(compute_cliff);
        work_units.push_back(WorkUnitSpec{
            .name = "cliff", .kernels = {READER, WRITER, COMPUTE_CLIFF}, .target_nodes = core_range_cliff});
    }

    // Runtime args — keep the legacy node-first loop; AddRuntimeArgsForNode transposes into name-first.
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};

    uint32_t ncores_full = ncores - has_cliff;
    uint32_t tile_start_id = 0;
    uint32_t page_start_id = 0;
    const auto& cores = corerange_to_cores(available_grid);
    for (uint32_t i = 0; i < ncores_full; ++i) {
        const CoreCoord& core = cores[i];
        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core,
            {{"num_rows", nblocks_per_core * TILE_HEIGHT},
             {"num_tiles_per_block", ntiles_per_block},
             {"block_width_size", page_size},
             {"num_full_blocks_in_row", 1u},
             {"start_page_id", page_start_id}});
        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values,
            core,
            {{"num_pages", ntiles_per_block * nblocks_per_core}, {"start_id", tile_start_id}});
        tile_start_id += ntiles_per_block * nblocks_per_core;
        page_start_id += TILE_HEIGHT * nblocks_per_core * num_pages_in_row;
    }
    if (has_cliff) {
        const CoreCoord& core = cores[ncores_full];
        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core,
            {{"num_rows", nblocks_per_core_cliff * TILE_HEIGHT},
             {"num_tiles_per_block", ntiles_per_block},
             {"block_width_size", page_size},
             {"num_full_blocks_in_row", 1u},
             {"start_page_id", page_start_id}});
        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values,
            core,
            {{"num_pages", ntiles_per_block * nblocks_per_core_cliff}, {"start_id", tile_start_id}});
    }

    ProgramSpec spec{
        .name = "tilize_multi_core",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters =
            {{.unique_id = INPUT, .spec = a.tensor_spec()}, {.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_ra), std::move(writer_ra)};
    run_args.tensor_args = {
        {INPUT, TensorArgument{a.mesh_tensor()}},
        {OUTPUT, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
