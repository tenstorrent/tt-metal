// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_default_program_factory.hpp"

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {
const TensorParamName MC_INPUT_TENSOR{"input"};
const TensorParamName MC_OUTPUT_TENSOR{"output"};
const DFBSpecName MC_INPUT_DFB{"in"};
const DFBSpecName MC_OUTPUT_DFB{"out"};
const KernelSpecName MC_READER_KERNEL{"reader"};
const KernelSpecName MC_WRITER_KERNEL{"writer"};
const KernelSpecName MC_COMPUTE_G1{"compute_g1"};
const KernelSpecName MC_COMPUTE_G2{"compute_g2"};
}  // namespace

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

    // reader page sizing
    uint32_t page_size = src0_buffer->page_size();
    uint32_t num_pages_in_row = 1;
    uint32_t size_of_valid_data_in_last_page_in_row = page_size;
    if (a.is_sharded()) {
        uint32_t shard_width =
            a.shard_spec().has_value() ? a.shard_spec().value().shape[1] : a.nd_shard_spec().value().shard_shape[-1];
        num_pages_in_row = tt::div_up(logical_width, shard_width);  // Compute number of pages in one tensor row.
        uint32_t padding_size = (num_pages_in_row * page_size) - (a.logical_shape()[-1] * a.element_size());
        size_of_valid_data_in_last_page_in_row = page_size - padding_size;
    }

    // -- Spec --
    ProgramSpec spec;
    spec.name = "tilize_multi_core_default";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = MC_INPUT_TENSOR, .spec = a.tensor_spec()},
        TensorParameter{.unique_id = MC_OUTPUT_TENSOR, .spec = output.tensor_spec()},
    };

    spec.dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = MC_INPUT_DFB,
            .entry_size = input_single_tile_size,
            .num_entries = ntiles_per_block,
            .data_format_metadata = input_cb_data_format,
        },
        DataflowBufferSpec{
            .unique_id = MC_OUTPUT_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = ntiles_per_block,
            .data_format_metadata = output_cb_data_format,
        },
    };

    // -- Reader kernel (spans all_cores) --
    KernelSpec reader{
        .unique_id = MC_READER_KERNEL,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/tilize/device/kernels/dataflow/"
            "reader_unary_stick_layout_split_rows_multicore.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = MC_INPUT_DFB,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = MC_INPUT_TENSOR, .accessor_name = "src"}},
        .compile_time_args =
            {{"num_pages_in_row", num_pages_in_row},
             {"size_of_valid_data_in_last_page_in_row", size_of_valid_data_in_last_page_in_row}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_rows", "num_tiles_per_block", "block_width_size", "num_full_blocks_in_row", "start_page_id"}},
        .hw_config =
            ttnn::create_reader_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // -- Writer kernel (spans all_cores) --
    KernelSpec writer{
        .unique_id = MC_WRITER_KERNEL,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/tilize/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = MC_OUTPUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = MC_OUTPUT_TENSOR, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config =
            ttnn::create_writer_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // -- Compute kernels (preserved multiplicity: per-group CTAs) --
    ttnn::ComputeKernelConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_llk_acc};
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_config);
    if (fp32_llk_acc) {
        std::visit([&](auto& c) { c.unpack_modes.emplace(MC_INPUT_DFB, UnpackMode::UnpackToDest); }, compute_hw);
    }
    const char* compute_src = "ttnn/cpp/ttnn/operations/experimental/quasar/tilize/device/kernels/compute/tilize.cpp";
    auto make_compute = [&](const KernelSpecName& id, uint32_t nblocks_per_core_arg) {
        return KernelSpec{
            .unique_id = id,
            .source = compute_src,
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = MC_INPUT_DFB,
                     .accessor_name = "in",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = MC_OUTPUT_DFB,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 }},
            .compile_time_args =
                {{"per_core_block_cnt", nblocks_per_core_arg}, {"per_core_block_tile_cnt", ntiles_per_block}},
            .hw_config = compute_hw};
    };

    bool has_cliff = !core_range_cliff.empty();
    bool has_full = !core_range.ranges().empty();

    spec.kernels = {reader, writer};
    spec.work_units = {};
    if (has_full) {
        spec.kernels.push_back(make_compute(MC_COMPUTE_G1, nblocks_per_core));
        spec.work_units.push_back(WorkUnitSpec{
            .name = "tilize_default_wu_g1",
            .kernels = {MC_READER_KERNEL, MC_WRITER_KERNEL, MC_COMPUTE_G1},
            .target_nodes = core_range,
        });
    }
    if (has_cliff) {
        spec.kernels.push_back(make_compute(MC_COMPUTE_G2, nblocks_per_core_cliff));
        spec.work_units.push_back(WorkUnitSpec{
            .name = "tilize_default_wu_g2",
            .kernels = {MC_READER_KERNEL, MC_WRITER_KERNEL, MC_COMPUTE_G2},
            .target_nodes = core_range_cliff,
        });
    }

    // -- Run args (per-core, 1D distribution of blocks across cores) --
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = MC_READER_KERNEL};
    KernelRunArgs writer_run{.kernel = MC_WRITER_KERNEL};

    uint32_t ncores_full = ncores - has_cliff;
    uint32_t tile_start_id = 0;
    uint32_t page_start_id = 0;
    const auto& cores = corerange_to_cores(available_grid);
    KernelRunArgs::RuntimeArgValues& reader_rtas = reader_run.runtime_arg_values;
    KernelRunArgs::RuntimeArgValues& writer_rtas = writer_run.runtime_arg_values;
    for (uint32_t i = 0; i < ncores_full; ++i) {
        const CoreCoord& core = cores[i];
        AddRuntimeArgsForNode(
            reader_rtas,
            core,
            {
                {"num_rows", nblocks_per_core * TILE_HEIGHT},
                {"num_tiles_per_block", ntiles_per_block},
                {"block_width_size", page_size},
                {"num_full_blocks_in_row", 1u},
                {"start_page_id", page_start_id},
            });
        AddRuntimeArgsForNode(
            writer_rtas,
            core,
            {
                {"num_pages", ntiles_per_block * nblocks_per_core},
                {"start_id", tile_start_id},
            });
        tile_start_id += ntiles_per_block * nblocks_per_core;
        page_start_id += TILE_HEIGHT * nblocks_per_core * num_pages_in_row;
    }
    if (has_cliff) {
        const CoreCoord& core = cores[ncores_full];
        AddRuntimeArgsForNode(
            reader_rtas,
            core,
            {
                {"num_rows", nblocks_per_core_cliff * TILE_HEIGHT},
                {"num_tiles_per_block", ntiles_per_block},
                {"block_width_size", page_size},
                {"num_full_blocks_in_row", 1u},
                {"start_page_id", page_start_id},
            });
        AddRuntimeArgsForNode(
            writer_rtas,
            core,
            {
                {"num_pages", ntiles_per_block * nblocks_per_core_cliff},
                {"start_id", tile_start_id},
            });
    }

    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {MC_INPUT_TENSOR, TensorArgument{a.mesh_tensor()}},
        {MC_OUTPUT_TENSOR, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
