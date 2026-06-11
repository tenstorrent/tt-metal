// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_default_program_factory.hpp"

#include <filesystem>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental;

namespace {

// File-scope kernel paths. Names are unique across the tilize device/ sibling .cpp files to avoid
// unity-build collisions.
constexpr const char* MCD_READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
    "reader_unary_stick_layout_split_rows_multicore_m2.cpp";
constexpr const char* MCD_WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
    "writer_unary_interleaved_start_id_m2.cpp";
constexpr const char* MCD_COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_m2.cpp";

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

    const m2::DFBSpecName src0_dfb{"cb_id_in0"};
    const m2::DFBSpecName out0_dfb{"cb_id_out0"};

    m2::ProgramSpec spec;
    spec.name = "tilize_multi_core_default";

    // DFBs: src0 (formerly CB c_0) produced by the reader / consumed by compute; out0 (formerly CB c_16)
    // produced by compute / consumed by the writer. Both L1-allocated, one block deep.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = src0_dfb,
            .entry_size = input_single_tile_size,
            .num_entries = ntiles_per_block,
            .data_format_metadata = input_cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = out0_dfb,
            .entry_size = output_single_tile_size,
            .num_entries = ntiles_per_block,
            .data_format_metadata = output_cb_data_format,
        }};

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};

    /** reader compile-time args.
     * select_program_factory routes interleaved input here AND sharded input that can't use the optimized
     * sharded factories, so we must preserve the legacy width-split: a sharded row can span several pages
     * (num_pages_in_row) whose final page may carry width-padding (size_of_valid_data_in_last_page_in_row).
     * For interleaved input each row is exactly one full page (the values below stay 1 / page_size).
     */
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

    // Reader on NCRISC (RISCV_1 / NOC1), writer on BRISC.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{MCD_READER_KERNEL_PATH},
        .dfb_bindings = {m2::ProducerOf(src0_dfb, "cb_id_in0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src_args"}},
        .compile_time_args =
            {{"num_pages_in_row", num_pages_in_row},
             {"size_of_valid_data_in_last_page_in_row", size_of_valid_data_in_last_page_in_row}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_rows", "num_tiles_per_block", "block_width_size", "num_full_blocks_in_row", "start_page_id"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{MCD_WRITER_KERNEL_PATH},
        .dfb_bindings = {m2::ConsumerOf(out0_dfb, "output_cb")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    auto make_compute_hw = [&]() {
        m2::ComputeHardwareConfig hw{.fp32_dest_acc_en = fp32_llk_acc};
        // Legacy set UnpackToDestFp32 on c_0 whenever fp32_llk_acc; only valid for Float32 input DFB.
        if (fp32_llk_acc && input_cb_data_format == tt::DataFormat::Float32) {
            hw.unpack_to_dest_mode.emplace(src0_dfb, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32);
        }
        return hw;
    };
    auto make_compute = [&](const m2::KernelSpecName& name, uint32_t per_core_block_cnt) {
        return m2::KernelSpec{
            .unique_id = name,
            .source = std::filesystem::path{MCD_COMPUTE_KERNEL_PATH},
            .dfb_bindings = {m2::ConsumerOf(src0_dfb, "cb_id_in0"), m2::ProducerOf(out0_dfb, "cb_id_out0")},
            .compile_time_args =
                {{"per_core_block_cnt", per_core_block_cnt}, {"per_core_block_tile_cnt", ntiles_per_block}},
            .hw_config = make_compute_hw(),
        };
    };

    spec.kernels = {std::move(reader), std::move(writer)};

    // Local DFBs require their producer AND consumer to share a WorkUnitSpec. reader→src0→compute→out0→writer
    // is one chain, so reader/writer co-locate with the compute kernel of each core group (a KernelSpec may
    // belong to multiple WorkUnitSpecs — the reader/writer specs are reused across both groups).
    const bool has_full = !core_range.ranges().empty();
    const bool has_cliff = !core_range_cliff.empty();

    if (has_full) {
        spec.kernels.push_back(make_compute(m2::KernelSpecName{"compute_full"}, nblocks_per_core));
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "tilize_default_full",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute_full"}},
            .target_nodes = core_range});
    }
    if (has_cliff) {
        spec.kernels.push_back(make_compute(m2::KernelSpecName{"compute_cliff"}, nblocks_per_core_cliff));
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "tilize_default_cliff",
            .kernels =
                {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute_cliff"}},
            .target_nodes = core_range_cliff});
    }

    // 1D distribution of blocks across cores. Mirror the legacy per-core loop exactly.
    m2::ProgramRunArgs run_args;
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    uint32_t ncores_full = ncores - has_cliff;
    uint32_t tile_start_id = 0;
    uint32_t page_start_id = 0;
    const auto& cores = corerange_to_cores(available_grid);
    for (uint32_t i = 0; i < ncores_full; ++i) {
        const CoreCoord& core = cores[i];
        reader_args.runtime_arg_values.push_back(
            {core,
             {{"num_rows", nblocks_per_core * TILE_HEIGHT},
              {"num_tiles_per_block", ntiles_per_block},
              {"block_width_size", page_size},
              {"num_full_blocks_in_row", uint32_t{1}},
              {"start_page_id", page_start_id}}});
        writer_args.runtime_arg_values.push_back(
            {core, {{"num_pages", ntiles_per_block * nblocks_per_core}, {"start_id", tile_start_id}}});
        tile_start_id += ntiles_per_block * nblocks_per_core;
        page_start_id += TILE_HEIGHT * nblocks_per_core * num_pages_in_row;
    }
    if (has_cliff) {
        const CoreCoord& core = cores[ncores_full];
        reader_args.runtime_arg_values.push_back(
            {core,
             {{"num_rows", nblocks_per_core_cliff * TILE_HEIGHT},
              {"num_tiles_per_block", ntiles_per_block},
              {"block_width_size", page_size},
              {"num_full_blocks_in_row", uint32_t{1}},
              {"start_page_id", page_start_id}}});
        writer_args.runtime_arg_values.push_back(
            {core, {{"num_pages", ntiles_per_block * nblocks_per_core_cliff}, {"start_id", tile_start_id}}});
    }

    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));

    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(a.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
