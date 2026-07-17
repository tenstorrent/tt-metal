// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_default_program_factory.hpp"

#include <cmath>
#include <filesystem>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/experimental/quasar/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts TilizeWithValPaddingMultiCoreDefaultFactory::create_program_artifacts(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor, Tensor& tensor_return_value) {
    const Tensor& a = input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid =
        operation_attributes.sub_core_grids.has_value() ? operation_attributes.sub_core_grids.value() : default_grid;
    uint32_t tile_width = output.tensor_spec().tile().get_width();
    uint32_t tile_height = output.tensor_spec().tile().get_height();
    uint32_t num_blocks = output.physical_volume() / output.padded_shape()[-1] / tile_height;
    uint32_t num_tiles_per_row = output.padded_shape()[-1] / tile_width;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_blocks);

    bool has_cliff = !core_range_cliff.empty();

    uint32_t unpadded_row_size_bytes = a.logical_shape()[-1] * a.element_size();    // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = output.padded_shape()[-1] * a.element_size();  // Assuming bfloat16 dataformat

    // ---- reader compile-time values (mirror legacy) ----
    uint32_t packed_pad_value = detail::get_packed_value(a, operation_attributes.pad_value);
    // log2(tile_height * data_format_size_in_bytes)
    uint32_t shift_bits = static_cast<uint32_t>(std::log2(a.element_size() * tile_height));
    uint32_t elem_size = a.element_size();
    uint32_t num_pages_in_row = 1;
    uint32_t page_size = a.logical_shape()[-1] * a.element_size();
    uint32_t size_of_valid_data_in_last_page_in_row = a.logical_shape()[-1] * a.element_size();
    if (a.is_sharded() && a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        page_size = a.buffer()->page_size();
        uint32_t shard_width =
            a.shard_spec().has_value() ? a.shard_spec().value().shape[1] : a.nd_shard_spec().value().shard_shape[-1];
        num_pages_in_row = tt::div_up(a.logical_shape()[-1], shard_width);
        size_of_valid_data_in_last_page_in_row = unpadded_row_size_bytes - (num_pages_in_row - 1) * page_size;
    }

    // ---- Resource names ----
    const DFBSpecName IN{"in"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};

    // ---- DataflowBuffers (legacy CB c_0 / c_16) ----
    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = output_cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Reader (CTAs named; per-BlockRep groups are runtime varargs) ----
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/tilize_with_val_padding/device/kernels/dataflow/"
            "reader_unary_pad_dims_split_rows_multicore.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"tile_row_shift_bits", shift_bits},
             {"unpadded_X_size", unpadded_row_size_bytes},
             {"elem_size", elem_size},
             {"num_pages_in_row", num_pages_in_row},
             {"page_size", page_size},
             {"size_of_valid_data_in_last_page_in_row", size_of_valid_data_in_last_page_in_row}},
        .runtime_arg_schema = {.runtime_arg_names = {"padded_X_size", "pad_value", "start_page_id", "n_block_reps"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ---- Writer (Metal 2.0 fork) ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/tilize_with_val_padding/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ---- Compute (Metal 2.0 fork; full + cliff) ----
    auto make_compute_hw = [&]() -> ComputeHardwareConfig {
        ttnn::ComputeKernelConfig hw{
            .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_llk_acc};
        ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(device->arch(), hw);
        if (fp32_llk_acc) {
            std::visit(
                [&](auto& c) { c.unpack_modes.emplace(IN, tt::tt_metal::UnpackMode::UnpackToDest); }, compute_hw);
        }
        return compute_hw;
    };
    const std::filesystem::path compute_source(
        "ttnn/cpp/ttnn/operations/experimental/quasar/tilize_with_val_padding/device/kernels/compute/"
        "tilize_metal2.cpp");
    auto make_compute = [&](const KernelSpecName& id, uint32_t nblocks) {
        return KernelSpec{
            .unique_id = id,
            .source = compute_source,
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args = {{"per_core_block_cnt", nblocks}, {"per_core_block_tile_cnt", num_tiles_per_row}},
            .hw_config = make_compute_hw(),
        };
    };

    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;
    if (!core_range.empty()) {
        kernels.push_back(make_compute(COMPUTE_FULL, nblocks_per_core));
        work_units.push_back(
            WorkUnitSpec{.name = "wu_full", .kernels = {READER, WRITER, COMPUTE_FULL}, .target_nodes = core_range});
    }
    if (has_cliff) {
        kernels.push_back(make_compute(COMPUTE_CLIFF, nblocks_per_core_cliff));
        work_units.push_back(WorkUnitSpec{
            .name = "wu_cliff", .kernels = {READER, WRITER, COMPUTE_CLIFF}, .target_nodes = core_range_cliff});
    }

    // ---- Per-core runtime args (mirror legacy 1D block distribution) ----
    auto core_assignments = ttnn::distribute_work(
        output.logical_shape(),
        output.padded_shape(),
        ncores,
        nblocks_per_core,
        has_cliff,
        nblocks_per_core_cliff,
        tile_height);

    uint32_t tile_start_id = 0;
    uint32_t start_page_id = 0;
    const auto cores = corerange_to_cores(available_grid);

    KernelRunArgs::RuntimeArgValues reader_node_args;
    KernelRunArgs::RuntimeArgValues writer_node_args;
    Table<NodeCoord, AdvancedKernelRunArgs::Varargs> reader_varargs;
    uint32_t max_varargs = 0;

    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        const uint32_t core_start_page_id = start_page_id;

        // Build the per-BlockRep vararg tail (run-length encoded, mirroring the legacy prefix).
        AdvancedKernelRunArgs::Varargs reader_tail;
        uint32_t nblocks_per_core_local = 0;
        BlockRep ref_el = assignment[0];
        uint32_t count_repeated = 0;  // incremented on the first iteration
        for (const auto& el : assignment) {
            nblocks_per_core_local += el.block_count();
            start_page_id += el.data_row_count() * num_pages_in_row;
            if (compare_assignments(ref_el, el)) {
                count_repeated++;
            } else {
                reader_tail.push_back(ref_el.n_data);
                reader_tail.push_back(ref_el.n_mixed);
                reader_tail.push_back(ref_el.n_pads);
                reader_tail.push_back(ref_el.times);
                reader_tail.push_back(count_repeated);
                ref_el = el;
                count_repeated = 1;
            }
        }
        reader_tail.push_back(ref_el.n_data);
        reader_tail.push_back(ref_el.n_mixed);
        reader_tail.push_back(ref_el.n_pads);
        reader_tail.push_back(ref_el.times);
        reader_tail.push_back(count_repeated);

        const uint32_t n_block_reps = static_cast<uint32_t>(assignment.size());
        max_varargs = std::max<uint32_t>(max_varargs, static_cast<uint32_t>(reader_tail.size()));

        AddRuntimeArgsForNode(
            reader_node_args,
            core,
            {
                {"padded_X_size", padded_row_size_bytes},
                {"pad_value", packed_pad_value},
                {"start_page_id", core_start_page_id},
                {"n_block_reps", n_block_reps},
            });
        reader_varargs.emplace(core, std::move(reader_tail));

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core_local;
        AddRuntimeArgsForNode(
            writer_node_args,
            core,
            {
                {"num_pages", num_tiles_per_core},
                {"start_id", tile_start_id},
            });
        tile_start_id += num_tiles_per_core;
    }

    // Every node must supply exactly num_runtime_varargs words; pad each core's vararg vector up to
    // the max with zeros (never read — the reader loop is bounded by n_block_reps).
    for (auto& entry : reader_varargs) {
        entry.second.resize(max_varargs, 0u);
    }
    // Reader is kernels[0]; declare its uniform vararg count.
    kernels[0].advanced_options.num_runtime_varargs = max_varargs;

    ProgramSpec spec{
        .name = "tilize_with_val_padding_multi_core_default",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    KernelRunArgs reader_args{.kernel = READER, .runtime_arg_values = std::move(reader_node_args)};
    reader_args.advanced_options.runtime_varargs = std::move(reader_varargs);
    KernelRunArgs writer_args{.kernel = WRITER, .runtime_arg_values = std::move(writer_node_args)};
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args)};
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
