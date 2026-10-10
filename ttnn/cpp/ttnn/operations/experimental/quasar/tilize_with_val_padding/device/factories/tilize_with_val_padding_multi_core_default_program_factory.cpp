// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_default_program_factory.hpp"

#include <cmath>
#include <filesystem>
#include <optional>

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

namespace {

// Tilizes a row-major interleaved input into a tile interleaved output with 4 reader, 4 compute and 2
// writer threads, the data-movement ones with implicit sync. Each compute thread ("lane") tilizes a
// slice of the blocks (tile rows): with column lanes a quarter of every block's tiles, otherwise every
// 4th block whole. Reader thread t feeds lane t one lane-wide row segment per entry; the strided DFB
// places a lane's rows 4 slots apart, which the unpacker's row stride (full_ct_dim) follows. Rows past
// the input are read from a scratch row of pad values. Width padding is not supported, and height
// padding must sit in the last tile row. Returns std::nullopt when the tensors do not fit this path.
std::optional<ttnn::device_operation::ProgramArtifacts> create_tilize_split_program(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& a, const Tensor& output) {
    constexpr uint32_t num_lanes = 4;
    constexpr uint32_t num_writer_threads = 2;
    const bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                              output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;
    if (a.is_sharded() || output.is_sharded() || operation_attributes.sub_core_grids.has_value() || fp32_llk_acc ||
        a.element_size() != 2) {
        return std::nullopt;
    }
    const uint32_t tile_width = output.tensor_spec().tile().get_width();
    const uint32_t tile_height = output.tensor_spec().tile().get_height();
    const uint32_t width = output.padded_shape()[-1];
    const uint32_t row_bytes = width * a.element_size();
    if (a.logical_shape()[-1] != width || a.buffer()->page_size() != row_bytes) {
        return std::nullopt;
    }
    const uint32_t num_blocks = output.physical_volume() / width / tile_height;
    const uint32_t tiles_per_row = width / tile_width;
    // Padded row r is input row r: padding rows exist only past the last input row, which needs
    // either no padding or a single leading-dim slice.
    const uint32_t num_input_rows = a.logical_volume() / width;
    const uint32_t num_slices = output.physical_volume() / (output.padded_shape()[-2] * width);
    if (num_input_rows != num_blocks * tile_height &&
        !(num_slices == 1 && num_input_rows + tile_height > num_blocks * tile_height)) {
        return std::nullopt;
    }
    // Column lanes split every block's tiles four ways; otherwise each lane takes whole blocks, which
    // needs a multiple of 4 blocks per core. The output ring holds two blocks per lane, and implicit
    // sync keeps half of it per txn ID in a uint8.
    const bool column_lanes = tiles_per_row % num_lanes == 0;
    const uint32_t lane_tiles = column_lanes ? tiles_per_row / num_lanes : tiles_per_row;
    const uint32_t lane_bytes = lane_tiles * tile_width * a.element_size();
    if ((!column_lanes && num_blocks % num_lanes != 0) || num_lanes * lane_tiles > 255) {
        return std::nullopt;
    }
    const uint32_t blocks_per_unit = column_lanes ? 1 : num_lanes;

    MeshDevice* device = a.device();
    auto [ncores, all_cores, full_core_range, cliff_core_range, units_per_core, units_per_core_cliff] =
        ttnn::split_blocks_for_tilize(device->compute_with_storage_grid_size(), num_blocks / blocks_per_unit);

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const DFBSpecName IN{"in"};
    const DFBSpecName OUT{"out"};
    const ScratchpadSpecName PAD{"pad_row"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // One block of rows per lane in; two blocks of tiles per lane out.
    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = lane_bytes,
        .num_entries = num_lanes * tile_height,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = tt::tile_size(output_cb_data_format),
        .num_entries = 2 * num_lanes * lane_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    const std::filesystem::path kdir(
        "ttnn/cpp/ttnn/operations/experimental/quasar/tilize_with_val_padding/device/kernels/");
    KernelSpec::CompilerOptions::Defines dm_defines;
    dm_defines.emplace("COLUMN_LANES", column_lanes ? "1" : "0");
    KernelSpec reader{
        .unique_id = READER,
        .source = kdir / "dataflow/reader_unary_pad_rows_staged.cpp",
        .num_threads = num_lanes,
        .compiler_options = {.defines = dm_defines},
        .dfb_bindings = {ProducerOf(IN, "in")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args = {{"tile_height", tile_height}, {"lane_bytes", lane_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = {"first_block", "num_lane_blocks", "num_input_rows", "pad_value"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    reader.scratchpad_bindings = {ScratchpadBinding{.scratchpad_spec_name = PAD, .accessor_name = "pad"}};
    KernelSpec writer{
        .unique_id = WRITER,
        .source = kdir / "dataflow/writer_unary_staged_tiles_metal2.cpp",
        .num_threads = num_writer_threads,
        .compiler_options = {.defines = dm_defines},
        .dfb_bindings = {ConsumerOf(OUT, "out")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"tiles_per_row", tiles_per_row}, {"lane_tiles", lane_tiles}, {"num_lanes", num_lanes}},
        .runtime_arg_schema = {.runtime_arg_names = {"first_block", "num_lane_blocks"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kdir / "compute/tilize_lanes_metal2.cpp",
        .num_threads = num_lanes,
        .dfb_bindings = {ConsumerOf(IN, "in"), ProducerOf(OUT, "out")},
        .compile_time_args =
            {{"lane_tiles", lane_tiles},
             {"full_ct_dim", column_lanes ? tiles_per_row : num_lanes * tiles_per_row},
             {"block_rows", tile_height}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_lane_blocks"}},
        .hw_config = ttnn::to_compute_hardware_config(
            ttnn::ComputeKernelConfig{.math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false}),
    };

    const uint32_t packed_pad_value = detail::get_packed_value(a, operation_attributes.pad_value);
    KernelRunArgs::RuntimeArgValues reader_args;
    KernelRunArgs::RuntimeArgValues writer_args;
    KernelRunArgs::RuntimeArgValues compute_args;
    uint32_t first_block = 0;
    auto add_core = [&](const CoreCoord& core, uint32_t num_units) {
        const uint32_t num_lane_blocks = num_units;
        AddRuntimeArgsForNode(
            reader_args,
            core,
            {{"first_block", first_block},
             {"num_lane_blocks", num_lane_blocks},
             {"num_input_rows", num_input_rows},
             {"pad_value", packed_pad_value}});
        AddRuntimeArgsForNode(writer_args, core, {{"first_block", first_block}, {"num_lane_blocks", num_lane_blocks}});
        compute_args["num_lane_blocks"][core] = num_lane_blocks;
        first_block += num_units * blocks_per_unit;
    };
    for (const auto& core : corerange_to_cores(full_core_range, std::nullopt, true)) {
        add_core(core, units_per_core);
    }
    for (const auto& core : corerange_to_cores(cliff_core_range, std::nullopt, true)) {
        add_core(core, units_per_core_cliff);
    }

    ProgramSpec spec{
        .name = "tilize_with_val_padding_lanes",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()},
             TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {WorkUnitSpec{
            .name = "tilize_lanes", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}},
    };
    spec.scratchpads = {ScratchpadSpec{.unique_id = PAD, .size_per_node = num_lanes * lane_bytes}};
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_args)},
        KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_args)},
        KernelRunArgs{.kernel = COMPUTE, .runtime_arg_values = std::move(compute_args)}};
    run_args.tensor_args = {{INPUT, a.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TilizeWithValPaddingMultiCoreDefaultFactory::create_program_artifacts(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor, Tensor& tensor_return_value) {
    const Tensor& a = input_tensor;
    const Tensor& output = tensor_return_value;
    if (auto split = create_tilize_split_program(operation_attributes, a, output)) {
        return std::move(*split);
    }
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    MeshDevice* device = a.device();
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
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
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
        // Implicit sync: the kernel issues TXN_ID writes instead of wait_front / pop_front.
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // ---- Compute (Metal 2.0 fork; full + cliff) ----
    auto make_compute_hw = [&]() -> ComputeHardwareConfig {
        ttnn::ComputeKernelConfig hw{
            .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_llk_acc};
        ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(hw);
        if (fp32_llk_acc) {
            compute_hw.unpack_modes.emplace(IN, tt::tt_metal::UnpackMode::UnpackToDest);
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
