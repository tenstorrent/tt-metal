// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_default_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

// Each BlockRep run is encoded as this many positional vararg words on the reader kernel.
// Layout (must match reader_unary_pad_dims_split_rows_multicore.cpp): {n_data, n_mixed, n_pads,
// times, repeat_count}.
static constexpr uint32_t kWordsPerBlockRepGroup = 5;

ttnn::device_operation::ProgramArtifacts TilizeWithValPaddingMultiCoreDefaultFactory::create_program_artifacts(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor, Tensor& tensor_return_value) {
    const Tensor& a = input_tensor;
    const Tensor& output = tensor_return_value;

    const auto& input_mesh = a.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    // Metal 2.0 named resource handles.
    const DFBSpecName SRC_DFB{"src"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_FULL_KERNEL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF_KERNEL{"compute_cliff"};

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

    bool has_cliff = !core_range_cliff.ranges().empty();

    uint32_t unpadded_row_size_bytes = a.logical_shape()[-1] * a.element_size();
    uint32_t padded_row_size_bytes = output.padded_shape()[-1] * a.element_size();

    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // ------------------------------------------------------------------------
    // Dataflow buffers (formerly CB c_0 / c_16). One DFB per legacy CB.
    // ------------------------------------------------------------------------
    DataflowBufferSpec src_dfb_spec{
        .unique_id = SRC_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = output_cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_mesh.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_mesh.tensor_spec()};

    // ------------------------------------------------------------------------
    // Reader compile-time args. The legacy positional slot for aligned_page_size was unused by the
    // kernel and is dropped. num_pages_in_row / size_of_valid_data_in_last_page_in_row describe the
    // (optionally ND-sharded) input row layout.
    // ------------------------------------------------------------------------
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

    uint32_t packed_pad_value = detail::get_packed_value(a, operation_attributes.pad_value);

    // ------------------------------------------------------------------------
    // Per-core runtime args. This is the factory that motivated the vararg port: the reader's
    // per-block descriptors form a run-length-encoded BlockRep stream whose LENGTH differs core to
    // core. Metal 2.0 named RTAs require a fixed schema (every node on a KernelSpec supplies the
    // same named args), so the variable stream cannot be a named RTA. Instead:
    //   - the fixed scalars (padded_X_size, pad_value, start_page_id, n_block_reps) stay named RTAs;
    //   - the variable BlockRep stream is passed as positional "varargs"
    //     (KernelAdvancedOptions::num_runtime_varargs), read in the kernel via get_vararg().
    // Metal 2.0's named-RTA validation requires every node to supply EXACTLY num_runtime_varargs
    // words, so we first compute each core's stream, take the MAX length, declare that as the
    // uniform vararg count, and zero-pad shorter cores. n_block_reps bounds the kernel loop so the
    // trailing pad is never read. (We deliberately avoid the deprecated per-node-vararg-count
    // override.)
    // ------------------------------------------------------------------------
    auto core_assignments = ttnn::distribute_work(
        output.logical_shape(),
        output.padded_shape(),
        ncores,
        nblocks_per_core,
        has_cliff,
        nblocks_per_core_cliff,
        tile_height);

    struct ReaderCoreArgs {
        NodeCoord node;
        uint32_t start_page_id;
        uint32_t n_block_reps;
        std::vector<uint32_t> block_rep_stream;  // run-length-encoded BlockRep groups (varargs)
    };
    struct WriterCoreArgs {
        NodeCoord node;
        uint32_t num_pages;
        uint32_t start_id;
    };

    std::vector<ReaderCoreArgs> reader_core_args;
    std::vector<WriterCoreArgs> writer_core_args;
    reader_core_args.reserve(ncores);
    writer_core_args.reserve(ncores);

    uint32_t tile_start_id = 0;
    uint32_t start_page_id = 0;
    uint32_t max_stream_words = 0;

    const auto cores = corerange_to_cores(available_grid);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<ttnn::BlockRep>& assignment = core_assignments.at(i);

        // start_page_id is captured at the start of this core's work; the inner loop then advances
        // it for the next core.
        const uint32_t core_start_page_id = start_page_id;

        // Run-length-encode the assignment into {n_data, n_mixed, n_pads, times, repeat_count}
        // groups (identical to the legacy factory, but emitted into a vararg vector instead of
        // appended to a flat RTA list).
        std::vector<uint32_t> stream;
        stream.reserve(assignment.size() * kWordsPerBlockRepGroup);
        uint32_t nblocks_per_core_local = 0;
        ttnn::BlockRep ref_el = assignment[0];
        uint32_t count_repeated = 0;
        for (const auto& el : assignment) {
            nblocks_per_core_local += el.block_count();
            start_page_id += el.data_row_count() * num_pages_in_row;
            if (ttnn::compare_assignments(ref_el, el)) {
                count_repeated++;
            } else {
                stream.push_back(ref_el.n_data);
                stream.push_back(ref_el.n_mixed);
                stream.push_back(ref_el.n_pads);
                stream.push_back(ref_el.times);
                stream.push_back(count_repeated);
                ref_el = el;
                count_repeated = 1;
            }
        }
        stream.push_back(ref_el.n_data);
        stream.push_back(ref_el.n_mixed);
        stream.push_back(ref_el.n_pads);
        stream.push_back(ref_el.times);
        stream.push_back(count_repeated);

        max_stream_words = std::max(max_stream_words, static_cast<uint32_t>(stream.size()));

        const NodeCoord node = core;
        reader_core_args.push_back(ReaderCoreArgs{
            .node = node,
            .start_page_id = core_start_page_id,
            .n_block_reps = static_cast<uint32_t>(assignment.size()),
            .block_rep_stream = std::move(stream),
        });

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core_local;
        writer_core_args.push_back(
            WriterCoreArgs{.node = node, .num_pages = num_tiles_per_core, .start_id = tile_start_id});
        tile_start_id += num_tiles_per_core;
    }

    // ------------------------------------------------------------------------
    // Kernels.
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
                "reader_unary_pad_dims_split_rows_multicore.cpp"},
        .dfb_bindings = {ProducerOf(SRC_DFB, "in")},
        .tensor_bindings = {{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .compile_time_args =
            {{"tile_row_shift_bits", shift_bits},
             {"unpadded_X_size", unpadded_row_size_bytes},
             {"elem_size", elem_size},
             {"num_pages_in_row", num_pages_in_row},
             {"page_size", page_size},
             {"size_of_valid_data_in_last_page_in_row", size_of_valid_data_in_last_page_in_row}},
        .runtime_arg_schema = {.runtime_arg_names = {"padded_X_size", "pad_value", "start_page_id", "n_block_reps"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
        // Variable-length BlockRep stream. Sized to the longest per-core stream; shorter cores are
        // zero-padded below. See the per-core args comment above. (advanced_options is the last
        // KernelSpec field, so it must follow hw_config in this designated initializer.)
        .advanced_options = KernelAdvancedOptions{.num_runtime_varargs = max_stream_words},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_metal2.cpp"},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    ComputeHardwareConfig::UnpackToDestModes unpack_to_dest_modes;
    if (fp32_llk_acc && input_cb_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_modes.insert({SRC_DFB, UnpackToDestMode::UnpackToDestFp32});
    }

    // The legacy factory created a dedicated cliff compute kernel with a different compile-time
    // block count. That maps 1:1 to two compute KernelSpecs of the same source (full + cliff), each
    // in its own WorkUnitSpec, both binding the shared src/out DFBs. per_core_block_cnt stays a
    // compile-time arg (preserves loop unrolling).
    auto make_compute = [&](const KernelSpecName& id, uint32_t nblocks_this_group) {
        return KernelSpec{
            .unique_id = id,
            .source =
                std::filesystem::path{
                    "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_compute_metal2.cpp"},
            .dfb_bindings = {ConsumerOf(SRC_DFB, "in"), ProducerOf(OUT_DFB, "out")},
            .compile_time_args =
                {{"per_core_block_cnt", nblocks_this_group}, {"per_core_block_tile_cnt", num_tiles_per_row}},
            .hw_config =
                ComputeHardwareConfig{
                    .fp32_dest_acc_en = fp32_llk_acc,
                    .unpack_to_dest_mode = unpack_to_dest_modes,
                },
        };
    };

    std::vector<KernelSpec> kernels = {reader_spec, writer_spec, make_compute(COMPUTE_FULL_KERNEL, nblocks_per_core)};
    if (has_cliff) {
        kernels.push_back(make_compute(COMPUTE_CLIFF_KERNEL, nblocks_per_core_cliff));
    }

    // ------------------------------------------------------------------------
    // Per-node runtime args. Named scalars go through runtime_arg_values; the BlockRep stream goes
    // through advanced_options.runtime_varargs (zero-padded to max_stream_words).
    // ------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    reader_run.runtime_arg_values.reserve(ncores);
    writer_run.runtime_arg_values.reserve(ncores);

    for (auto& rca : reader_core_args) {
        reader_run.runtime_arg_values.push_back(
            {rca.node,
             {{"padded_X_size", padded_row_size_bytes},
              {"pad_value", packed_pad_value},
              {"start_page_id", rca.start_page_id},
              {"n_block_reps", rca.n_block_reps}}});

        std::vector<uint32_t> padded_stream = std::move(rca.block_rep_stream);
        padded_stream.resize(max_stream_words, 0u);  // uniform vararg count across all cores
        // runtime_varargs is a Table<NodeCoord, Varargs> (no push_back); insert one entry per node.
        reader_run.advanced_options.runtime_varargs[rca.node] = std::move(padded_stream);
    }
    for (const auto& wca : writer_core_args) {
        writer_run.runtime_arg_values.push_back({wca.node, {{"num_pages", wca.num_pages}, {"start_id", wca.start_id}}});
    }

    std::vector<WorkUnitSpec> work_units;
    work_units.push_back(WorkUnitSpec{
        .name = "tilize_with_val_padding_multi_core_default_full",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_FULL_KERNEL},
        .target_nodes = core_range,
    });
    if (has_cliff) {
        work_units.push_back(WorkUnitSpec{
            .name = "tilize_with_val_padding_multi_core_default_cliff",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_CLIFF_KERNEL},
            .target_nodes = core_range_cliff,
        });
    }

    ProgramSpec spec{
        .name = "tilize_with_val_padding_multi_core_default",
        .kernels = std::move(kernels),
        .dataflow_buffers = {src_dfb_spec, out_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_mesh)}}, {OUTPUT_TENSOR, TensorArgument{std::cref(output_mesh)}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
