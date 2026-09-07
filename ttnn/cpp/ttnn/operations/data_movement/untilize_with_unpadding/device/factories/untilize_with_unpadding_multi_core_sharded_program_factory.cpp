// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_sharded_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Spec names are prefixed per factory: all five factory .cpp files land in one unity-build
// translation unit, where every anonymous namespace merges into a single scope.
const KernelSpecName SH_READER{"sh_reader"};
const KernelSpecName SH_WRITER{"sh_writer"};
const KernelSpecName SH_COMPUTE{"sh_compute"};
const DFBSpecName SH_IN{"sh_in"};
const DFBSpecName SH_OUT{"sh_out"};
const DFBSpecName SH_SHARDED_OUT{"sh_sharded_out"};
const TensorParamName SH_INPUT{"sh_input"};
const TensorParamName SH_OUTPUT{"sh_output"};

}  // namespace

ttnn::device_operation::ProgramArtifacts UntilizeWithUnpaddingMultiCoreShardedProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();
    // WIDTH_SHARDED <-> BLOCK_SHARDED with a matching column shard width (enforced in validate()).
    // Unlike the same-shard-type out_sharded path below (a same-core L1-to-L1 copy via a dataflow
    // buffer borrowed onto the output buffer), the executing core here may not be the physically-
    // owning core of the output shard, so a dedicated writer addresses the destination via
    // TensorAccessor page-id routing instead.
    bool cross_shard_type = out_sharded && output.memory_config().memory_layout() != a.memory_config().memory_layout();
    // Special handling for tensors of W=16 and H%32==0
    // In this case skip untilizing on compute and in writer kernel just copy face0 and face2,
    // and skip face1 and face3. Only writer_unary_unpad_width_16_sharded.cpp knows how to extract
    // faces 0 and 2 from the tiled output emitted by eltwise_copy.cpp; that writer is only reached
    // inside the `else if (out_sharded)` branch. The interleaved-output writers
    // (writer_unary_unpad_sharded_to_interleaved.cpp and
    // writer_unary_stick_layout_interleaved_blocks_metal2.cpp) expect the normal untilized row-major
    // rows produced by untilize.cpp and cannot consume the tiled data, so the fast path must be gated
    // on out_sharded in addition to !cross_shard_type.
    bool unpad_tensor_w_16 = out_sharded && !cross_shard_type && output.padded_shape()[-1] == 16 &&
                             output.padded_shape()[-2] % TILE_HEIGHT == 0;
    tt::DataFormat input_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_dfb_data_format);
    tt::DataFormat output_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_dfb_data_format);

    uint32_t num_rows_block = 0, block_row_size = 0, output_row_size = 0, last_block_row_size_unpadded = 0,
             num_output_rows_unpadded = 0;
    CoreCoord end_core;
    uint32_t last_idx = 0;
    auto shard_spec = a.shard_spec().value();

    // I am not sure it is correct to ever use the shard_spec here.
    auto out_shard_spec = output.shard_spec().has_value() ? output.shard_spec().value() : shard_spec;

    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto all_cores = shard_spec.grid;
    uint32_t ntiles_per_block = shard_spec.shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t global_batch = a.physical_volume() / (a.padded_shape()[-2] * a.padded_shape()[-1]);
    uint32_t batch =
        a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED
            ? std::max(1u, (shard_spec.shape[0] * shard_spec.shape[1]) / (a.padded_shape()[-2] * a.padded_shape()[-1]))
            : global_batch;
    uint32_t ntiles_per_batch = ntiles_per_block * nblocks_per_core / batch;

    num_rows_block = out_shard_spec.shape[0];
    block_row_size = out_shard_spec.shape[1] * output.element_size();     // in0_block_w * TILE_WIDTH * dtype_nbytes
    output_row_size = output.padded_shape()[-1] * output.element_size();  // output row size bytes
    last_block_row_size_unpadded = block_row_size - (tt::round_up(output.padded_shape()[-1], out_shard_spec.shape[1]) -
                                                     output.padded_shape()[-1]) *
                                                        output.element_size();
    uint32_t num_output_rows = output.physical_volume() / output.padded_shape()[-1];
    num_output_rows_unpadded =
        num_rows_block - (tt::round_up(num_output_rows, out_shard_spec.shape[0]) - num_output_rows);
    if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        last_idx = tt::div_up(output.padded_shape()[-1], out_shard_spec.shape[1]) - 1;
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        last_idx = tt::div_up(num_output_rows, out_shard_spec.shape[0]) - 1;
    } else {
        end_core = {
            tt::div_up(output.padded_shape()[-1], out_shard_spec.shape[1]) - 1,
            tt::div_up(num_output_rows, out_shard_spec.shape[0]) - 1};
    }
    if (!row_major) {
        std::swap(end_core.x, end_core.y);
    }

    // Per-core output column-shard index and starting output row, derived from the INPUT's grid
    // geometry independently of the output's (their grids may differ in shape/orientation between
    // WIDTH_SHARDED's 1xKW grid and BLOCK_SHARDED's KHxKW grid). cross_kw is the column-shard count,
    // identical on both sides since validate() requires matching column shard width.
    struct CrossShardCoreInfo {
        uint32_t col_shard_id = 0;
        uint32_t row_start_id = 0;
    };
    std::vector<CrossShardCoreInfo> cross_shard_core_infos;
    uint32_t cross_kw = 0;
    if (cross_shard_type) {
        auto in_cores = corerange_to_cores(all_cores, std::nullopt, row_major);
        cross_shard_core_infos.reserve(in_cores.size());
        if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            cross_kw = static_cast<uint32_t>(in_cores.size());
            for (uint32_t i = 0; i < in_cores.size(); ++i) {
                cross_shard_core_infos.push_back({i, 0});
            }
        } else {
            // BLOCK_SHARDED input. For COL_MAJOR orientation the physical x/y grid axes swap
            // which one is the logical row-shard (KH) vs column-shard (KW) axis - same convention
            // as compute_output_specs()'s BLOCK_SHARDED shard-shape derivation above. Once grid_cols
            // is the KW axis and grid_rows is the KH axis (post-swap), a single division formula
            // works for both orientations, since corerange_to_cores enumerates row_major as
            // i = y*grid_cols_raw + x (x fastest) and !row_major as i = x*grid_rows_raw + y (y
            // fastest) - i.e. i / grid_cols(post-swap) and i % grid_cols(post-swap) recover (kh, kw)
            // in both cases.
            CoreRange bbox = all_cores.bounding_box();
            uint32_t grid_cols = bbox.end_coord.x - bbox.start_coord.x + 1;
            uint32_t grid_rows = bbox.end_coord.y - bbox.start_coord.y + 1;
            if (!row_major) {
                std::swap(grid_cols, grid_rows);
            }
            cross_kw = grid_cols;
            for (uint32_t i = 0; i < in_cores.size(); ++i) {
                uint32_t kh = i / grid_cols;
                uint32_t kw = i % grid_cols;
                cross_shard_core_infos.push_back({kw, kh * shard_spec.shape[0]});
            }
        }
    }

    uint32_t num_input_tiles = ntiles_per_block * nblocks_per_core;
    // Input buffer: sharded → borrowed onto the input tensor's own L1 allocation; the backing
    // address resolves at runtime from the corresponding TensorArgument, on every dispatch.
    DataflowBufferSpec in_dfb{
        .unique_id = SH_IN,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_dfb_data_format,
        .borrowed_from = src_sharded ? std::optional<TensorParamName>{SH_INPUT} : std::nullopt,
    };

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_output_tiles = out_sharded ? (unpad_tensor_w_16 ? 16 : ntiles_per_batch * 2) : ntiles_per_block * 2;
    uint32_t aligned_page_size = static_cast<uint32_t>(output.buffer()->aligned_page_size());
    DataflowBufferSpec out_dfb{
        .unique_id = SH_OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_dfb_data_format,
    };

    Group<DataflowBufferSpec> dataflow_buffers = {std::move(in_dfb), std::move(out_dfb)};
    const bool has_sharded_out_dfb = out_sharded && !cross_shard_type;
    if (has_sharded_out_dfb) {
        // The kernel advances the write pointer by aligned_page_size (which may be
        // larger than block_row_size due to buffer alignment padding), so the buffer's
        // entry size must match to avoid overflow.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = SH_SHARDED_OUT,
            .entry_size = aligned_page_size,
            .num_entries = num_output_rows_unpadded,
            .data_format_metadata = output_dfb_data_format,
            .borrowed_from = SH_OUTPUT,
        });
    }

    /** reader
     */
    KernelSpec reader{
        .unique_id = SH_READER,
        .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SH_IN,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    };

    /** writer
     */
    KernelSpec writer{
        .unique_id = SH_WRITER,
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    };
    if (cross_shard_type) {
        writer.source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_unpad_cross_sharded.cpp";
        writer.dfb_bindings = {DFBBinding{
            .dfb_spec_name = SH_OUT,
            .accessor_name = "untilize_out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }};
        writer.tensor_bindings = {TensorBinding{
            .tensor_parameter_name = SH_OUTPUT,
            .accessor_name = "dst",
        }};
        writer.runtime_arg_schema = {
            .runtime_arg_names = {
                "num_padded_tiles_per_batch",
                "num_unpadded_rows_per_batch",
                "padded_block_row_size_bytes",
                "unpadded_block_row_size_bytes",
                "batch",
                "col_byte_offset",
                "row_start_id"}};
    } else if (out_sharded) {
        writer.source = unpad_tensor_w_16
                            ? "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                              "writer_unary_unpad_width_16_sharded.cpp"
                            : "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                              "writer_unary_unpad_batch_rows_sharded.cpp";
        // SH_SHARDED_OUT is borrowed onto the output shard and this writer is its only toucher — it
        // fills it by write pointer and nothing downstream drains it — so the writer binds both
        // endpoints (a self-loop). Legal on Gen1 for a data-movement kernel; the kernel is unchanged.
        writer.dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = SH_OUT,
                .accessor_name = "untilize_out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = SH_SHARDED_OUT,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = SH_SHARDED_OUT,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }};
        writer.compile_time_args = {{"aligned_page_size", aligned_page_size}};
        if (unpad_tensor_w_16) {
            writer.runtime_arg_schema = {
                .runtime_arg_names = {"num_unpadded_output_rows", "num_padded_tiles_per_core"}};
        } else {
            writer.runtime_arg_schema = {
                .runtime_arg_names = {
                    "num_unpadded_output_rows",
                    "num_padded_tiles_per_batch",
                    "num_unpadded_rows_per_batch",
                    "padded_block_row_size_bytes",
                    "unpadded_block_row_size_bytes",
                    "batch"}};
        }
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        // Height-sharded -> interleaved uses a dedicated writer that walks each core's absolute rows
        // and drops both interior (row) and column padding per matrix. It handles any alignment of
        // matrices to core boundaries (whole matrices per core, a single matrix split across cores,
        // or a batch whose matrices straddle cores), so there is no unbatched restriction.
        writer.source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_unpad_sharded_to_interleaved.cpp";
        writer.dfb_bindings = {DFBBinding{
            .dfb_spec_name = SH_OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }};
        writer.tensor_bindings = {TensorBinding{
            .tensor_parameter_name = SH_OUTPUT,
            .accessor_name = "dst",
        }};
        writer.runtime_arg_schema = {
            .runtime_arg_names = {
                "start_padded_row",
                "num_rows",
                "matrix_h_padded",
                "matrix_h_logical",
                "block_row_size",
                "row_size_unpadded",
                "ntiles_per_row"}};
    } else {
        writer.source = "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks_metal2.cpp";
        writer.dfb_bindings = {DFBBinding{
            .dfb_spec_name = SH_OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }};
        writer.tensor_bindings = {TensorBinding{
            .tensor_parameter_name = SH_OUTPUT,
            .accessor_name = "dst",
        }};
        writer.compile_time_args = {
            {"float32_dtype",
             (uint32_t)(input_dfb_data_format == tt::DataFormat::Float32 or
                        input_dfb_data_format == tt::DataFormat::UInt32 or
                        input_dfb_data_format == tt::DataFormat::Int32)},
            {"output_row_size", output_row_size}};
        writer.runtime_arg_schema = {
            .runtime_arg_names = {
                "num_rows_block",
                "block_row_size",
                "batch",
                "num_blocks_h",
                "num_blocks_w",
                "last_block_row_size_unpadded",
                "num_output_rows_unpadded",
                "block_start_row_id",
                "block_start_row_offset"}};
    }

    /** compute
     */
    KernelSpec::CompilerOptions::Defines compute_kernel_defines;
    if (input_dfb_data_format == tt::DataFormat::Int32 || input_dfb_data_format == tt::DataFormat::UInt32 ||
        input_dfb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.emplace("DST_ACCUM_MODE", "1");
    }
    ComputeGen1Config compute_hw_config{.enable_32_bit_dest = fp32_dest_acc_en};
    if (fp32_dest_acc_en) {
        compute_hw_config.unpack_modes = {{SH_IN, UnpackMode::UnpackToDest}};
    }

    KernelSpec compute{
        .unique_id = SH_COMPUTE,
        .compiler_options =
            {
                .defines = std::move(compute_kernel_defines),
                .opt_level = KernelBuildOptLevel::O3,
            },
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = SH_IN,
                 // The copy kernel names its input `in`; the untilize kernels name theirs `src`.
                 .accessor_name = unpad_tensor_w_16 ? "in" : "src",
                 .endpoint_type = DFBEndpointType::CONSUMER,
             },
             DFBBinding{
                 .dfb_spec_name = SH_OUT,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .hw_config = compute_hw_config,
    };
    if (unpad_tensor_w_16) {
        // Use copy compute kernel just for a potential data type conversion.
        compute.source = "ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp";
        compute.compile_time_args = {{"per_core_tile_cnt", num_input_tiles}};
    } else {
        compute.source = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_metal2.cpp";
        compute.compile_time_args = {
            {"per_core_block_cnt", (uint32_t)nblocks_per_core},
            {"per_core_block_tile_cnt", (uint32_t)ntiles_per_block}};
    }

    // Runtime args: legacy code uses SetRuntimeArgs(program, kernel, all_cores, args) which broadcasts
    // the same args to every core. Metal 2.0 named RTAs are per node; enumerate cores and emit one
    // entry per core, as the descriptor form already did.
    const std::vector<CoreCoord> all_core_coords = corerange_to_cores(all_cores, std::nullopt, row_major);

    KernelRunArgs reader_run_args{.kernel = SH_READER};
    for (const auto& core : all_core_coords) {
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"num_tiles_per_core", ntiles_per_block * nblocks_per_core}});
    }

    KernelRunArgs writer_run_args{.kernel = SH_WRITER};
    if (cross_shard_type) {
        // Per-core row/column-shard indices are independent of the executing core's own position
        // (see cross_shard_core_infos above), so runtime args differ per core rather than
        // broadcasting one set to every core like the same-shard-type branch below.
        uint32_t num_output_rows_cross = output.physical_volume() / output.padded_shape()[-1];
        for (uint32_t i = 0; i < all_core_coords.size(); ++i) {
            const auto& core = all_core_coords[i];
            const auto& info = cross_shard_core_infos[i];
            uint32_t this_core_rows = 0;
            if (info.row_start_id < num_output_rows_cross) {
                this_core_rows = std::min(shard_spec.shape[0], num_output_rows_cross - info.row_start_id);
            }
            uint32_t row_size_unpadded =
                (info.col_shard_id == cross_kw - 1) ? last_block_row_size_unpadded : block_row_size;
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {// batch==1, cross-type is unbatched-only
                 {"num_padded_tiles_per_batch", ntiles_per_batch},
                 {"num_unpadded_rows_per_batch", this_core_rows},
                 // source buffer stride, always full
                 {"padded_block_row_size_bytes", block_row_size},
                 // trimmed for the last column shard
                 {"unpadded_block_row_size_bytes", row_size_unpadded},
                 {"batch", 1u},
                 // row->page split done by TensorAccessor
                 {"col_byte_offset", info.col_shard_id * block_row_size},
                 {"row_start_id", info.row_start_id}});
        }
    } else if (out_sharded) {
        for (const auto& core : all_core_coords) {
            if (unpad_tensor_w_16) {
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"num_unpadded_output_rows", num_output_rows_unpadded},
                     {"num_padded_tiles_per_core", num_input_tiles}});
            } else {
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"num_unpadded_output_rows", num_output_rows_unpadded},
                     {"num_padded_tiles_per_batch", ntiles_per_batch},
                     {"num_unpadded_rows_per_batch", out_shard_spec.shape[0] / batch},
                     {"padded_block_row_size_bytes", shard_spec.shape[1] * output.element_size()},
                     {"unpadded_block_row_size_bytes", block_row_size},
                     {"batch", batch}});
            }
        }
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        // General height-sharded -> interleaved. Each core owns the absolute padded rows
        // [i * shard_h, (i + 1) * shard_h) of the flattened [global_batch * H_padded, W_padded] row
        // space (enumerated in shard/core order). The kernel maps every row to its (matrix,
        // row-in-matrix) and writes the real rows to their logical interleaved page, dropping both
        // interior (row) and column padding. This is correct for any alignment of matrices to core
        // boundaries, so there is no unbatched restriction.
        const uint32_t matrix_h_padded = a.padded_shape()[-2];
        const uint32_t matrix_h_logical = output.logical_shape()[-2];
        const uint32_t shard_height = shard_spec.shape[0];
        const uint32_t buffer_row_size = shard_spec.shape[1] * output.element_size();  // row stride (padded width)
        const uint32_t row_size_unpadded =
            output.logical_shape()[-1] * output.element_size();  // bytes written per row (logical width)

        for (uint32_t i = 0; i < all_core_coords.size(); ++i) {
            const CoreCoord& core = all_core_coords[i];
            const uint32_t start_padded_row = i * shard_height;
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"start_padded_row", start_padded_row},
                 {"num_rows", shard_height},
                 {"matrix_h_padded", matrix_h_padded},
                 {"matrix_h_logical", matrix_h_logical},
                 {"block_row_size", buffer_row_size},
                 {"row_size_unpadded", row_size_unpadded},
                 {"ntiles_per_row", ntiles_per_block}});
        }
    } else {
        for (uint32_t i = 0; i < all_core_coords.size(); ++i) {
            CoreCoord core = all_core_coords[i];

            // writer runtime args
            uint32_t block_start_row_offset;
            uint32_t block_start_row_id_offset;
            uint32_t row_size_unpadded = block_row_size;
            uint32_t num_rows_unpadded = num_rows_block;
            if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
                block_start_row_offset = i * block_row_size;
                block_start_row_id_offset = 0;
                if (i > last_idx) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                } else {
                    num_rows_unpadded = num_output_rows_unpadded;
                    if (i == last_idx) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                }
            } else {
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }
                if (core.x > end_core.x || core.y > end_core.y) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                }
            }

            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"num_rows_block", num_rows_block},
                 {"block_row_size", block_row_size},
                 {"batch", 1u},
                 {"num_blocks_h", 1u},
                 {"num_blocks_w", 1u},
                 {"last_block_row_size_unpadded", row_size_unpadded},
                 {"num_output_rows_unpadded", num_rows_unpadded},
                 {"block_start_row_id", block_start_row_id_offset},
                 {"block_start_row_offset", block_start_row_offset}});
        }
    }

    Group<TensorParameter> tensor_parameters;
    ProgramRunArgs run_args;
    if (src_sharded) {
        // Declared for the borrowed input buffer above; no kernel binds it directly.
        tensor_parameters.push_back(TensorParameter{.unique_id = SH_INPUT, .spec = input_mesh_tensor.tensor_spec()});
        run_args.tensor_args.emplace(SH_INPUT, input_mesh_tensor);
    }
    tensor_parameters.push_back(TensorParameter{.unique_id = SH_OUTPUT, .spec = output_mesh_tensor.tensor_spec()});
    run_args.tensor_args.emplace(SH_OUTPUT, output_mesh_tensor);

    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_sharded",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {SH_READER, SH_WRITER, SH_COMPUTE},
            .target_nodes = all_cores,
        }},
    };

    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
