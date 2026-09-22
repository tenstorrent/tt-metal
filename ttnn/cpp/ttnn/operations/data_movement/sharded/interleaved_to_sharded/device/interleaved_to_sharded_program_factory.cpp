// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

// Hardcoded for non-partial interleaved_to_sharded operation
// to keep backward compatibility after migration to new infra
// https://github.com/tenstorrent/tt-metal/issues/32752
constexpr uint32_t num_slices = 1;
constexpr uint32_t slice_index = 0;

ttnn::device_operation::ProgramArtifacts InterleavedToShardedProgramFactory::create_program_artifacts(
    const InterleavedToShardedParams& /*operation_attributes*/,
    const InterleavedToShardedInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    // Keep explicit bool init to match legacy behavior which forced it true
    bool keep_l1_aligned = true;  // operation_attributes.keep_l1_aligned;

    uint32_t num_units_per_shard = 0;
    uint32_t input_unit_size = 0;
    uint32_t output_unit_size = 0;
    uint32_t num_units_per_shard_width = 0;
    uint32_t num_units_per_shard_height = 0;
    uint32_t num_units_offset = 0;
    uint32_t num_units_per_row = 0;
    uint32_t num_units_per_shard_height_last = 0;
    uint32_t num_units_per_shard_width_last = 0;
    uint32_t padded_offset_bytes = 0;

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto shard_spec = output.shard_spec().value();
    auto shard_strategy = output.memory_config().memory_layout();

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    auto cores = get_optimal_worker_cores_for_sharded_tensor(output);
    auto all_cores = CoreRangeSet(ttsl::Span<const CoreCoord>(cores));
    CoreCoord end_core = cores.back();

    const bool is_tile = input.layout() == Layout::TILE;
    bool convert_df = input_data_format != output_data_format;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);
    bool is_quasar = (input.device()->arch() == tt::ARCH::QUASAR);

    if (is_tile) {
        input_unit_size = tt::tile_size(input_data_format);
        output_unit_size = tt::tile_size(output_data_format);
        TT_FATAL(
            shard_spec.shape[0] % TILE_HEIGHT == 0 && shard_spec.shape[1] % TILE_WIDTH == 0,
            "Shard shape {} must be tile {}x{} sized!",
            shard_spec.shape,
            TILE_HEIGHT,
            TILE_WIDTH);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.padded_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height = (input.physical_volume() / input.padded_shape()[-1]) / TILE_HEIGHT;
        num_units_per_shard_height_last =
            num_units_per_shard_height -
            (tt::round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width -
            (tt::round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
        padded_offset_bytes = (num_units_per_shard_width - num_units_per_shard_width_last) * input_unit_size;
    } else {
        input_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * input.element_size());
        output_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * output.element_size());
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = static_cast<uint32_t>(input.logical_shape()[-1] * input.element_size());
        num_units_offset = 1;
        uint32_t num_units_height = static_cast<uint32_t>(input.logical_volume() / input.logical_shape()[-1]);
        num_units_per_shard_height_last =
            num_units_per_shard_height -
            (tt::round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        // TODO: Use a different variable name. Units refers to pages, but this is being used as size
        num_units_per_shard_width_last =
            input_unit_size - (tt::round_up(num_units_per_row, input_unit_size) - num_units_per_row);
        // Adjust accordingly to l1 alignment, do it for all archs
        if (keep_l1_aligned) {
            padded_offset_bytes = tt::align(input_unit_size, hal::get_l1_alignment());
        } else {
            padded_offset_bytes = tt::align(input_unit_size, input.buffer()->alignment());
        }
    }

    uint32_t num_input_units = num_units_per_shard;
    uint32_t output_page_size = tt::align(output_unit_size, dst_buffer->alignment());

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};            // interleaved pages streamed in; exists only when converting formats
    const DFBSpecName OUT_DFB{"out"};          // the output shard, in the output's data format
    const ScratchpadSpecName SCRATCH{"scratch"};  // DRAM->L1 alignment staging; row-major only
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    Group<DataflowBufferSpec> dataflow_buffers;
    Group<ScratchpadSpec> scratchpads;

    // Output DFB. When the destination is sharded (non-DRAM) it is built on the output shard's own L1
    // buffer (borrowed memory), so the pages the program produces land directly in the output tensor and
    // the backing address re-resolves from the OUTPUT tensor argument on every enqueue — which is what
    // lets a program-cache hit pick up a reallocated output shard. When the destination is DRAM there is
    // nothing resident to borrow, so the DFB allocates its own L1 and the writer drains it over the NoC.
    // Without format conversion this is also the buffer the reader fills: legacy pointed the input and
    // output buffer indices at one allocation, and a single DFB bound by two kernels says the same
    // thing in Metal 2.0.
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = output_page_size,
        .num_entries = num_input_units,
        .data_format_metadata = output_data_format,
        .borrowed_from = dst_is_dram ? std::nullopt : std::optional<TensorParamName>{OUTPUT},
    });

    if (convert_df) {
        // Separate input DFB for the reader to stream interleaved pages into, in the *input* format, so
        // the compute kernel has both formats in front of it. Never borrowed — the input is interleaved,
        // so nothing outside the program backs it.
        uint32_t input_page_size = tt::align(input_unit_size, src_buffer->alignment());
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN_DFB,
            .entry_size = input_page_size,
            .num_entries = num_input_units,
            .data_format_metadata = input_data_format,
        });
    }

    uint32_t dram_alignment = hal::get_dram_alignment();
    uint32_t l1_alignment = hal::get_l1_alignment();
    uint32_t num_trids = 4;
    // The scratchpad stages DRAM (64B) reads down to L1 (16B) alignment. See issue #34414. Only the
    // row-major reader has a use for it -- the tile reader never receives a handle to it -- so the
    // tile path declares no spec at all rather than an unbound one.
    const bool needs_scratch =
        (src_is_dram && (input_unit_size % dram_alignment != 0)) || (is_blackhole || is_quasar) || keep_l1_aligned;
    const bool has_scratch = !is_tile && needs_scratch;
    if (has_scratch) {
        uint32_t scratch_page_size = tt::align(input_unit_size + dram_alignment, dram_alignment);
        // Whole region the former DFB reserved on each node: entry_size * num_entries.
        scratchpads.push_back(ScratchpadSpec{
            .unique_id = SCRATCH,
            .size_per_node = scratch_page_size * num_trids,
        });
    }

    // Reader kernel. Produces into the input DFB when converting formats, otherwise straight into the
    // output DFB. The row-major reader additionally drives the alignment scratchpad -- a private
    // Scratchpad it fills and drains itself.
    KernelSpec reader{
        .unique_id = READER,
        .dfb_bindings =
            {DFBBinding{
                .dfb_spec_name = convert_df ? IN_DFB : OUT_DFB,
                .accessor_name = "in",
                .endpoint_type = DFBEndpointType::PRODUCER,
            }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        // The reader drives its `in` DFB with explicit reserve_back/push_back (and does many sub-tile
        // stick reads), so opt every bound DFB out of Gen2 implicit-sync credit accounting; the flag is
        // ignored on Gen1. Matches the sibling sharded_to_interleaved factory.
        .hw_config = ttnn::create_reader_datamovement_config(
            input.device()->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };
    if (is_tile) {
        reader.source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "reader_unary_sharded_blocks_interleaved_start_id_metal2.cpp";
        reader.compile_time_args = {{"num_readers", all_cores.num_cores()}};
        reader.runtime_arg_schema.runtime_arg_names = {
            "block_height_tiles",
            "block_width_tiles",
            "padded_offset_bytes",
            "input_width_offset_tiles",
            "block_num_tiles",
            "start_id_offset",
            "start_id_base"};
    } else {
        reader.source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "reader_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp";
        reader.compile_time_args = {{"num_trids", num_trids}};
        // The kernel reads no equivalent of the legacy arg 1 (num_units_per_row), so the named schema
        // has no slot for it.
        reader.runtime_arg_schema.runtime_arg_names = {
            "block_height",
            "block_width_bytes",
            "padded_block_width_bytes",
            "aligned",
            "aligned_input_width_offset_bytes",
            "aligned_block_width_bytes",
            "aligned_offset",
            "start_id"};
        if (has_scratch) {
            reader.scratchpad_bindings.push_back(ScratchpadBinding{
                .scratchpad_spec_name = SCRATCH,
                .accessor_name = "scratch",
            });
        }
    }

    // Writer kernel. Every variant drains the output DFB as dfb::out; only the DRAM-destination ones
    // carry it further, over the NoC into the interleaved output tensor.
    KernelSpec writer{
        .unique_id = WRITER,
        .dfb_bindings =
            {DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
        // The writer drains its `out` DFB with explicit wait_front/pop_front, so opt out of Gen2
        // implicit-sync credit accounting (ignored on Gen1), matching sharded_to_interleaved.
        .hw_config = ttnn::create_writer_datamovement_config(
            input.device()->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };
    if (dst_is_dram) {
        writer.tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}};
        if (is_tile) {
            writer.source =
                "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                "writer_unary_sharded_blocks_start_id_metal2.cpp";
            writer.runtime_arg_schema.runtime_arg_names = {
                "block_height_tiles",
                "block_width_tiles",
                "padded_offset",
                "block_width_padded_num_tiles",
                "output_width_tiles",
                "start_id_offset",
                "start_id_base"};
        } else {
            writer.source =
                "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                "writer_unary_sharded_stick_layout_start_id_metal2.cpp";
            writer.runtime_arg_schema.runtime_arg_names = {
                "block_height", "block_width_bytes", "padded_block_width_bytes", "start_id", "output_width_in_pages"};
        }
    } else {
        // Output is sharded in place; the writer only handshakes on the DFB. This binds the shared
        // Metal 2.0 fork already in this directory, so its accessor and argument names are that
        // kernel's interface, not this op's choice.
        writer.source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "writer_unary_sharded_metal2.cpp";
        writer.runtime_arg_schema.runtime_arg_names = {"num_units"};
    }

    Group<KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));

    Group<KernelSpecName> work_unit_kernels{READER, WRITER};

    // Optional compute kernel for data-format conversion. This binds the sibling Metal 2.0 fork in this
    // family's kernel pool, which reads per_core_tile_cnt as a *runtime* argument -- the count differs on
    // the end core, so the other fork of this kernel, which reads it as a compile-time constant, does not
    // fit.
    if (convert_df) {
        kernels.push_back(KernelSpec{
            .unique_id = COMPUTE,
            .source = "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy_metal2.cpp",
            // The legacy ComputeConfigDescriptor set no opt_level, which resolves to O3 for a compute
            // kernel; Metal 2.0's CompilerOptions defaults to O2, so O3 has to be stated here.
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = IN_DFB,
                     .accessor_name = "in",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 }},
            .runtime_arg_schema = {.runtime_arg_names = {"per_core_tile_cnt"}},
            // Every field of the legacy ComputeConfigDescriptor{} was left at its default, and the
            // Metal 2.0 Gen1 compute defaults match those field for field (HiFi4; math_approx_mode
            // false = Precise SFPU; bfp8_pack_precise false = Approximate pack; fp32_dest_acc_en
            // false; dst_full_sync_en false = double_buffer_dest true), so an all-default Gen1 config
            // reproduces the legacy settings exactly.
            .hw_config = ComputeHardwareConfig{ComputeGen1Config{}},
        });
        work_unit_kernels.push_back(COMPUTE);
    }

    uint32_t starting_idx_h =
        operations::data_movement::detail::calculate_starting_idx_h(input, num_slices, slice_index);
    uint32_t curr_idx_h = 0;
    uint32_t curr_idx_w = 0;

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    for (const auto& core : cores) {
        uint32_t curr_num_units_per_shard = num_units_per_shard;
        if (is_tile) {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = num_units_per_shard_width;
            uint32_t padded_offset = 0;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core == end_core) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core == end_core) {
                    shard_width = num_units_per_shard_width_last;
                    padded_offset = padded_offset_bytes;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                        padded_offset = padded_offset_bytes;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                        padded_offset = padded_offset_bytes;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            curr_num_units_per_shard = shard_height * num_units_per_shard_width;

            // Reader run-time args. The source base address is no longer among them -- it arrives with
            // the INPUT tensor binding.
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"block_height_tiles", shard_height},
                 {"block_width_tiles", shard_width},
                 {"padded_offset_bytes", padded_offset},
                 {"input_width_offset_tiles", num_units_offset},
                 {"block_num_tiles", curr_num_units_per_shard},
                 {"start_id_offset", curr_idx_h + curr_idx_w},
                 {"start_id_base", starting_idx_h}});

            // Writer run-time args
            uint32_t pad_offset = (num_units_per_shard_width - shard_width) * output_unit_size;
            if (dst_is_dram) {
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"block_height_tiles", shard_height},
                     {"block_width_tiles", shard_width},
                     {"padded_offset", pad_offset},
                     {"block_width_padded_num_tiles", curr_num_units_per_shard},
                     {"output_width_tiles", num_units_offset},
                     {"start_id_offset", curr_idx_h + curr_idx_w},
                     {"start_id_base", starting_idx_h}});
            } else {
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values, core, {{"num_units", curr_num_units_per_shard}});
            }

            // Update indexing
            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = input_unit_size;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                    curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                        curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                        curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                    }
                }
            }

            bool aligned = false;
            if (src_is_dram) {
                aligned = (curr_idx_w % dram_alignment == 0) && (padded_offset_bytes % dram_alignment == 0);
            } else if (is_blackhole || is_quasar) {
                aligned = (curr_idx_w % l1_alignment == 0) && (padded_offset_bytes % l1_alignment == 0);
            } else {
                aligned = true;
            }
            uint32_t aligned_width_offset = 0;
            uint32_t aligned_shard_width = 0;
            uint32_t aligned_offset = 0;
            if (!aligned) {
                // TODO: is this right, leaving non BH case the same for now, should investigate
                if (!(is_blackhole || is_quasar)) {
                    aligned_width_offset = tt::round_down(curr_idx_w, dram_alignment);
                } else {
                    if (src_is_dram) {
                        aligned_width_offset = tt::round_down(curr_idx_w, dram_alignment);
                    } else {
                        aligned_width_offset = tt::round_down(curr_idx_w, l1_alignment);
                    }
                }
                aligned_offset = curr_idx_w - aligned_width_offset;
                aligned_shard_width = aligned_offset + shard_width;
            } else {
                aligned_width_offset = curr_idx_w;
                aligned_shard_width = shard_width;
                aligned_offset = 0;
            }

            // Reader run-time args. The source base address is no longer among them -- it arrives with
            // the INPUT tensor binding.
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"block_height", shard_height},
                 {"block_width_bytes", shard_width},
                 {"padded_block_width_bytes", padded_offset_bytes},
                 {"aligned", static_cast<uint32_t>(aligned)},
                 {"aligned_input_width_offset_bytes", aligned_width_offset},
                 {"aligned_block_width_bytes", aligned_shard_width},
                 {"aligned_offset", aligned_offset},
                 {"start_id", curr_idx_h}});

            // Writer run-time args
            if (dst_is_dram) {
                uint32_t page_id_within_row = curr_idx_w / input_unit_size;
                uint32_t output_width_in_pages = tt::div_up(num_units_per_row, input_unit_size);
                uint32_t start_id = (curr_idx_h * output_width_in_pages) + page_id_within_row;
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"block_height", shard_height},
                     {"block_width_bytes", shard_width},
                     {"padded_block_width_bytes", padded_offset_bytes},
                     {"start_id", start_id},
                     {"output_width_in_pages", output_width_in_pages}});
            } else {
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values, core, {{"num_units", curr_num_units_per_shard}});
            }

            // Update indexing
            curr_idx_w += input_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
        if (convert_df) {
            AddRuntimeArgsForNode(
                compute_run_args.runtime_arg_values, core, {{"per_core_tile_cnt", curr_num_units_per_shard}});
        }
    }

    ProgramSpec spec{
        .name = "interleaved_to_sharded",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .scratchpads = std::move(scratchpads),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {WorkUnitSpec{
            .name = "interleaved_to_sharded",
            .kernels = std::move(work_unit_kernels),
            .target_nodes = all_cores,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    if (convert_df) {
        run_args.kernel_run_args.push_back(std::move(compute_run_args));
    }
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
