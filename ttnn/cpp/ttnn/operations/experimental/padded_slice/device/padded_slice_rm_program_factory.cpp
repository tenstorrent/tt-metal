// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "padded_slice_rm_program_factory.hpp"
#include "padded_slice_utils.hpp"

#include "hostdevcommon/kernel_structs.h"
#include "optional"
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include <algorithm>
#include <cstdint>
#include <ranges>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>
#include <vector>
#include <tt-metalium/tt_align.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::experimental::prim {

// Spec names. Op-scoped namespace rather than an anonymous one: the two padded_slice factories share a
// CMake target, where same-named anonymous-namespace constants collide under a unity build.
namespace padded_slice_rm_metal2 {
const KernelSpecName READER{"reader"};
const KernelSpecName WRITER{"writer"};

const DFBSpecName OUT{"out"};
const DFBSpecName TEMP_PAD{"temp_pad"};
const DFBSpecName NON_ALIGNED{"non_aligned"};

const ScratchpadSpecName ID_PER_DIM{"id_per_dim"};

const TensorParamName INPUT{"input"};
const TensorParamName OUTPUT{"output"};
}  // namespace padded_slice_rm_metal2

namespace {

// Layout of the per-core reader argument vector built below: nine named scalars, then the three
// num_dims-long vararg blocks. The input base address is a tensor binding, not an argument.
constexpr uint32_t kPaddedSliceRmReaderUnpaddedStickSizeArgIdx = 0;
constexpr uint32_t kPaddedSliceRmReaderStartIdArgIdx = 3;
constexpr uint32_t kPaddedSliceRmReaderSrcOffsetArgIdx = 7;
constexpr uint32_t kPaddedSliceRmReaderMisalignmentArgIdx = 8;
constexpr uint32_t kPaddedSliceRmReaderVarargsBase = 9;

// The reader builds its TensorAccessor on the input buffer's base address and applies this core's
// read-window offset to every read, so the address it receives is a plain buffer base.
struct PaddedSliceRmReaderOffset {
    uint32_t src_offset_bytes;  // W-begin plus this core's width offset, rounded down to the source alignment
    uint32_t misalignment;      // bytes between that aligned offset and the exact start
};

PaddedSliceRmReaderOffset padded_slice_rm_reader_offset(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const CoreCoord& core,
    uint32_t core_index) {
    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "Input buffer should be allocated on device");
    TT_FATAL(output_buffer != nullptr, "Output buffer should be allocated on device!");

    const auto output_shard_spec = output_tensor.shard_spec().value();
    const bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const bool is_block_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool is_width_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const uint32_t output_row_size_bytes = output_shard_spec.shape[1] * input_tensor.element_size();

    const auto src_buffer_alignment =
        input_buffer->buffer_type() == BufferType::DRAM ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const auto dst_buffer_alignment =
        output_buffer->buffer_type() == BufferType::DRAM ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const uint32_t begins_bytes = output_tensor_start[-1] * input_tensor.element_size();
    const uint32_t output_row_size_bytes_offset = tt::round_up(output_row_size_bytes, dst_buffer_alignment);

    uint32_t core_w_index = 0;
    if (is_block_sharded) {
        core_w_index = rm_orientation ? core.x : core.y;
    } else if (is_width_sharded) {
        core_w_index = core_index;
    }
    const uint32_t width_offset = core_w_index * output_row_size_bytes_offset;
    // When the exact start isn't aligned to the source buffer, create_program_artifacts enables the reader's
    // non-aligned path, which reads from the aligned offset below it and skips `misalignment` bytes.
    const uint32_t exact_offset = begins_bytes + width_offset;
    const uint32_t misalignment = exact_offset % src_buffer_alignment;
    return {.src_offset_bytes = exact_offset - misalignment, .misalignment = misalignment};
}

}  // namespace

static std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>
get_padded_slice_runtime_args_rm_sharded_output(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& actual_output_shape,
    const std::vector<CoreCoord>& cores) {
    auto input_shape = input_tensor.logical_shape();
    auto output_shard_spec = output_tensor.shard_spec().value();
    auto output_shard_shape = output_shard_spec.shape;

    auto num_cores_total = cores.size();

    bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    bool is_block_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    bool is_width_sharded = output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    [[maybe_unused]] uint32_t num_cores_channels =
        ttnn::operations::experimental::detail::get_num_cores_channels_from_sharded_tensor(output_tensor);
    int input_page_size = input_shape[-1] * input_tensor.element_size();
    // Sliced last-dim width. The reader base already includes the slice's start offset.
    const int slice_row_size_bytes = static_cast<int>(actual_output_shape[-1] * input_tensor.element_size());
    [[maybe_unused]] uint32_t input_row_size_bytes =
        tt::div_up(input_shape[-1], num_cores_channels) * input_tensor.element_size();

    uint32_t output_row_size_bytes = output_shard_shape[1] * input_tensor.element_size();
    uint32_t output_row_size_elems = output_shard_shape[1];

    log_debug(
        tt::LogOp,
        "input_row_size_bytes: {}, input_page_size: {}, output_row_size_bytes: {}",
        input_row_size_bytes,
        input_page_size,
        output_row_size_bytes);
    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_output_sticks_per_dim[0] = 1;
    num_input_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    log_debug(tt::LogOp, "Output Shape : {}, Input Shape : {}", actual_output_shape, input_shape);
    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_output_dim = actual_output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_input_dim = (num_total_dim - num_output_dim) * accumulated_total_per_dim[i - 1];
        num_output_sticks_per_dim[i] = num_output_dim;
        num_input_sticks_per_dim[i] = num_input_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    for (int i = 0; i < num_dims; i++) {
        log_debug(
            tt::LogOp,
            "i = {}, num_output_sticks_per_dim: {}, num_input_sticks_per_dim: {}, accumulated_total_per_dim: {}",
            i,
            num_output_sticks_per_dim[i],
            num_input_sticks_per_dim[i],
            accumulated_total_per_dim[i]);
    }
    auto dst_buffer_alignment = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();

    uint32_t output_row_size_bytes_offset = tt::round_up(output_row_size_bytes, dst_buffer_alignment);
    // The input base address is a tensor binding, and the reader's accessor takes its page size from
    // that binding, so this list is scalars only.
    std::vector<uint32_t> common_reader_kernel_args = {
        output_row_size_bytes,  // unpadded_stick_size, set per core
        output_row_size_bytes_offset,
        num_dims,
        0,   // start_id, set per core
        0,   // num_sticks_per_core, set per core
        0,   // num_sticks_per_core_read, set per core
        0,   // num_read_per_barrier, set per core
        0u,  // src_offset_bytes, set per core
        0u,  // misalignment, set per core
    };

    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_output_sticks_per_dim.begin(), num_output_sticks_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_input_sticks_per_dim.begin(), num_input_sticks_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    const uint32_t num_sticks_per_core = output_shard_spec.shape[0];

    log_debug(tt::LogOp, "num_stick_per_core: {}", num_sticks_per_core);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, output_tensor_start);

    uint32_t core_index = 0;
    for (const auto& core : cores) {
        uint32_t core_w_index = 0;
        uint32_t core_h_index = core_index;
        if (is_block_sharded) {
            core_w_index = rm_orientation ? core.x : core.y;
            core_h_index = rm_orientation ? core.y : core.x;
        } else if (is_width_sharded) {
            core_h_index = 0;
            core_w_index = core_index;
        }

        const uint32_t num_sticks_written = core_h_index * num_sticks_per_core;
        const int width_offset = core_w_index * output_row_size_bytes_offset;

        id_per_dim[0] = num_sticks_written % num_output_sticks_per_dim[0];
        uint32_t output_written = num_sticks_written / num_output_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;
        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = output_written % num_output_sticks_per_dim[j];
            output_written = output_written / num_output_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        int this_input_row_size_bytes =
            std::max(std::min<int>(output_row_size_bytes, slice_row_size_bytes - width_offset), 0);
        uint32_t this_core_num_sticks = num_sticks_per_core;
        if (this_input_row_size_bytes == 0) {
            this_core_num_sticks = 0;
        }
        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        reader_kernel_args[kPaddedSliceRmReaderUnpaddedStickSizeArgIdx] = this_input_row_size_bytes;
        uint32_t addr_offset = kPaddedSliceRmReaderStartIdArgIdx;
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset++] = this_core_num_sticks;
        reader_kernel_args[addr_offset++] = this_core_num_sticks;
        reader_kernel_args[addr_offset] = this_core_num_sticks;
        const auto reader_offset =
            padded_slice_rm_reader_offset(input_tensor, output_tensor, output_tensor_start, core, core_index);
        reader_kernel_args[kPaddedSliceRmReaderSrcOffsetArgIdx] = reader_offset.src_offset_bytes;
        reader_kernel_args[kPaddedSliceRmReaderMisalignmentArgIdx] = reader_offset.misalignment;
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        log_trace(
            tt::LogOp,
            "For Core {}, start_id : {}, src_offset_bytes : {}, misalignment : {}, width_offset : {}, "
            "this_core_num_sticks : {}, this_input_row_size_bytes : {}",
            core,
            start_id,
            reader_offset.src_offset_bytes,
            reader_offset.misalignment,
            width_offset,
            this_core_num_sticks,
            this_input_row_size_bytes);

        std::vector<uint32_t> writer_kernel_args = {
            this_core_num_sticks, output_row_size_elems, this_input_row_size_bytes, output_row_size_bytes};
        ret_val[core_index] = {reader_kernel_args, writer_kernel_args};
        core_index++;
    }

    return ret_val;
}

ttnn::device_operation::ProgramArtifacts PaddedSliceRMProgramFactory::create_program_artifacts(
    const PaddedSliceParams& operation_attributes, const PaddedSliceInputs& tensor_args, Tensor& output) {
    using namespace padded_slice_rm_metal2;

    const auto& a = tensor_args.input;
    const auto& output_tensor_start = operation_attributes.padded_slice_start;
    const auto& output_tensor_end = operation_attributes.padded_slice_end;
    const auto& input_mesh = a.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    const ttnn::Shape output_shape = output.logical_shape();
    ttnn::Shape actual_output_shape = output_tensor_end;
    for (int i = 0; i < output_shape.rank(); i++) {
        actual_output_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }

    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    TT_FATAL(src0_buffer != nullptr, "Input buffer should be allocated on device");

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    TT_FATAL(output.is_sharded(), "Output Tensor must be sharded.");
    auto output_shard_spec = output.shard_spec().value();

    uint32_t output_row_size_bytes = output_shard_spec.shape[1] * output.element_size();

    CoreRangeSet total_cores = output.shard_spec().value().grid;
    bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    std::vector<CoreCoord> iter_cores = corerange_to_cores(total_cores, std::nullopt, rm_orientation);

    uint32_t num_cores_channels =
        ttnn::operations::experimental::detail::get_num_cores_channels_from_sharded_tensor(output);

    bool pad_output_row = false;
    log_debug(tt::LogOp, "Input Shape {}, Padded Shape : {}", a.logical_shape(), a.padded_shape());

    uint32_t slice_row_size_bytes = actual_output_shape[-1] * a.element_size();
    slice_row_size_bytes = slice_row_size_bytes / num_cores_channels;

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    TT_FATAL(
        dst_buffer->buffer_type() == tt::tt_metal::BufferType::L1,
        "Output buffer should be L1 for padded_slice operation with tiled inputs");

    auto src_buffer_alignment = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();

    TT_FATAL(
        output_row_size_bytes % dst_buffer_alignment == 0,
        "Output row size {} must be aligned to the destination buffer {} alignment {}",
        output_row_size_bytes,
        dst_buffer->buffer_type(),
        dst_buffer_alignment);
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    // A start that isn't aligned to the source buffer needs the non-aligned path too: the reader reads from
    // the aligned offset below the start and skips the misalignment on-device.
    const uint32_t begins_bytes = output_tensor_start[-1] * a.element_size();
    auto is_non_aligned = false;
    if (output_row_size_bytes % alignment || begins_bytes % src_buffer_alignment) {
        is_non_aligned = true;
    }

    // The kernel advances the write pointer by the aligned row size (stick_size_offset),
    // so the DFB entry size must match to avoid overflow.
    uint32_t output_dfb_page_size =
        is_non_aligned ? tt::round_up(output_row_size_bytes, dst_buffer_alignment) : output_row_size_bytes;

    uint32_t num_output_sticks_per_core = output_shard_spec.shape[0];

    // ---- Dataflow buffers ----
    // The output DFB is a window onto the resident output shard; its backing address comes from the
    // OUTPUT tensor argument on every dispatch.
    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT,
        .entry_size = output_dfb_page_size,
        .num_entries = num_output_sticks_per_core,
        .data_format_metadata = dfb_data_format,
        .borrowed_from = OUTPUT,
    });
    if (output_row_size_bytes > slice_row_size_bytes) {
        pad_output_row = true;
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = TEMP_PAD,
            .entry_size = output_row_size_bytes,
            .num_entries = 1,
            .data_format_metadata = dfb_data_format,
        });
    }
    uint32_t num_trids = 2;
    if (is_non_aligned) {
        // Scratch page must accommodate padded_stick_size + worst-case misalignment.
        const uint32_t scratch_page_size =
            tt::align((a.logical_shape()[-1] * a.element_size()) + src_buffer_alignment, src_buffer_alignment);
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = NON_ALIGNED,
            .entry_size = scratch_page_size,
            .num_entries = num_trids,
            .data_format_metadata = dfb_data_format,
        });
    }

    // Scratchpad for the reader's per-dimension index counter: seeded from the id_per_dim vararg block,
    // then advanced on the device.
    const std::uint32_t num_dims = static_cast<std::uint32_t>(a.logical_shape().rank());
    ScratchpadSpec id_per_dim_scratch{
        .unique_id = ID_PER_DIM,
        .size_per_node = num_dims * static_cast<uint32_t>(sizeof(uint32_t)),
    };

    // ---- Kernels ----
    // The non-aligned scratch exists only when the read window isn't aligned to the source buffer; the
    // reader compiles that path, and binds the scratch, under NON_ALIGNED.
    Group<DFBBinding> reader_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
    };
    KernelSpec::CompilerOptions::Defines reader_defines;
    if (is_non_aligned) {
        // The reader alone fills and drains the scratch (TRID staging), so it binds both ends.
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = NON_ALIGNED,
            .accessor_name = "non_aligned",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = NON_ALIGNED,
            .accessor_name = "non_aligned",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        reader_defines.emplace("NON_ALIGNED", "1");
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
            "padded_slice_reader_rm_interleaved_start_id.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .scratchpad_bindings =
            {
                ScratchpadBinding{
                    .scratchpad_spec_name = ID_PER_DIM,
                    .accessor_name = "id_per_dim",
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "src",
                },
            },
        .compile_time_args = {{"num_trids", num_trids}},
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {
                        "unpadded_stick_size",
                        "stick_size_offset",
                        "num_dims",
                        "start_id",
                        "num_sticks_per_core",
                        "num_sticks_per_core_read",
                        "num_read_per_barrier",
                        "src_offset_bytes",
                        "misalignment",
                    },
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
        // Three num_dims-long blocks: num_unpadded_sticks, num_padded_sticks, then the id_per_dim seed.
        .advanced_options = {.num_runtime_varargs = 3 * num_dims},
    };

    // pad_output_row selects the writer source, as before: the op's pad writer when the output row is
    // wider than the sliced row, otherwise the Metal 2.0 fork of data_movement/sharded's plain writer,
    // whose binding and argument names (dfb::out, num_units) are the fork's interface, inherited here.
    // The output DFB is fill-only on the pad-writer path (the output is resident); the CONSUMER label
    // there only completes the reader's PRODUCER end.
    const DFBBinding writer_out_binding{
        .dfb_spec_name = OUT,
        .accessor_name = "out",
        .endpoint_type = DFBEndpointType::CONSUMER,
    };
    KernelSpec writer =
        pad_output_row
            ? KernelSpec{
                  .unique_id = WRITER,
                  .source =
                      "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
                      "writer_unary_sharded_padded_rm.cpp",
                  .dfb_bindings =
                      {
                          writer_out_binding,
                          // The pad writer alone zero-fills the pad row and NoC-reads from it, so it
                          // binds both ends.
                          DFBBinding{
                              .dfb_spec_name = TEMP_PAD,
                              .accessor_name = "temp_pad",
                              .endpoint_type = DFBEndpointType::PRODUCER,
                          },
                          DFBBinding{
                              .dfb_spec_name = TEMP_PAD,
                              .accessor_name = "temp_pad",
                              .endpoint_type = DFBEndpointType::CONSUMER,
                          },
                      },
                  .compile_time_args = {{"output_elem_size", output.element_size()}},
                  .runtime_arg_schema =
                      {
                          .runtime_arg_names =
                              {
                                  "num_units",
                                  "num_elements_per_row",
                                  "unpadded_row_size_bytes",
                                  "padded_row_size_bytes",
                              },
                      },
                  .hw_config = ttnn::create_writer_datamovement_config(),
              }
            : KernelSpec{
                  .unique_id = WRITER,
                  .source =
                      "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
                      "writer_unary_sharded_metal2.cpp",
                  .dfb_bindings = {writer_out_binding},
                  .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
                  .hw_config = ttnn::create_writer_datamovement_config(),
              };

    // ---- Per-core runtime args ----
    auto all_runtime_args = get_padded_slice_runtime_args_rm_sharded_output(
        a, output, output_tensor_start, actual_output_shape, iter_cores);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    uint32_t i = 0;
    for (const auto& core : iter_cores) {
        const std::vector<uint32_t>& r = all_runtime_args[i].first;
        const std::vector<uint32_t>& w = all_runtime_args[i].second;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"unpadded_stick_size", r[0]},
             {"stick_size_offset", r[1]},
             {"num_dims", r[2]},
             {"start_id", r[3]},
             {"num_sticks_per_core", r[4]},
             {"num_sticks_per_core_read", r[5]},
             {"num_read_per_barrier", r[6]},
             {"src_offset_bytes", r[7]},
             {"misalignment", r[8]}});
        // The three per-dimension blocks follow the scalars in the same order the kernel reads them.
        reader_run_args.advanced_options.runtime_varargs[core] =
            std::vector<uint32_t>(r.begin() + kPaddedSliceRmReaderVarargsBase, r.end());

        if (pad_output_row) {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"num_units", w[0]},
                 {"num_elements_per_row", w[1]},
                 {"unpadded_row_size_bytes", w[2]},
                 {"padded_row_size_bytes", w[3]}});
        } else {
            // The plain writer reads only the unit count.
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"num_units", w[0]}});
        }
        i++;
    }

    ProgramSpec spec{
        .name = "padded_slice_rm",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .scratchpads = {std::move(id_per_dim_scratch)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER},
                    .target_nodes = total_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, input_mesh},
        {OUTPUT, output_mesh},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

tt::tt_metal::experimental::ProgramRunArgs PaddedSliceRMProgramFactory::override_runtime_arguments(
    const PaddedSliceParams& /*operation_attributes*/,
    const PaddedSliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*coord*/) {
    using namespace padded_slice_rm_metal2;

    const auto& src_tensor = tensor_args.input;
    auto& dst_tensor = output;
    TT_FATAL(dst_tensor.is_sharded(), "Output tensor must be sharded");
    auto* src_buffer = src_tensor.buffer();
    TT_FATAL(src_buffer != nullptr, "Input buffer should be allocated on device");
    auto* dst_buffer = dst_tensor.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // The read-window offsets follow from the slice bounds and the output sharding, both part of the
    // program-cache key; only the input buffer's base and the output shard backing the output DFB can
    // differ between dispatches. Both travel as tensor bindings, so a cache hit refreshes exactly those.
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {INPUT, src_tensor.mesh_tensor()},
        {OUTPUT, dst_tensor.mesh_tensor()},
    };
    return run_args;
}

}  // namespace ttnn::experimental::prim
