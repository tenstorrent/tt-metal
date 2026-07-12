// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal-2.0 (Quasar) row-major padded_slice factory. Emits a ProgramSpec + QuasarDataMovementKernel
// (Quasar rejects the legacy DataMovementKernel the descriptor path builds -- kernel.hpp:382).
// Structure mirrors the quasar interleaved_to_sharded factory: the OUTPUT DFB is borrowed onto the
// sharded-L1 output buffer; the reader PRODUCES the sliced sticks into it (from the interleaved src
// TensorAccessor), and a writer CONSUMES/commits it (reusing the ported i2s writer_unary_sharded for
// the non-pad case). The per-dim slice geometry (num_output/num_input sticks + id_per_dim) is passed as
// positional runtime varargs (Metal-2 has no array-valued named args), read via get_vararg in the kernel.

#include "padded_slice_rm_program_factory.hpp"
#include "padded_slice_utils.hpp"

#include "optional"
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/math.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

using namespace tt::tt_metal::experimental;

// Per-core reader/writer arg vectors (geometry unchanged from the shared RM factory). Reader vector:
//   [0]=src abs addr (aligned-down)+width_offset, [1]=padded_stick, [2]=unpadded_stick, [3]=stick_offset,
//   [4]=num_dims, [5]=start_id, [6..8]=num_sticks(x3), [9..)=num_output_sticks_per_dim, num_input_sticks_
//   per_dim, id_per_dim (each num_dims). Writer vector: {num_sticks, out_row_elems, in_row_bytes, out_row_bytes}.
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

    int input_page_size = input_shape[-1] * input_tensor.element_size();

    uint32_t output_row_size_bytes = output_shard_shape[1] * input_tensor.element_size();
    uint32_t output_row_size_elems = output_shard_shape[1];

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_output_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_input_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);
    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    num_output_sticks_per_dim[0] = 1;
    num_input_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < static_cast<int32_t>(num_dims); i++) {
        uint32_t num_output_dim = actual_output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_input_dim = (num_total_dim - num_output_dim) * accumulated_total_per_dim[i - 1];
        num_output_sticks_per_dim[i] = num_output_dim;
        num_input_sticks_per_dim[i] = num_input_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();
    auto dst_buffer_alignment = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? hal::get_dram_alignment()
                                    : hal::get_l1_alignment();

    uint32_t begins_bytes = output_tensor_start[-1] * input_tensor.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;

    uint32_t output_row_size_bytes_offset = tt::round_up(output_row_size_bytes, dst_buffer_alignment);
    uint32_t start_addr = input_tensor.buffer()->address();
    std::vector<uint32_t> common_reader_kernel_args = {
        start_addr + begins_bytes - misalignment,
        input_page_size,
        output_row_size_bytes,
        output_row_size_bytes_offset,
        num_dims,
        0,
        0,
        0,
        0};

    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_output_sticks_per_dim.begin(), num_output_sticks_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_input_sticks_per_dim.begin(), num_input_sticks_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    const uint32_t num_sticks_per_core = output_shard_spec.shape[0];
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
            std::max(std::min<int>(output_row_size_bytes, input_page_size - width_offset), 0);
        uint32_t this_core_num_sticks = num_sticks_per_core;
        if (this_input_row_size_bytes == 0) {
            this_core_num_sticks = 0;
        }
        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        reader_kernel_args[0] += width_offset;
        reader_kernel_args[2] = this_input_row_size_bytes;
        uint32_t addr_offset = 5;
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset++] = this_core_num_sticks;
        reader_kernel_args[addr_offset++] = this_core_num_sticks;
        reader_kernel_args[addr_offset] = this_core_num_sticks;
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        std::vector<uint32_t> writer_kernel_args = {
            this_core_num_sticks, output_row_size_elems, (uint32_t)this_input_row_size_bytes, output_row_size_bytes};
        ret_val[core_index] = {reader_kernel_args, writer_kernel_args};
        core_index++;
    }

    return ret_val;
}

namespace {
const TensorParamName PS_SRC{"padded_slice_src"};
const TensorParamName PS_OUT{"padded_slice_out"};
const DFBSpecName PS_OUT_DFB{"padded_slice_out_dfb"};
const ScratchpadSpecName PS_SCRATCH{"padded_slice_scratch"};
const KernelSpecName PS_READER{"padded_slice_reader"};
const KernelSpecName PS_WRITER{"padded_slice_writer"};
constexpr uint32_t kNumTrids = 2;
}  // namespace

ttnn::device_operation::ProgramArtifacts PaddedSliceRMProgramFactory::create_program_artifacts(
    const PaddedSliceParams& operation_attributes, const PaddedSliceInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& output_tensor_start = operation_attributes.padded_slice_start;
    const auto& output_tensor_end = operation_attributes.padded_slice_end;

    const ttnn::Shape output_shape = output.logical_shape();
    ttnn::Shape actual_output_shape = output_tensor_end;
    for (int i = 0; i < output_shape.rank(); i++) {
        actual_output_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    TT_FATAL(output.is_sharded(), "padded_slice output tensor must be sharded.");
    auto output_shard_spec = output.shard_spec().value();
    uint32_t output_row_size_bytes = output_shard_spec.shape[1] * output.element_size();

    CoreRangeSet total_cores = output_shard_spec.grid;
    bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    std::vector<CoreCoord> iter_cores = corerange_to_cores(total_cores, std::nullopt, rm_orientation);

    uint32_t num_cores_channels =
        ttnn::operations::experimental::quasar::detail::get_num_cores_channels_from_sharded_tensor(output);
    uint32_t input_row_size_bytes = (a.logical_shape()[-1] * a.element_size()) / num_cores_channels;

    TT_FATAL(
        output.buffer()->buffer_type() == tt::tt_metal::BufferType::L1,
        "padded_slice output buffer must be L1 (sharded).");

    auto src_buffer_alignment = a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    TT_FATAL(
        output_row_size_bytes % dst_buffer_alignment == 0,
        "padded_slice output row size {} must be aligned to {}",
        output_row_size_bytes,
        dst_buffer_alignment);
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    const bool is_non_aligned = (output_row_size_bytes % alignment) != 0;

    // pad_output_row: output row wider than the input row -> the padding writer. Not yet ported to Quasar
    // (the resnet stem is non-pad: sliced C == input C). The pad path can be added later mirroring the
    // shared writer_unary_sharded_padded_rm kernel.
    const bool pad_output_row = output_row_size_bytes > input_row_size_bytes;
    TT_FATAL(!pad_output_row, "Quasar padded_slice: pad-row path (output_row > input_row) not yet ported.");

    const uint32_t output_cb_page_size =
        is_non_aligned ? tt::round_up(output_row_size_bytes, dst_buffer_alignment) : output_row_size_bytes;
    const uint32_t num_output_sticks_per_core = output_shard_spec.shape[0];
    const uint32_t num_dims = static_cast<uint32_t>(a.logical_shape().rank());

    // ---- ProgramSpec ----
    ProgramSpec spec;
    spec.name = "padded_slice_rm";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = PS_SRC, .spec = a.tensor_spec()},
        TensorParameter{.unique_id = PS_OUT, .spec = output.tensor_spec()},
    };

    // OUTPUT DFB borrowed onto the sharded-L1 output buffer (reader produces, writer commits).
    DataflowBufferSpec out_dfb{
        .unique_id = PS_OUT_DFB,
        .entry_size = output_cb_page_size,
        .num_entries = num_output_sticks_per_core,
        .data_format_metadata = cb_data_format,
    };
    out_dfb.borrowed_from = PS_OUT;
    spec.dataflow_buffers.push_back(out_dfb);

    // TRID staging scratchpad (used only by the non-aligned path; reader fills+drains -> not a DFB / self-loop).
    // Declared UNCONDITIONALLY: the kernel references `scratch::pad` inside `if constexpr(is_non_aligned)`, and a
    // non-template if-constexpr still name-checks its discarded branch, so the binding must always exist even when
    // aligned. Unused (tiny) on the aligned path.
    {
        const uint32_t scratch_page =
            tt::align((a.logical_shape()[-1] * a.element_size()) + src_buffer_alignment, src_buffer_alignment);
        spec.scratchpads.push_back(ScratchpadSpec{.unique_id = PS_SCRATCH, .size_per_node = scratch_page * kNumTrids});
    }

    // Reader: interleaved src -> OUTPUT DFB.
    KernelSpec reader{
        .unique_id = PS_READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/padded_slice/device/kernels/dataflow/"
            "padded_slice_reader_rm_interleaved_start_id.cpp"};
    reader.tensor_bindings = {TensorBinding{.tensor_parameter_name = PS_SRC, .accessor_name = "src"}};
    reader.dfb_bindings = {ProducerOf(PS_OUT_DFB, "in0")};
    // Always bound (see the ScratchpadSpec note above): the kernel's `scratch::pad` reference must resolve
    // even when is_non_aligned is false (non-template if-constexpr still name-checks the discarded branch).
    reader.scratchpad_bindings = {ScratchpadBinding{.scratchpad_spec_name = PS_SCRATCH, .accessor_name = "pad"}};
    reader.compile_time_args = {
        {"is_non_aligned", is_non_aligned ? 1u : 0u},
        {"src_buffer_alignment", static_cast<uint32_t>(src_buffer_alignment)},
        {"num_trids", kNumTrids}};
    reader.runtime_arg_schema = {
        .runtime_arg_names = {
            "src_byte_offset",
            "padded_stick_size",
            "unpadded_stick_size",
            "stick_size_offset",
            "num_dims",
            "start_id",
            "num_sticks_per_core",
            "num_sticks_per_core_read",
            "num_read_per_barrier"}};
    // Per-dim geometry tail (num_output_sticks_per_dim, num_input_sticks_per_dim, id_per_dim).
    reader.advanced_options.num_runtime_varargs = 3 * num_dims;
    reader.hw_config = DataMovementHardwareConfig{
        .role = DataMovementRoleHint::READER,
        .gen2_config = DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true}};

    // Writer: drain/commit the OUTPUT DFB (non-pad). Reuses the ported i2s sharded writer.
    KernelSpec writer{
        .unique_id = PS_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/interleaved_to_sharded/device/kernels/dataflow/"
            "writer_unary_sharded.cpp"};
    writer.dfb_bindings = {ConsumerOf(PS_OUT_DFB, "out")};
    writer.runtime_arg_schema = {.runtime_arg_names = {"num_units"}};
    writer.hw_config = DataMovementHardwareConfig{
        .role = DataMovementRoleHint::WRITER,
        .gen2_config = DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true}};

    spec.kernels = {reader, writer};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {PS_READER, PS_WRITER}, .target_nodes = total_cores}};

    // ---- Per-core runtime args ----
    auto per_core = get_padded_slice_runtime_args_rm_sharded_output(
        a, output, output_tensor_start, actual_output_shape, iter_cores);

    const uint32_t start_addr = a.buffer()->address();
    const uint32_t begins_bytes = output_tensor_start[-1] * a.element_size();
    const uint32_t misalignment = begins_bytes % src_buffer_alignment;

    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = PS_READER};
    KernelRunArgs writer_run{.kernel = PS_WRITER};

    uint32_t i = 0;
    for (const auto& core : iter_cores) {
        const auto& r = per_core[i].first;
        const auto& w = per_core[i].second;
        // r[0] is the aligned-down absolute src addr (+width_offset); recover the per-core byte offset from
        // the tensor base (the kernel re-derives misalignment and reads from the aligned-down page).
        const uint32_t src_byte_offset = (r[0] - start_addr) + misalignment;
        reader_run.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core,
            .args = {
                {"src_byte_offset", src_byte_offset},
                {"padded_stick_size", r[1]},
                {"unpadded_stick_size", r[2]},
                {"stick_size_offset", r[3]},
                {"num_dims", r[4]},
                {"start_id", r[5]},
                {"num_sticks_per_core", r[6]},
                {"num_sticks_per_core_read", r[7]},
                {"num_read_per_barrier", r[8]}}});
        // Vararg tail: r[9 ..) == num_output_sticks_per_dim, num_input_sticks_per_dim, id_per_dim.
        reader_run.advanced_options.runtime_varargs[core] = std::vector<uint32_t>(r.begin() + 9, r.end());

        writer_run.runtime_arg_values.push_back(
            KernelRunArgs::NodeRuntimeArgs{.node = core, .args = {{"num_units", w[0]}}});
        i++;
    }

    run_args.kernel_run_args.push_back(reader_run);
    run_args.kernel_run_args.push_back(writer_run);
    run_args.tensor_args.emplace(PS_SRC, TensorArgument{a.mesh_tensor()});
    run_args.tensor_args.emplace(PS_OUT, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
