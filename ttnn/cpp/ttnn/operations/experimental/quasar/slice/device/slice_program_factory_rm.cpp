// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_program_factory_rm.hpp"

#include <optional>
#include <vector>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/spec_run_args.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::operations::experimental::quasar {

namespace {

// Per-core runtime arguments for the row-major slice, split into the named scalar
// fields (mapped to named RTAs) and the per-core id_per_dim runtime varargs.
struct SliceRmPerCoreArgs {
    uint32_t start_id = 0;
    uint32_t num_sticks_per_core = 0;
    uint32_t num_sticks_per_core_read = 0;
    uint32_t num_read_per_barrier = 0;
    uint32_t num_sticks_written = 0;  // writer start_id
    std::vector<uint32_t> id_per_dim;
};

// Common (all-core) runtime arguments for the row-major slice.
struct SliceRmCommonArgs {
    // Reader scalar fields (the legacy fixed reader slots [0..5], minus the buffer
    // address which is now bound via the INPUT TensorParameter):
    //   begins_offset_bytes = begins_bytes - misalignment (the W-dim slice start, applied
    //                         to the accessor page address in-kernel; see kernel comment),
    uint32_t begins_offset_bytes = 0;
    uint32_t unpadded_stick_size = 0;
    uint32_t stick_size_offset = 0;  // unpadded_row_size_bytes_offset (L1 stride between sticks)
    uint32_t misalignment = 0;
    // Writer scalar fields:
    uint32_t unpadded_row_size_bytes = 0;
    uint32_t unpadded_row_size_bytes_offset = 0;
    // Common runtime varargs (read by a runtime-varying index in the reader's inner loop):
    //   [0, num_dims)        -> num_unpadded_sticks_per_dim
    //   [num_dims, 2*num_dims)-> num_padded_sticks_per_dim
    std::vector<uint32_t> reader_common_varargs;
};

// Computes the per-core and common runtime args, preserving the legacy stick-walk EXACTLY.
inline std::pair<SliceRmCommonArgs, std::vector<SliceRmPerCoreArgs>> get_slice_runtime_args_rm(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    uint32_t num_cores,
    const std::vector<CoreCoord>& all_cores_vec,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_1,
    uint32_t num_sticks_per_core_group_2,
    uint32_t max_read_size) {
    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    uint32_t unpadded_row_size_bytes = output_shape[-1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    uint32_t begins_bytes = output_tensor_start[-1] * input_tensor.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;
    uint32_t unpadded_row_size_bytes_offset = tt::round_up(unpadded_row_size_bytes, alignment);

    // NOTE: under the spec ABI the per-shard split (shard_W * elem for B/W-sharded, full row
    // otherwise) is driven by the BOUND TensorAccessor's own aligned page size, derived from each
    // tensor's TensorSpec — so the legacy per-shard page-size overrides are no longer dispatched.

    SliceRmCommonArgs common;
    // begins_offset_bytes: the W-dim slice start (rounded down to the nearest aligned address),
    // formerly folded into the source buffer base address (legacy reader slot [0] =
    // start_addr + begins_bytes - misalignment). The base address is now bound via the INPUT
    // TensorParameter, so the kernel applies this byte offset to the accessor page address.
    common.begins_offset_bytes = begins_bytes - misalignment;
    common.unpadded_stick_size = unpadded_row_size_bytes;
    common.stick_size_offset = unpadded_row_size_bytes_offset;
    common.misalignment = misalignment;
    common.unpadded_row_size_bytes = unpadded_row_size_bytes;
    common.unpadded_row_size_bytes_offset = unpadded_row_size_bytes_offset;
    // Common runtime varargs layout: [0, num_dims) num_unpadded, [num_dims, 2*num_dims) num_padded.
    common.reader_common_varargs.reserve(2 * num_dims);
    common.reader_common_varargs.insert(
        common.reader_common_varargs.end(), num_unpadded_sticks_per_dim.begin(), num_unpadded_sticks_per_dim.end());
    common.reader_common_varargs.insert(
        common.reader_common_varargs.end(), num_padded_sticks_per_dim.begin(), num_padded_sticks_per_dim.end());

    std::vector<SliceRmPerCoreArgs> ret_val;
    ret_val.reserve(num_cores);

    uint32_t start_offset =
        ttnn::operations::experimental::quasar::get_rm_start_offset(input_tensor, output_tensor_start);
    uint32_t num_sticks_written = 0;
    for (const auto& core : all_cores_vec) {
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            num_sticks_per_core = 0;
        }

        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            auto num_sticks_per_core_pad32 = num_sticks_per_core + ((32 - num_sticks_per_core % 32) % 32);
            num_sticks_per_core_read = tt::tt_metal::merge_num_sticks_to_read(
                num_sticks_per_core_pad32, unpadded_row_size_bytes_offset, max_read_size);
            num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
        }

        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        SliceRmPerCoreArgs args;
        args.start_id = start_id;
        args.num_sticks_per_core = num_sticks_per_core;
        args.num_sticks_per_core_read = num_sticks_per_core_read;
        args.num_read_per_barrier = num_read_per_barrier;
        args.num_sticks_written = num_sticks_written;
        args.id_per_dim.assign(id_per_dim.begin(), id_per_dim.end());

        num_sticks_written += num_sticks_per_core;
        ret_val.push_back(std::move(args));
    }

    return {std::move(common), std::move(ret_val)};
}

constexpr uint32_t MAX_READ_SIZE = 4096;

std::tuple<uint32_t, uint32_t, uint32_t> compute_cb_size(
    const Tensor& input,
    const Tensor& output,
    const Shape& output_tensor_start,
    const uint32_t num_sticks_per_core_group_1,
    const uint32_t num_sticks_per_core_group_2) {
    auto src_buffer_alignment = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    const auto single_alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    auto alignment = single_alignment;

    uint32_t begins_bytes = output_tensor_start[-1] * input.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;

    if (misalignment != 0) {
        alignment *= 2;
    }
    const uint32_t unpadded_row_size_bytes = output.padded_shape()[-1] * input.element_size();
    const uint32_t cb_page_size = tt::round_up(unpadded_row_size_bytes, alignment);
    const uint32_t stick_stride_for_merge = tt::round_up(unpadded_row_size_bytes, single_alignment);
    const uint32_t num_input_pages = num_sticks_per_core_group_1 > num_sticks_per_core_group_2
                                         ? num_sticks_per_core_group_1
                                         : num_sticks_per_core_group_2;
    uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
    if (num_input_pages != 0) {
        auto num_sticks_per_core_pad32 = num_input_pages + ((32 - num_input_pages % 32) % 32);
        num_sticks_per_core_read =
            tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, stick_stride_for_merge, MAX_READ_SIZE);
        num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
    }

    return std::make_tuple(cb_page_size, num_read_per_barrier, misalignment);
}

}  // namespace

}  // namespace ttnn::operations::experimental::quasar

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramSpecArtifacts SliceRmProgramFactory::create_program_spec(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();

    uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_sticks)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    // Resource names (prefixed to avoid unity-build collisions with the other slice factories).
    const DFBSpecName C0{"slice_rm_c0"};  // legacy CB index 0: stick stream (reader -> writer)
    const TensorParamName INPUT{"slice_rm_input"};
    const TensorParamName OUTPUT{"slice_rm_output"};
    const KernelSpecName READER{"slice_rm_reader"};
    const KernelSpecName WRITER{"slice_rm_writer"};

    // --- DataflowBuffer (legacy CB c_0, normal/non-borrowed) ---
    // The DFB's entry_size / num_entries depend on slice_start (via misalignment /
    // unpadded_row_size_bytes), so each unique slice layout produces a distinct ProgramSpec
    // (the spec is the cache key). entry_size * num_entries == legacy total_size.
    const auto [cb_page_size, num_read_per_barrier_cb, misalignment_cb] =
        ttnn::operations::experimental::quasar::compute_cb_size(
            input, output, args.slice_start, num_sticks_per_core_group_1, num_sticks_per_core_group_2);

    DataflowBufferSpec c0_dfb{
        .unique_id = C0,
        .entry_size = cb_page_size,
        .num_entries = num_read_per_barrier_cb * 2,
        .data_format_metadata = cb_data_format,
    };

    std::uint32_t num_dims = static_cast<std::uint32_t>(input.padded_shape().rank());

    // --- Reader KernelSpec ---
    // Legacy reader appended TensorAccessorArgs(*src0_buffer) as CTAs and dispatched the
    // buffer address as runtime slot [0]; under the spec ABI the input buffer is bound via
    // the INPUT TensorParameter and accessed through TensorAccessor(tensor::in), so the
    // address RTA and the TensorAccessorArgs CTAs are dropped.
    //
    // The fixed reader slots [0..9] are remapped as follows:
    //   slot [0] (buffer addr + W-dim begins) -> base addr bound via INPUT; the W-dim byte
    //            offset becomes the begins_offset_bytes RTA (applied as the NOC read offset).
    //   slot [1] (padded_stick_size, per-shard page-size override) -> DROPPED (page size now
    //            derived from the bound TensorAccessor's TensorSpec).
    //   slot [4] (num_dims) -> CTA (same on every core, bounds the vararg loops).
    //   slots [2,3,5] (unpadded_stick_size, stick_size_offset, misalignment) -> named RTAs (common values).
    //   slots [6..9] (start_id, num_sticks_per_core, num_sticks_per_core_read, num_read_per_barrier)
    //            -> named RTAs (per-core values).
    // num_unpadded_sticks_per_dim / num_padded_sticks_per_dim -> common runtime varargs (2*num_dims);
    // id_per_dim -> per-core runtime varargs (num_dims).
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/"
            "slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = C0, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "in"}},
        // num_dims is the same on every core and indexes the vararg loops, so it is a CTA
        // (the legacy slot [4]); the remaining fixed reader slots map to named RTAs.
        .compile_time_args = {{"num_dims", num_dims}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"begins_offset_bytes",
                  "unpadded_stick_size",
                  "stick_size_offset",
                  "misalignment",
                  "start_id",
                  "num_sticks_per_core",
                  "num_sticks_per_core_read",
                  "num_read_per_barrier"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
        .advanced_options = {.num_runtime_varargs = num_dims, .num_common_runtime_varargs = 2 * num_dims},
    };

    // --- Writer KernelSpec ---
    // Legacy writer CTA slot [0] = src0_cb_index (now the bound DFB) and slot [1..] =
    // TensorAccessorArgs(*dst_buffer); both are dropped under the spec ABI (DFB via
    // DataflowBuffer(dfb::cb_out), output buffer via TensorAccessor(tensor::out)). The legacy
    // writer runtime slot [0] (output buffer address) is likewise dropped; the per-shard
    // page-size override (slot [7]) is dropped because the bound TensorAccessor derives the
    // per-shard page size from the OUTPUT tensor's TensorSpec.
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/"
            "slice_writer_unary_stick_layout_interleaved_start_id.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = C0, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"unpadded_row_size_bytes",
                  "unpadded_row_size_bytes_offset",
                  "num_sticks_per_core",
                  "num_sticks_per_core_read",
                  "num_read_per_barrier",
                  "start_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // --- Per-core runtime args (the stick walk, preserved exactly from legacy) ---
    auto all_cores_vec = corerange_to_cores(all_cores);
    auto [common, per_core] = ttnn::operations::experimental::quasar::get_slice_runtime_args_rm(
        input,
        output,
        args.slice_start,
        num_cores,
        all_cores_vec,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        ttnn::operations::experimental::quasar::MAX_READ_SIZE);

    // Reader: named per-core args + per-core id_per_dim varargs + common varargs.
    Group<KernelRunArgs::NodeRuntimeArgs> reader_node_args;
    Group<KernelRunArgs::NodeRuntimeArgs> writer_node_args;
    AdvancedKernelRunArgs reader_run_advanced;
    reader_node_args.reserve(all_cores_vec.size());
    writer_node_args.reserve(all_cores_vec.size());

    for (size_t i = 0; i < all_cores_vec.size(); ++i) {
        const NodeCoord node = all_cores_vec[i];
        const auto& pc = per_core[i];

        // Every core (including no-op cores, where num_sticks_per_core == 0) gets reader +
        // writer args and a full-length id_per_dim vararg vector, matching the legacy emit.
        reader_node_args.push_back(
            {.node = node,
             .args =
                 {{"begins_offset_bytes", common.begins_offset_bytes},
                  {"unpadded_stick_size", common.unpadded_stick_size},
                  {"stick_size_offset", common.stick_size_offset},
                  {"misalignment", common.misalignment},
                  {"start_id", pc.start_id},
                  {"num_sticks_per_core", pc.num_sticks_per_core},
                  {"num_sticks_per_core_read", pc.num_sticks_per_core_read},
                  {"num_read_per_barrier", pc.num_read_per_barrier}}});
        reader_run_advanced.runtime_varargs.emplace(node, pc.id_per_dim);

        writer_node_args.push_back(
            {.node = node,
             .args =
                 {{"unpadded_row_size_bytes", common.unpadded_row_size_bytes},
                  {"unpadded_row_size_bytes_offset", common.unpadded_row_size_bytes_offset},
                  {"num_sticks_per_core", pc.num_sticks_per_core},
                  {"num_sticks_per_core_read", pc.num_sticks_per_core_read},
                  {"num_read_per_barrier", pc.num_read_per_barrier},
                  {"start_id", pc.num_sticks_written}}});
    }

    // --- TensorParameters ---
    TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // --- Assemble ProgramSpec ---
    ProgramSpec spec;
    spec.name = "slice_rm";
    spec.kernels = {reader, writer};
    spec.dataflow_buffers = {c0_dfb};
    spec.tensor_parameters = {input_param, output_param};
    spec.work_units = {WorkUnitSpec{
        .name = "slice_rm",
        .kernels = {READER, WRITER},
        .target_nodes = all_cores,
    }};

    // --- Assemble ProgramRunArgs ---
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = std::move(reader_node_args),
            .common_runtime_arg_values = {},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = std::move(reader_run_advanced.runtime_varargs),
                    .common_runtime_varargs = std::move(common.reader_common_varargs)},
        },
        KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = std::move(writer_node_args),
        },
    };
    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramSpecArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
