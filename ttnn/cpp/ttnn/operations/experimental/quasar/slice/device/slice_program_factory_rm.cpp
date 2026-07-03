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
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::operations::experimental::quasar {

namespace {

constexpr uint32_t MAX_READ_SIZE = 4096;

// Per-core runtime args computed by the host. The fixed reader fields shared by all cores
// (src_addr, reader_page_size, unpadded_row_size_bytes, stick_size_offset, misalignment) are
// supplied as common runtime args; the per-core fields (start_id, num_sticks_per_core,
// num_sticks_per_core_read, num_read_per_barrier) and the id_per_dim varargs are per-core.
struct SliceRmRuntimeArgs {
    // Common (shared) values.
    uint32_t addr_offset = 0;  // begins_bytes - misalignment (byte offset into the input buffer)
    uint32_t reader_page_size = 0;
    uint32_t unpadded_row_size_bytes = 0;
    uint32_t unpadded_row_size_bytes_offset = 0;
    uint32_t misalignment = 0;
    uint32_t num_dims = 0;
    std::vector<uint32_t> num_unpadded_sticks_per_dim;
    std::vector<uint32_t> num_padded_sticks_per_dim;

    // Per-core values.
    std::vector<uint32_t> start_id;
    std::vector<uint32_t> num_sticks_per_core;
    std::vector<uint32_t> num_sticks_per_core_read;
    std::vector<uint32_t> num_read_per_barrier;
    std::vector<std::vector<uint32_t>> id_per_dim;  // per-core array of length num_dims

    // Writer per-core values.
    uint32_t writer_page_size = 0;
    std::vector<uint32_t> num_sticks_written;
};

SliceRmRuntimeArgs get_slice_runtime_args_rm(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const std::vector<CoreCoord>& all_cores_vec,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_1,
    uint32_t num_sticks_per_core_group_2,
    uint32_t max_read_size) {
    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    uint32_t padded_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);

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

    // shard_W * elem_size for B/W-sharded (splits row across shards); full row otherwise.
    // Fallback is padded for the reader tensor, unpadded for the writer tensor.
    const auto per_shard_page_size_bytes = [&](const Tensor& t, uint32_t row_bytes) -> uint32_t {
        const auto& mc = t.memory_config();
        if (mc.is_sharded() && (mc.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                                mc.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED)) {
            const auto& spec = mc.shard_spec().value();
            return spec.shape[1] * t.element_size();
        }
        return row_bytes;
    };
    const uint32_t reader_page_size = per_shard_page_size_bytes(input_tensor, padded_row_size_bytes);

    SliceRmRuntimeArgs ret;
    ret.addr_offset = begins_bytes - misalignment;  // read from nearest aligned address (relative to buffer base)
    ret.reader_page_size = reader_page_size;
    ret.unpadded_row_size_bytes = unpadded_row_size_bytes;
    ret.unpadded_row_size_bytes_offset = unpadded_row_size_bytes_offset;
    ret.misalignment = misalignment;
    ret.num_dims = num_dims;
    ret.num_unpadded_sticks_per_dim = num_unpadded_sticks_per_dim;
    ret.num_padded_sticks_per_dim = num_padded_sticks_per_dim;
    ret.writer_page_size = per_shard_page_size_bytes(output_tensor, unpadded_row_size_bytes);

    const size_t num_cores = all_cores_vec.size();
    ret.start_id.reserve(num_cores);
    ret.num_sticks_per_core.reserve(num_cores);
    ret.num_sticks_per_core_read.reserve(num_cores);
    ret.num_read_per_barrier.reserve(num_cores);
    ret.id_per_dim.reserve(num_cores);
    ret.num_sticks_written.reserve(num_cores);

    std::vector<uint32_t> id_per_dim(num_dims);
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

        ret.start_id.push_back(start_id);
        ret.num_sticks_per_core.push_back(num_sticks_per_core);
        ret.num_sticks_per_core_read.push_back(num_sticks_per_core_read);
        ret.num_read_per_barrier.push_back(num_read_per_barrier);
        ret.id_per_dim.push_back(id_per_dim);
        ret.num_sticks_written.push_back(num_sticks_written);

        num_sticks_written += num_sticks_per_core;
    }

    return ret;
}

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

ttnn::device_operation::ProgramArtifacts SliceRmProgramFactory::create_program_artifacts(
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

    // DFB sizing varies with slice_start; padded_shape folds into compute_program_hash() so each
    // unique DFB layout gets its own cache entry (entry_size/num_entries are not patched on cache hit).
    const auto [cb_page_size, num_read_per_barrier, misalignment] =
        ttnn::operations::experimental::quasar::compute_cb_size(
            input, output, args.slice_start, num_sticks_per_core_group_1, num_sticks_per_core_group_2);

    // Resource names
    const DFBSpecName C0{"c0"};  // src->dst staging FIFO (legacy CB src0_cb_index = c_0)
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    // --- DataflowBuffer (legacy src0 CB, double-buffered worth of one barrier's reads) ---
    DataflowBufferSpec c0_dfb{
        .unique_id = C0,
        .entry_size = cb_page_size,
        .num_entries = num_read_per_barrier * 2,
        .data_format_metadata = cb_data_format,
    };

    auto all_cores_vec = corerange_to_cores(all_cores);
    auto rt = ttnn::operations::experimental::quasar::get_slice_runtime_args_rm(
        input,
        output,
        args.slice_start,
        all_cores_vec,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        ttnn::operations::experimental::quasar::MAX_READ_SIZE);

    const uint32_t num_dims = rt.num_dims;

    // Reader common runtime varargs layout: [0, num_dims) num_unpadded_sticks, [num_dims, 2*num_dims)
    // num_padded_sticks.
    std::vector<uint32_t> reader_common_varargs;
    reader_common_varargs.reserve(2 * num_dims);
    reader_common_varargs.insert(
        reader_common_varargs.end(), rt.num_unpadded_sticks_per_dim.begin(), rt.num_unpadded_sticks_per_dim.end());
    reader_common_varargs.insert(
        reader_common_varargs.end(), rt.num_padded_sticks_per_dim.begin(), rt.num_padded_sticks_per_dim.end());

    // --- Reader KernelSpec ---
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/"
            "slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp",
        .compiler_options = {},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = C0, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "in"}},
        .compile_time_args = {{"num_dims", num_dims}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"start_id", "num_sticks_per_core", "num_sticks_per_core_read", "num_read_per_barrier"},
             .common_runtime_arg_names =
                 {"addr_offset", "padded_stick_size", "unpadded_stick_size", "stick_size_offset", "misalignment"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = num_dims, .num_common_runtime_varargs = 2 * num_dims},
    };

    // --- Writer KernelSpec ---
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/"
            "slice_writer_unary_stick_layout_interleaved_start_id.cpp",
        .compiler_options = {},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = C0, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"stick_size",
                  "stick_size_offset",
                  "num_sticks_per_core",
                  "num_sticks_per_core_read",
                  "num_read_per_barrier",
                  "start_id",
                  "page_size_override"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // --- Per-core runtime args ---
    Group<KernelRunArgs::NodeRuntimeArgs> reader_node_args;
    Group<KernelRunArgs::NodeRuntimeArgs> writer_node_args;
    AdvancedKernelRunArgs reader_run_advanced;

    for (size_t i = 0; i < all_cores_vec.size(); ++i) {
        const auto& core = all_cores_vec[i];
        reader_node_args.push_back(
            {.node = core,
             .args = {
                 {"start_id", rt.start_id[i]},
                 {"num_sticks_per_core", rt.num_sticks_per_core[i]},
                 {"num_sticks_per_core_read", rt.num_sticks_per_core_read[i]},
                 {"num_read_per_barrier", rt.num_read_per_barrier[i]}}});
        reader_run_advanced.runtime_varargs.emplace(core, rt.id_per_dim[i]);

        writer_node_args.push_back(
            {.node = core,
             .args = {
                 {"stick_size", rt.unpadded_row_size_bytes},
                 {"stick_size_offset", rt.unpadded_row_size_bytes_offset},
                 {"num_sticks_per_core", rt.num_sticks_per_core[i]},
                 {"num_sticks_per_core_read", rt.num_sticks_per_core_read[i]},
                 {"num_read_per_barrier", rt.num_read_per_barrier[i]},
                 {"start_id", rt.num_sticks_written[i]},
                 {"page_size_override", rt.writer_page_size}}});
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
        .name = "slice_rm_wu",
        .kernels = {READER, WRITER},
        .target_nodes = all_cores,
    }};

    // --- Assemble ProgramRunArgs ---
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = std::move(reader_node_args),
            .common_runtime_arg_values =
                {{"addr_offset", rt.addr_offset},
                 {"padded_stick_size", rt.reader_page_size},
                 {"unpadded_stick_size", rt.unpadded_row_size_bytes},
                 {"stick_size_offset", rt.unpadded_row_size_bytes_offset},
                 {"misalignment", rt.misalignment}},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = std::move(reader_run_advanced.runtime_varargs),
                    .common_runtime_varargs = std::move(reader_common_varargs)},
        },
        KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = std::move(writer_node_args),
        },
    };
    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
