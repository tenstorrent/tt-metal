// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_multi_core_default_program_factory.hpp"

#include <algorithm>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;
static const uint32_t max_read_size = 2048;  // max read size in bytes for reader and writer kernels

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {

uint32_t get_num_stick_per_barrier(uint32_t stick_size_padded_aligned) {
    return std::max(tt::div_up(max_read_size, stick_size_padded_aligned), 1u);
}

}  // namespace

ttnn::device_operation::ProgramArtifacts PadRmReaderWriterMultiCoreDefaultProgramFactory::create_program_spec(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    const auto& input_tensor_start = operation_attributes.input_tensor_start;

    const auto& a_shape = a.logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    uint32_t W_padded = output_padded_shape[3], H_padded = output_padded_shape[2], C_padded = output_padded_shape[1],
             N_padded = output_padded_shape[0];
    uint32_t NCH_padded = H_padded * C_padded * N_padded;

    const auto& front_pad = input_tensor_start;

    auto stick_size = W * a.element_size();
    auto stick_size_padded = W_padded * a.element_size();
    auto stick_size_padded_front = front_pad[-1] * a.element_size();
    auto stick_size_padded_end = stick_size_padded - stick_size - stick_size_padded_front;
    uint32_t stick_size_padded_aligned = tt::align(stick_size_padded, hal::get_l1_alignment());
    uint32_t stick_size_padded_DRAM_aligned = tt::align(stick_size_padded, hal::get_dram_alignment());
    [[maybe_unused]] uint32_t row_major_min_bytes = 16;

    // Input page-based addressing
    uint32_t num_input_pages_in_row = 1;
    uint32_t input_page_size = a.buffer()->page_size();
    uint32_t size_of_valid_data_in_last_input_page_in_row = a.buffer()->page_size();
    if (a.is_sharded()) {
        uint32_t shard_width =
            a.shard_spec().has_value() ? a.shard_spec().value().shape[1] : a.nd_shard_spec().value().shard_shape[-1];
        num_input_pages_in_row = tt::div_up(a.logical_shape()[-1], shard_width);
        size_of_valid_data_in_last_input_page_in_row = stick_size - (num_input_pages_in_row - 1) * input_page_size;
    }

    // Output page-based addressing
    uint32_t num_output_pages_in_row = 1;
    uint32_t output_page_size = output.buffer()->page_size();
    uint32_t size_of_valid_data_in_last_output_page_in_row = output.buffer()->page_size();
    if (output.is_sharded()) {
        uint32_t output_shard_width = output.shard_spec().has_value() ? output.shard_spec().value().shape[1]
                                                                      : output.nd_shard_spec().value().shard_shape[-1];
        num_output_pages_in_row = tt::div_up(W_padded, output_shard_width);
        size_of_valid_data_in_last_output_page_in_row =
            stick_size_padded - (num_output_pages_in_row - 1) * output_page_size;
    }

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    IDevice* device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_sticks_padded_per_core_group_1,
         num_sticks_padded_per_core_group_2] =
            sub_core_grids.has_value() ? tt::tt_metal::split_work_to_cores(sub_core_grids.value(), NCH_padded)
                                       : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, NCH_padded);

    auto cores_in_order = corerange_to_cores(all_cores, num_cores, true);

    // construct const buffer with the pad_value
    bool not_pad_by_zero = pad_value != 0;

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
    }

    bool unaligned = stick_size_padded_aligned % hal::get_dram_alignment() != 0;
    // c_2 (pad_align) is allocated (and bound) only when there is front padding or an unaligned
    // stick; the kernel references dfb::pad_align only under FRONT_PAD_OR_UNALIGNED to match.
    bool front_pad_or_unaligned = (stick_size_padded_front != 0 || unaligned);

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "pad_rm_reader_writer_multi_core_default";

    uint32_t num_sticks_per_barrier = get_num_stick_per_barrier(stick_size_padded_aligned);
    const uint32_t buffer_reader_writer_async_factor = 16;

    // c_0 (in0): reader fills, writer drains.
    // c_1 (pad): pad-value fill scratch — fake CB the reader reads only as an address source.
    // c_2 (pad_align): pad-align scratch — fake CB, allocated only when front_pad_or_unaligned.
    spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = m2::DFBSpecName{"in0"},
        .entry_size = stick_size_padded_aligned,
        .num_entries = buffer_reader_writer_async_factor * num_sticks_per_barrier,
        .data_format_metadata = cb_data_format});
    spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = m2::DFBSpecName{"pad"},
        .entry_size = stick_size_padded_DRAM_aligned,
        .num_entries = 1,
        .data_format_metadata = cb_data_format});
    if (front_pad_or_unaligned) {
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"pad_align"},
            .entry_size = stick_size_padded_DRAM_aligned,
            .num_entries = 1,
            .data_format_metadata = cb_data_format});
    }

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"dst"}, .spec = output.tensor_spec()},
    };

    // Reader DFB bindings: in0 PRODUCER, pad self-loop, and pad_align self-loop (when allocated).
    m2::Group<m2::DFBBinding> reader_dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in0"},
            .accessor_name = "in0",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"pad"},
            .accessor_name = "pad",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"pad"},
            .accessor_name = "pad",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
    };
    if (front_pad_or_unaligned) {
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"pad_align"},
            .accessor_name = "pad_align",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"pad_align"},
            .accessor_name = "pad_align",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "reader_pad_dims_rm_interleaved_v2.cpp"},
        .compiler_options =
            {.defines = front_pad_or_unaligned ? m2::Table<std::string, std::string>{{"FRONT_PAD_OR_UNALIGNED", "1"}}
                                               : m2::Table<std::string, std::string>{}},
        .dfb_bindings = reader_dfb_bindings,
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"src"}, .accessor_name = "src"}},
        .compile_time_args =
            {{"N", (std::uint32_t)N + front_pad[-4]},
             {"H", (std::uint32_t)H + front_pad[-2]},
             {"C", (std::uint32_t)C + front_pad[-3]},
             {"stick_size_bytes", (std::uint32_t)stick_size},
             {"N_padded", (std::uint32_t)N_padded},
             {"H_padded", (std::uint32_t)H_padded},
             {"C_padded", (std::uint32_t)C_padded},
             {"stick_size_padded", (std::uint32_t)stick_size_padded},
             {"stick_size_padded_front", (std::uint32_t)stick_size_padded_front},
             {"stick_size_padded_end", (std::uint32_t)stick_size_padded_end},
             {"num_zero_pad_sticks_read", (std::uint32_t)tt::div_up(stick_size_padded, 512)},  // max zero size is 512B
             {"last_zero_stick_size", (std::uint32_t)(stick_size_padded % 512 == 0 ? 512 : stick_size_padded % 512)},
             {"not_pad_by_zero", (std::uint32_t)not_pad_by_zero},
             {"packed_pad_value", (std::uint32_t)packed_pad_value},
             {"stick_size_padded_aligned", (std::uint32_t)stick_size_padded_aligned},
             {"unaligned", (std::uint32_t)unaligned},
             {"num_input_pages_in_row", (std::uint32_t)num_input_pages_in_row},
             {"input_page_size", (std::uint32_t)input_page_size},
             {"size_of_valid_data_in_last_input_page_in_row",
              (std::uint32_t)size_of_valid_data_in_last_input_page_in_row}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks_per_core",
                  "num_sticks_per_barrier",
                  "start_page_id",
                  "front_pad_n",
                  "front_pad_c",
                  "front_pad_h"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };
    // start_dim_offset[4]: per-core starting dim indices {0, curr_h, curr_c, curr_n}, passed as
    // runtime varargs (the kernel indexes [1],[2],[3], matching the legacy get_arg_addr layout).
    constexpr std::uint32_t kNumDimOffsets = 4;
    reader.advanced_options.num_runtime_varargs = kNumDimOffsets;

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "writer_pad_dims_rm_interleaved_v2.cpp"},
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in0"},
            .accessor_name = "in0",
            .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"dst"}, .accessor_name = "dst"}},
        .compile_time_args =
            {{"stick_size_bytes", (std::uint32_t)stick_size_padded},
             {"stick_size_padded_aligned", (std::uint32_t)stick_size_padded_aligned},
             {"num_output_pages_in_row", (std::uint32_t)num_output_pages_in_row},
             {"output_page_size", (std::uint32_t)output_page_size},
             {"size_of_valid_data_in_last_output_page_in_row",
              (std::uint32_t)size_of_valid_data_in_last_output_page_in_row}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_sticks_per_core", "num_sticks_per_barrier", "start_page_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    spec.kernels = {reader, writer};

    // Local/self-loop DFBs require producer and consumer to share a WorkUnitSpec.  Both kernels run
    // on all_cores, so one WorkUnitSpec hosts both.
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "multi_core",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = all_cores},
    };

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    // The legacy helper used input_tensor.padded_shape() for the H/C/N bounds — mirror that here.
    // H_padded/C_padded already use the output padded shape.
    auto input_padded_shape = a.padded_shape();
    uint32_t H_in = input_padded_shape[2], C_in = input_padded_shape[1], N_in = input_padded_shape[0];
    std::vector<uint32_t> start_dim_offset(kNumDimOffsets, 0);

    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    uint32_t curr_sticks_read = 0;
    uint32_t curr_sticks_write = 0;
    for (const auto& core : cores_in_order) {
        const m2::NodeCoord node{core};
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_padded_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_padded_per_core_group_2;
        } else {
            // no-op
            num_sticks_per_core = 0;
        }

        // The legacy buffer-address RTA slot (0=src/dst; 0u on idle cores) is now carried by the
        // src/dst TensorBindings; idle cores short-circuit on num_sticks_per_core == 0.
        reader_run.runtime_arg_values.push_back(
            {node,
             {{"num_sticks_per_core", num_sticks_per_core},
              {"num_sticks_per_barrier", num_sticks_per_barrier},
              {"start_page_id", curr_sticks_read * num_input_pages_in_row},
              {"front_pad_n", static_cast<uint32_t>(front_pad[-4])},
              {"front_pad_c", static_cast<uint32_t>(front_pad[-3])},
              {"front_pad_h", static_cast<uint32_t>(front_pad[-2])}}});

        m2::AdvancedKernelRunArgs::Varargs reader_varargs;
        reader_varargs.reserve(kNumDimOffsets);
        for (uint32_t v : start_dim_offset) {
            reader_varargs.push_back(v);
        }
        reader_run.advanced_options.runtime_varargs[node] = std::move(reader_varargs);

        writer_run.runtime_arg_values.push_back(
            {node,
             {{"num_sticks_per_core", num_sticks_per_core},
              {"num_sticks_per_barrier", num_sticks_per_barrier},
              {"start_page_id", curr_sticks_write * num_output_pages_in_row}}});

        curr_sticks_write += num_sticks_per_core;

        for (uint32_t k = 0; k < num_sticks_per_core; ++k) {
            if ((curr_h >= front_pad[-2] and curr_h < (H_in + front_pad[-2])) and
                (curr_c >= front_pad[-3] and curr_c < (C_in + front_pad[-3])) and
                (curr_n >= front_pad[-4] and curr_n < (N_in + front_pad[-4]))) {
                curr_sticks_read++;
            }

            curr_h++;
            if (curr_h == H_padded) {
                curr_c++;
                curr_h = 0;
                if (curr_c == C_padded) {
                    curr_n++;
                    curr_c = 0;
                }
            }
        }

        start_dim_offset = {0, curr_h, curr_c, curr_n};
    }

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"src"}, a.mesh_tensor()},
        {m2::TensorParamName{"dst"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
