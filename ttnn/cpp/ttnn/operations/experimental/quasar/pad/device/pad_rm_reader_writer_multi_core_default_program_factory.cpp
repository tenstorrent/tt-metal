// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_multi_core_default_program_factory.hpp"

#include <algorithm>
#include <filesystem>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

static const uint32_t max_read_size = 2048;  // max read size in bytes for reader and writer kernels

namespace ttnn::prim::qsr {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {

uint32_t get_num_stick_per_barrier(uint32_t stick_size_padded_aligned) {
    return std::max(tt::div_up(max_read_size, stick_size_padded_aligned), 1u);
}

}  // namespace

ttnn::device_operation::ProgramArtifacts PadRmReaderWriterMultiCoreDefaultProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN0{"cb_in0"};              // legacy c_0: input stream (reader produces, writer consumes)
    const DFBSpecName CB_PAD{"cb_pad"};              // legacy c_1: pad-value scratchpad (reader self-loop)
    const DFBSpecName CB_PAD_ALIGN{"cb_pad_align"};  // legacy c_2: pad-align scratchpad (reader self-loop, optional)

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/"
        "reader_pad_dims_rm_interleaved_v2.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/"
        "writer_pad_dims_rm_interleaved_v2.cpp";

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
    bool has_pad_align = (stick_size_padded_front != 0) || unaligned;

    uint32_t num_sticks_per_barrier = get_num_stick_per_barrier(stick_size_padded_aligned);
    const uint32_t buffer_reader_writer_async_factor = 16;

    // ------------------------------------------------------------------------
    // DataflowBufferSpecs (legacy CBs c_0 / c_1 / [c_2]).
    //   cb_in0:      reader->writer FIFO stream.
    //   cb_pad:      reader self-loop pad-value scratchpad.
    //   cb_pad_align: reader self-loop pad-align scratchpad (only when front padding or unaligned).
    // ------------------------------------------------------------------------
    DataflowBufferSpec cb_in0_spec{
        .unique_id = CB_IN0,
        .entry_size = stick_size_padded_aligned,
        .num_entries = buffer_reader_writer_async_factor * num_sticks_per_barrier,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec cb_pad_spec{
        .unique_id = CB_PAD,
        .entry_size = stick_size_padded_DRAM_aligned,
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec cb_pad_align_spec{
        .unique_id = CB_PAD_ALIGN,
        .entry_size = stick_size_padded_DRAM_aligned,
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    };

    // ------------------------------------------------------------------------
    // Tensor parameters (Case-1 page access on both kernels).
    // ------------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()};

    // ------------------------------------------------------------------------
    // Reader KernelSpec: cb_in0 PRODUCER, cb_pad self-loop, [cb_pad_align self-loop].
    // The legacy reader's slot-0 buffer-address RTA is dropped (reached via TensorBinding tensor::src);
    // the legacy get_arg_addr(7) start-dim-offset array is read by constant indices, so it becomes
    // three named scalar RTAs.
    // ------------------------------------------------------------------------
    KernelSpec::CompileTimeArgs reader_cta{
        {"N", static_cast<uint32_t>(N + front_pad[-4])},
        {"H", static_cast<uint32_t>(H + front_pad[-2])},
        {"C", static_cast<uint32_t>(C + front_pad[-3])},
        {"stick_size_bytes", static_cast<uint32_t>(stick_size)},
        {"N_padded", N_padded},
        {"H_padded", H_padded},
        {"C_padded", C_padded},
        {"stick_size_padded", static_cast<uint32_t>(stick_size_padded)},
        {"stick_size_padded_front", static_cast<uint32_t>(stick_size_padded_front)},
        {"stick_size_padded_end", static_cast<uint32_t>(stick_size_padded_end)},
        {"num_zero_pad_sticks_read", static_cast<uint32_t>(tt::div_up(stick_size_padded, 512u))},
        {"last_zero_stick_size", static_cast<uint32_t>(stick_size_padded % 512 == 0 ? 512 : stick_size_padded % 512)},
        {"not_pad_by_zero", static_cast<uint32_t>(not_pad_by_zero)},
        {"packed_pad_value", packed_pad_value},
        {"stick_size_padded_aligned", stick_size_padded_aligned},
        {"unaligned", static_cast<uint32_t>(unaligned)},
        {"num_input_pages_in_row", num_input_pages_in_row},
        {"input_page_size", input_page_size},
        {"size_of_valid_data_in_last_input_page_in_row", size_of_valid_data_in_last_input_page_in_row},
    };

    Group<DFBBinding> reader_dfb_bindings{
        ProducerOf(CB_IN0, "cb_in0"),
        ProducerOf(CB_PAD, "cb_pad"),
        ConsumerOf(CB_PAD, "cb_pad"),
    };
    if (has_pad_align) {
        reader_dfb_bindings.push_back(ProducerOf(CB_PAD_ALIGN, "cb_pad_align"));
        reader_dfb_bindings.push_back(ConsumerOf(CB_PAD_ALIGN, "cb_pad_align"));
    }

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .dfb_bindings = reader_dfb_bindings,
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
        .compile_time_args = reader_cta,
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks_per_core",
                  "num_sticks_per_barrier",
                  "start_page_id",
                  "front_pad_n",
                  "front_pad_c",
                  "front_pad_h",
                  "start_dim_h",
                  "start_dim_c",
                  "start_dim_n"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    if (has_pad_align) {
        reader_spec.compiler_options.defines = {{"HAS_PAD_ALIGN", "1"}};
    }

    // ------------------------------------------------------------------------
    // Writer KernelSpec: cb_in0 CONSUMER (accessor cb_out0), Case-1 output.
    // ------------------------------------------------------------------------
    KernelSpec::CompileTimeArgs writer_cta{
        {"stick_size_bytes", static_cast<uint32_t>(stick_size_padded)},
        {"stick_size_padded_aligned", stick_size_padded_aligned},
        {"num_output_pages_in_row", num_output_pages_in_row},
        {"output_page_size", output_page_size},
        {"size_of_valid_data_in_last_output_page_in_row", size_of_valid_data_in_last_output_page_in_row},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .dfb_bindings = {ConsumerOf(CB_IN0, "cb_out0")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
        .compile_time_args = writer_cta,
        .runtime_arg_schema = {.runtime_arg_names = {"num_sticks_per_core", "num_sticks_per_barrier", "start_page_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ------------------------------------------------------------------------
    // Per-core runtime args. Mirrors the legacy get_runtime_args_rm() accounting: the input region
    // is walked stick-by-stick to derive each core's start_page_id and start dim offsets.
    // ------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};

    auto input_padded_shape = a.padded_shape();
    uint32_t H_in = input_padded_shape[2], C_in = input_padded_shape[1], N_in = input_padded_shape[0];

    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    uint32_t curr_sticks_read = 0;
    uint32_t curr_sticks_write = 0;
    uint32_t off_h = 0, off_c = 0, off_n = 0;  // start_dim_offset[1..3] for the current core

    for (const auto& core : cores_in_order) {
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_padded_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_padded_per_core_group_2;
        } else {
            num_sticks_per_core = 0;  // no-op (cores_in_order never contains idle cores)
        }

        const NodeCoord node = core;

        KernelRunArgs::RuntimeArgValues& reader_rtas = reader_run.runtime_arg_values;
        AddRuntimeArgsForNode(
            reader_rtas,
            node,
            {
                {"num_sticks_per_core", num_sticks_per_core},
                {"num_sticks_per_barrier", num_sticks_per_barrier},
                {"start_page_id", curr_sticks_read * num_input_pages_in_row},
                {"front_pad_n", static_cast<uint32_t>(front_pad[-4])},
                {"front_pad_c", static_cast<uint32_t>(front_pad[-3])},
                {"front_pad_h", static_cast<uint32_t>(front_pad[-2])},
                {"start_dim_h", off_h},
                {"start_dim_c", off_c},
                {"start_dim_n", off_n},
            });

        KernelRunArgs::RuntimeArgValues& writer_rtas = writer_run.runtime_arg_values;
        AddRuntimeArgsForNode(
            writer_rtas,
            node,
            {
                {"num_sticks_per_core", num_sticks_per_core},
                {"num_sticks_per_barrier", num_sticks_per_barrier},
                {"start_page_id", curr_sticks_write * num_output_pages_in_row},
            });

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

        // start_dim_offset for the next core = {0, curr_h, curr_c, curr_n}
        off_h = curr_h;
        off_c = curr_c;
        off_n = curr_n;
    }

    WorkUnitSpec wu{
        .name = "pad_rm_multicore_default",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = all_cores,
    };

    Group<DataflowBufferSpec> dfbs{cb_in0_spec, cb_pad_spec};
    if (has_pad_align) {
        dfbs.push_back(cb_pad_align_spec);
    }

    ProgramSpec spec{
        .name = "pad_rm_multicore_default",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = dfbs,
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(a.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
