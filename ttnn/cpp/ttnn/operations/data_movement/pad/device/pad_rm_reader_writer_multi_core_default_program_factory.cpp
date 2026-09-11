// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_reader_writer_multi_core_default_program_factory.hpp"

#include <algorithm>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
static const uint32_t max_read_size = 2048;  // max read size in bytes for reader and writer kernels

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {

uint32_t get_num_stick_per_barrier(uint32_t stick_size_padded_aligned) {
    return std::max(tt::div_up(max_read_size, stick_size_padded_aligned), 1u);
}

// Names are prefixed per factory: all seven pad factories land in one unity-build
// translation unit, where every anonymous namespace is merged into a single scope.
const KernelSpecName RM_DEF_READER{"reader"};
const KernelSpecName RM_DEF_WRITER{"writer"};
const DFBSpecName RM_DEF_IN0{"in0"};
const DFBSpecName RM_DEF_PAD{"pad"};
const DFBSpecName RM_DEF_PAD_ALIGN{"pad_align"};
const TensorParamName RM_DEF_INPUT{"input"};
const TensorParamName RM_DEF_OUTPUT{"output"};

}  // namespace

ttnn::device_operation::ProgramArtifacts PadRmReaderWriterMultiCoreDefaultProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    const auto& input_tensor_start = operation_attributes.input_tensor_start;

    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    const auto& a_shape = a.logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    uint32_t W_padded = output_padded_shape[3], H_padded = output_padded_shape[2], C_padded = output_padded_shape[1],
             N_padded = output_padded_shape[0];
    uint32_t NCH_padded = H_padded * C_padded * N_padded;

    const auto& front_pad = input_tensor_start;

    auto stick_size = W * a.element_size();
    auto stick_size_padded = W_padded * a.element_size();
    auto stick_size_padded_front = front_pad[-1] * a.element_size();
    uint32_t stick_size_padded_aligned = tt::align(stick_size_padded, hal::get_l1_alignment());
    uint32_t stick_size_padded_DRAM_aligned = tt::align(stick_size_padded, hal::get_dram_alignment());

    // Input page-based addressing
    uint32_t num_input_pages_in_row = 1;
    if (a.is_sharded()) {
        uint32_t shard_width =
            a.shard_spec().has_value() ? a.shard_spec().value().shape[1] : a.nd_shard_spec().value().shard_shape[-1];
        num_input_pages_in_row = tt::div_up(a.logical_shape()[-1], shard_width);
    }

    // Output page-based addressing
    uint32_t num_output_pages_in_row = 1;
    if (output.is_sharded()) {
        uint32_t output_shard_width = output.shard_spec().has_value() ? output.shard_spec().value().shape[1]
                                                                      : output.nd_shard_spec().value().shard_shape[-1];
        num_output_pages_in_row = tt::div_up(W_padded, output_shard_width);
    }

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

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

    Group<DataflowBufferSpec> dataflow_buffers;

    // pad: reused for pad-value scratch on every dispatch. Only the reader touches it — it fills
    // the entry with baby-RISCV stores and loop-back-reads it per stick, with no FIFO ops — so the
    // reader binds both endpoints (self-loop).
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RM_DEF_PAD,
        .entry_size = stick_size_padded_DRAM_aligned,
        .num_entries = 1,
        .data_format_metadata = dfb_data_format,
    });

    // pad_align: a realignment staging area, allocated only when the reader can actually take one
    // of the two branches that use it. The kernel gates its construction and every reference to it
    // on the matching PAD_ALIGN_DFB define, because a kernel may not name a DFB it has not bound.
    bool unaligned = stick_size_padded_aligned % hal::get_dram_alignment() != 0;
    const bool needs_pad_align_dfb = stick_size_padded_front != 0 || unaligned;
    if (needs_pad_align_dfb) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RM_DEF_PAD_ALIGN,
            .entry_size = stick_size_padded_DRAM_aligned,
            .num_entries = 1,
            .data_format_metadata = dfb_data_format,
        });
    }

    Group<DFBBinding> reader_dfb_bindings = {
        DFBBinding{
            .dfb_spec_name = RM_DEF_IN0,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = RM_DEF_PAD,
            .accessor_name = "pad",
            .endpoint_type = DFBEndpointType::PRODUCER,
        },
        DFBBinding{
            .dfb_spec_name = RM_DEF_PAD,
            .accessor_name = "pad",
            .endpoint_type = DFBEndpointType::CONSUMER,
        },
    };
    KernelSpec::CompilerOptions::Defines reader_defines;
    if (needs_pad_align_dfb) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_DEF_PAD_ALIGN,
            .accessor_name = "pad_align",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RM_DEF_PAD_ALIGN,
            .accessor_name = "pad_align",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        reader_defines.emplace("PAD_ALIGN_DFB", "1");
    }

    KernelSpec reader{
        .unique_id = RM_DEF_READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
            "reader_pad_dims_rm_interleaved_v2.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = RM_DEF_INPUT,
                    .accessor_name = "src",
                },
            },
        .compile_time_args =
            {
                {"N", static_cast<uint32_t>(N + front_pad[-4])},
                {"H", static_cast<uint32_t>(H + front_pad[-2])},
                {"C", static_cast<uint32_t>(C + front_pad[-3])},
                {"stick_size_bytes", static_cast<uint32_t>(stick_size)},
                {"N_padded", N_padded},
                {"H_padded", H_padded},
                {"C_padded", C_padded},
                {"stick_size_padded", static_cast<uint32_t>(stick_size_padded)},
                {"stick_size_padded_front", static_cast<uint32_t>(stick_size_padded_front)},
                {"not_pad_by_zero", static_cast<uint32_t>(not_pad_by_zero)},
                {"packed_pad_value", packed_pad_value},
                {"stick_size_padded_aligned", stick_size_padded_aligned},
                {"unaligned", static_cast<uint32_t>(unaligned)},
                {"num_input_pages_in_row", num_input_pages_in_row},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_sticks_per_core",
                     "num_sticks_per_barrier",
                     "start_page_id",
                     "front_pad_n",
                     "front_pad_c",
                     "front_pad_h",
                     "start_dim_offset_h",
                     "start_dim_offset_c",
                     "start_dim_offset_n"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = RM_DEF_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
            "writer_pad_dims_rm_interleaved_v2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = RM_DEF_IN0,
                    .accessor_name = "out0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = RM_DEF_OUTPUT,
                    .accessor_name = "dst",
                },
            },
        .compile_time_args =
            {
                {"stick_size_bytes", static_cast<uint32_t>(stick_size_padded)},
                {"stick_size_padded_aligned", stick_size_padded_aligned},
                {"num_output_pages_in_row", num_output_pages_in_row},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_sticks_per_core", "num_sticks_per_barrier", "start_page_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // Build per-core runtime args inline (legacy path called get_runtime_args_rm()
    // which produced the same data).
    // The legacy helper used input_tensor.padded_shape() for the H/C/N bounds —
    // mirror that here.  H_padded/C_padded already use the output padded shape.
    auto input_padded_shape = a.padded_shape();
    uint32_t H_in = input_padded_shape[2], C_in = input_padded_shape[1], N_in = input_padded_shape[0];

    uint32_t num_sticks_per_barrier = get_num_stick_per_barrier(stick_size_padded_aligned);

    KernelRunArgs reader_run_args{.kernel = RM_DEF_READER};
    KernelRunArgs writer_run_args{.kernel = RM_DEF_WRITER};

    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    uint32_t curr_sticks_read = 0;
    uint32_t curr_sticks_write = 0;
    for (const auto& core : cores_in_order) {
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_padded_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_padded_per_core_group_2;
        } else {
            // no-op
            num_sticks_per_core = 0;
        }

        // curr_h / curr_c / curr_n are this core's starting position in the padded output; they
        // are advanced by the loop below only after these args are recorded.
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_sticks_per_core", num_sticks_per_core},
             {"num_sticks_per_barrier", num_sticks_per_barrier},
             {"start_page_id", curr_sticks_read * num_input_pages_in_row},
             {"front_pad_n", static_cast<uint32_t>(front_pad[-4])},
             {"front_pad_c", static_cast<uint32_t>(front_pad[-3])},
             {"front_pad_h", static_cast<uint32_t>(front_pad[-2])},
             {"start_dim_offset_h", curr_h},
             {"start_dim_offset_c", curr_c},
             {"start_dim_offset_n", curr_n}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_sticks_per_core", num_sticks_per_core},
             {"num_sticks_per_barrier", num_sticks_per_barrier},
             {"start_page_id", curr_sticks_write * num_output_pages_in_row}});

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
    }

    uint32_t dfb_npages = get_num_stick_per_barrier(stick_size_padded_aligned);
    const uint32_t buffer_reader_writer_async_factor = 16;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RM_DEF_IN0,
        .entry_size = stick_size_padded_aligned,
        .num_entries = buffer_reader_writer_async_factor * dfb_npages,
        .data_format_metadata = dfb_data_format,
    });

    ProgramSpec spec{
        .name = "pad_rm_reader_writer_multi_core_default",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = RM_DEF_INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = RM_DEF_OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {RM_DEF_READER, RM_DEF_WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {RM_DEF_INPUT, TensorArgument{input_mesh_tensor}},
        {RM_DEF_OUTPUT, TensorArgument{output_mesh_tensor}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
