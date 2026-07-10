// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_tile_multicore_program_factory.hpp"

#include <algorithm>
#include <filesystem>
#include <vector>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {
using ttnn::operations::data_movement::get_num_pages;

namespace {
int advance_tensor_index(std::vector<uint32_t>& idx, const ttnn::Shape& dims, uint32_t ndims) {
    // increment least-significant dim first
    for (int32_t d = ndims - 1; d >= 0; d--) {
        uint32_t v = idx[d] + 1;
        if (v < dims[d]) {
            idx[d] = v;
            return 1;
        }
        idx[d] = 0;  // wrap and carry
    }
    return 0;  // overflowed most-significant dim
}
}  // namespace

ttnn::device_operation::ProgramArtifacts PadTileMulticoreProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN0{"cb_in0"};  // legacy c_0: input stream (reader produces, writer consumes)
    const DFBSpecName CB_PAD{"cb_pad"};  // legacy c_2: pad-value scratchpad (writer self-loop)
    // (legacy c_1 output CB was unused by the kernels and is dropped.)

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    constexpr const char* READER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/reader_pad_tiled.cpp";
    constexpr const char* WRITER_PATH =
        "ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/writer_pad_tiled.cpp";

    const auto& a = tensor_args.input;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;

    const auto& a_shape = a.logical_shape();
    uint32_t num_pages = get_num_pages(output);

    IDevice* device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        sub_core_grids.has_value() ? tt::tt_metal::split_work_to_cores(sub_core_grids.value(), num_pages)
                                   : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);

    auto cores_in_order = corerange_to_cores(all_cores, num_cores, true);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t page_size = output.buffer()->page_size();
    uint32_t multi_buffering_size = 2;

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    uint32_t packed_pad_value;
    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    switch (a.dtype()) {
        case DataType::INT32:
        case DataType::UINT32: packed_pad_value = pad_value; break;
        case DataType::BFLOAT16:
            packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});
            break;
        case DataType::UINT16:
            packed_pad_value = ttnn::operations::data_movement::pack_two_uint16_into_uint32(
                {ttnn::operations::data_movement::float_to_uint16(pad_value),
                 ttnn::operations::data_movement::float_to_uint16(pad_value)});
            break;
        case DataType::FLOAT32: packed_pad_value = std::bit_cast<uint32_t>(pad_value); break;
        default:
            packed_pad_value = 0;
            TT_ASSERT(
                false,
                "Unsupported datatype for pad tile multicore, can only support INT32, UINT32, BFLOAT16, UINT16, "
                "FLOAT32");
    }

    const uint32_t num_dims = static_cast<uint32_t>(output_padded_shape.rank());

    // ------------------------------------------------------------------------
    // DataflowBufferSpecs. cb_in0 = reader->writer FIFO; cb_pad = writer self-loop pad scratchpad.
    // ------------------------------------------------------------------------
    DataflowBufferSpec cb_in0_spec{
        .unique_id = CB_IN0,
        .entry_size = page_size,
        .num_entries = multi_buffering_size,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec cb_pad_spec{
        .unique_id = CB_PAD,
        .entry_size = page_size,
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()};

    // ------------------------------------------------------------------------
    // Reader KernelSpec: cb_in0 PRODUCER, Case-1 input. The four per-dim arrays come in as varargs.
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{READER_PATH},
        .dfb_bindings = {ProducerOf(CB_IN0, "cb_input")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
        .compile_time_args = {{"num_dims", num_dims}, {"page_size", page_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages_to_write", "start_offset"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };
    reader_spec.advanced_options.num_runtime_varargs = 4 * num_dims;

    // ------------------------------------------------------------------------
    // Writer KernelSpec: cb_in0 CONSUMER, cb_pad self-loop, Case-1 output.
    // ------------------------------------------------------------------------
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_PATH},
        .dfb_bindings =
            {ConsumerOf(CB_IN0, "cb_input"), ProducerOf(CB_PAD, "cb_pad_val"), ConsumerOf(CB_PAD, "cb_pad_val")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
        .compile_time_args =
            {{"num_dims", num_dims},
             {"page_size", page_size},
             {"pad_value", packed_pad_value},
             {"element_size", static_cast<uint32_t>(output.element_size())}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages_to_write", "start_offset"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };
    writer_spec.advanced_options.num_runtime_varargs = 4 * num_dims;

    // ------------------------------------------------------------------------
    // Per-core runtime args + varargs. Mirrors the legacy page-walk: a core reads an input tile only
    // when its running input id_per_dim is within the output's input region.
    // ------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};

    std::vector<uint32_t> input_id_per_dim(a_shape.rank(), 0);
    std::vector<uint32_t> output_id_per_dim(output_padded_shape.rank(), 0);

    auto input_page_shape = a.padded_shape();
    auto output_page_shape = output_padded_shape;
    input_page_shape[-1] /= tt::constants::TILE_HEIGHT;
    input_page_shape[-2] /= tt::constants::TILE_HEIGHT;
    output_page_shape[-1] /= tt::constants::TILE_HEIGHT;
    output_page_shape[-2] /= tt::constants::TILE_HEIGHT;

    bool within_input_region;
    uint32_t input_page_offset = 0;
    uint32_t output_page_offset = 0;

    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core = cores_in_order[i];
        const NodeCoord node = core;

        uint32_t num_pages_per_core;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            num_pages_per_core = 0;  // no-op (cores_in_order never contains idle cores)
        }

        // Vararg payload (identical for reader and writer): the four per-dim arrays back to back.
        std::vector<uint32_t> varargs;
        varargs.reserve(4 * num_dims);
        for (uint32_t d = 0; d < num_dims; ++d) {
            varargs.push_back(input_page_shape[d]);
        }
        for (uint32_t d = 0; d < num_dims; ++d) {
            varargs.push_back(output_page_shape[d]);
        }
        for (uint32_t v : input_id_per_dim) {
            varargs.push_back(v);
        }
        for (uint32_t v : output_id_per_dim) {
            varargs.push_back(v);
        }

        reader_run.runtime_arg_values.push_back(
            {node, {{"num_pages_to_write", num_pages_per_core}, {"start_offset", input_page_offset}}});
        reader_run.advanced_options.runtime_varargs[node] = varargs;

        writer_run.runtime_arg_values.push_back(
            {node, {{"num_pages_to_write", num_pages_per_core}, {"start_offset", output_page_offset}}});
        writer_run.advanced_options.runtime_varargs[node] = varargs;

        // Advance the running per-dim indices by this core's page count (input id only advances while
        // within the input region).
        for (uint32_t p = 0; p < num_pages_per_core; p++) {
            within_input_region = true;
            for (uint32_t d = 0; d < input_id_per_dim.size(); d++) {
                if (input_id_per_dim[d] < output_id_per_dim[d]) {
                    within_input_region = false;
                    break;
                }
            }
            if (within_input_region) {
                advance_tensor_index(input_id_per_dim, input_page_shape, input_id_per_dim.size());
                input_page_offset++;
            }
            advance_tensor_index(output_id_per_dim, output_page_shape, output_id_per_dim.size());
            output_page_offset++;
        }
    }

    WorkUnitSpec wu{
        .name = "pad_tile_multicore",
        .kernels = {READER_KERNEL, WRITER_KERNEL},
        .target_nodes = all_cores,
    };

    ProgramSpec spec{
        .name = "pad_tile_multicore",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {cb_in0_spec, cb_pad_spec},
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
