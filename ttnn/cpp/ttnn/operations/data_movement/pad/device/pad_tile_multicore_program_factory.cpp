// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_tile_multicore_program_factory.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {
using ttnn::operations::data_movement::get_num_pages;

static inline int advance_tensor_index(std::vector<uint32_t>& idx, const ttnn::Shape& dims, uint32_t ndims) {
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

ttnn::device_operation::ProgramArtifacts PadTileMulticoreProgramFactory::create_program_spec(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input;
    Tensor& output = tensor_return_value;
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

    const uint32_t num_dims = output_padded_shape.rank();

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "pad_tile_multicore";

    // Legacy CB layout, preserved 1:1:
    //   c_0 input  (2 pages): reader PRODUCER -> writer CONSUMER.
    //   c_1 output (2 pages): allocated by the legacy descriptor but never used by the kernel
    //                         as a FIFO; preserved as a writer self-loop so the L1 footprint is
    //                         unchanged and the validator's producer+consumer rule is satisfied.
    //   c_2 pad    (1 page) : output-dtype L1 scratch the writer fills with the pad value once
    //                         and reuses as the NoC source for every padded tile; it has no FIFO
    //                         peer, so it is a writer self-loop (producer + consumer).
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"in0"},
            .entry_size = page_size,
            .num_entries = multi_buffering_size,
            .data_format_metadata = cb_data_format},
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"out"},
            .entry_size = page_size,
            .num_entries = multi_buffering_size,
            .data_format_metadata = cb_data_format},
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"pad"},
            .entry_size = page_size,
            .num_entries = 1,
            .data_format_metadata = cb_data_format},
    };

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"dst"}, .spec = output.tensor_spec()},
    };

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "reader_pad_tiled.cpp"},
        .dfb_bindings = {m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in0"},
            .accessor_name = "in0",
            .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"src"}, .accessor_name = "src"}},
        .compile_time_args = {{"page_size", page_size}, {"num_dims", num_dims}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages_to_write", "start_offset"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };
    // Per-core per-dim arrays [input_page_shape, output_page_shape, input_id_per_dim, output_id_per_dim],
    // each num_dims long, passed as runtime varargs (read positionally in the kernel).
    reader.advanced_options.num_runtime_varargs = num_dims * 4;

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "writer_pad_tiled.cpp"},
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"in0"},
                 .accessor_name = "in0",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER},
             // "out" self-loop: legacy output CB, allocated but unused by the kernel as a FIFO.
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"out"},
                 .accessor_name = "out",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"out"},
                 .accessor_name = "out",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER},
             // "pad" self-loop: the writer both reserves (producer) and consumes (consumer) the
             // pad-scratch DFB; it has no other endpoint.
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"pad"},
                 .accessor_name = "pad",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"pad"},
                 .accessor_name = "pad",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"dst"}, .accessor_name = "dst"}},
        .compile_time_args =
            {{"page_size", page_size},
             {"num_dims", num_dims},
             {"pad_value", packed_pad_value},
             {"element_size", static_cast<uint32_t>(output.element_size())}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages_to_write", "start_offset"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };
    // Same per-core per-dim varargs as the reader.
    writer.advanced_options.num_runtime_varargs = num_dims * 4;

    spec.kernels = {reader, writer};

    // Local DFBs (in0, out, pad) require their producer and consumer KernelSpecs to share the
    // same WorkUnitSpec. Both kernels run on all_cores, so a single WorkUnitSpec hosts both.
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

    /*
    See the original single-core/legacy comment block: this factory walks the output tile space,
    deciding per output tile whether it falls inside the input region (copy a tile) or outside it
    (write a pad tile), and advances the input index only inside the input region. The host
    pre-computes, per core, the input/output starting indices and page offsets.
    */

    std::vector<uint32_t> input_id_per_dim, output_id_per_dim;  // input and output id_per_dims
    // initialize id_per_dims to vectors of length num_dims filled with 0
    input_id_per_dim.resize(a_shape.rank(), 0);
    output_id_per_dim.resize(output_padded_shape.rank(), 0);
    // instantiate the input and output tensor padded shapes
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
        CoreCoord core = cores_in_order[i];
        const m2::NodeCoord node{core};

        uint32_t num_pages_per_core;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            num_pages_per_core = 0;  // no-op
        }

        // The legacy buffer-address RTA slot (0u on idle cores, Buffer* on active ones) is now
        // carried by the src/dst TensorBindings; idle cores short-circuit on num_pages_to_write == 0.
        reader_run.runtime_arg_values.push_back(
            {node, {{"num_pages_to_write", num_pages_per_core}, {"start_offset", input_page_offset}}});
        writer_run.runtime_arg_values.push_back(
            {node, {{"num_pages_to_write", num_pages_per_core}, {"start_offset", output_page_offset}}});

        // Per-core varargs: [input_page_shape..., output_page_shape..., input_id_per_dim..., output_id_per_dim...].
        // Every core gets the same input and output tile shapes, plus its own starting indices.
        m2::AdvancedKernelRunArgs::Varargs per_dim;
        per_dim.reserve(num_dims * 4);
        for (auto v : input_page_shape) {
            per_dim.push_back(v);
        }
        for (auto v : output_page_shape) {
            per_dim.push_back(v);
        }
        for (uint32_t v : input_id_per_dim) {
            per_dim.push_back(v);
        }
        for (uint32_t v : output_id_per_dim) {
            per_dim.push_back(v);
        }
        reader_run.advanced_options.runtime_varargs[node] = per_dim;
        writer_run.advanced_options.runtime_varargs[node] = std::move(per_dim);

        // We now need to increment the input and output id_per_dims by the number of pages this core is processing
        // Similarly to in the kernel, we only increment the input id_per_dim if we are within the input region
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
        // The input and output id_per_dim should now be set correctly for the next core
    }

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"src"}, a.mesh_tensor()},
        {m2::TensorParamName{"dst"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
