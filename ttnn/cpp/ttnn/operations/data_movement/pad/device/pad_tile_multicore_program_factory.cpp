// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_tile_multicore_program_factory.hpp"

#include <bit>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {
using ttnn::operations::data_movement::get_num_pages;
namespace m2 = tt::tt_metal::experimental;

namespace {

// File-scope constants must be unique across sibling .cpp under the unity build, hence the _PTMC suffix.
constexpr const char* READER_KERNEL_PATH_PTMC =
    "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_tiled_m2.cpp";
constexpr const char* WRITER_KERNEL_PATH_PTMC =
    "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_tiled_m2.cpp";

// increment least-significant dim first; wrap and carry. Identical to the kernel-side helper.
inline int advance_tensor_index_host(std::vector<uint32_t>& idx, const ttnn::Shape& dims, uint32_t ndims) {
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
    const auto& a = tensor_args.input;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;

    const auto& a_shape = a.logical_shape();
    uint32_t num_pages = get_num_pages(output);

    auto compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
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

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramSpec
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec;
    spec.name = "pad_tile_multicore";

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};

    // src0 DFB: double-buffered, produced by the reader and consumed by the writer.
    // pad_val DFB: a writer-local one-page scratchpad that holds a tile full of the pad value.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"cb_input"},
            .entry_size = page_size,
            .num_entries = multi_buffering_size,
            .data_format_metadata = cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"cb_pad_val"},
            .entry_size = page_size,
            .num_entries = 1,
            .data_format_metadata = cb_data_format,
        }};

    // 4 per-dim arrays (input_page_shape, output_page_shape, input_id_per_dim, output_id_per_dim).
    const uint32_t num_varargs = 4 * num_dims;

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{READER_KERNEL_PATH_PTMC},
        .dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"cb_input"}, "cb_input")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src_args"}},
        .compile_time_args = {{"page_size", page_size}, {"num_dims", num_dims}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages_to_write", "start_offset"}},
        // Reader on NCRISC (RISCV_1 / NOC1), writer on BRISC — so the two data-movement kernels
        // don't collide on the same DM processor.
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
        .advanced_options = m2::KernelAdvancedOptions{.num_runtime_varargs = num_varargs},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{WRITER_KERNEL_PATH_PTMC},
        .dfb_bindings =
            {m2::ConsumerOf(m2::DFBSpecName{"cb_input"}, "cb_input"),
             m2::ProducerOf(m2::DFBSpecName{"cb_pad_val"}, "cb_pad_val"),
             m2::ConsumerOf(m2::DFBSpecName{"cb_pad_val"}, "cb_pad_val")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
        .compile_time_args =
            {{"page_size", page_size},
             {"num_dims", num_dims},
             {"pad_value", packed_pad_value},
             {"element_size", static_cast<uint32_t>(output.element_size())}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages_to_write", "start_offset"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
        .advanced_options = m2::KernelAdvancedOptions{.num_runtime_varargs = num_varargs},
    };

    spec.kernels = {std::move(reader), std::move(writer)};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "pad_tile_multicore",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
        .target_nodes = all_cores}};

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramRunArgs (degenerate: full set)
    ////////////////////////////////////////////////////////////////////////////
    // Per-core work split. The host walks the same input/output id-per-dim state machine as the kernels:
    // it only advances the input index while inside the input region, so each core's starting indices
    // line up with the tiles it must read/write. (See the legacy factory's worked example.)
    std::vector<uint32_t> input_id_per_dim, output_id_per_dim;
    input_id_per_dim.resize(a_shape.rank(), 0);
    output_id_per_dim.resize(output_padded_shape.rank(), 0);
    auto input_page_shape = a.padded_shape();
    auto output_page_shape = output_padded_shape;
    input_page_shape[-1] /= tt::constants::TILE_HEIGHT;
    input_page_shape[-2] /= tt::constants::TILE_HEIGHT;
    output_page_shape[-1] /= tt::constants::TILE_HEIGHT;
    output_page_shape[-2] /= tt::constants::TILE_HEIGHT;
    bool within_input_region;
    uint32_t input_page_offset = 0;
    uint32_t output_page_offset = 0;

    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = cores_in_order[i];

        uint32_t num_pages_per_core;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            num_pages_per_core = 0;  // no-op
        }

        // Varargs: same input/output tile shapes and this core's starting indices, in the order the
        // kernel reads them: input_page_shape, output_page_shape, input_id_per_dim, output_id_per_dim.
        std::vector<uint32_t> varargs;
        varargs.reserve(num_varargs);
        for (auto v : input_page_shape) {
            varargs.push_back(v);
        }
        for (auto v : output_page_shape) {
            varargs.push_back(v);
        }
        for (uint32_t v : input_id_per_dim) {
            varargs.push_back(v);
        }
        for (uint32_t v : output_id_per_dim) {
            varargs.push_back(v);
        }

        reader_args.runtime_arg_values.push_back(
            {core, {{"num_pages_to_write", num_pages_per_core}, {"start_offset", input_page_offset}}});
        writer_args.runtime_arg_values.push_back(
            {core, {{"num_pages_to_write", num_pages_per_core}, {"start_offset", output_page_offset}}});
        reader_args.advanced_options.runtime_varargs.emplace(core, varargs);
        writer_args.advanced_options.runtime_varargs.emplace(core, varargs);

        // Advance the host-side state machine by this core's page count, mirroring the kernels.
        for (uint32_t p = 0; p < num_pages_per_core; p++) {
            within_input_region = true;
            for (uint32_t d = 0; d < input_id_per_dim.size(); d++) {
                if (input_id_per_dim[d] < output_id_per_dim[d]) {
                    within_input_region = false;
                    break;
                }
            }
            if (within_input_region) {
                advance_tensor_index_host(input_id_per_dim, input_page_shape, input_id_per_dim.size());
                input_page_offset++;
            }
            advance_tensor_index_host(output_id_per_dim, output_page_shape, output_id_per_dim.size());
            output_page_offset++;
        }
    }

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));
    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(a.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
