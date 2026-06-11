// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_tile_program_factory.hpp"

#include <cstdint>
#include <filesystem>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;
namespace m2 = tt::tt_metal::experimental;

namespace {

// File-scope constants must be unique across sibling .cpp under the unity build, hence the _PTC suffix.
constexpr const char* READER_KERNEL_PATH_PTC =
    "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_unary_interleaved_m2.cpp";
constexpr const char* WRITER_KERNEL_PATH_PTC =
    "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_unary_pad_dims_interleaved_m2.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts PadTileCoreProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_shape = operation_attributes.output_padded_shape;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
    }

    uint32_t num_unpadded_Xt = a.padded_shape()[3] / TILE_WIDTH;
    uint32_t num_total_Xt = output_shape[3] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = a.padded_shape()[2] / TILE_HEIGHT;
    uint32_t num_total_Yt = output_shape[2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;
    uint32_t num_unpadded_Z = a.padded_shape()[1];
    uint32_t num_total_Z = output_shape[1];
    uint32_t num_padded_Zt = (num_total_Z - num_unpadded_Z) * num_total_Yt * num_total_Xt;
    uint32_t num_unpadded_W = a.padded_shape()[0];
    uint32_t num_total_W = output_shape[0];
    uint32_t num_padded_Wt = (num_total_W - num_unpadded_W) * num_total_Z * num_total_Yt * num_total_Xt;

    uint32_t num_unpadded_tiles = a.physical_volume() / TILE_HW;

    const CoreRangeSet core_ranges{CoreRange{{0, 0}, {0, 0}}};

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramSpec
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec;
    spec.name = "pad_tile_single_core";

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};

    // src0 DFB (c_0): produced by the reader (cb_id_in0), consumed by the writer (cb_id_out0).
    // pad DFB (c_1): a writer-local one-tile scratchpad holding the pad value.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = single_tile_size,
            .num_entries = 2,  // num_input_tiles
            .data_format_metadata = cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"pad"},
            .entry_size = single_tile_size,
            .num_entries = 1,  // num_pad_tiles
            .data_format_metadata = cb_data_format,
        }};

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{READER_KERNEL_PATH_PTC},
        .dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"src0"}, "cb_id_in0")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src_args"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        // Reader on NCRISC (RISCV_1 / NOC1), writer on BRISC.
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{WRITER_KERNEL_PATH_PTC},
        .dfb_bindings =
            {m2::ConsumerOf(m2::DFBSpecName{"src0"}, "cb_id_out0"),
             m2::ProducerOf(m2::DFBSpecName{"pad"}, "cb_id_out1"),
             m2::ConsumerOf(m2::DFBSpecName{"pad"}, "cb_id_out1")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_unpadded_W",
                  "num_padded_Wt",
                  "num_unpadded_Z",
                  "num_padded_Zt",
                  "num_unpadded_Yt",
                  "num_padded_Yt",
                  "num_unpadded_Xt",
                  "num_padded_Xt",
                  "pad_value"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    spec.kernels = {std::move(reader), std::move(writer)};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "pad_tile_single_core",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
        .target_nodes = core_ranges}};

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramRunArgs (degenerate: full set)
    ////////////////////////////////////////////////////////////////////////////
    const CoreCoord core{0, 0};
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    reader_args.runtime_arg_values.push_back({core, {{"num_pages", num_unpadded_tiles}, {"start_id", uint32_t{0}}}});
    writer_args.runtime_arg_values.push_back(
        {core,
         {{"num_unpadded_W", num_unpadded_W},
          {"num_padded_Wt", num_padded_Wt},
          {"num_unpadded_Z", num_unpadded_Z},
          {"num_padded_Zt", num_padded_Zt},
          {"num_unpadded_Yt", num_unpadded_Yt},
          {"num_padded_Yt", num_padded_Yt},
          {"num_unpadded_Xt", num_unpadded_Xt},
          {"num_padded_Xt", num_padded_Xt},
          {"pad_value", packed_pad_value}}});

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
