// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_interleaved_program_factory.hpp"
#include "transpose_utils.hpp"

#include "ttnn/operations/math.hpp"

#include <bit>
#include <filesystem>

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace ttnn::prim {

namespace {

// Unique file-scope constants per .cpp (unity build) — prefix with the variant name.
constexpr const char* TRANSPOSE_HC_TILED_READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
    "reader_unary_transpose_hc_interleaved_tiled_padding_aware_m2.cpp";
constexpr const char* TRANSPOSE_HC_TILED_WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
    "writer_unary_transpose_hc_interleaved_tiled_padding_aware_m2.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeHCTiledInterleavedProgramFactory::create_program_artifacts(
    const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    // pad_value is always defined at API level; padding is decided purely by shape
    const float pad_value = operation_attributes.pad_value;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    auto tile = input_tensor.tensor_spec().tile();
    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    uint32_t C = input_tensor.logical_shape()[1];
    bool needs_padding = (C % tile_shape[1] != 0);

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto max_padding_write = face_shape[0] * face_shape[1];

    uint32_t element_size = input_tensor.element_size();
    uint32_t padding_val_packed = 0;
    uint32_t num_writes = 0;
    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];

    if (C % tile_shape[1] != 0) {
        uint32_t num_packed_values = sizeof(uint32_t) / element_size;
        num_writes = max_padding_write / num_packed_values;
        switch (input_tensor.dtype()) {
            case DataType::INT32: padding_val_packed = std::bit_cast<uint32_t>(pad_value); break;
            case DataType::UINT32: padding_val_packed = pad_value; break;
            case DataType::BFLOAT16:
                padding_val_packed = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
                break;
            case DataType::UINT16:
                padding_val_packed =
                    pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
                break;
            case DataType::FLOAT32: padding_val_packed = std::bit_cast<uint32_t>(pad_value); break;
            default:
                padding_val_packed = 0;
                TT_ASSERT(
                    false,
                    "Unsupported datatype for pad tile multicore, can only support INT32, UINT32, BFLOAT16, UINT16, "
                    "FLOAT32");
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramSpec
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec;
    spec.name = "transpose_hc_tiled_interleaved";

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output_tensor.tensor_spec()}};

    // Two local L1 DFBs, both produced by the reader and consumed by the writer (one producer + one
    // consumer in the same work unit):
    //   - src0:    the tile staging buffer (double-buffered).
    //   - padding: the single-entry padding tile. Always declared so the dfb::padding token always
    //              exists in both kernels; when needs_padding is false the kernels' `if constexpr`
    //              guards leave it unused (this avoids the conditional-DFB-token gap).
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = single_tile_size,
            .num_entries = 2,
            .data_format_metadata = cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"padding"},
            .entry_size = max_padding_write * element_size,
            .num_entries = 1,
            .data_format_metadata = cb_data_format,
        }};

    // Reader named compile-time args (formerly KernelDescriptor::NamedCompileTimeArgs). swap_hw / H / W /
    // accumulated_outer_dims / tile_height / tile_width are the degenerate values the legacy descriptor
    // passed for the transpose-HC use of this shared padding-aware reader.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{TRANSPOSE_HC_TILED_READER_KERNEL_PATH},
        .dfb_bindings =
            {m2::ProducerOf(m2::DFBSpecName{"src0"}, "src0"), m2::ProducerOf(m2::DFBSpecName{"padding"}, "padding")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src_args"}},
        .compile_time_args =
            {{"num_writes", num_writes},
             {"padding_val_packed", padding_val_packed},
             {"needs_padding", static_cast<uint32_t>(needs_padding)},
             {"swap_hw", 0u},
             {"H", 1u},
             {"W", 1u},
             {"accumulated_outer_dims", 1u},
             {"tile_height", 1u},
             {"tile_width", 1u}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        // Reader on NCRISC (RISCV_1 / NOC1), writer on BRISC — so the two data-movement kernels don't
        // collide on the same DM processor.
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{TRANSPOSE_HC_TILED_WRITER_KERNEL_PATH},
        .dfb_bindings =
            {m2::ConsumerOf(m2::DFBSpecName{"src0"}, "src0"), m2::ConsumerOf(m2::DFBSpecName{"padding"}, "padding")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
        .compile_time_args =
            {{"element_size", element_size},
             {"C", C},
             {"H", H},
             {"W", W},
             {"tile_height", tile_shape[0]},
             {"tile_width", tile_shape[1]},
             {"face_height", face_shape[0]},
             {"face_width", face_shape[1]},
             {"needs_padding", static_cast<uint32_t>(needs_padding)}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"start_tile_idx", "end_tile_idx", "start_padding_tile_idx", "end_padding_tile_idx"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    spec.kernels = {std::move(reader), std::move(writer)};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "transpose_hc_tiled_interleaved",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
        .target_nodes = total_cores}};

    ////////////////////////////////////////////////////////////////////////////
    //                      Per-core runtime args (full grid)
    ////////////////////////////////////////////////////////////////////////////
    // The work split uses two parallel partitions (unpadded vs padded tile counts) so each core needs a
    // (start, end) pair from both. Writer also tracks the padded range; reader only the unpadded count.
    auto tile_hw = tile_shape[0] * tile_shape[1];
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / tile_hw;
    uint32_t num_output_tiles = output_tensor.physical_volume() / tile_hw;
    uint32_t padded_num_tensor_tiles =
        num_output_tiles / (output_tensor.padded_shape()[2] / tile_shape[0]);  // only last row of Ct should have padding

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);
    auto
        [padded_num_cores,
         padded_all_cores,
         padded_core_group_1,
         padded_core_group_2,
         padded_num_tiles_per_core_group_1,
         padded_num_tiles_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, padded_num_tensor_tiles);

    m2::ProgramRunArgs run_args;
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    uint32_t start_idx = 0;
    uint32_t padded_start_idx = 0;
    // Need to set runtime args for all cores, not just the ones doing work.
    for (const auto& core : total_cores) {
        uint32_t num_tiles_per_core;
        uint32_t padded_tiles_per_core;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            num_tiles_per_core = 0;
        }

        if (padded_core_group_1.contains(core)) {
            padded_tiles_per_core = padded_num_tiles_per_core_group_1;
        } else if (padded_core_group_2.contains(core)) {
            padded_tiles_per_core = padded_num_tiles_per_core_group_2;
        } else {
            padded_tiles_per_core = 0;
        }

        uint32_t end_idx = start_idx + num_tiles_per_core;
        uint32_t padded_end_idx = padded_start_idx + padded_tiles_per_core;

        reader_args.runtime_arg_values.push_back(
            {core, {{"num_tiles", num_tiles_per_core}, {"start_id", start_idx}}});
        writer_args.runtime_arg_values.push_back(
            {core,
             {{"start_tile_idx", start_idx},
              {"end_tile_idx", end_idx},
              {"start_padding_tile_idx", padded_start_idx},
              {"end_padding_tile_idx", padded_end_idx}}});

        start_idx = end_idx;
        padded_start_idx = padded_end_idx;
    }
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));

    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(input_tensor.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output_tensor.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
