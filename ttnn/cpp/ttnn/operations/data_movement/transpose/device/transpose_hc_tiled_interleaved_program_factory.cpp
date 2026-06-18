// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_tiled_interleaved_program_factory.hpp"
#include "transpose_utils.hpp"

#include "ttnn/operations/math.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Per-core runtime args for the reader + writer kernels. The work split uses
// two parallel partitions (unpadded vs padded tile counts) so each core needs a
// (start, end) pair from both. Writer also tracks the padded range; reader only
// the unpadded count. Only the dispatch channel changes (named RTAs); the legacy
// buffer-address RTA (reader/writer slot 0) is replaced by the TensorBindings.
void emit_runtime_args_hc_tiled_interleaved(
    m2::KernelRunArgs& reader_run,
    m2::KernelRunArgs& writer_run,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const CoreRange& total_cores) {
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    auto tile_hw = tile_shape[0] * tile_shape[1];
    uint32_t num_tensor_tiles = input_tensor.physical_volume() / tile_hw;
    uint32_t num_output_tiles = output_tensor.physical_volume() / tile_hw;
    uint32_t padded_num_tensor_tiles = num_output_tiles / (output_tensor.padded_shape()[2] /
                                                           tile_shape[0]);  // only last row of Ct should have padding

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
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

    all_cores = num_cores > padded_num_cores ? all_cores : padded_all_cores;
    auto cores = corerange_to_cores(all_cores, std::nullopt);

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

        reader_run.runtime_arg_values.push_back({core, {{"num_tiles", num_tiles_per_core}, {"start_id", start_idx}}});
        writer_run.runtime_arg_values.push_back(
            {core,
             {{"start_tile_idx", start_idx},
              {"end_tile_idx", end_idx},
              {"start_padding_tile_idx", padded_start_idx},
              {"end_padding_tile_idx", padded_end_idx}}});

        start_idx = end_idx;
        padded_start_idx = padded_end_idx;
    }
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeHCTiledInterleavedProgramFactory::create_program_spec(
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

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "transpose_hc_tiled_interleaved";

    // src0 DFB (legacy CB c_0): the reader produces tiles into it, the writer consumes them.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = single_tile_size,
            .num_entries = 2,
            .data_format_metadata = cb_data_format,
        },
    };

    auto max_padding_write = face_shape[0] * face_shape[1];
    // padding DFB (legacy CB c_1): only present when needs_padding. The reader produces one padding
    // entry, the writer consumes it. The conditional binding is honored kernel-side via #ifdef
    // NEEDS_PADDING.
    if (needs_padding) {
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"padding"},
            .entry_size = max_padding_write * input_tensor.element_size(),
            .num_entries = 1,
            .data_format_metadata = cb_data_format,
        });
    }

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

    // Reader: forked from reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp (shared with
    // the unmigrated permute_tiled factory) to reader_unary_transpose_hc_interleaved_tiled_padding_aware_m2.cpp.
    // The legacy factory built the input accessor with TensorAccessorArgs(RuntimeTensorShape) and
    // plumbed the buffer address through RTA slot 0; both collapse to the TensorBinding below. The
    // legacy named CTA `needs_padding` is promoted to the NEEDS_PADDING define (gates dfb::padding).
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                        "reader_unary_transpose_hc_interleaved_tiled_padding_aware_m2.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "src0",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "input"},
            },
        .compile_time_args =
            {
                {"num_writes", num_writes},
                {"padding_val_packed", padding_val_packed},
                {"swap_hw", 0u},
                {"H", 1u},
                {"W", 1u},
                {"accumulated_outer_dims", 1u},
                {"tile_height", 1u},
                {"tile_width", 1u},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_tiles", "start_id"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    // Writer (in place — used only by this factory). The legacy writer carried the output CB index
    // (= c_0, the shared src0 DFB) as CTA slot 1; that magic index is replaced by the dfb::out binding
    // (accessor_name "out", consuming the src0 DFB). The legacy `needs_padding` CTA is promoted to the
    // NEEDS_PADDING define (gates dfb::padding).
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                        "writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "out",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "output"},
            },
        .compile_time_args =
            {
                {"element_size", element_size},
                {"C", C},
                {"H", H},
                {"W", W},
                {"tile_height", tile_shape[0]},
                {"tile_width", tile_shape[1]},
                {"face_height", face_shape[0]},
                {"face_width", face_shape[1]},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"start_tile_idx", "end_tile_idx", "start_padding_tile_idx", "end_padding_tile_idx"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    if (needs_padding) {
        // The padding DFB is bound on both kernels; emit the matching define to both so the kernel-side
        // dfb::padding references compile.
        reader.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"padding"},
            .accessor_name = "padding",
            .endpoint_type = m2::DFBEndpointType::PRODUCER,
        });
        reader.compiler_options.defines = {{"NEEDS_PADDING", "1"}};
        writer.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"padding"},
            .accessor_name = "padding",
            .endpoint_type = m2::DFBEndpointType::CONSUMER,
        });
        writer.compiler_options.defines = {{"NEEDS_PADDING", "1"}};
    }

    spec.kernels = {reader, writer};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output_tensor.tensor_spec()},
    };
    // reader (produces src0/padding) and writer (consumes src0/padding) share one WorkUnitSpec — every
    // node hosting a DFB hosts both its endpoints. The legacy factory launches both on the full grid
    // (total_cores); cores outside the work split get num_tiles = 0.
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "transpose_hc_tiled_interleaved",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = total_cores,
        },
    };

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    emit_runtime_args_hc_tiled_interleaved(reader_run, writer_run, input_tensor, output_tensor, total_cores);

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"input"}, input_tensor.mesh_tensor()},
        {m2::TensorParamName{"output"}, output_tensor.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
