// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.hpp"

#include <optional>
#include <filesystem>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor SliceTileTensorArgsProgramFactory::create_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    const auto& start_tensor = tensor_args.start_tensor.value();
    const auto& end_tensor = tensor_args.end_tensor.value();
    tt::tt_metal::IDevice* device = input_tensor.device();
    ProgramDescriptor desc;

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt::tt_metal::Buffer* src_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* start_buffer = start_tensor.buffer();
    tt::tt_metal::Buffer* end_buffer = end_tensor.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();

    TT_FATAL(src_buffer != nullptr, "Input buffer should be allocated on device!");
    TT_FATAL(start_buffer != nullptr, "Start buffer should be allocated on device!");
    TT_FATAL(end_buffer != nullptr, "End buffer should be allocated on device!");
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    constexpr uint8_t src0_cb_index = 0;
    constexpr uint8_t tensor_cb_index = 1;
    constexpr uint32_t num_input_tiles = 2;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tensor_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_tensor.padded_shape().rank());
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_width = tile_shape[1];
    uint32_t tile_height = tile_shape[0];

    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index, tensor_cb_index, num_dims, tile_width, tile_height};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*start_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*end_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Reader common runtime args layout (matches kernel):
    //   [src_addr, start_buf_addr, end_buf_addr,
    //    num_unpadded_per_dim..., num_padded_per_dim..., input_shape...]
    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output.padded_shape();
    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    std::vector<uint32_t> reader_common_args(3 + (num_dims * 3));
    reader_common_args[0] = src_buffer->address();
    reader_common_args[1] = start_buffer->address();
    reader_common_args[2] = end_buffer->address();
    uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 3;
    uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
    uint32_t* input_shape_args = num_padded_tiles_per_dim + num_dims;
    num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    num_padded_tiles_per_dim[0] = num_padded_Xt;
    num_padded_tiles_per_dim[1] = num_padded_Yt;
    for (int32_t i = 2; i < static_cast<int32_t>(num_dims); ++i) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        num_padded_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }
    for (int32_t i = 0; i < static_cast<int32_t>(num_dims); ++i) {
        input_shape_args[i] = input_shape[i];
    }

    // Reader per-core runtime args: [start_id, num_tiles, id_per_dim...]
    // Writer per-core runtime args: [dst_addr, num_tiles, start_id]
    constexpr uint32_t start_offset = 0;
    KernelDescriptor::RuntimeArgs reader_runtime_args;
    KernelDescriptor::RuntimeArgs writer_runtime_args;
    uint32_t num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op core
            std::vector<uint32_t> reader_args(2 + num_dims, 0);
            reader_runtime_args.emplace_back(core, std::move(reader_args));
            writer_runtime_args.emplace_back(core, std::vector<uint32_t>{0, 0, 0});
            continue;
        }

        std::vector<uint32_t> reader_args(2 + num_dims);
        reader_args[2] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = reader_args[2] + start_offset;
        for (uint32_t j = 1; j < num_dims; ++j) {
            reader_args[2 + j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += reader_args[2 + j] * accumulated_total_per_dim[j - 1];
        }
        reader_args[0] = start_id;
        reader_args[1] = num_tiles_per_core;

        reader_runtime_args.emplace_back(core, std::move(reader_args));
        writer_runtime_args.emplace_back(
            core, std::vector<uint32_t>{dst_buffer->address(), num_tiles_per_core, num_tiles_written});
        num_tiles_written += num_tiles_per_core;
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.runtime_args = std::move(reader_runtime_args);
    reader_desc.common_runtime_args = std::move(reader_common_args);
    reader_desc.config = ReaderConfigDescriptor{};
    desc.kernels.push_back(std::move(reader_desc));

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.runtime_args = std::move(writer_runtime_args);
    writer_desc.config = WriterConfigDescriptor{};
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

// Metal 2.0 (ProgramSpec) port of the TILE tensor-args factory. Mirrors create_descriptor's
// work split and per-core / common runtime-arg computation exactly, but expresses it with the
// Metal 2.0 host API (ProgramSpec + ProgramRunArgs) and points at the forked *_m2 kernels.
//
// Arg mapping vs the legacy descriptor:
//   - The src0 CB (index 0, double-buffered) becomes DataflowBufferSpec "cb_in", bound
//     reader=PRODUCER ("cb_in") and writer=CONSUMER ("cb_out").
//   - The tensor CB (index 1, single-tile staging for start/end reads) becomes
//     DataflowBufferSpec "cb_tensor", bound as a reader self-loop (PRODUCER + CONSUMER on the
//     reader) — the reader is its only user (it reads start/end into it, then reads them back).
//   - Reader and writer share the one WorkUnitSpec on all_cores (local-DFB rule for cb_in).
//   - src/start/end/dst buffer addresses move from raw runtime args into
//     TensorParameter/TensorBinding (ta::src / ta::start / ta::end / ta::dst); their
//     TensorAccessor config is supplied by the binding rather than appended TensorAccessorArgs CTAs.
//   - Reader: num_dims/tile_width/tile_height are named CTAs; start_id/num_tiles are named
//     per-core RTAs; the per-dim id_per_dim[] array becomes runtime varargs; the common
//     [num_unpadded.., num_padded.., input_shape..] arrays become common varargs.
//   - Writer: num_pages/start_id are named per-core RTAs (reuses the shared *_m2 writer fork).
ttnn::device_operation::ProgramArtifacts SliceTileTensorArgsSpecProgramFactory::create_program_spec(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    const auto& start_tensor = tensor_args.start_tensor.value();
    const auto& end_tensor = tensor_args.end_tensor.value();
    tt::tt_metal::IDevice* device = input_tensor.device();

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    // Buffer addresses are no longer threaded through runtime args (they flow via the
    // src/start/end/dst TensorParameter bindings), so we only assert the output is allocated.
    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    constexpr uint32_t num_input_tiles = 2;

    m2::ProgramSpec spec;
    spec.name = "slice_tile_tensor_args";

    // --- DFBs (was: CB index 0 double-buffered FIFO, CB index 1 single-tile staging) ---
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"cb_in"},
            .entry_size = single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"cb_tensor"},
            .entry_size = single_tile_size,
            .num_entries = 1,
            .data_format_metadata = cb_data_format,
        },
    };

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_tensor.padded_shape().rank());
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_width = tile_shape[1];
    uint32_t tile_height = tile_shape[0];

    // --- Common (broadcast) reader args (matches kernel):
    //   [num_unpadded_per_dim..., num_padded_per_dim..., input_shape...] ---
    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output.padded_shape();
    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    std::vector<uint32_t> num_unpadded_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_padded_tiles_per_dim(num_dims);
    std::vector<uint32_t> input_shape_args(num_dims);
    num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    num_padded_tiles_per_dim[0] = num_padded_Xt;
    num_padded_tiles_per_dim[1] = num_padded_Yt;
    for (int32_t i = 2; i < static_cast<int32_t>(num_dims); ++i) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        num_padded_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }
    for (int32_t i = 0; i < static_cast<int32_t>(num_dims); ++i) {
        input_shape_args[i] = input_shape[i];
    }

    // --- Kernel specs ---
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
                                        "reader_unary_unpad_dims_interleaved_start_id_tensor_args_m2.cpp"},
        .compiler_options = {},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"cb_in"},
                    .accessor_name = "cb_in",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
                // cb_tensor self-loop: the reader is its only user (writes start/end in, reads back).
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"cb_tensor"},
                    .accessor_name = "cb_tensor",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"cb_tensor"},
                    .accessor_name = "cb_tensor",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"src"}, .accessor_name = "src"},
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"start"}, .accessor_name = "start"},
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"end"}, .accessor_name = "end"},
            },
        .compile_time_args = {{"num_dims", num_dims}, {"tile_width", tile_width}, {"tile_height", tile_height}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"start_id", "num_tiles"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };
    // Per-dim id_per_dim[] running indices are passed as runtime varargs.
    reader.advanced_options.num_runtime_varargs = num_dims;
    // [num_unpadded_per_dim..., num_padded_per_dim..., input_shape...] are passed as common varargs.
    reader.advanced_options.num_common_runtime_varargs = num_dims * 3;

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_m2.cpp"},
        .compiler_options = {},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"cb_in"},
                    .accessor_name = "cb_out",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"dst"}, .accessor_name = "dst"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    spec.kernels = {reader, writer};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"start"}, .spec = start_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"end"}, .spec = end_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"dst"}, .spec = output.tensor_spec()},
    };
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "slice_tile_tensor_args",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = all_cores,
        },
    };

    // --- ProgramRunArgs ---
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    // Common reader varargs: [num_unpadded_per_dim..., num_padded_per_dim..., input_shape...]
    m2::AdvancedKernelRunArgs::Varargs common_varargs;
    common_varargs.reserve(num_dims * 3);
    for (uint32_t j = 0; j < num_dims; ++j) {
        common_varargs.push_back(num_unpadded_tiles_per_dim[j]);
    }
    for (uint32_t j = 0; j < num_dims; ++j) {
        common_varargs.push_back(num_padded_tiles_per_dim[j]);
    }
    for (uint32_t j = 0; j < num_dims; ++j) {
        common_varargs.push_back(input_shape_args[j]);
    }
    reader_run.advanced_options.common_runtime_varargs = std::move(common_varargs);

    // The legacy tensor-args reader uses a fixed base start_id of 0 (start_offset is constexpr 0
    // there); the per-region offset is computed kernel-side from the start/end tensors.
    constexpr uint32_t start_offset = 0;
    uint32_t num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        const m2::NodeCoord node{core};
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op core: zero-filled args (matches legacy descriptor).
            reader_run.runtime_arg_values.push_back({node, {{"start_id", 0u}, {"num_tiles", 0u}}});
            reader_run.advanced_options.runtime_varargs[node] = m2::AdvancedKernelRunArgs::Varargs(num_dims, 0u);
            writer_run.runtime_arg_values.push_back({node, {{"num_pages", 0u}, {"start_id", 0u}}});
            continue;
        }

        // Per-dim starting indices for this core (the legacy reader_args[2..] / id_per_dim varargs).
        m2::AdvancedKernelRunArgs::Varargs id_per_dim(num_dims);
        id_per_dim[0] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;
        for (uint32_t j = 1; j < num_dims; ++j) {
            id_per_dim[j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        reader_run.runtime_arg_values.push_back({node, {{"start_id", start_id}, {"num_tiles", num_tiles_per_core}}});
        reader_run.advanced_options.runtime_varargs[node] = std::move(id_per_dim);

        writer_run.runtime_arg_values.push_back(
            {node, {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_written}}});

        num_tiles_written += num_tiles_per_core;
    }

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"src"}, input_tensor.mesh_tensor()},
        {m2::TensorParamName{"start"}, start_tensor.mesh_tensor()},
        {m2::TensorParamName{"end"}, end_tensor.mesh_tensor()},
        {m2::TensorParamName{"dst"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
