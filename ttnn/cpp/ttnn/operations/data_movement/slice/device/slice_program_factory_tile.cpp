// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"

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

tt::tt_metal::ProgramDescriptor SliceTileProgramFactory::create_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt::tt_metal::Buffer* src0_buffer = input.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    // --- CB Descriptor ---
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;

    tt::tt_metal::ProgramDescriptor program_descriptor;

    tt::tt_metal::CBDescriptor cb_desc;
    cb_desc.total_size = num_input_tiles * single_tile_size;
    cb_desc.core_ranges = all_cores;
    cb_desc.format_descriptors.push_back(tt::tt_metal::CBFormatDescriptor{
        .buffer_index = static_cast<uint8_t>(src0_cb_index),
        .data_format = cb_data_format,
        .page_size = single_tile_size});
    program_descriptor.cbs.push_back(std::move(cb_desc));

    // --- Reader Kernel Descriptor ---
    // CB index via named compile-time arg (essential for fusion CB remapping).
    std::vector<uint32_t> reader_compile_time_args = {num_dims};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    // Reader common runtime args: [src_addr, num_unpadded_per_dim..., num_padded_per_dim...]
    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    std::vector<uint32_t> reader_common_args(1 + (num_dims * 2));
    reader_common_args[0] = src0_buffer->address();
    uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 1;
    uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
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

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input, args.slice_start);

    // Reader per-core runtime args: [start_id, num_tiles, id_per_dim...]
    tt::tt_metal::KernelDescriptor::RuntimeArgs reader_runtime_args;
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
            continue;
        }

        std::vector<uint32_t> reader_args(2 + num_dims);
        // Compute per-dim indices for this core's starting position
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
        num_tiles_written += num_tiles_per_core;
    }

    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "reader_unary_unpad_dims_interleaved_start_id.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = reader_compile_time_args;
    reader_kernel_desc.named_compile_time_args = {{"cb_in", src0_cb_index}};
    reader_kernel_desc.runtime_args = std::move(reader_runtime_args);
    reader_kernel_desc.common_runtime_args = std::move(reader_common_args);
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};
    program_descriptor.kernels.push_back(std::move(reader_kernel_desc));

    // --- Writer Kernel Descriptor ---
    // CB index via named compile-time arg (essential for fusion CB remapping).
    std::vector<uint32_t> writer_compile_time_args = {};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    // Writer per-core runtime args: [dst_addr, num_tiles, start_id]
    tt::tt_metal::KernelDescriptor::RuntimeArgs writer_runtime_args;
    num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op core
            writer_runtime_args.emplace_back(core, std::vector<uint32_t>{0, 0, 0});
            continue;
        }

        writer_runtime_args.emplace_back(
            core, std::vector<uint32_t>{dst_buffer->address(), num_tiles_per_core, num_tiles_written});
        num_tiles_written += num_tiles_per_core;
    }

    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = writer_compile_time_args;
    writer_kernel_desc.named_compile_time_args = {{"cb_out", src0_cb_index}};
    writer_kernel_desc.runtime_args = std::move(writer_runtime_args);
    writer_kernel_desc.config = tt::tt_metal::WriterConfigDescriptor{};
    program_descriptor.kernels.push_back(std::move(writer_kernel_desc));

    return program_descriptor;
}

// Metal 2.0 (ProgramSpec) port of the TILE non-strided factory. Mirrors create_descriptor's
// work split and per-core / common runtime-arg computation exactly, but expresses it with the
// Metal 2.0 host API (ProgramSpec + ProgramRunArgs) and points at the forked *_m2 kernels.
//
// Arg mapping vs the legacy descriptor:
//   - The single src0 CB (index 0) becomes one DataflowBufferSpec, bound reader=PRODUCER
//     (accessor "cb_in") and writer=CONSUMER (accessor "cb_out"). Reader and writer share the
//     one WorkUnitSpec on all_cores (local-DFB rule).
//   - src/dst buffer addresses move from raw runtime args (src_addr/dst_addr) into
//     TensorParameter/TensorBinding (ta::src / ta::dst); their TensorAccessor config is supplied
//     by the binding rather than appended TensorAccessorArgs CTAs.
//   - Reader: num_dims is a named CTA; start_id/num_tiles are named per-core RTAs; the per-dim
//     id_per_dim[] array becomes runtime varargs; the common [num_unpadded.., num_padded..]
//     arrays become common varargs.
//   - Writer: num_pages/start_id are named per-core RTAs.
ttnn::device_operation::ProgramArtifacts SliceTileSpecProgramFactory::create_program_spec(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    // Buffer addresses are no longer threaded through runtime args (they flow via the
    // src/dst TensorParameter bindings), so we only need the output buffer for the assert.
    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    // --- DFB (was: CB index 0, double-buffered single-tile FIFO) ---
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;

    m2::ProgramSpec spec;
    spec.name = "slice_tile";

    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"cb_in_out"},
            .entry_size = single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = cb_data_format,
        },
    };
    (void)src0_cb_index;  // index is now implied by the DFB binding

    // --- Common (broadcast) reader args: [num_unpadded_per_dim..., num_padded_per_dim...] ---
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

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input, args.slice_start);

    // --- Kernel specs ---
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
                                        "reader_unary_unpad_dims_interleaved_start_id_m2.cpp"},
        .compiler_options = {},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"cb_in_out"},
                    .accessor_name = "cb_in",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"src"}, .accessor_name = "src"},
            },
        .compile_time_args = {{"num_dims", num_dims}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"start_id", "num_tiles"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };
    // Per-dim id_per_dim[] running indices are passed as runtime varargs.
    reader.advanced_options.num_runtime_varargs = num_dims;
    // [num_unpadded_per_dim..., num_padded_per_dim...] are passed as common varargs.
    reader.advanced_options.num_common_runtime_varargs = num_dims * 2;

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_m2.cpp"},
        .compiler_options = {},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"cb_in_out"},
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
        m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = input.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"dst"}, .spec = output.tensor_spec()},
    };
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "slice_tile",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = all_cores,
        },
    };

    // --- ProgramRunArgs ---
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    // Common reader varargs: [num_unpadded_per_dim..., num_padded_per_dim...]
    m2::AdvancedKernelRunArgs::Varargs common_varargs;
    common_varargs.reserve(num_dims * 2);
    for (uint32_t j = 0; j < num_dims; ++j) {
        common_varargs.push_back(num_unpadded_tiles_per_dim[j]);
    }
    for (uint32_t j = 0; j < num_dims; ++j) {
        common_varargs.push_back(num_padded_tiles_per_dim[j]);
    }
    reader_run.advanced_options.common_runtime_varargs = std::move(common_varargs);

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
        {m2::TensorParamName{"src"}, input.mesh_tensor()},
        {m2::TensorParamName{"dst"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
