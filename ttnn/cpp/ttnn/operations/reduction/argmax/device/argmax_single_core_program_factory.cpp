// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "argmax_device_operation.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <string>
#include <utility>

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

static std::tuple<uint32_t, uint32_t> get_page_sizes_single_core(
    const MeshTensor& input, const MeshTensor& output, bool keepdim, bool reduce_all) {
    const auto& input_shape = input.padded_shape();
    const uint32_t rank = input_shape.size();

    switch (input.layout()) {
        case Layout::ROW_MAJOR: {
            const uint32_t red_dim_units = input_shape[rank - 1];
            const uint32_t input_unit_size = input.element_size();
            const uint32_t output_unit_size = output.element_size();
            const uint32_t output_last_dim = reduce_all or keepdim or (rank < 2) ? 1 : input_shape[rank - 2];

            return {red_dim_units * input_unit_size, output_last_dim * output_unit_size};
        }
        case Layout::TILE: {
            TT_FATAL(
                output.layout() == Layout::ROW_MAJOR,
                "For TILE input layout, only ROW_MAJOR output is supported, got output layout: {}",
                output.layout());

            return {input.tensor_spec().compute_page_size_bytes(), output.tensor_spec().compute_page_size_bytes()};
        }
        default:
            TT_FATAL(
                false,
                "Unsupported input layout {} for argmax single-core. Supported: ROW_MAJOR, TILE",
                input.layout());
    }
}

// Named compile-time arguments for the selected reader.
//
// The argument names are the reader kernels' own variable names, so a reader reads each one as
// `get_arg(args::<name>)`. The ROW_MAJOR and TILE readers take different argument sets, matching
// the branch that selects the kernel source.
static KernelSpec::CompileTimeArgs get_ctime_args_single_core(
    const MeshTensor& input,
    uint32_t src_page_size,
    uint32_t dst_page_size,
    bool keepdim,
    bool reduce_all,
    bool tile_reader_omit_reduce_all_keepdim) {
    const auto& input_shape = input.padded_shape();
    const uint32_t rank = input_shape.size();

    switch (input.layout()) {
        case Layout::ROW_MAJOR: {
            const uint32_t red_dim_units = input_shape[rank - 1];
            const uint32_t output_last_dim = reduce_all or keepdim or (rank < 2) ? 1 : input_shape[rank - 2];
            const uint32_t inner_dim_units = output_last_dim;
            const uint32_t outer_dim_units = input.logical_volume() / inner_dim_units / red_dim_units;

            return {
                {"src_page_size", src_page_size},
                {"dst_page_size", dst_page_size},
                {"outer_dim_units", outer_dim_units},
                {"inner_dim_units", inner_dim_units},
                {"red_dim_units", red_dim_units},
                {"reduce_all", static_cast<uint32_t>(reduce_all)},
            };
        }
        case Layout::TILE: {
            const uint32_t logical_rank = input.logical_shape().size();
            const uint32_t tile_width = input.tensor_spec().tile().get_width();
            const uint32_t tile_height = input.tensor_spec().tile().get_height();
            const uint32_t w_tiles = input_shape[rank - 1] / tile_width;
            const uint32_t h_tiles = input_shape[rank - 2] / tile_height;
            const uint32_t w_logical = input.logical_shape()[logical_rank - 1];
            const uint32_t h_logical = logical_rank > 1 ? input.logical_shape()[logical_rank - 2] : 1;
            const uint32_t outer_dim_units = input.logical_volume() / (h_logical * w_logical);

            KernelSpec::CompileTimeArgs cta = {
                {"src_page_size", src_page_size},
                // Neither TILE reader reads dst_page_size: both derive the write size from
                // output_page_elements instead. Emitted anyway so the two reader families keep a
                // single argument builder, exactly as the pre-Metal-2.0 factory did.
                {"dst_page_size", dst_page_size},
                {"tile_height", tile_height},
                {"tile_width", tile_width},
                {"input_height", h_tiles},
                {"input_width", w_tiles},
                {"logical_height", h_logical},
                {"logical_width", w_logical},
                {"outer_dim_size", outer_dim_units},
            };
            // Width-reduction reader uses reduce_all/keepdim; height-reduction reader does not.
            if (!tile_reader_omit_reduce_all_keepdim) {
                cta.emplace("reduce_all", static_cast<uint32_t>(reduce_all));
                cta.emplace("keepdim", static_cast<uint32_t>(keepdim));
            }
            return cta;
        }
        default:
            TT_FATAL(
                false,
                "Unsupported input layout {} for argmax single-core. Supported: ROW_MAJOR, TILE",
                input.layout());
    }
}

ttnn::device_operation::ProgramArtifacts ArgMaxSingleCoreProgramFactory::create_program_artifacts(
    const ArgmaxParams& operation_attributes, const ArgmaxInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const auto& dim = operation_attributes.dim;
    const bool keepdim = operation_attributes.keepdim;

    const tt::tt_metal::IDevice* device = &output.mutable_device();
    const bool reduce_all = not dim.has_value();

    // Resource names. Declared function-locally: both argmax factories share a unity-build
    // translation unit, and namespace-scope `const` objects would collide by name there.
    const KernelSpecName READER{"reader"};
    const DFBSpecName SRC{"src"};
    const DFBSpecName DST{"dst"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    const auto grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_units = 1;  // single-core
    auto [num_cores, all_cores, unused_1, unused_2, unused_3, unused_4] =
        tt::tt_metal::split_work_to_cores(grid_size, num_units);

    TT_FATAL(num_cores > 0, "Argmax single-core split requires at least one core ");
    validate_reduce_op_program_grid(
        "Argmax single-core", all_cores, device->compute_with_storage_grid_size(), nullptr, true, {});

    const tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const auto [src_page_size, dst_page_size] = get_page_sizes_single_core(input, output, keepdim, reduce_all);

    // Input DFB: one entry holding a whole input page.
    DataflowBufferSpec dfb_src{
        .unique_id = SRC,
        .entry_size = src_page_size,
        .num_entries = 1,
        .data_format_metadata = input_data_format,
    };

    // Output DFB: one entry holding a whole output page.
    DataflowBufferSpec dfb_dst{
        .unique_id = DST,
        .entry_size = dst_page_size,
        .num_entries = 1,
        .data_format_metadata = output_data_format,
    };

    const int32_t rank_i = static_cast<int32_t>(input.logical_shape().size());
    const int32_t nd = dim.has_value() ? (dim.value() < 0 ? dim.value() + rank_i : dim.value()) : -1;
    const bool dim_is_h = dim.has_value() && rank_i >= 2 && nd == rank_i - 2;

    // Compile-time args
    const bool tile_reader_omit_reduce_all_keepdim = input.layout() == Layout::TILE && dim_is_h;
    KernelSpec::CompileTimeArgs ctime_args = get_ctime_args_single_core(
        input, src_page_size, dst_page_size, keepdim, reduce_all, tile_reader_omit_reduce_all_keepdim);

    std::string kernel_path;
    if (input.layout() == Layout::ROW_MAJOR) {
        // The Metal 2.0 fork; the legacy source beside it still serves a ProgramDescriptor
        // consumer that cannot supply named bindings.
        kernel_path = "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_metal2.cpp";
    } else {
        // TILE: reduction on H (rank-2) uses the height reader
        kernel_path = dim_is_h
                          ? "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_tile_layout_h.cpp"
                          : "ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_tile_layout.cpp";
    }

    // Both DFBs are touched by this one reader and by nothing else: it takes a raw write pointer
    // into each and never runs a FIFO operation on either. A single toucher cannot present a
    // producer and a consumer on distinct kernels, so the reader is bound as both.
    KernelSpec reader{
        .unique_id = READER,
        .source = kernel_path,
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = SRC, .accessor_name = "src", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = SRC, .accessor_name = "src", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = DST, .accessor_name = "dst", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = DST, .accessor_name = "dst", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"},
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"},
            },
        .compile_time_args = std::move(ctime_args),
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    ProgramSpec spec{
        .name = "argmax_single_core",
        .kernels = {std::move(reader)},
        .dataflow_buffers = {std::move(dfb_src), std::move(dfb_dst)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{.name = "main", .kernels = {READER}, .target_nodes = all_cores},
            },
    };

    // The reader declares no runtime or common runtime arguments — the two tensor base addresses
    // it used to read as runtime args now arrive through its tensor bindings — so it needs no
    // KernelRunArgs entry.
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {INPUT, input},
        {OUTPUT, output},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
