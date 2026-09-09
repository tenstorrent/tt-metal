// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_align.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts Fold::MultiCore::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const Tensor& input = tensor_args.input_tensor;
    const Tensor& output = output_tensor;
    const uint32_t stride_h = operation_attributes.stride_h;
    const uint32_t stride_w = operation_attributes.stride_w;

    // Metal 2.0 resource + kernel names, declared local to the factory (not at namespace scope) so
    // they can't collide with a sibling factory in the same unity-build translation unit.
    const DFBSpecName SRC0{"src0"};
    const DFBSpecName DST0{"dst0"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName READER{"reader"};

    constexpr const char* FOLD_KERNEL =
        "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_dfb2s_row_major.cpp";

    auto all_cores = input.shard_spec()->grid;
    auto shard_shape = input.shard_spec()->shape;

    tt::DataFormat dfb_data_format = datatype_to_dataformat_converter(input.dtype());

    uint32_t pixel_size = shard_shape[1] * input.element_size();
    uint32_t num_pixels = shard_shape[0];
    uint32_t num_dst_pixels = num_pixels / (stride_h * stride_w);

    // chunk consists of channel values of stride_w neighboring pixels along the W dimension
    uint32_t width = input.padded_shape()[2];
    uint32_t chunk_size = stride_w * pixel_size;
    uint32_t dst_pixel_size = stride_h * chunk_size;
    uint32_t num_dst_rows = num_pixels / (width * stride_h);
    uint32_t pixels_per_dst_row = stride_h * width;

    const uint32_t aligned_pixel_size = tt::align(pixel_size, hal::get_l1_alignment());
    const uint32_t aligned_dst_pixel_size = tt::align(dst_pixel_size, hal::get_l1_alignment());

    // Input DFB — borrowed onto the sharded input buffer.
    // The backing L1 address resolves at runtime from the INPUT tensor argument.
    DataflowBufferSpec src0_dfb{
        .unique_id = SRC0,
        .entry_size = aligned_pixel_size,
        .num_entries = num_pixels,
        .data_format_metadata = dfb_data_format,
        .borrowed_from = INPUT,
    };

    // Output DFB — borrowed onto the sharded output buffer.
    DataflowBufferSpec dst0_dfb{
        .unique_id = DST0,
        .entry_size = aligned_dst_pixel_size,
        .num_entries = num_dst_pixels,
        .data_format_metadata = dfb_data_format,
        .borrowed_from = OUTPUT,
    };

    TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // Named compile-time args shared by both instances. The legacy `is_reader` CTA (a per-instance
    // literal that splits the output columns between the two same-source instances) is appended
    // per instance below. The two magic CB-index CTAs are gone — replaced by DFB bindings.
    const KernelSpec::CompileTimeArgs common_cta{
        {"pixel_size", pixel_size},
        {"aligned_pixel_size", aligned_pixel_size},
        {"aligned_dst_pixel_size", aligned_dst_pixel_size},
        {"aligned_chunk_size", stride_w * aligned_pixel_size},
        {"aligned_row_size", width * aligned_pixel_size},
        {"stride_h", stride_h},
        {"stride_w", stride_w},
        {"num_dst_rows", num_dst_rows},
        {"num_dst_cols", width / stride_w},
        {"dst_row_offset", pixels_per_dst_row * aligned_pixel_size},
        {"element_size", input.element_size()},
    };

    // Dual-instance work-split: one kernel source instantiated twice over the same grid,
    // splitting the output columns. Both instances raw-touch both borrowed DFBs (no FIFO ops),
    // so the endpoints are role-free — assign 1P+1C per DFB (cosmetic on Gen1), not multi-binding.
    // Build optimization level Os was faster than O2 when this kernel was originally tuned.
    auto make_cta = [&](uint32_t is_reader) {
        KernelSpec::CompileTimeArgs cta = common_cta;
        cta.insert({"is_reader", is_reader});
        return cta;
    };

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path{FOLD_KERNEL},
        .compiler_options = {.opt_level = KernelBuildOptLevel::Os},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "src0", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = DST0, .accessor_name = "dst0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args = make_cta(/*is_reader=*/1),
        .hw_config = ttnn::create_writer_datamovement_config(input.device()->arch()),
    };

    KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path{FOLD_KERNEL},
        .compiler_options = {.opt_level = KernelBuildOptLevel::Os},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "src0", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = DST0, .accessor_name = "dst0", .endpoint_type = DFBEndpointType::CONSUMER}},
        .compile_time_args = make_cta(/*is_reader=*/0),
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch()),
    };

    ProgramSpec spec{
        .name = "fold_multi_core_sharded",
        .kernels = {writer_spec, reader_spec},
        .dataflow_buffers = {src0_dfb, dst0_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {WRITER, READER},
            .target_nodes = all_cores,
        }},
    };

    ProgramRunArgs run_args;
    // No runtime args on either kernel; provide empty entries so every kernel has a KernelRunArgs.
    run_args.kernel_run_args = {KernelRunArgs{.kernel = WRITER}, KernelRunArgs{.kernel = READER}};
    run_args.tensor_args = {
        {INPUT, TensorArgument{input.mesh_tensor()}},
        {OUTPUT, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::data_movement
