// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"

#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/math.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::experimental::quasar {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts Fold::MultiCore::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const MeshTensor& input = tensor_args.input_tensor.mesh_tensor();
    const MeshTensor& output = output_tensor.mesh_tensor();
    const uint32_t stride_h = operation_attributes.stride_h;
    const uint32_t stride_w = operation_attributes.stride_w;

    auto all_cores = tensor_args.input_tensor.shard_spec()->grid;
    auto shard_shape = tensor_args.input_tensor.shard_spec()->shape;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());

    uint32_t pixel_size = shard_shape[1] * tensor_args.input_tensor.element_size();
    uint32_t num_pixels = shard_shape[0];
    uint32_t num_dst_pixels = num_pixels / (stride_h * stride_w);

    // chunk consists of channel values of stride_w neighboring pixels along the W dimension
    uint32_t width = tensor_args.input_tensor.padded_shape()[2];
    uint32_t chunk_size = stride_w * pixel_size;
    uint32_t dst_pixel_size = stride_h * chunk_size;
    uint32_t num_dst_rows = num_pixels / (width * stride_h);
    uint32_t pixels_per_dst_row = stride_h * width;

    const uint32_t aligned_pixel_size = tt::align(pixel_size, hal::get_l1_alignment());
    const uint32_t aligned_dst_pixel_size = tt::align(dst_pixel_size, hal::get_l1_alignment());

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName SRC0{"src0"};
    const DFBSpecName DST0{"dst0"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName READER{"reader"};

    const std::filesystem::path kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/fold/device/kernels/dataflow/writer_cb2s_row_major.cpp";

    // ---- Tensor parameters ----
    TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Dataflow buffers (borrowed from the io tensors) ----
    // Input DFB — globally allocated to the sharded input buffer (legacy c_0).
    DataflowBufferSpec src0_dfb{
        .unique_id = SRC0,
        .entry_size = aligned_pixel_size,
        .num_entries = num_pixels,
        .data_format_metadata = cb_data_format,
        .borrowed_from = INPUT,
    };
    // Output DFB — globally allocated to the sharded output buffer (legacy c_16).
    DataflowBufferSpec dst0_dfb{
        .unique_id = DST0,
        .entry_size = aligned_dst_pixel_size,
        .num_entries = num_dst_pixels,
        .data_format_metadata = cb_data_format,
        .borrowed_from = OUTPUT,
    };

    // ---- Compile-time args (shared by both kernel roles, except is_reader) ----
    // Legacy CTA slots 0,1 (src_cb, dst_cb) become DFB bindings; the rest are named CTAs.
    auto make_cta = [&](uint32_t is_reader) {
        return KernelSpec::CompileTimeArgs{
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
            {"element_size", tensor_args.input_tensor.element_size()},
            {"is_reader", is_reader},
        };
    };

    // Both kernels read SRC0 (input) and write DST0 (output) purely by base pointer
    // (borrowed-memory DFBs; no FIFO ops). To satisfy the spec validator's
    // one-producer / one-consumer-per-node invariant for each DFB while letting BOTH
    // kernels access BOTH buffers, the fake producer/consumer roles are split across
    // the two co-resident kernels: SRC0 gets reader=PRODUCER + writer=CONSUMER;
    // DST0 gets writer=PRODUCER + reader=CONSUMER. (Self-loop fake-CB workaround;
    // see METAL2_PORT_REPORT.md — neither kernel performs real FIFO operations.)

    // Writer-role kernel: shares the same source as the reader (a single kernel file
    // selects its role from the is_reader compile-time arg). Build optimization level Os
    // was faster than O2 when this kernel was originally tuned.
    KernelSpec writer{
        .unique_id = WRITER,
        .source = kernel_source,
        .compiler_options = {.opt_level = KernelBuildOptLevel::Os},
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "src0", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = DST0, .accessor_name = "dst0", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .compile_time_args = make_cta(/*is_reader=*/1),
        .hw_config = ttnn::create_writer_datamovement_config(input.device().arch()),
    };

    KernelSpec reader{
        .unique_id = READER,
        .source = kernel_source,
        .compiler_options = {.opt_level = KernelBuildOptLevel::Os},
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "src0", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = DST0, .accessor_name = "dst0", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .compile_time_args = make_cta(/*is_reader=*/0),
        .hw_config = ttnn::create_reader_datamovement_config(input.device().arch()),
    };

    // ---- Assemble the spec ----
    ProgramSpec spec;
    spec.name = "fold_multi_core_sharded";
    spec.kernels = {std::move(writer), std::move(reader)};
    spec.dataflow_buffers = {std::move(src0_dfb), std::move(dst0_dfb)};
    spec.tensor_parameters = {std::move(input_param), std::move(output_param)};
    spec.work_units = {WorkUnitSpec{.name = "wu", .kernels = {WRITER, READER}, .target_nodes = all_cores}};

    // ---- Run args ----
    // Neither kernel has runtime args; provide empty per-kernel entries to satisfy the
    // "a KernelRunArgs for every kernel" contract.
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{.kernel = WRITER},
        ProgramRunArgs::KernelRunArgs{.kernel = READER},
    };
    run_args.tensor_args.insert({INPUT, input});
    run_args.tensor_args.insert({OUTPUT, output});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::experimental::quasar
