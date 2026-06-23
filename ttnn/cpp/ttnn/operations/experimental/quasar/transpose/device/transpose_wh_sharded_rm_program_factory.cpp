// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_rm_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <vector>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts TransposeWHShardedRMProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN0{"cb_in0"};              // legacy c_0: input shard (borrowed)
    const DFBSpecName CB_IN{"cb_in"};                // legacy c_24: reader -> compute tile staging
    const DFBSpecName CB_TILIZE{"cb_tilize"};        // legacy c_25: tilize self-loop intermediate
    const DFBSpecName CB_OUT_STAGE{"cb_out_stage"};  // legacy c_27: compute -> writer staging (ht>8)
    const DFBSpecName CB_OUT0{"cb_out0"};            // legacy c_16: output shard (borrowed)

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    const uint32_t stick_size_bytes = W * input_tensor.element_size();
    const uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    // Output page geometry + per-row pack page counts (verbatim from legacy). pack_num_pages and
    // pack_num_pages_last_row were dead in the SHARDED compute kernel and are not forwarded.
    uint32_t output_page_size, pack_num_pages_last_col, pack_num_pages_last_row_col;
    if ((W % TILE_WIDTH) != 0 and (H % TILE_HEIGHT) != 0) {
        output_page_size = (W % TILE_WIDTH) * (H % TILE_HEIGHT) * output_tensor.element_size();
        auto output_page_size_last_col = TILE_WIDTH * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages_last_col = dst_single_tile_size / output_page_size_last_col;
        pack_num_pages_last_row_col = 1;
    } else if ((W % TILE_WIDTH) != 0 and (H % TILE_HEIGHT) == 0) {
        output_page_size = (W % TILE_WIDTH) * (TILE_HEIGHT)*output_tensor.element_size();
        pack_num_pages_last_col = dst_single_tile_size / output_page_size;
        pack_num_pages_last_row_col = 1;
    } else if ((W % TILE_WIDTH) == 0 and (H % TILE_HEIGHT) != 0) {
        output_page_size = (TILE_WIDTH) * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages_last_col = 1;
        pack_num_pages_last_row_col = 1;
    } else {
        output_page_size = dst_single_tile_size;
        pack_num_pages_last_col = 1;
        pack_num_pages_last_row_col = 1;
    }

    const auto shard_spec = input_tensor.shard_spec().value();
    const uint32_t shard_height = shard_spec.shape[0];
    const uint32_t num_hw_blocks_per_core = shard_height / H;

    const bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;
    const auto& all_cores = shard_spec.grid;

    const bool ht_gt_8 = ht > 8;

    // Reader/writer geometry (CTAs), verbatim from legacy.
    const uint32_t H_per_tile = H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT;
    const uint32_t H_per_tile_last = H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT;
    const uint32_t W_per_tile = W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH;
    const uint32_t W_per_tile_last = W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH;
    const uint32_t last_output_row_num_datums = H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT;

    log_debug(tt::LogOp, "transpose_wh_sharded_rm");
    log_debug(tt::LogOp, "ht: {}, wt: {}, num_hw_blocks_per_core: {}", ht, wt, num_hw_blocks_per_core);
    log_debug(tt::LogOp, "output_page_size: {}", output_page_size);

    // ------------------------------------------------------------------------
    // DataflowBufferSpecs. cb_in0 / cb_out0 are borrowed-memory DFBs aliasing the input/output shard
    // buffers (legacy CBDescriptor::buffer); cb_in / cb_tilize are program-scope L1 intermediates;
    // cb_out_stage exists only on the ht>8 path (compute -> writer staging).
    // ------------------------------------------------------------------------
    std::vector<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = CB_IN0,
        .entry_size = stick_size_bytes,
        .num_entries = shard_height,
        .data_format_metadata = src0_cb_data_format,
        .borrowed_from = INPUT_TENSOR,
    });
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = CB_IN,
        .entry_size = src0_single_tile_size,
        .num_entries = wt * 2,  // double buffer
        .data_format_metadata = src0_cb_data_format,
    });
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = CB_TILIZE,
        .entry_size = src0_single_tile_size,
        .num_entries = ht * wt,
        .data_format_metadata = src0_cb_data_format,
    });
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = CB_OUT0,
        .entry_size = output_page_size,
        .num_entries = (stick_size_bytes * shard_height) / output_page_size,
        .data_format_metadata = dst_cb_data_format,
        .borrowed_from = OUTPUT_TENSOR,
    });
    if (ht_gt_8) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = CB_OUT_STAGE,
            .entry_size = dst_single_tile_size,
            .num_entries = ht * 2,  // double buffer
            .data_format_metadata = dst_cb_data_format,
        });
    }

    // ------------------------------------------------------------------------
    // Tensor parameters. Each is used only as a DFB borrowed_from backing store (no kernel-side
    // TensorAccessor on the sharded path), which the framework counts as a legitimate use.
    // ------------------------------------------------------------------------
    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_tensor.tensor_spec()};

    // ------------------------------------------------------------------------
    // Kernels. reader gathers rows from the borrowed input shard (cb_in0, read by address) into the
    // tile-staging cb_in; compute tilizes + transposes + pack-untilizes into cb_out (mapped to the
    // borrowed output shard on ht<=8, or to cb_out_stage on ht>8); the ht>8 writer drains cb_out_stage
    // into the borrowed output shard (cb_out0, written by address).
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source =
            std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
                                  "reader_unary_transpose_wh_sharded_rm.cpp"},
        // cb_in0: read-by-address borrowed shard; single-producer-single-consumer self-loop (no FIFO ops).
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = CB_IN0, .accessor_name = "cb_src", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = CB_IN0, .accessor_name = "cb_src", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = CB_IN, .accessor_name = "cb_dst", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"num_hw_blocks_per_core", num_hw_blocks_per_core},
             {"Ht", ht},
             {"H_per_tile", H_per_tile},
             {"H_per_tile_last", H_per_tile_last},
             {"Wt", wt},
             {"W_size_bytes", stick_size_bytes},
             {"l1_write_offset_bytes", wt * input_tensor.element_size() * TILE_WIDTH}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    ComputeHardwareConfig compute_cfg{.fp32_dest_acc_en = fp32_dest_acc_en};
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        // Keep both the tilize input (cb_in) and its output (cb_tilize, which feeds the transpose)
        // in full Float32 on the unpack-to-dest path; otherwise the unpacker falls back to tf32.
        compute_cfg.unpack_to_dest_mode = {
            {CB_IN, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32},
            {CB_TILIZE, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32}};
    }

    // Output binding: ht<=8 -> compute self-loops the borrowed output shard directly; ht>8 -> compute
    // is PRODUCER-only of the staging DFB (the compute kernel's producer-side wait_front does not make
    // it a consumer; the writer is the real consumer).
    std::vector<DFBBinding> compute_bindings = {
        DFBBinding{.dfb_spec_name = CB_IN, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = CB_TILIZE, .accessor_name = "cb_tilize", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = CB_TILIZE, .accessor_name = "cb_tilize", .endpoint_type = DFBEndpointType::CONSUMER}};
    if (ht_gt_8) {
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = CB_OUT_STAGE, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER});
    } else {
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = CB_OUT0, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = CB_OUT0, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::CONSUMER});
    }

    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/compute/"
                                        "transpose_wh_rm_sharded.cpp"},
        .dfb_bindings = std::move(compute_bindings),
        .compile_time_args =
            {{"Ht", ht},
             {"Wt", wt},
             {"HtWt", ht * wt},
             {"num_hw_blocks_per_core", num_hw_blocks_per_core},
             {"last_output_row_num_datums", last_output_row_num_datums},
             {"pack_num_pages_last_col", pack_num_pages_last_col},
             {"pack_num_pages_last_row_col", pack_num_pages_last_row_col}},
        .hw_config = compute_cfg,
    };

    std::vector<KernelSpec> kernels;
    std::vector<KernelSpecName> wu_kernels;
    kernels.push_back(std::move(reader_spec));
    kernels.push_back(std::move(compute_spec));
    wu_kernels.push_back(READER_KERNEL);
    wu_kernels.push_back(COMPUTE_KERNEL);

    if (ht_gt_8) {
        KernelSpec writer_spec{
            .unique_id = WRITER_KERNEL,
            .source =
                std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
                                      "writer_unary_transpose_wh_sharded_rm.cpp"},
            // cb_out_stage: real consumer of the compute staging output.
            // cb_out0: write-by-address borrowed shard; single-producer-single-consumer self-loop.
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = CB_OUT_STAGE,
                     .accessor_name = "cb_src",
                     .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = CB_OUT0, .accessor_name = "cb_dst", .endpoint_type = DFBEndpointType::PRODUCER},
                 DFBBinding{
                     .dfb_spec_name = CB_OUT0, .accessor_name = "cb_dst", .endpoint_type = DFBEndpointType::CONSUMER}},
            .compile_time_args =
                {{"num_hw_blocks_per_core", num_hw_blocks_per_core},
                 {"Ht", ht},
                 {"Wt", wt},
                 {"W_per_tile", W_per_tile},
                 {"W_per_tile_last", W_per_tile_last},
                 {"H_size_bytes", H * output_tensor.element_size()},
                 {"l1_read_offset_bytes", ht * output_tensor.element_size() * TILE_HEIGHT}},
            .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
        };
        kernels.push_back(std::move(writer_spec));
        wu_kernels.push_back(WRITER_KERNEL);
    }

    WorkUnitSpec wu{
        .name = "transpose_wh_sharded_rm",
        .kernels = std::move(wu_kernels),
        .target_nodes = all_cores,
    };

    ProgramSpec spec{
        .name = "transpose_wh_sharded_rm",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    // Sharded RM kernels carry no runtime args and no kernel-side TensorAccessor, so no
    // kernel_run_args entries are needed — only the two borrowed tensors flow through tensor_args.
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_tensor.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
