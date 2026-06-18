// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_rm_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <algorithm>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TransposeWHShardedRMProgramFactory::create_program_spec(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    uint32_t stick_size_bytes = W * input_tensor.element_size();
    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    uint32_t output_page_size, pack_num_pages, pack_num_pages_last_col, pack_num_pages_last_row,
        pack_num_pages_last_row_col;
    if ((W % TILE_WIDTH) != 0 and (H % TILE_HEIGHT) != 0) {
        output_page_size = (W % TILE_WIDTH) * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        auto output_page_size_last_col = TILE_WIDTH * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages_last_col = dst_single_tile_size / output_page_size_last_col;
        auto output_page_size_last_row = TILE_HEIGHT * (W % TILE_WIDTH) * output_tensor.element_size();
        pack_num_pages_last_row = dst_single_tile_size / output_page_size_last_row;
        pack_num_pages_last_row_col = 1;
    } else if ((W % TILE_WIDTH) != 0 and (H % TILE_HEIGHT) == 0) {
        output_page_size = (W % TILE_WIDTH) * (TILE_HEIGHT)*output_tensor.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        pack_num_pages_last_col = pack_num_pages;
        pack_num_pages_last_row = 1;
        pack_num_pages_last_row_col = 1;
    } else if ((W % TILE_WIDTH) == 0 and (H % TILE_HEIGHT) != 0) {
        output_page_size = (TILE_WIDTH) * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        pack_num_pages_last_col = 1;
        pack_num_pages_last_row = pack_num_pages;
        pack_num_pages_last_row_col = 1;
    } else {
        output_page_size = dst_single_tile_size;
        pack_num_pages = 1;
        pack_num_pages_last_col = 1;
        pack_num_pages_last_row = 1;
        pack_num_pages_last_row_col = 1;
    }

    log_debug(tt::LogOp, "output_page_size: {}", output_page_size);
    log_debug(tt::LogOp, "pack_num_pages: {}", pack_num_pages);
    log_debug(tt::LogOp, "pack_num_pages_last_col: {}", pack_num_pages_last_col);
    log_debug(tt::LogOp, "pack_num_pages_last_row: {}", pack_num_pages_last_row);
    log_debug(tt::LogOp, "pack_num_pages_last_row_col: {}", pack_num_pages_last_row_col);

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t num_hw_blocks_per_core = shard_height / H;

    log_debug(tt::LogOp, "shard_height: {}", shard_height);
    log_debug(tt::LogOp, "dst_single_tile_size: {}", dst_single_tile_size);

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;

    auto& all_cores = shard_spec.grid;
    [[maybe_unused]] uint32_t num_cores = shard_spec.num_cores();

    log_debug(tt::LogOp, "all_cores: {}", all_cores);
    log_debug(tt::LogOp, "num_cores: {}", num_cores);

    const bool out_stage = ht > 8;

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "transpose_wh_sharded_rm";

    // src0 (legacy c_0): borrowed input shard, read by base pointer only (reader self-loop).
    // out (legacy c_16): borrowed output shard. When Ht <= 8 the compute pack-untilizes directly into
    //   it (compute self-loop FIFO); when Ht > 8 the writer drains the staging DFB into it by base
    //   pointer (writer self-loop).
    // in_scratch (legacy c_24): allocated double-buffered intermediate; reader produces, compute
    //   consumes (tilize input).
    // tilize (legacy c_25): allocated; compute self-loop (tilize writes, transpose reads back).
    // out_stage (legacy c_27, only Ht > 8): allocated; compute produces (pack-untilize), writer consumes.
    // The legacy c_26 ("im2") DFB is dead — no kernel reads or writes it — so it is omitted (a Metal 2.0
    //   DFB requires >=1 producer and >=1 consumer).
    uint32_t num_in_tiles = wt * 2;  // double buffer
    uint32_t num_im_tiles = ht * wt;
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = stick_size_bytes,
            .num_entries = shard_height,
            .data_format_metadata = src0_cb_data_format,
            .borrowed_from = m2::TensorParamName{"input"},
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"out"},
            .entry_size = output_page_size,
            .num_entries = shard_height,
            .data_format_metadata = dst_cb_data_format,
            .borrowed_from = m2::TensorParamName{"output"},
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"in_scratch"},
            .entry_size = src0_single_tile_size,
            .num_entries = num_in_tiles,
            .data_format_metadata = src0_cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"tilize"},
            .entry_size = src0_single_tile_size,
            .num_entries = num_im_tiles,
            .data_format_metadata = src0_cb_data_format,
        },
    };
    if (out_stage) {
        uint32_t num_out_tiles = ht * 2;  // double buffer
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"out_stage"},
            .entry_size = dst_single_tile_size,
            .num_entries = num_out_tiles,
            .data_format_metadata = dst_cb_data_format,
        });
    }

    // ---- Reader (in place) ----
    // src0 read by base pointer only -> self-loop. in_scratch produced for the compute.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                        "reader_unary_transpose_wh_sharded_rm.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "src0",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "src0",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER},
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"in_scratch"},
                    .accessor_name = "in_scratch",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {
                {"num_hw_blocks_per_core", num_hw_blocks_per_core},
                {"Ht", ht},
                {"H_per_tile", H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT},
                {"H_per_tile_last", H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT},
                {"Wt", wt},
                {"W_size_bytes", stick_size_bytes},
                {"l1_write_offset_bytes", wt * input_tensor.element_size() * TILE_WIDTH},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    // ---- Compute (reuses transpose_wh_rm.cpp SHARDED branch) ----
    m2::ComputeHardwareConfig compute_hw{.fp32_dest_acc_en = fp32_dest_acc_en};
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        // Keep both the tilize input (in_scratch / c_24) and its output (tilize / c_25, which feeds the
        // transpose) in full Float32 on the unpack-to-dest path; otherwise the unpacker falls back to
        // tf32 and drops the low mantissa bits.
        compute_hw.unpack_to_dest_mode.insert(
            {m2::DFBSpecName{"in_scratch"}, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32});
        compute_hw.unpack_to_dest_mode.insert(
            {m2::DFBSpecName{"tilize"}, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32});
    }

    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/"
                                        "transpose_wh_rm.cpp"},
        .compile_time_args =
            {
                {"Ht", ht},
                {"Wt", wt},
                {"HtWt", ht * wt},
                {"num_hw_blocks_per_core", num_hw_blocks_per_core},
                {"last_output_row_num_datums", H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT},
                {"pack_num_pages", pack_num_pages},
                {"pack_num_pages_last_col", pack_num_pages_last_col},
                {"pack_num_pages_last_row", pack_num_pages_last_row},
                {"pack_num_pages_last_row_col", pack_num_pages_last_row_col},
            },
        .hw_config = compute_hw,
    };
    // in_scratch consumer (tilize input), tilize self-loop, and output PRODUCER+CONSUMER (pack-untilize
    // both reserves/pushes and — on the non-narrow path — waits on the output FIFO).
    compute.dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in_scratch"},
            .accessor_name = "in_scratch",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"tilize"},
            .accessor_name = "tilize",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"tilize"},
            .accessor_name = "tilize",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
    };
    compute.compiler_options.defines = {{"SHARDED", "1"}};
    // DST_ACCUM_MODE is derived from ComputeHardwareConfig.fp32_dest_acc_en (genfiles), matching the
    // legacy descriptor path: int types get fp32_dest_acc_en=false -> DST_ACCUM_MODE=0 (max_bct=8). Do
    // NOT force it to 1 for int here (that would halve max_bct to 4 — a behavior change vs the original).
    if (out_stage) {
        // compute pack-untilizes into the staging DFB (producer; consumer too on the non-narrow path).
        compute.compiler_options.defines.insert({"OUT_STAGE", "1"});
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"out_stage"},
            .accessor_name = "out_stage",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"out_stage"},
            .accessor_name = "out_stage",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
    } else {
        // compute pack-untilizes directly into the borrowed output (producer + self-consumer).
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"out"},
            .accessor_name = "out",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"out"},
            .accessor_name = "out",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }

    // ---- Writer (in place, only when Ht > 8) ----
    m2::KernelSpec writer;
    if (out_stage) {
        writer = m2::KernelSpec{
            .unique_id = m2::KernelSpecName{"writer"},
            .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                            "writer_unary_transpose_wh_sharded_rm.cpp"},
            .compile_time_args =
                {
                    {"num_hw_blocks_per_core", num_hw_blocks_per_core},
                    {"Ht", ht},
                    {"Wt", wt},
                    {"W_per_tile", W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH},
                    {"W_per_tile_last", W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH},
                    {"H_size_bytes", H * output_tensor.element_size()},
                    {"l1_read_offset_bytes", ht * output_tensor.element_size() * TILE_HEIGHT},
                },
            .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
        };
        // writer drains out_stage (consumer) and writes the borrowed output by base pointer (self-loop).
        writer.dfb_bindings = {
            m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{"out_stage"},
                .accessor_name = "out_stage",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{"out"},
                .accessor_name = "out",
                .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{"out"},
                .accessor_name = "out",
                .endpoint_type = m2::DFBEndpointType::CONSUMER},
        };
    }

    std::vector<m2::KernelSpecName> wu_kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"compute"}};
    if (out_stage) {
        spec.kernels = {reader, compute, writer};
        wu_kernels.push_back(m2::KernelSpecName{"writer"});
    } else {
        spec.kernels = {reader, compute};
    }
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output_tensor.tensor_spec()},
    };
    // The DFBs link reader->compute (in_scratch) and compute->writer (out_stage), so all kernels share
    // one WorkUnitSpec on all_cores (Local-DFB rule).
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "transpose_wh_sharded_rm",
            .kernels = std::move(wu_kernels),
            .target_nodes = all_cores,
        },
    };

    // ---- ProgramRunArgs (mutable) ----
    // No per-kernel runtime args: every count is a compile-time arg in this factory (the legacy factory
    // emitted no runtime_args at all).
    m2::ProgramRunArgs run;
    run.tensor_args = {
        {m2::TensorParamName{"input"}, input_tensor.mesh_tensor()},
        {m2::TensorParamName{"output"}, output_tensor.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
