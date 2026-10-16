// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_rm_program_factory.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts TransposeWHShardedRMProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Declared function-locally: this op's factories share one translation unit in the unity
    // build, so file-scope names would collide across them.
    const DFBSpecName IN{"in"};
    const DFBSpecName TILIZE{"tilize"};
    const DFBSpecName OUT_STAGE{"out_stage"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

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

    // pack_num_pages and pack_num_pages_last_row were never read by the sharded compute path.
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

    log_debug(tt::LogOp, "output_page_size: {}", output_page_size);
    log_debug(tt::LogOp, "pack_num_pages_last_col: {}", pack_num_pages_last_col);
    log_debug(tt::LogOp, "pack_num_pages_last_row_col: {}", pack_num_pages_last_row_col);

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t num_hw_blocks_per_core = shard_height / H;

    log_debug(tt::LogOp, "shard_height: {}", shard_height);
    log_debug(tt::LogOp, "dst_single_tile_size: {}", dst_single_tile_size);

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;

    auto& all_cores = shard_spec.grid;

    log_debug(tt::LogOp, "all_cores: {}", all_cores);
    log_debug(tt::LogOp, "num_cores: {}", shard_spec.num_cores());

    // Quasar has no narrow-row pack, so it always stages: compute packs whole tiles and the writer copies
    // each output row out of the padded tile rows.
    const bool stage_output = ht > 8 || input_tensor.device()->arch() == tt::ARCH::QUASAR;

    std::vector<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = IN,
        .entry_size = src0_single_tile_size,
        .num_entries = wt * 2,  // double buffer
        .data_format_metadata = src0_cb_data_format,
    });
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = TILIZE,
        .entry_size = src0_single_tile_size,
        .num_entries = ht * wt,
        .data_format_metadata = src0_cb_data_format,
    });
    if (stage_output) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = OUT_STAGE,
            .entry_size = dst_single_tile_size,
            .num_entries = ht * 2,  // double buffer
            .data_format_metadata = dst_cb_data_format,
        });
    } else {
        // Compute packs straight into the output shard.
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = OUT,
            .entry_size = output_page_size,
            .num_entries = (stick_size_bytes * shard_height) / output_page_size,
            .data_format_metadata = dst_cb_data_format,
            .borrowed_from = OUTPUT,
        });
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "reader_unary_transpose_wh_sharded_rm.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN, .accessor_name = "cb_dst", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"num_hw_blocks_per_core", num_hw_blocks_per_core},
             {"Ht", ht},
             {"H_per_tile", H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT},
             {"H_per_tile_last", H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT},
             {"Wt", wt},
             {"W_size_bytes", stick_size_bytes},
             {"l1_write_offset_bytes", wt * input_tensor.element_size() * TILE_WIDTH}},
        .hw_config = create_reader_datamovement_config(),
    };

    // Legacy built a ComputeConfigDescriptor directly, setting only fp32_dest_acc_en and
    // unpack_to_dest_mode; ComputeHardwareConfig's defaults reproduce the rest.
    ComputeHardwareConfig compute_hw{.enable_32_bit_dest = fp32_dest_acc_en};
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        // Keep both the tilize input and its output (which feeds the transpose) in full Float32
        // on the unpack-to-dest path; otherwise the unpacker falls back to tf32.
        compute_hw.unpack_modes.insert({IN, UnpackMode::UnpackToDest});
        compute_hw.unpack_modes.insert({TILIZE, UnpackMode::UnpackToDest});
    }

    std::vector<DFBBinding> compute_bindings = {
        DFBBinding{.dfb_spec_name = IN, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = TILIZE, .accessor_name = "cb_tilize", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = TILIZE, .accessor_name = "cb_tilize", .endpoint_type = DFBEndpointType::CONSUMER}};
    if (stage_output) {
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUT_STAGE, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER});
    } else {
        // The output stays resident in the shard, so compute is both ends of the borrowed DFB.
        compute_bindings.push_back(
            DFBBinding{.dfb_spec_name = OUT, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER});
        compute_bindings.push_back(
            DFBBinding{.dfb_spec_name = OUT, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::CONSUMER});
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm_sharded.cpp",
        // Legacy left opt_level unset on a ComputeConfigDescriptor, which resolves to O3;
        // a Metal 2.0 KernelSpec defaults to O2, so the level is restated explicitly here.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_bindings),
        .compile_time_args =
            {{"Ht", ht},
             {"Wt", wt},
             {"HtWt", ht * wt},
             {"num_hw_blocks_per_core", num_hw_blocks_per_core},
             {"last_output_row_num_datums", H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT},
             {"pack_num_pages_last_col", pack_num_pages_last_col},
             {"pack_num_pages_last_row_col", pack_num_pages_last_row_col}},
        .hw_config = compute_hw,
    };

    std::vector<KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(compute));
    std::vector<KernelSpecName> wu_kernels = {READER, COMPUTE};

    if (stage_output) {
        KernelSpec writer{
            .unique_id = WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                "writer_unary_transpose_wh_sharded_rm.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = OUT_STAGE, .accessor_name = "cb_src", .endpoint_type = DFBEndpointType::CONSUMER}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
            .compile_time_args =
                {{"num_hw_blocks_per_core", num_hw_blocks_per_core},
                 {"Ht", ht},
                 {"Wt", wt},
                 {"W_per_tile", W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH},
                 {"W_per_tile_last", W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH},
                 {"H_size_bytes", H * output_tensor.element_size()},
                 {"l1_read_offset_bytes", ht * output_tensor.element_size() * TILE_HEIGHT}},
            .hw_config = create_writer_datamovement_config(),
        };
        kernels.push_back(std::move(writer));
        wu_kernels.push_back(WRITER);
    }

    ProgramSpec spec{
        .name = "transpose_wh_sharded_rm",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()},
             TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()}},
        .work_units = {WorkUnitSpec{
            .name = "transpose_wh_sharded_rm", .kernels = std::move(wu_kernels), .target_nodes = all_cores}},
    };

    // All kernels take compile-time args only; the two shard buffers are the whole per-dispatch state.
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {INPUT, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {OUTPUT, TensorArgument{std::cref(output_tensor.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
