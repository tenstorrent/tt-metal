// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

const DFBSpecName WHS_SRC0_DFB{"whs_src0"};  // c_0 (borrowed from input)
const DFBSpecName WHS_OUT_DFB{"whs_out"};    // c_16 (borrowed from output)
const TensorParamName WHS_INPUT{"whs_input"};
const TensorParamName WHS_OUTPUT{"whs_output"};
const KernelSpecName WHS_READER{"whs_reader"};
const KernelSpecName WHS_COMPUTE{"whs_compute"};
const KernelSpecName WHS_WRITER{"whs_writer"};

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeWHShardedProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const auto tile = input_tensor.tensor_spec().tile();
    const uint32_t tile_hw = tile.get_tile_hw();

    IDevice* device = input_tensor.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto shard_spec = input_tensor.shard_spec().value();
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    auto& all_cores = shard_spec.grid;
    uint32_t num_tiles_per_shard = shard_spec.numel() / tile_hw;

    // ---- ProgramSpec ----
    ProgramSpec spec;
    spec.name = "transpose_wh_sharded";

    spec.tensor_parameters = {
        TensorParameter{.unique_id = WHS_INPUT, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = WHS_OUTPUT, .spec = output_tensor.tensor_spec()},
    };

    // Sharded CBs become borrowed-memory DFBs: they draw their backing L1 address from the
    // corresponding tensor_arg at runtime (the Metal 2.0 form of the legacy .buffer =
    // UpdateDynamicCircularBufferAddress re-application on cache hit).
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WHS_SRC0_DFB,
        .entry_size = src0_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = src0_cb_data_format,
        .borrowed_from = WHS_INPUT,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WHS_OUT_DFB,
        .entry_size = dst_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = dst_cb_data_format,
        .borrowed_from = WHS_OUTPUT,
    });

    ComputeGen1Config compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        compute_cfg.unpack_modes.emplace(WHS_SRC0_DFB, UnpackMode::UnpackToDest);
    }

    // Borrowed donor kernels: each is forked beside its own legacy original (eltwise/unary and
    // data_movement/sharded respectively), since the legacy sources still serve many other ops.
    KernelSpec reader{
        .unique_id = WHS_READER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "reader_unary_sharded_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = WHS_SRC0_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = WHS_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "writer_unary_sharded_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = WHS_OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    KernelSpec compute{
        .unique_id = WHS_COMPUTE,
        // Lent kernel: transpose_wh_sharded.cpp is cross-op shared with legacy peers
        // (create_qkv_heads, create_qkv_heads_from_separate_tensors,
        // split_query_key_value_and_split_heads_sharded), so the legacy source must stay
        // non-Metal-2.0 for them; this factory binds the Metal 2.0 fork beside it.
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/"
            "transpose_wh_sharded_metal2.cpp",
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = WHS_SRC0_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = WHS_OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"NHtWt", "HtWt", "N", "Ht", "Wt"}},
        .hw_config = ComputeHardwareConfig{compute_cfg},
    };

    spec.kernels = {reader, writer, compute};
    spec.work_units = {
        WorkUnitSpec{.name = "main", .kernels = {WHS_READER, WHS_WRITER, WHS_COMPUTE}, .target_nodes = total_cores}};

    // ---- work distribution ----
    auto padded_shape = input_tensor.padded_shape();
    auto shard_shape = shard_spec.shape;

    uint32_t H = padded_shape[2];
    uint32_t Hs = shard_shape[0], Ws = shard_shape[1];

    uint32_t Hts = Hs / tile.get_height();
    uint32_t Wts = Ws / tile.get_width();

    uint32_t Ht = H / tile.get_height();
    uint32_t Ht_per_shard = std::min(Ht, Hts);

    uint32_t num_hw_blocks_per_shard = Hts > Ht ? Hts / Ht : 1;

    uint32_t HtWt_tile_size = Ht_per_shard * Wts;
    uint32_t num_blocks = num_hw_blocks_per_shard * HtWt_tile_size;

    auto bbox = all_cores.bounding_box();
    std::vector<CoreCoord> cores =
        grid_to_cores_with_noop(bbox.end_coord.x, bbox.end_coord.y, num_cores_x, num_cores_y, row_major);

    const uint32_t num_active = all_cores.num_cores();

    // ---- ProgramRunArgs ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = WHS_READER};
    KernelRunArgs compute_run{.kernel = WHS_COMPUTE};
    KernelRunArgs writer_run{.kernel = WHS_WRITER};

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        if (i < num_active) {
            AddRuntimeArgsForNode(reader_run.runtime_arg_values, core, {{"num_tiles", num_blocks}});
            AddRuntimeArgsForNode(
                compute_run.runtime_arg_values,
                core,
                {{"NHtWt", num_blocks},
                 {"HtWt", HtWt_tile_size},
                 {"N", num_hw_blocks_per_shard},
                 {"Ht", Ht_per_shard},
                 {"Wt", Wts}});
            AddRuntimeArgsForNode(writer_run.runtime_arg_values, core, {{"num_units", num_blocks}});
        } else {
            // No-op tail core: zero-filled, matching legacy std::vector<uint32_t>(N) rows.
            AddRuntimeArgsForNode(reader_run.runtime_arg_values, core, {{"num_tiles", 0u}});
            AddRuntimeArgsForNode(
                compute_run.runtime_arg_values, core, {{"NHtWt", 0u}, {"HtWt", 0u}, {"N", 0u}, {"Ht", 0u}, {"Wt", 0u}});
            AddRuntimeArgsForNode(writer_run.runtime_arg_values, core, {{"num_units", 0u}});
        }
    }

    run_args.kernel_run_args = {reader_run, writer_run, compute_run};
    run_args.tensor_args.emplace(WHS_INPUT, TensorArgument{input_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(WHS_OUTPUT, TensorArgument{output_tensor.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
