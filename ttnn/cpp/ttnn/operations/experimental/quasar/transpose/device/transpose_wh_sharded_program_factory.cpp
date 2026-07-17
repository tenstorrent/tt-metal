// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <algorithm>
#include <vector>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts TransposeWHShardedProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN0{"cb_in0"};    // legacy c_0: input shard (borrowed)
    const DFBSpecName CB_OUT0{"cb_out0"};  // legacy c_16: output shard (borrowed)

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

    const auto tile = input_tensor.tensor_spec().tile();
    const uint32_t tile_hw = tile.get_tile_hw();

    const bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;

    const auto shard_spec = input_tensor.shard_spec().value();
    const bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    const auto& all_cores = shard_spec.grid;
    const uint32_t num_tiles_per_shard = shard_spec.numel() / tile_hw;

    // ------------------------------------------------------------------------
    // Borrowed-memory DFBs. The input/output shards already reside in L1, so instead of allocating
    // fresh DFB storage we build the DFBs directly on the input/output tensor shard buffers (legacy
    // CBDescriptor::buffer = input/output_tensor.buffer()). The shard's L1 base address is uniform
    // across the shard grid and its per-core size equals entry_size * num_entries.
    // ------------------------------------------------------------------------
    std::vector<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = CB_IN0,
        .entry_size = src0_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = src0_cb_data_format,
        .borrowed_from = INPUT_TENSOR,
    });
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = CB_OUT0,
        .entry_size = dst_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = dst_cb_data_format,
        .borrowed_from = OUTPUT_TENSOR,
    });

    // ------------------------------------------------------------------------
    // Tensor parameters. Each is used only as a DFB borrowed_from backing store (no kernel-side
    // TensorAccessor on the sharded path), which the framework counts as a legitimate use.
    // ------------------------------------------------------------------------
    TensorParameter input_param{
        .unique_id = INPUT_TENSOR,
        .spec = input_tensor.tensor_spec(),
    };
    TensorParameter output_param{
        .unique_id = OUTPUT_TENSOR,
        .spec = output_tensor.tensor_spec(),
    };

    // ------------------------------------------------------------------------
    // Per-shard work geometry (identical for every active shard core).
    // ------------------------------------------------------------------------
    const auto padded_shape = input_tensor.padded_shape();
    const auto shard_shape = shard_spec.shape;

    const uint32_t H = padded_shape[2];
    const uint32_t Hs = shard_shape[0], Ws = shard_shape[1];

    const uint32_t Hts = Hs / tile.get_height();
    const uint32_t Wts = Ws / tile.get_width();

    const uint32_t Ht = H / tile.get_height();
    const uint32_t Ht_per_shard = std::min(Ht, Hts);

    const uint32_t num_hw_blocks_per_shard = Hts > Ht ? Hts / Ht : 1;

    const uint32_t HtWt_tile_size = Ht_per_shard * Wts;
    const uint32_t num_blocks = num_hw_blocks_per_shard * HtWt_tile_size;

    // ------------------------------------------------------------------------
    // Kernels. Reader/writer are trivial sharded stubs (push_back / wait_front on the borrowed DFB);
    // compute transposes each tile within the resident shard.
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source =
            std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
                                  "reader_unary_sharded.cpp"},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = ttnn::create_reader_datamovement_config(input_tensor.device()->arch()),
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source =
            std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
                                  "writer_unary_sharded.cpp"},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_OUT0, .accessor_name = "cb_out0", .endpoint_type = DFBEndpointType::CONSUMER}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(input_tensor.device()->arch()),
    };

    ttnn::ComputeKernelConfig compute_cfg{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(input_tensor.device()->arch(), compute_cfg);
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        std::visit(
            [&](auto& c) { c.unpack_modes.emplace(CB_IN0, tt::tt_metal::UnpackMode::UnpackToDest); }, compute_hw);
    }

    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/compute/"
                                        "transpose_wh_sharded.cpp"},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = CB_OUT0, .accessor_name = "cb_out0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"NHtWt", "HtWt", "N", "Ht", "Wt"}},
        .hw_config = compute_hw,
    };

    std::vector<KernelSpec> kernels = {std::move(reader_spec), std::move(writer_spec), std::move(compute_spec)};

    // ------------------------------------------------------------------------
    // Runtime args. Every active shard core does identical work, so the values are uniform. Work is
    // placed on exactly the shard grid (all_cores) — there are no legacy no-op cores here because a
    // borrowed-memory DFB may only be placed where its backing shard buffer is actually allocated.
    // ------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    KernelRunArgs compute_run{.kernel = COMPUTE_KERNEL};

    const std::vector<NodeCoord> cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (const NodeCoord& node : cores) {
        reader_run.runtime_arg_values["num_tiles"][node] = num_blocks;
        writer_run.runtime_arg_values["num_units"][node] = num_blocks;
        KernelRunArgs::RuntimeArgValues& compute_rtas = compute_run.runtime_arg_values;
        AddRuntimeArgsForNode(
            compute_rtas,
            node,
            {
                {"NHtWt", num_blocks},
                {"HtWt", HtWt_tile_size},
                {"N", num_hw_blocks_per_shard},
                {"Ht", Ht_per_shard},
                {"Wt", Wts},
            });
    }

    WorkUnitSpec wu{
        .name = "transpose_wh_sharded",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = all_cores,
    };

    ProgramSpec spec{
        .name = "transpose_wh_sharded",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run), std::move(compute_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_tensor.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
