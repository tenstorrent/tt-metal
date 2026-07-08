// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) program factory for binary_ng.
//
// This is the generic Metal 2.0 translation of binary_ng's ProgramDescriptor factory
// (binary_ng_program_factory.cpp). It is invoked for the slice that
// BinaryNgDeviceOperation::matches_metal_v2_slice() admits (the default path for that slice; the DFB
// path is arch-portable — CB-backed on Wormhole/Blackhole, overlay-backed on Quasar):
//
//   SHARDED no-broadcast ADD (the ResNet50 residual config: bf8/bf16, height/block-sharded L1,
//   in-place, optional fused RELU). This is the faithful translation of the descriptor factory's
//   worker-grid split: each input/output shard is borrowed by a DFB (DataflowBufferSpec::
//   borrowed_from -> the tensor's TensorParameter; the DFB analog of the CB factory's
//   `.buffer = tensor.buffer()` global allocation). Reader/writer do no NoC work for resident
//   shards; the compute kernel runs in local L1.

#include "binary_ng_device_operation.hpp"
#include "binary_ng_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::operations::experimental::quasar::binary_ng {

namespace {

using ttnn::device_operation::ProgramArtifacts;

// Kernel sources for the sharded no-broadcast DFB path.
constexpr const char* kShardedReaderDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_dfb/dataflow/"
    "reader_sharded_no_bcast_dfb.cpp";
constexpr const char* kShardedWriterDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_dfb/dataflow/"
    "writer_sharded_no_bcast_dfb.cpp";
constexpr const char* kShardedComputeDfb =
    "ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/device/kernels_dfb/compute/eltwise_binary_no_bcast_dfb.cpp";

// Tile count for one full shard: (tile-rounded) shard height-in-tiles * width-in-tiles. Every core
// processes this same count, including a partial end core under an uneven shard. That is correct
// because a sharded buffer allocates the full rounded-up shard on every core: the trailing tiles on a
// partial core are allocated L1 with no host page mapped, so computing the add on them is in-bounds and
// never reaches the logical output. The selector pins equal shard shapes across a, b, and the output
// memory_config (and the TT_FATAL below cross-checks the per-tensor counts), so the three tensors
// over-allocate identically. This matches the descriptor factory, which also uses the full shard tile
// count when the a/b/c shard specs are equal.
uint32_t full_shard_tiles(const Tensor& tensor, const ShardSpec& shard_spec) {
    const uint32_t tile_h = tensor.tensor_spec().tile().get_height();
    const uint32_t tile_w = tensor.tensor_spec().tile().get_width();
    const uint32_t shard_ht = tt::round_up(shard_spec.shape[0], tile_h) / tile_h;
    const uint32_t shard_wt = tt::round_up(shard_spec.shape[1], tile_w) / tile_w;
    return shard_ht * shard_wt;
}

ProgramArtifacts create_sharded_artifacts(
    const BinaryNgDeviceOperation::operation_attributes_t& op,
    const BinaryNgDeviceOperation::tensor_args_t& tensor_args,
    Tensor& c) {
    const Tensor& a = tensor_args.input_tensor_a;
    const Tensor& b = *tensor_args.input_tensor_b;

    TT_FATAL(
        a.is_sharded() && b.is_sharded() && c.is_sharded(),
        "binary_ng Metal 2.0 sharded factory requires a, b, and output to all be sharded");

    Buffer* a_buffer = a.buffer();
    Buffer* b_buffer = b.buffer();
    Buffer* c_buffer = c.buffer();
    TT_FATAL(
        a_buffer != nullptr && b_buffer != nullptr && c_buffer != nullptr,
        "binary_ng Metal 2.0 sharded factory requires allocated device buffers");

    // Data formats / tile sizes per tensor (bf8 and bf16 supported).
    const tt::DataFormat a_df = datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat b_df = datatype_to_dataformat_converter(b.dtype());
    const tt::DataFormat c_df = datatype_to_dataformat_converter(c.dtype());
    const tt::tt_metal::Tile a_tile = a.tensor_spec().tile();
    const tt::tt_metal::Tile b_tile = b.tensor_spec().tile();
    const tt::tt_metal::Tile c_tile = c.tensor_spec().tile();
    const uint32_t a_tile_bytes = static_cast<uint32_t>(a_tile.get_tile_size(a_df));
    const uint32_t b_tile_bytes = static_cast<uint32_t>(b_tile.get_tile_size(b_df));
    const uint32_t c_tile_bytes = static_cast<uint32_t>(c_tile.get_tile_size(c_df));

    const ShardSpec a_shard = *a.shard_spec();
    const ShardSpec b_shard = *b.shard_spec();
    const ShardSpec c_shard = *c.shard_spec();

    // Place work on exactly the shard grid -- the cores that physically own a shard -- NOT
    // op.worker_grid. get_worker_grid() returns the full sub-device worker set for a sharded output (a
    // superset of the shard grid); the descriptor factory only tolerates that by sending dummy zero
    // runtime args to non-shard cores. This DFB factory places a kernel on every target core, so a
    // non-shard core would run a compute that writes a full shard into a borrowed L1 address the tensor
    // does not own there. The selector (matches_metal_v2_slice) pins a_shard.grid == b_shard.grid and
    // a_shard.grid == the output memory_config's shard grid, and for in-place add_ the output aliases a,
    // so a_shard.grid is the common core set for a, b, and the output. The per-tensor full_shard_tiles
    // equality TT_FATAL below additionally guards an explicit output tensor whose shard shape diverges.
    const CoreRangeSet& grid = a_shard.grid;
    const auto cores = corerange_to_cores(grid, std::nullopt, /*row_wise=*/true);
    TT_FATAL(!cores.empty(), "binary_ng Metal 2.0 sharded factory: empty shard grid");

    // -----------------------------------------------------------------------------------------
    // Tensor parameters: declare a, b, c with their exact specs (ValidateTensorArgs requires full
    // TensorSpec equality). For in-place add_ (a aliases c) the adapter still binds both names to the
    // same MeshTensor; we declare distinct parameter names so each DFB borrows independently.
    // -----------------------------------------------------------------------------------------
    const m2::TensorParamName T_A{"binary_ng_a"};
    const m2::TensorParamName T_B{"binary_ng_b"};
    const m2::TensorParamName T_C{"binary_ng_c"};

    const m2::DFBSpecName IN0{"binary_ng_in0_dfb"};
    const m2::DFBSpecName IN1{"binary_ng_in1_dfb"};
    const m2::DFBSpecName OUT{"binary_ng_out_dfb"};
    const m2::KernelSpecName READER{"binary_ng_sharded_reader"};
    const m2::KernelSpecName WRITER{"binary_ng_sharded_writer"};
    const m2::KernelSpecName COMPUTE{"binary_ng_sharded_compute"};

    // DFB num_entries == one full shard so the borrowed ring spans the resident shard. The selector
    // guarantees a/b/c share one shard spec, so each tensor's full shard tile count is uniform across
    // the grid. entry_size == that tensor's tile bytes.
    const uint32_t a_shard_tiles = full_shard_tiles(a, a_shard);
    const uint32_t b_shard_tiles = full_shard_tiles(b, b_shard);
    const uint32_t c_shard_tiles = full_shard_tiles(c, c_shard);
    TT_FATAL(
        a_shard_tiles == b_shard_tiles && a_shard_tiles == c_shard_tiles,
        "binary_ng Metal 2.0 sharded factory: a/b/c shard tile counts must match (no-broadcast) but got {} {} {}",
        a_shard_tiles,
        b_shard_tiles,
        c_shard_tiles);
    const uint32_t shard_tiles = a_shard_tiles;

    // Tiles processed per DST register acquire. The compute engine's DST can hold a bounded number
    // of tiles per tile_regs_acquire/commit; the descriptor factory uses 8 for 16-bit FPU types (see
    // the num_tiles_per_cycle selection in binary_ng_program_factory.cpp). The compute kernel loops
    // over the shard in chunks of this many tiles, so a shard wider/taller than the DST capacity is
    // handled correctly. Must not exceed the shard tile count.
    const uint32_t num_tiles_per_cycle = std::min<uint32_t>(8u, shard_tiles);

    auto make_dfb = [](const m2::DFBSpecName& name,
                       const m2::TensorParamName& borrowed,
                       uint32_t entry_size,
                       uint32_t num_entries,
                       tt::DataFormat df,
                       const tt::tt_metal::Tile& tile) {
        return m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = df,
            .tile_format_metadata = tile,
            .borrowed_from = borrowed,
        };
    };
    m2::DataflowBufferSpec in0_dfb = make_dfb(IN0, T_A, a_tile_bytes, shard_tiles, a_df, a_tile);
    m2::DataflowBufferSpec in1_dfb = make_dfb(IN1, T_B, b_tile_bytes, shard_tiles, b_df, b_tile);
    m2::DataflowBufferSpec out_dfb = make_dfb(OUT, T_C, c_tile_bytes, shard_tiles, c_df, c_tile);

    // -----------------------------------------------------------------------------------------
    // Compute defines: ADD via the same OpConfig the descriptor factory uses, plus fused RELU /
    // post-activation chain. Mirrors binary_ng_program_factory.cpp's compute_kernel_defines.
    // -----------------------------------------------------------------------------------------
    const DataType input_dtype = op.input_dtype;
    OpConfig op_config(op.binary_op_type, std::in_place_type<OpConfig::FpuBinaryOp>, input_dtype);
    std::map<std::string, std::string> compute_defines = op_config.as_defines(input_dtype);

    // The DFB compute kernel uses ELTWISE_OP (the `_2_0` convention) rather than BINARY_OP; alias it
    // to whatever OpConfig produced for the FPU binary op.
    {
        auto it = compute_defines.find("BINARY_OP");
        if (it != compute_defines.end()) {
            compute_defines["ELTWISE_OP"] = it->second;
        }
        auto it_type = compute_defines.find("BINARY_OP_TYPE");
        if (it_type != compute_defines.end()) {
            compute_defines["ELTWISE_OP_TYPE"] = it_type->second;
        }
    }

    // Fused post-activations (RELU fast path). Same logic as the descriptor factory: a single RELU
    // post-activation with no broadcast uses the packer (PACK_RELU); other chains expand to SFPU.
    const auto& post_acts = op.post_activations;
    if (op.lhs_activations.empty() && op.rhs_activations.empty() && post_acts.size() == 1 &&
        post_acts[0].type() == unary::UnaryOpType::RELU) {
        compute_defines["PROCESS_POST_ACTIVATIONS(i)"] = "";
        compute_defines["PACK_RELU"] = "1";
        ttnn::operations::unary::utils::update_macro_defines(
            ttnn::operations::unary::UnaryOpType::RELU, compute_defines);
    } else if (!post_acts.empty()) {
        add_activation_defines(compute_defines, post_acts, "POST", input_dtype);
    }

    m2::KernelSpec::CompilerOptions::Defines compute_defines_tbl;
    for (const auto& [k, v] : compute_defines) {
        compute_defines_tbl.emplace(k, v);
    }

    // -----------------------------------------------------------------------------------------
    // Kernels.
    // -----------------------------------------------------------------------------------------
    // Reader/writer take NO tensor bindings: the input/output shards are borrowed by the DFBs
    // (DataflowBufferSpec::borrowed_from already registers each TensorParameter as used), and the
    // kernels do no NoC work — they only publish/drain the resident shards. So no TensorAccessor.
    m2::KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(kShardedReaderDfb),
        .num_threads = 1,
        .dfb_bindings = {m2::ProducerOf(IN0, "in0"), m2::ProducerOf(IN1, "in1")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    };

    m2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path(kShardedWriterDfb),
        .num_threads = 1,
        .dfb_bindings = {m2::ConsumerOf(OUT, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    };

    // bf8/bf16 outputs do not need fp32 dest accumulation (matches descriptor factory: false unless
    // an operand/output is fp32/int32/uint32, which this slice excludes).
    const bool fp32_dest_acc_en = false;

    m2::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(kShardedComputeDfb),
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines_tbl},
        .dfb_bindings = {m2::ConsumerOf(IN0, "in0"), m2::ConsumerOf(IN1, "in1"), m2::ProducerOf(OUT, "out")},
        .compile_time_args = {{"num_tiles_per_cycle", num_tiles_per_cycle}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
        .hw_config = ttnn::to_compute_hardware_config(
            a.device()->arch(),
            ttnn::ComputeKernelConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .math_approx_mode = false,
                .fp32_dest_acc_en = fp32_dest_acc_en,
            }),
    };

    // -----------------------------------------------------------------------------------------
    // Work unit over the shard grid.
    // -----------------------------------------------------------------------------------------
    std::set<CoreRange> target_ranges;
    for (const auto& core : cores) {
        target_ranges.insert(CoreRange(core, core));
    }
    m2::NodeRangeSet target_nodes(target_ranges);

    m2::WorkUnitSpec wu{
        .name = "binary_ng_metal_v2_sharded",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = target_nodes,
    };

    m2::ProgramSpec spec{
        .name = "binary_ng_metal_v2_sharded_add",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in0_dfb, in1_dfb, out_dfb},
        .tensor_parameters =
            {{.unique_id = T_A, .spec = a.tensor_spec()},
             {.unique_id = T_B, .spec = b.tensor_spec()},
             {.unique_id = T_C, .spec = c.tensor_spec()}},
        .work_units = {wu},
    };

    // -----------------------------------------------------------------------------------------
    // Per-core runtime args (num_tiles per shard) + tensor bindings.
    // -----------------------------------------------------------------------------------------
    m2::Group<m2::ProgramRunArgs::KernelRunArgs::NodeRuntimeArgs> reader_args, writer_args, compute_args;
    reader_args.reserve(cores.size());
    writer_args.reserve(cores.size());
    compute_args.reserve(cores.size());
    for (const auto& core : cores) {
        const m2::NodeCoord node{static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)};
        const uint32_t n = c_shard_tiles;
        reader_args.push_back({node, {{"num_tiles", n}}});
        writer_args.push_back({node, {{"num_tiles", n}}});
        compute_args.push_back({node, {{"num_tiles", n}}});
    }

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_args)},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_args)},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE, .runtime_arg_values = std::move(compute_args)},
    };
    // Bind each TensorParameter to its MeshTensor. The adapter resolves these against the io tensors
    // (a, b, c) by pointer identity. For in-place add_ a and c are the same MeshTensor — both bindings
    // resolve to it, and both DFBs borrow the same shard (aliased borrow; permitted).
    run_params.tensor_args.emplace(T_A, m2::ProgramRunArgs::TensorArgument{a.mesh_tensor()});
    run_params.tensor_args.emplace(T_B, m2::ProgramRunArgs::TensorArgument{b.mesh_tensor()});
    run_params.tensor_args.emplace(T_C, m2::ProgramRunArgs::TensorArgument{c.mesh_tensor()});

    return ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
        .op_owned_tensors = {},
    };
}

}  // namespace

ttnn::device_operation::ProgramArtifacts BinaryNgDeviceOperation::ProgramFactoryMetalV2::create_program_artifacts(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, tensor_return_value_t& c) {
    // matches_metal_v2_slice() admits only fully-sharded configs, so all tensors are L1 shards here.
    return create_sharded_artifacts(operation_attributes, tensor_args, c);
}

}  // namespace ttnn::operations::experimental::quasar::binary_ng
