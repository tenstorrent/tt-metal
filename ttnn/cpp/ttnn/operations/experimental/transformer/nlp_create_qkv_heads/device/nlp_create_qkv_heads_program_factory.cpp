// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "nlp_create_qkv_heads_device_operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::experimental::transformer {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// Single source of truth for the Interleaved factory's per-core work split.  create_program_artifacts()
// walks `cores` in this order when it emits the per-core runtime args.
struct InterleavedWorkSplit {
    std::vector<CoreCoord> cores;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_blocks_per_core_group_1 = 0;
    uint32_t num_blocks_per_core_group_2 = 0;
    // Q-only head creation with fewer sequence blocks than cores splits the work per (batch, head) instead:
    // a block is then one head row rather than one full input row.
    bool head_parallel = false;
    // Otherwise, without transpose_k_heads or the Q head split, a block is one tile of the flattened (tile row,
    // tile in row) space (*_tiles.cpp kernels) instead of a whole tile row, so shapes with few tile rows still fill
    // the grid.
    bool tile_split = false;
};

InterleavedWorkSplit build_interleaved_work_split(
    const NlpCreateHeadsDeviceOperation::operation_attributes_t& operation_attributes, const Tensor& input_tensor) {
    const auto& input_shape = input_tensor.padded_shape();
    const CoreCoord grid = input_tensor.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    const uint32_t sequence_blocks = input_shape[0] * input_shape[1] * input_shape[2] / TILE_HEIGHT;
    // Split heads only when the Q-only sequence split would leave cores idle.
    const bool head_parallel = operation_attributes.num_kv_heads == 0 && operation_attributes.num_q_heads > 1 &&
                               !operation_attributes.transpose_k_heads && sequence_blocks < grid.x * grid.y;
    const bool tile_split =
        !head_parallel && !operation_attributes.transpose_k_heads && !operation_attributes.q_head_split.has_value();
    const uint32_t row_tiles = (operation_attributes.num_q_heads + 2 * operation_attributes.num_kv_heads) *
                               (operation_attributes.head_dim / TILE_WIDTH);
    const uint32_t num_blocks = tile_split ? sequence_blocks * row_tiles
                                           : sequence_blocks * (head_parallel ? operation_attributes.num_q_heads : 1);
    auto [num_cores, all_cores, core_group_1, core_group_2, blocks_group_1, blocks_group_2] =
        tt::tt_metal::split_work_to_cores(grid, num_blocks);

    InterleavedWorkSplit split;
    split.head_parallel = head_parallel;
    split.tile_split = tile_split;
    split.all_cores = std::move(all_cores);
    split.core_group_1 = std::move(core_group_1);
    split.core_group_2 = std::move(core_group_2);
    split.num_blocks_per_core_group_1 = blocks_group_1;
    split.num_blocks_per_core_group_2 = blocks_group_2;
    split.cores.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        split.cores.push_back(CoreCoord{i / num_cores_y, i % num_cores_y});
    }
    return split;
}

// The five tensor arguments both factories bind: the Q input, the optional separate KV input and the three
// outputs.  The names must match the TensorParameter unique_ids the factories declare.  Shared by
// create_program_artifacts and override_runtime_arguments of both factories so a renamed or added tensor
// parameter cannot drift between them.
using TensorRunArgs = decltype(tt::tt_metal::experimental::ProgramRunArgs::tensor_args);
TensorRunArgs build_qkv_tensor_run_args(
    const NlpCreateHeadsDeviceOperation::tensor_args_t& tensor_args,
    NlpCreateHeadsDeviceOperation::tensor_return_value_t& output) {
    const TensorParamName INPUT_Q{"input_q"};
    const TensorParamName INPUT_KV{"input_kv"};
    const TensorParamName Q{"q"};
    const TensorParamName K{"k"};
    const TensorParamName V{"v"};

    TensorRunArgs tensor_run_args = {
        {INPUT_Q, tensor_args.input_tensor_q.mesh_tensor()},
        {Q, std::get<0>(output).mesh_tensor()},
        {K, std::get<1>(output).mesh_tensor()},
        {V, std::get<2>(output).mesh_tensor()},
    };
    if (tensor_args.input_tensor_kv.has_value()) {
        tensor_run_args.emplace(INPUT_KV, tensor_args.input_tensor_kv->mesh_tensor());
    }
    return tensor_run_args;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts NlpCreateHeadsDeviceOperation::Interleaved::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Spec resource names (function-local: the two factories share one unity translation unit).
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};
    const DFBSpecName QV{"qv"};        // Q and V head tiles, reader -> writer (K too unless transpose_k_heads)
    const DFBSpecName K_IN{"k_in"};    // K head tiles, reader -> compute (transpose_k_heads only)
    const DFBSpecName K_OUT{"k_out"};  // transposed K head tiles, compute -> writer (transpose_k_heads only)
    const TensorParamName INPUT_Q{"input_q"};
    const TensorParamName INPUT_KV{"input_kv"};
    const TensorParamName Q{"q"};
    const TensorParamName K{"k"};
    const TensorParamName V{"v"};

    const Tensor& input_tensor = tensor_args.input_tensor_q;
    // Read through the reference: the KV tensor argument below must name the tensor the framework sees.
    const std::optional<Tensor>& input_tensor_kv = tensor_args.input_tensor_kv;
    const uint32_t num_q_heads = operation_attributes.num_q_heads;
    const uint32_t num_kv_heads = operation_attributes.num_kv_heads;
    const uint32_t head_dim = operation_attributes.head_dim;
    const bool transpose_k_heads = operation_attributes.transpose_k_heads;
    auto& output = tensor_return_value;

    const auto& input_shape = input_tensor.padded_shape();

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();

    uint32_t single_tile_size = tt::tile_size(data_format);
    TT_ASSERT(input_tensor.buffer()->size() % single_tile_size == 0);

    if (read_from_input_tensor_kv) {
        TT_ASSERT(input_tensor_kv->buffer()->size() % single_tile_size == 0);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_w_tiles = input_shape[3] / TILE_WIDTH;
    uint32_t in1_w_tiles = 0;
    if (read_from_input_tensor_kv) {
        in1_w_tiles = input_tensor_kv->padded_shape()[3] / TILE_WIDTH;
    }

    // Per output tensor args
    // Output shape for Q is: [B, num_q_heads, s, head_dim], shuffled from [B, 1, s, num_q_heads * head_dim]
    // Output shape for K/V is: [B, num_kv_heads, s, head_dim], shuffled from [B, 1, s, num_kv_heads * head_dim]
    // NOTE: Output h and w dims are identical for Q, K, V, so any arg that is related to these dims for q_* can be
    // shared for K, V
    uint32_t q_out_h_tiles = input_shape[2] / TILE_HEIGHT;
    uint32_t q_out_w_tiles = head_dim / TILE_WIDTH;  // tiles along head_dim
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = num_q_heads * q_out_HtWt;
    uint32_t kv_out_CHtWt = num_kv_heads * q_out_HtWt;
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;
    uint32_t kv_num_tiles = num_kv_heads * q_out_w_tiles;

    const auto split = build_interleaved_work_split(operation_attributes, input_tensor);
    const auto& core_group_1 = split.core_group_1;
    const auto& core_group_2 = split.core_group_2;
    const uint32_t num_blocks_per_core_group_1 = split.num_blocks_per_core_group_1;
    const uint32_t num_blocks_per_core_group_2 = split.num_blocks_per_core_group_2;

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    ttnn::Tensor& q = std::get<0>(output);
    ttnn::Tensor& k = std::get<1>(output);
    ttnn::Tensor& v = std::get<2>(output);

    TT_ASSERT(q.buffer() != nullptr, "Output q buffer should be allocated on device!");
    TT_ASSERT(k.buffer() != nullptr, "Output k buffer should be allocated on device!");
    TT_ASSERT(v.buffer() != nullptr, "Output v buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = input_tensor.device();
    const tt::ARCH arch = device->arch();

    // Tensor parameters: the Q input, the optional separate KV input, and the three outputs.  The kernels
    // reach them through TensorAccessor(tensor::<name>); the base addresses ride the bindings.
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = INPUT_Q, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = Q, .spec = q.tensor_spec()},
        TensorParameter{.unique_id = K, .spec = k.tensor_spec()},
        TensorParameter{.unique_id = V, .spec = v.tensor_spec()},
    };
    if (read_from_input_tensor_kv) {
        tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_KV, .spec = input_tensor_kv->tensor_spec()});
    }

    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (transpose_k_heads) {
        reader_defines.emplace("TRANSPOSE_K_HEADS", "1");
        writer_defines.emplace("TRANSPOSE_K_HEADS", "1");
    }
    if (read_from_input_tensor_kv) {
        reader_defines.emplace("READ_FROM_INPUT_TENSOR_KV", "1");
    }
    if (operation_attributes.kv_tied) {
        reader_defines.emplace("KV_TIED", "1");
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_create_qkv_heads.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = QV,
                    .accessor_name = "qv",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT_Q,
                    .accessor_name = "input_q",
                },
            },
        .compile_time_args =
            {
                {"q_num_tiles", q_num_tiles},
                {"kv_num_tiles", kv_num_tiles},
                {"head_parallel", static_cast<uint32_t>(split.head_parallel)},
                {"head_tiles", q_out_w_tiles},
                {"seq_tiles", q_out_h_tiles},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_blocks", "in0_tensor_tile_id", "in1_tensor_tile_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };
    // TODO: Q, K, V doesn't necessarily need to be the same output mem config
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
            "writer_tm_tile_layout_nlp_create_qkv_heads.cpp",
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = QV,
                    .accessor_name = "qv",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = Q,
                    .accessor_name = "q",
                },
                TensorBinding{
                    .tensor_parameter_name = K,
                    .accessor_name = "k",
                },
                TensorBinding{
                    .tensor_parameter_name = V,
                    .accessor_name = "v",
                },
            },
        .compile_time_args =
            {
                {"q_out_h_tiles", q_out_h_tiles},
                {"q_out_w_tiles", q_out_w_tiles},
                {"q_out_HtWt", q_out_HtWt},
                {"q_out_c", num_q_heads},
                {"kv_out_c", num_kv_heads},
                {"head_parallel", static_cast<uint32_t>(split.head_parallel)},
                // Non-zero only for the Q head split: output 0 takes the first split_width tiles of every
                // head row and output 1 (bound as K) the rest.
                {"split_width", static_cast<uint32_t>(operation_attributes.q_head_split.value_or(0) / TILE_WIDTH)},
            },
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_blocks", "q_out_h_dim", "q_out_tensor_tile_id", "k_out_tensor_tile_id", "v_out_tensor_tile_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };
    if (read_from_input_tensor_kv) {
        reader.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = INPUT_KV,
            .accessor_name = "input_kv",
        });
    }
    constexpr uint32_t tile_chunk = 8;  // *_tiles.cpp kernels: tiles per NoC barrier (the DFB holds 2 chunks)
    if (split.tile_split) {
        reader.source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
            "reader_tm_tile_layout_nlp_create_qkv_heads_tiles.cpp";
        reader.compile_time_args.insert({"in0_w_tiles", in0_w_tiles});
        reader.compile_time_args.insert({"in1_w_tiles", in1_w_tiles});
        reader.compile_time_args.insert({"chunk", tile_chunk});
        reader.runtime_arg_schema.runtime_arg_names = {"num_tiles", "start_tile"};
        writer.source =
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
            "writer_tm_tile_layout_nlp_create_qkv_heads_tiles.cpp";
        writer.compile_time_args.insert({"chunk", tile_chunk});
        writer.runtime_arg_schema.runtime_arg_names = {"num_tiles", "start_tile"};
    }

    // Dataflow buffers
    // Four-tile capacity: quadruple buffering for the one-tile paths, one batched head transfer otherwise.
    uint32_t dfb_num_tiles = 4;

    // TODO: Investigate perf allocating full in0_w_tiles with double buffer
    // uint32_t qv_num_tiles = in0_w_tiles * 2; // double buffer; this runs out of space for generic shapes
    uint32_t qv_num_tiles = split.tile_split ? 2 * tile_chunk : dfb_num_tiles;
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = QV,
            .entry_size = single_tile_size,
            .num_entries = qv_num_tiles,
            .data_format_metadata = data_format,
        },
    };

    Group<KernelSpecName> work_unit_kernels_g1 = {READER, WRITER};
    Group<KernelSpecName> work_unit_kernels_g2 = {READER, WRITER};
    Group<KernelSpec> kernels;

    // If we transpose_k_heads:
    // - reader will write K heads to k_in, instead of qv
    // - compute will wait on k_in and write to k_out
    // - writer will wait on k_out, instead of qv
    // Neither K buffer exists otherwise; the generic transpose_wh compute kernel binds the two of them.
    if (transpose_k_heads) {
        // For FLOAT32 input, enable fp32 dest accumulation so the JIT data-format selection
        // resolves the unpack-dst buffer to Tf32 (10-bit mantissa) instead of Float16_b (7-bit
        // mantissa). Mirrors the per-dtype promotion in eltwise unary/binary primitives.
        const bool fp32_dest_acc_en = input_tensor.dtype() == tt_metal::DataType::FLOAT32;

        ComputeGen1Config compute_hw{.enable_32_bit_dest = fp32_dest_acc_en};
        if (fp32_dest_acc_en) {
            // The legacy descriptor left unpack_to_dest_mode empty (unpack to SrcA/B).  With a 32-bit dest
            // and a Float32 input buffer the mode has to be stated explicitly; this is the same mode.
            compute_hw.unpack_modes.emplace(K_IN, UnpackMode::UnpackToSrc);
        }

        // One compute spec per work-split core group: the block count is a compile-time argument, so
        // each group compiles its own instance.
        auto make_compute = [&](const KernelSpecName& unique_id, uint32_t NHtWt) {
            return KernelSpec{
                .unique_id = unique_id,
                .source = "ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp",
                // The legacy compute descriptor resolved to O3; Metal 2.0 defaults every kernel to O2.
                .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
                .dfb_bindings =
                    {
                        DFBBinding{
                            .dfb_spec_name = K_IN,
                            .accessor_name = "in",
                            .endpoint_type = DFBEndpointType::CONSUMER,
                        },
                        DFBBinding{
                            .dfb_spec_name = K_OUT,
                            .accessor_name = "out",
                            .endpoint_type = DFBEndpointType::PRODUCER,
                        },
                    },
                .compile_time_args = {{"NHtWt", NHtWt}},
                .hw_config = ComputeHardwareConfig{compute_hw},
            };
        };
        kernels.push_back(make_compute(COMPUTE_G1, num_blocks_per_core_group_1 * kv_num_tiles));
        work_unit_kernels_g1.push_back(COMPUTE_G1);
        if (core_group_2.num_cores() > 0) {
            kernels.push_back(make_compute(COMPUTE_G2, num_blocks_per_core_group_2 * kv_num_tiles));
            work_unit_kernels_g2.push_back(COMPUTE_G2);
        }

        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = K_IN,
            .accessor_name = "k",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = K_OUT,
            .accessor_name = "k",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });

        uint32_t k_in_num_tiles = dfb_num_tiles;
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = K_IN,
            .entry_size = single_tile_size,
            .num_entries = k_in_num_tiles,
            .data_format_metadata = data_format,
        });

        uint32_t k_out_num_tiles = dfb_num_tiles;
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = K_OUT,
            .entry_size = single_tile_size,
            .num_entries = k_out_num_tiles,
            .data_format_metadata = data_format,
        });
    }

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    uint32_t num_blocks_written = 0;
    for (const CoreCoord& core : split.cores) {
        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        if (split.tile_split) {
            // here a block is one tile: the core owns tiles [num_blocks_written, + num_blocks_per_core)
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"num_tiles", num_blocks_per_core}, {"start_tile", num_blocks_written}});
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"num_tiles", num_blocks_per_core}, {"start_tile", num_blocks_written}});
            num_blocks_written += num_blocks_per_core;
            continue;
        }
        uint32_t q_out_h_dim = num_blocks_written % q_out_h_tiles;
        uint32_t q_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * q_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t v_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t k_out_tensor_tile_id = transpose_k_heads
                                            ? (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + q_out_h_dim
                                            : v_out_tensor_tile_id;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {
                {"num_blocks", num_blocks_per_core},
                // Head-parallel blocks are (batch, head, row) indices the reader decodes itself.
                {"in0_tensor_tile_id", split.head_parallel ? num_blocks_written : num_blocks_written * in0_w_tiles},
                {"in1_tensor_tile_id", num_blocks_written * in1_w_tiles},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {
                {"num_blocks", num_blocks_per_core},
                {"q_out_h_dim", q_out_h_dim},
                {"q_out_tensor_tile_id",
                 split.head_parallel ? num_blocks_written * q_out_w_tiles : q_out_tensor_tile_id},
                {"k_out_tensor_tile_id", k_out_tensor_tile_id},
                {"v_out_tensor_tile_id", v_out_tensor_tile_id},
            });

        num_blocks_written += num_blocks_per_core;
    }

    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));

    // One work unit per work-split core group; the reader and writer run on both, the per-group compute
    // instance (when present) only on its own group.
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{
            .name = "core_group_1",
            .kernels = std::move(work_unit_kernels_g1),
            .target_nodes = core_group_1,
        },
    };
    if (core_group_2.num_cores() > 0) {
        work_units.push_back(WorkUnitSpec{
            .name = "core_group_2",
            .kernels = std::move(work_unit_kernels_g2),
            .target_nodes = core_group_2,
        });
    }

    ProgramSpec spec{
        .name = "nlp_create_qkv_heads_interleaved",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = build_qkv_tensor_run_args(tensor_args, output);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

namespace {

// Per-core Q-side work of one kernel instance.  The reader-config instance takes the first per_risc0
// heads of the core's Q output shard, the writer-config instance the remaining per_risc1 heads.
struct ShardedQArgs {
    uint32_t num_q_heads = 0;              // heads this instance reads
    uint32_t remote_q_head_start_idx = 0;  // first head inside the source shard it starts from
    uint32_t start_q_x = 0;                // source core, as indices into the NoC coordinate tables
    uint32_t start_q_y = 0;
    uint32_t q_offset = 0;  // byte offset into the Q output shard where this instance writes
};

struct ShardedCoreArgs {
    CoreCoord core;
    bool read_kv_heads = false;  // this core also holds a K/V output shard
    ShardedQArgs reader_q;
    ShardedQArgs writer_q;
    // K/V-side start (both instances walk the same heads; the reader instance reads the K section, the
    // writer instance the V section).
    uint32_t remote_kv_head_start_idx = 0;
    uint32_t start_kv_x = 0;
    uint32_t start_kv_y = 0;
};

// Single source of truth for the Sharded per-core reader/writer runtime args.  The values below `cores`
// are identical on every core; they are still delivered per core, as the legacy factory did.
struct ShardedArgs {
    uint32_t head_size = 0;
    uint32_t per_core_in_q_heads = 0;
    uint32_t per_core_out_kv_heads = 0;
    uint32_t per_core_in_kv_heads = 0;
    uint32_t k_section_offset = 0;  // byte offset of the K section inside the (fused or separate) input shard
    uint32_t v_section_offset = 0;  // byte offset of the V section
    uint32_t k_num_tiles = 0;
    uint32_t num_cores_x = 0;
    std::vector<uint32_t> noc_x_coords;
    std::vector<uint32_t> noc_y_coords;
    std::vector<ShardedCoreArgs> cores;
};

ShardedArgs build_sharded_core_args(
    const NlpCreateHeadsDeviceOperation::operation_attributes_t& operation_attributes,
    const NlpCreateHeadsDeviceOperation::tensor_args_t& tensor_args,
    NlpCreateHeadsDeviceOperation::tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input_tensor_q;
    const auto& input_tensor_kv = tensor_args.input_tensor_kv;
    auto head_dim = operation_attributes.head_dim;
    auto num_q_heads = operation_attributes.num_q_heads;
    auto num_kv_heads = operation_attributes.num_kv_heads;

    tt_metal::IDevice* device = input_tensor.device();
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();
    uint32_t single_tile_size = tt::tile_size(data_format);
    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    auto q_shard_spec = std::get<0>(output).shard_spec().value();
    auto q_cores = q_shard_spec.grid;

    uint32_t per_core_out_q_heads = num_q_heads / q_cores.num_cores();
    uint32_t per_risc0_out_q_heads = div_up(per_core_out_q_heads, 2);
    uint32_t per_risc1_out_q_heads = per_core_out_q_heads / 2;
    uint32_t per_core_in_q_heads = num_q_heads / input_tensor.shard_spec().value().num_cores();

    auto k_shard_spec = std::get<1>(output).shard_spec().value();
    auto k_cores = k_shard_spec.grid;
    auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    uint32_t per_core_out_kv_heads = num_kv_heads / k_cores.num_cores();
    uint32_t per_core_in_kv_heads =
        num_kv_heads / (read_from_input_tensor_kv ? input_tensor_kv.value().shard_spec().value().num_cores()
                                                  : input_tensor.shard_spec().value().num_cores());

    // The input shard bases ride the tensor bindings.  The K/V columns live either in the separate KV
    // tensor's shard (section offset 0 for K) or after the Q heads inside the fused shard; that section
    // start is passed as a byte offset the kernel adds to the base on device.
    const uint32_t k_section_offset =
        read_from_input_tensor_kv ? 0 : per_core_in_q_heads * head_tiles * single_tile_size;
    // Tied: V is K's own columns, so the writer reads from K's section rather than the one after
    // it. The per-core head offsets are added on device from remote_kv_head_start_idx.
    const uint32_t v_section_offset = operation_attributes.kv_tied
                                          ? k_section_offset
                                          : k_section_offset + (per_core_in_kv_heads * head_tiles * single_tile_size);

    uint32_t num_cores = std::max(q_cores.num_cores(), k_cores.num_cores());
    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    ShardedArgs args;
    args.head_size = head_size;
    args.per_core_in_q_heads = per_core_in_q_heads;
    args.per_core_out_kv_heads = per_core_out_kv_heads;
    args.per_core_in_kv_heads = per_core_in_kv_heads;
    args.k_section_offset = k_section_offset;
    args.v_section_offset = v_section_offset;
    args.k_num_tiles = k_num_tiles;
    args.num_cores_x = num_cores_x;

    args.noc_x_coords.reserve(num_cores_x);
    for (uint32_t x = 0; x < num_cores_x; ++x) {
        args.noc_x_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    args.noc_y_coords.reserve(num_cores_y);
    for (uint32_t y = 0; y < num_cores_y; ++y) {
        args.noc_y_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    uint32_t remote_q_head_start_idx = 0;
    uint32_t remote_kv_head_start_idx = 0;
    uint32_t q_x = 0, q_y = 0, kv_x = 0, kv_y = 0;

    uint32_t remote_q_read = 0;
    uint32_t remote_kv_read = 0;

    args.cores.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        ShardedCoreArgs e;
        e.core = cores[i];
        e.read_kv_heads = i < k_cores.num_cores();
        // The kernel derives its start addresses itself: q = q shard base + remote_q_head_start_idx *
        // head_size, kv = kv shard base + kv_section_offset + remote_kv_head_start_idx * head_size.

        // Reader-config instance: the first per_risc0 heads of this core's Q output shard.
        e.reader_q = ShardedQArgs{
            .num_q_heads = per_risc0_out_q_heads,
            .remote_q_head_start_idx = remote_q_head_start_idx,
            .start_q_x = q_x,
            .start_q_y = q_y,
            .q_offset = 0,
        };

        remote_q_read += per_risc0_out_q_heads;
        q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
        q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
        remote_q_head_start_idx = (remote_q_head_start_idx + per_risc0_out_q_heads) % per_core_in_q_heads;

        // Writer-config instance: the remaining heads, written after the reader instance's.
        e.writer_q = ShardedQArgs{
            .num_q_heads = per_risc1_out_q_heads,
            .remote_q_head_start_idx = remote_q_head_start_idx,
            .start_q_x = q_x,
            .start_q_y = q_y,
            .q_offset = per_risc0_out_q_heads * head_size,
        };

        if (per_risc1_out_q_heads > 0) {
            remote_q_read += per_risc1_out_q_heads;
            q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
            q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
            remote_q_head_start_idx = (per_risc1_out_q_heads + remote_q_head_start_idx) % per_core_in_q_heads;
        }

        e.remote_kv_head_start_idx = remote_kv_head_start_idx;
        e.start_kv_x = kv_x;
        e.start_kv_y = kv_y;

        if (e.read_kv_heads) {
            remote_kv_read += per_core_out_kv_heads;
            kv_y = (remote_kv_read / per_core_in_kv_heads) / num_cores_x;
            kv_x = (remote_kv_read / per_core_in_kv_heads) % num_cores_x;
            remote_kv_head_start_idx = (remote_kv_head_start_idx + per_core_out_kv_heads) % per_core_in_kv_heads;
        }

        args.cores.push_back(e);
    }

    return args;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts NlpCreateHeadsDeviceOperation::Sharded::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Spec resource names (function-local: the two factories share one unity translation unit).
    // One kernel source, four specs: the reader-config and writer-config instances, each once for the cores
    // that also hold a K/V output shard and once for the cores that hold a Q output shard only.
    const KernelSpecName READER_KV{"reader_kv"};
    const KernelSpecName WRITER_KV{"writer_kv"};
    const KernelSpecName READER_Q{"reader_q"};
    const KernelSpecName WRITER_Q{"writer_q"};
    const DFBSpecName Q_OUT{"q_out"};
    const DFBSpecName K_OUT{"k_out"};
    const DFBSpecName V_OUT{"v_out"};
    const TensorParamName INPUT_Q{"input_q"};
    const TensorParamName INPUT_KV{"input_kv"};
    const TensorParamName Q{"q"};
    const TensorParamName K{"k"};
    const TensorParamName V{"v"};

    const auto& input_tensor = tensor_args.input_tensor_q;
    const auto& input_tensor_kv = tensor_args.input_tensor_kv;
    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();
    auto& output = tensor_return_value;

    IDevice* device = input_tensor.device();
    const tt::ARCH arch = device->arch();

    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(data_format);

    auto q_shard_spec = std::get<0>(output).shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;

    auto k_shard_spec = std::get<1>(output).shard_spec().value();
    auto k_cores = k_shard_spec.grid;
    auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    auto v_shard_spec = std::get<2>(output).shard_spec().value();
    auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    // The three output buffers are borrowed as dataflow buffers: each kernel instance writes its heads
    // straight into the output shard.  Their L1 addresses resolve from the q/k/v tensor arguments.
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = Q_OUT,
            .entry_size = single_tile_size,
            .num_entries = q_num_tiles,
            .data_format_metadata = data_format,
            .borrowed_from = Q,
        },
        DataflowBufferSpec{
            .unique_id = K_OUT,
            .entry_size = single_tile_size,
            .num_entries = k_num_tiles,
            .data_format_metadata = data_format,
            .borrowed_from = K,
        },
        DataflowBufferSpec{
            .unique_id = V_OUT,
            .entry_size = single_tile_size,
            .num_entries = v_num_tiles,
            .data_format_metadata = data_format,
            .borrowed_from = V,
        },
    };

    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = INPUT_Q, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = Q, .spec = std::get<0>(output).tensor_spec()},
        TensorParameter{.unique_id = K, .spec = std::get<1>(output).tensor_spec()},
        TensorParameter{.unique_id = V, .spec = std::get<2>(output).tensor_spec()},
    };
    if (read_from_input_tensor_kv) {
        tensor_parameters.push_back(TensorParameter{.unique_id = INPUT_KV, .spec = input_tensor_kv->tensor_spec()});
    }

    // Build the per-core reader/writer runtime args via the shared builder.  The input shard bases ride
    // the tensor bindings; every section and head offset is a separate scalar the kernel adds on device.
    const auto args = build_sharded_core_args(operation_attributes, tensor_args, tensor_return_value);
    // The NoC coordinate tables of the source grid (x-coordinates, then y-coordinates) are indexed by the
    // kernel's data walk, so they ride the positional vararg block; every node gets the same table.
    const uint32_t num_varargs = args.noc_x_coords.size() + args.noc_y_coords.size();
    AdvancedKernelRunArgs::Varargs noc_coords;
    noc_coords.reserve(num_varargs);
    noc_coords.insert(noc_coords.end(), args.noc_x_coords.begin(), args.noc_x_coords.end());
    noc_coords.insert(noc_coords.end(), args.noc_y_coords.begin(), args.noc_y_coords.end());

    // The K/V output shards exist on k_cores only (a prefix of q_cores in row-major order), and the kernel
    // reads K/V heads exactly there.  A dataflow buffer lives on every node its kernels run on, so the K/V
    // buffers are bound from kernel specs placed on k_cores alone; the remaining Q cores get specs that
    // bind the Q output only.
    const CoreRangeSet q_only_cores = q_cores.subtract(k_cores);
    const bool has_q_only_cores = q_only_cores.num_cores() > 0;

    auto make_instance = [&](const KernelSpecName& unique_id, bool is_reader_instance, bool reads_kv_heads) {
        KernelSpec instance{
            .unique_id = unique_id,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
                "reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp",
            // Both instances write disjoint head ranges of the Q output shard through its write pointer,
            // with no FIFO traffic: the producer / consumer roles only satisfy the one-of-each rule.
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = Q_OUT,
                        .accessor_name = "q_out",
                        .endpoint_type = is_reader_instance ? DFBEndpointType::PRODUCER : DFBEndpointType::CONSUMER,
                    },
                },
            .tensor_bindings =
                {
                    TensorBinding{
                        .tensor_parameter_name = INPUT_Q,
                        .accessor_name = "input_q",
                    },
                },
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"head_size",
                      "num_q_heads",
                      "num_q_heads_per_core",
                      "remote_q_head_start_idx",
                      "start_q_x",
                      "start_q_y",
                      "q_offset",
                      "num_x"}},
            .hw_config = is_reader_instance ? ttnn::create_reader_datamovement_config(arch)
                                            : ttnn::create_writer_datamovement_config(arch),
            .advanced_options = {.num_runtime_varargs = num_varargs},
        };
        if (reads_kv_heads) {
            instance.compiler_options.defines.emplace("READ_KV_HEADS", "1");
            // The reader instance fills the K output shard, the writer instance the V output shard; nothing
            // drains either (they are the outputs), so each is bound at both ends by its one writer.
            const DFBSpecName& kv_out = is_reader_instance ? K_OUT : V_OUT;
            instance.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = kv_out,
                .accessor_name = "kv_out",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            instance.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = kv_out,
                .accessor_name = "kv_out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
            if (read_from_input_tensor_kv) {
                instance.compiler_options.defines.emplace("READ_FROM_INPUT_TENSOR_KV", "1");
                instance.tensor_bindings.push_back(TensorBinding{
                    .tensor_parameter_name = INPUT_KV,
                    .accessor_name = "input_kv",
                });
            }
            for (const char* name :
                 {"num_kv_heads",
                  "num_kv_heads_per_core",
                  "remote_kv_head_start_idx",
                  "start_kv_x",
                  "start_kv_y",
                  "kv_section_offset",
                  "num_kv_tiles"}) {
                instance.runtime_arg_schema.runtime_arg_names.push_back(name);
            }
        }
        return instance;
    };

    Group<KernelSpec> kernels = {
        make_instance(READER_KV, /*is_reader_instance=*/true, /*reads_kv_heads=*/true),
        make_instance(WRITER_KV, /*is_reader_instance=*/false, /*reads_kv_heads=*/true),
    };
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{
            .name = "kv_cores",
            .kernels = {READER_KV, WRITER_KV},
            .target_nodes = k_cores,
        },
    };
    if (has_q_only_cores) {
        kernels.push_back(make_instance(READER_Q, /*is_reader_instance=*/true, /*reads_kv_heads=*/false));
        kernels.push_back(make_instance(WRITER_Q, /*is_reader_instance=*/false, /*reads_kv_heads=*/false));
        work_units.push_back(WorkUnitSpec{
            .name = "q_only_cores",
            .kernels = {READER_Q, WRITER_Q},
            .target_nodes = q_only_cores,
        });
    }

    KernelRunArgs reader_kv_run_args{.kernel = READER_KV};
    KernelRunArgs writer_kv_run_args{.kernel = WRITER_KV};
    KernelRunArgs reader_q_run_args{.kernel = READER_Q};
    KernelRunArgs writer_q_run_args{.kernel = WRITER_Q};
    for (const auto& e : args.cores) {
        // Same arg set for both instances; only the instance's Q-side values and (on kv cores) the
        // K vs V section offset differ.
        auto emit = [&](KernelRunArgs& run_args, const ShardedQArgs& q, uint32_t kv_section_offset) {
            AddRuntimeArgsForNode(
                run_args.runtime_arg_values,
                e.core,
                {
                    {"head_size", args.head_size},
                    {"num_q_heads", q.num_q_heads},
                    {"num_q_heads_per_core", args.per_core_in_q_heads},
                    {"remote_q_head_start_idx", q.remote_q_head_start_idx},
                    {"start_q_x", q.start_q_x},
                    {"start_q_y", q.start_q_y},
                    {"q_offset", q.q_offset},
                    {"num_x", args.num_cores_x},
                });
            if (e.read_kv_heads) {
                AddRuntimeArgsForNode(
                    run_args.runtime_arg_values,
                    e.core,
                    {
                        {"num_kv_heads", args.per_core_out_kv_heads},
                        {"num_kv_heads_per_core", args.per_core_in_kv_heads},
                        {"remote_kv_head_start_idx", e.remote_kv_head_start_idx},
                        {"start_kv_x", e.start_kv_x},
                        {"start_kv_y", e.start_kv_y},
                        {"kv_section_offset", kv_section_offset},
                        {"num_kv_tiles", args.k_num_tiles},
                    });
            }
            run_args.advanced_options.runtime_varargs.emplace(e.core, noc_coords);
        };
        if (e.read_kv_heads) {
            emit(reader_kv_run_args, e.reader_q, args.k_section_offset);
            emit(writer_kv_run_args, e.writer_q, args.v_section_offset);
        } else {
            emit(reader_q_run_args, e.reader_q, 0);
            emit(writer_q_run_args, e.writer_q, 0);
        }
    }

    ProgramSpec spec{
        .name = "nlp_create_qkv_heads_sharded",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_kv_run_args), std::move(writer_kv_run_args)};
    if (has_q_only_cores) {
        run_args.kernel_run_args.push_back(std::move(reader_q_run_args));
        run_args.kernel_run_args.push_back(std::move(writer_q_run_args));
    }
    run_args.tensor_args = build_qkv_tensor_run_args(tensor_args, output);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

// Only per-dispatch state is re-applied: the tensor bindings, which carry the input shard bases and
// re-point the three borrowed output buffers.  Every other runtime arg derives from the operation
// attributes or the input/output TensorSpecs, which the program hash covers, so a cache hit means they are
// identical by construction.
tt::tt_metal::experimental::ProgramRunArgs NlpCreateHeadsDeviceOperation::Sharded::override_runtime_arguments(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramRunArgs params;
    params.tensor_args = build_qkv_tensor_run_args(tensor_args, tensor_return_value);
    return params;
}

// The reader takes the input (and optional KV input), the writer the three outputs; all five ride the
// tensor bindings.  The Interleaved dataflow buffers are not borrowed, so there is nothing to re-point there.
tt::tt_metal::experimental::ProgramRunArgs NlpCreateHeadsDeviceOperation::Interleaved::override_runtime_arguments(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramRunArgs params;
    params.tensor_args = build_qkv_tensor_run_args(tensor_args, tensor_return_value);
    return params;
}

}  // namespace ttnn::operations::experimental::transformer
