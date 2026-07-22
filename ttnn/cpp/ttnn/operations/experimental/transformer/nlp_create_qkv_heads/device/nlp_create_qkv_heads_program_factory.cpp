// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include "nlp_create_qkv_heads_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// Metal 2.0 named resource handles for the Interleaved ProgramSpec.
// Op-prefixed to stay distinct under unity builds (Pattern: Unity-build hygiene for
// anonymous-namespace symbols). The kernel-source tokens (accessor_name strings) are the
// short names "qv"/"k_in"/"k_out", "in0"/"in1"/"q"/"k"/"v".
const DFBSpecName QKV_IV_QV_DFB{"nlpcqh_iv_qv"};
const DFBSpecName QKV_IV_KIN_DFB{"nlpcqh_iv_k_in"};
const DFBSpecName QKV_IV_KOUT_DFB{"nlpcqh_iv_k_out"};

const TensorParamName QKV_IV_IN0{"nlpcqh_iv_in0"};
const TensorParamName QKV_IV_IN1{"nlpcqh_iv_in1"};
const TensorParamName QKV_IV_Q{"nlpcqh_iv_q"};
const TensorParamName QKV_IV_K{"nlpcqh_iv_k"};
const TensorParamName QKV_IV_V{"nlpcqh_iv_v"};

const KernelSpecName QKV_IV_READER{"nlpcqh_iv_reader"};
const KernelSpecName QKV_IV_WRITER{"nlpcqh_iv_writer"};
const KernelSpecName QKV_IV_COMPUTE_G1{"nlpcqh_iv_compute_g1"};
const KernelSpecName QKV_IV_COMPUTE_G2{"nlpcqh_iv_compute_g2"};

// Metal 2.0 named resource handles for the Sharded ProgramSpec.
// Kernel-source tokens (accessor names): dfb "q_out"/"kv_out", tensor "in_q"/"in_kv".
const DFBSpecName QKV_SH_Q_DFB{"nlpcqh_sh_q_out"};  // c_16, borrowed from Q output
const DFBSpecName QKV_SH_K_DFB{"nlpcqh_sh_k_out"};  // c_17, borrowed from K output
const DFBSpecName QKV_SH_V_DFB{"nlpcqh_sh_v_out"};  // c_18, borrowed from V output

const TensorParamName QKV_SH_IN_Q{"nlpcqh_sh_in_q"};
const TensorParamName QKV_SH_IN_KV{"nlpcqh_sh_in_kv"};
const TensorParamName QKV_SH_Q_OUT{"nlpcqh_sh_q_out_t"};
const TensorParamName QKV_SH_K_OUT{"nlpcqh_sh_k_out_t"};
const TensorParamName QKV_SH_V_OUT{"nlpcqh_sh_v_out_t"};

const KernelSpecName QKV_SH_READER{"nlpcqh_sh_reader"};
const KernelSpecName QKV_SH_WRITER{"nlpcqh_sh_writer"};

}  // namespace

ttnn::device_operation::ProgramArtifacts NlpCreateHeadsDeviceOperation::Interleaved::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& input_tensor = tensor_args.input_tensor_q;
    const bool read_from_input_tensor_kv = tensor_args.input_tensor_kv.has_value();
    const uint32_t num_q_heads = operation_attributes.num_q_heads;
    const uint32_t num_kv_heads = operation_attributes.num_kv_heads;
    const uint32_t head_dim = operation_attributes.head_dim;
    const bool transpose_k_heads = operation_attributes.transpose_k_heads;

    Tensor& q = std::get<0>(tensor_return_value);
    Tensor& k = std::get<1>(tensor_return_value);
    Tensor& v = std::get<2>(tensor_return_value);

    IDevice* device = input_tensor.device();
    CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto& input_shape = input_tensor.padded_shape();
    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_w_tiles = input_shape[3] / TILE_WIDTH;
    uint32_t in1_w_tiles = 0;
    if (read_from_input_tensor_kv) {
        in1_w_tiles = tensor_args.input_tensor_kv.value().padded_shape()[3] / TILE_WIDTH;
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

    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of in0_w_tiles per core
    uint32_t num_blocks = input_shape[0] * input_shape[1] * input_shape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);
    const bool group_2_present = core_group_2.num_cores() > 0;

    // For FLOAT32 input, enable fp32 dest accumulation so the JIT data-format selection
    // resolves the unpack-dst CB to Tf32 (10-bit mantissa) instead of Float16_b (7-bit
    // mantissa). Mirrors the per-dtype promotion in eltwise unary/binary primitives.
    const bool fp32_dest_acc_en = input_tensor.dtype() == tt_metal::DataType::FLOAT32;

    // Quadruple-buffer everything (micro_block_size == 1).
    uint32_t cb_num_tiles = 4;

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramSpec
    ////////////////////////////////////////////////////////////////////////////
    ProgramSpec spec;
    spec.name = "nlp_create_qkv_heads_interleaved";

    // Tensor parameters (Case 1). The legacy buffer-address RTAs + TensorAccessorArgs CTAs collapse to bindings.
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = QKV_IV_IN0, .spec = input_tensor.tensor_spec()});
    if (read_from_input_tensor_kv) {
        spec.tensor_parameters.push_back(
            TensorParameter{.unique_id = QKV_IV_IN1, .spec = tensor_args.input_tensor_kv.value().tensor_spec()});
    }
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = QKV_IV_Q, .spec = q.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = QKV_IV_K, .spec = k.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = QKV_IV_V, .spec = v.tensor_spec()});

    // Dataflow buffers (formerly cb1 / cb0 / cb16). cb1 (QV, reader->writer FIFO for Q and V, and K on the
    // non-transpose path) is always present. cb0 (K_IN, reader->compute) and cb16 (K_OUT, compute->writer)
    // exist only when transposing K heads.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = QKV_IV_QV_DFB,
        .entry_size = single_tile_size,
        .num_entries = cb_num_tiles,
        .data_format_metadata = cb_data_format,
    });
    if (transpose_k_heads) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = QKV_IV_KIN_DFB,
            .entry_size = single_tile_size,
            .num_entries = cb_num_tiles,
            .data_format_metadata = cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = QKV_IV_KOUT_DFB,
            .entry_size = single_tile_size,
            .num_entries = cb_num_tiles,
            .data_format_metadata = cb_data_format,
        });
    }

    // Reader kernel (single instance across all_cores). Produces QV (and K_IN under transpose).
    KernelSpec::CompilerOptions::Defines reader_defines;
    if (transpose_k_heads) {
        reader_defines.insert({"TRANSPOSE_K_HEADS", "1"});
    }
    if (read_from_input_tensor_kv) {
        reader_defines.insert({"READ_FROM_INPUT_TENSOR_KV", "1"});
    }
    KernelSpec reader{
        .unique_id = QKV_IV_READER,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
                "reader_tm_tile_layout_nlp_create_qkv_heads.cpp"},
        .compiler_options = {.defines = reader_defines},
        .compile_time_args = {{"q_num_tiles", q_num_tiles}, {"kv_num_tiles", kv_num_tiles}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    reader.dfb_bindings.push_back(
        DFBBinding{.dfb_spec_name = QKV_IV_QV_DFB, .accessor_name = "qv", .endpoint_type = DFBEndpointType::PRODUCER});
    if (transpose_k_heads) {
        reader.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = QKV_IV_KIN_DFB, .accessor_name = "k_in", .endpoint_type = DFBEndpointType::PRODUCER});
    }
    reader.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = QKV_IV_IN0, .accessor_name = "in0"});
    if (read_from_input_tensor_kv) {
        reader.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = QKV_IV_IN1, .accessor_name = "in1"});
    }
    reader.runtime_arg_schema.runtime_arg_names.push_back("num_blocks");
    reader.runtime_arg_schema.runtime_arg_names.push_back("in0_tensor_tile_id");
    if (read_from_input_tensor_kv) {
        reader.runtime_arg_schema.runtime_arg_names.push_back("in1_tensor_tile_id");
    }

    // Writer kernel (single instance across all_cores). Consumes QV (and K_OUT under transpose).
    // TODO: Q, K, V doesn't necessarily need to be the same output mem config
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (transpose_k_heads) {
        writer_defines.insert({"TRANSPOSE_K_HEADS", "1"});
    }
    KernelSpec writer{
        .unique_id = QKV_IV_WRITER,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
                "writer_tm_tile_layout_nlp_create_qkv_heads.cpp"},
        .compiler_options = {.defines = writer_defines},
        .compile_time_args =
            {{"q_out_h_tiles", q_out_h_tiles},
             {"q_out_w_tiles", q_out_w_tiles},
             {"q_out_HtWt", q_out_HtWt},
             {"q_out_c", num_q_heads},
             {"kv_out_c", num_kv_heads}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };
    writer.dfb_bindings.push_back(
        DFBBinding{.dfb_spec_name = QKV_IV_QV_DFB, .accessor_name = "qv", .endpoint_type = DFBEndpointType::CONSUMER});
    if (transpose_k_heads) {
        writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = QKV_IV_KOUT_DFB, .accessor_name = "k_out", .endpoint_type = DFBEndpointType::CONSUMER});
    }
    writer.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = QKV_IV_Q, .accessor_name = "q"});
    writer.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = QKV_IV_K, .accessor_name = "k"});
    writer.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = QKV_IV_V, .accessor_name = "v"});
    for (const char* name :
         {"num_blocks", "q_out_h_dim", "q_out_tensor_tile_id", "k_out_tensor_tile_id", "v_out_tensor_tile_id"}) {
        writer.runtime_arg_schema.runtime_arg_names.push_back(name);
    }

    spec.kernels.push_back(reader);
    spec.kernels.push_back(writer);

    // Compute kernel (transpose only): one KernelSpec per non-empty core group (preserved multiplicity).
    // K_IN CONSUMER + K_OUT PRODUCER. The per-group CTA NHtWt stays a compile-time constant per group.
    if (transpose_k_heads) {
        auto make_compute = [&](const KernelSpecName& id, uint32_t nblocks_per_core) {
            ComputeGen1Config cfg{.enable_32_bit_dest = fp32_dest_acc_en};
            // Metal 2.0 requires an explicit unpack_modes entry when a compute kernel consumes a Float32 DFB with
            // enable_32_bit_dest = true. Legacy set no unpack_to_dest_mode -> Default -> UnpackToSrc; mirror that.
            if (fp32_dest_acc_en) {
                cfg.unpack_modes.insert({QKV_IV_KIN_DFB, UnpackMode::UnpackToSrc});
            }
            KernelSpec compute{
                .unique_id = id,
                .source =
                    std::filesystem::path{
                        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/compute/"
                        "transpose_wh_metal2.cpp"},
                .compile_time_args = {{"NHtWt", nblocks_per_core * kv_num_tiles}},
                .hw_config = ComputeHardwareConfig{cfg},
            };
            compute.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = QKV_IV_KIN_DFB, .accessor_name = "k_in", .endpoint_type = DFBEndpointType::CONSUMER});
            compute.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = QKV_IV_KOUT_DFB,
                .accessor_name = "k_out",
                .endpoint_type = DFBEndpointType::PRODUCER});
            return compute;
        };
        spec.kernels.push_back(make_compute(QKV_IV_COMPUTE_G1, num_blocks_per_core_group_1));
        if (group_2_present) {
            spec.kernels.push_back(make_compute(QKV_IV_COMPUTE_G2, num_blocks_per_core_group_2));
        }
    }

    // Work units. On the transpose path, one WU per core group (each running reader + writer + its compute
    // kernel); reader/writer are single KernelSpecs shared across both WUs (their union is all_cores). On the
    // non-transpose path a single WU on all_cores suffices.
    if (transpose_k_heads) {
        Group<KernelSpecName> g1_kernels = {QKV_IV_READER, QKV_IV_WRITER, QKV_IV_COMPUTE_G1};
        spec.work_units.push_back(WorkUnitSpec{.name = "iv_g1", .kernels = g1_kernels, .target_nodes = core_group_1});
        if (group_2_present) {
            Group<KernelSpecName> g2_kernels = {QKV_IV_READER, QKV_IV_WRITER, QKV_IV_COMPUTE_G2};
            spec.work_units.push_back(
                WorkUnitSpec{.name = "iv_g2", .kernels = g2_kernels, .target_nodes = core_group_2});
        }
    } else {
        spec.work_units.push_back(
            WorkUnitSpec{.name = "iv_main", .kernels = {QKV_IV_READER, QKV_IV_WRITER}, .target_nodes = all_cores});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramRunArgs (per-core runtime args)
    ////////////////////////////////////////////////////////////////////////////
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = QKV_IV_READER};
    KernelRunArgs writer_run{.kernel = QKV_IV_WRITER};

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t q_out_h_dim = num_blocks_written % q_out_h_tiles;
        uint32_t q_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * q_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t v_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t k_out_tensor_tile_id = transpose_k_heads
                                            ? (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + q_out_h_dim
                                            : v_out_tensor_tile_id;

        reader_run.runtime_arg_values["num_blocks"][core] = num_blocks_per_core;
        reader_run.runtime_arg_values["in0_tensor_tile_id"][core] = num_blocks_written * in0_w_tiles;
        if (read_from_input_tensor_kv) {
            reader_run.runtime_arg_values["in1_tensor_tile_id"][core] = num_blocks_written * in1_w_tiles;
        }

        writer_run.runtime_arg_values["num_blocks"][core] = num_blocks_per_core;
        writer_run.runtime_arg_values["q_out_h_dim"][core] = q_out_h_dim;
        writer_run.runtime_arg_values["q_out_tensor_tile_id"][core] = q_out_tensor_tile_id;
        writer_run.runtime_arg_values["k_out_tensor_tile_id"][core] = k_out_tensor_tile_id;
        writer_run.runtime_arg_values["v_out_tensor_tile_id"][core] = v_out_tensor_tile_id;

        num_blocks_written += num_blocks_per_core;
    }

    run_args.kernel_run_args.push_back(reader_run);
    run_args.kernel_run_args.push_back(writer_run);
    // Compute kernels carry only a CTA (NHtWt) and no runtime args; still list them so every kernel has an entry.
    if (transpose_k_heads) {
        run_args.kernel_run_args.push_back(KernelRunArgs{.kernel = QKV_IV_COMPUTE_G1});
        if (group_2_present) {
            run_args.kernel_run_args.push_back(KernelRunArgs{.kernel = QKV_IV_COMPUTE_G2});
        }
    }

    run_args.tensor_args.emplace(QKV_IV_IN0, TensorArgument{input_tensor.mesh_tensor()});
    if (read_from_input_tensor_kv) {
        run_args.tensor_args.emplace(QKV_IV_IN1, TensorArgument{tensor_args.input_tensor_kv.value().mesh_tensor()});
    }
    run_args.tensor_args.emplace(QKV_IV_Q, TensorArgument{q.mesh_tensor()});
    run_args.tensor_args.emplace(QKV_IV_K, TensorArgument{k.mesh_tensor()});
    run_args.tensor_args.emplace(QKV_IV_V, TensorArgument{v.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

namespace {

// Runtime-arg slot layout for the Sharded reader/writer.  Slots 6 and 15 hold the *clean* input-buffer
// bases, which the Metal 2.0 port carries as Case 2 tensor bindings (tensor::in_q / tensor::in_kv) rather
// than runtime args — build_sharded_core_args still writes 0 placeholders there and create_program_artifacts
// maps every other slot to a named RTA (skipping 6/15).  Slots 7 and 16 carry the *byte offsets* of the Q
// and K/V regions within a remote input shard; the kernel adds base + region offset + per-core head offset.

struct ShardedCoreArgs {
    CoreCoord core;
    std::vector<uint32_t> reader_args;
    std::vector<uint32_t> writer_args;
};

// Single source of truth for the Sharded per-core reader/writer runtime args.  The base-address slots
// (6, 15) hold a placeholder 0 here; the Metal 2.0 port carries those bases as Case 2 tensor bindings
// (tensor::in_q / tensor::in_kv) instead, so create_program_artifacts skips slots 6/15 when mapping named
// RTAs; slots 7/16 hold the Q / K-or-V region byte offset within a shard, which the kernel adds to the base.
std::vector<ShardedCoreArgs> build_sharded_core_args(
    const NlpCreateHeadsDeviceOperation::operation_attributes_t& operation_attributes,
    const NlpCreateHeadsDeviceOperation::tensor_args_t& tensor_args,
    NlpCreateHeadsDeviceOperation::tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input_tensor_q;
    const auto& input_tensor_kv = tensor_args.input_tensor_kv;
    auto head_dim = operation_attributes.head_dim;
    auto num_q_heads = operation_attributes.num_q_heads;
    auto num_kv_heads = operation_attributes.num_kv_heads;

    tt_metal::IDevice* device = input_tensor.device();
    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
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

    // Region byte-offsets of Q / K / V within a remote input shard.  Q always starts at offset 0.
    // Fused-QKV shard layout is [Q | K | V]; with a separate KV tensor the KV shard is [K | V].  These
    // replace the old host-folded `base + region` addresses: the kernel adds base + region offset itself.
    uint32_t q_region_offset = 0;
    uint32_t k_region_offset = read_from_input_tensor_kv ? 0u : per_core_in_q_heads * head_size;
    uint32_t v_region_offset = read_from_input_tensor_kv ? per_core_in_kv_heads * head_size
                                                         : (per_core_in_q_heads + per_core_in_kv_heads) * head_size;

    uint32_t num_cores = std::max(q_cores.num_cores(), k_cores.num_cores());
    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(num_cores_x);
    for (uint32_t x = 0; x < num_cores_x; ++x) {
        noc_x_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(num_cores_y);
    for (uint32_t y = 0; y < num_cores_y; ++y) {
        noc_y_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    uint32_t remote_q_head_start_idx = 0;
    uint32_t remote_kv_head_start_idx = 0;
    uint32_t q_x = 0, q_y = 0, kv_x = 0, kv_y = 0;

    uint32_t remote_q_read = 0;
    uint32_t remote_kv_read = 0;

    std::vector<ShardedCoreArgs> result;
    result.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        bool read_kv_heads = i < k_cores.num_cores();
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(18 + num_cores_x + num_cores_y);
        reader_runtime_args = {
            head_size,                 // 0
            per_risc0_out_q_heads,     // 1
            per_core_in_q_heads,       // 2
            remote_q_head_start_idx,   // 3
            q_x,                       // 4
            q_y,                       // 5
            0,                         // 6  q buffer base -> Buffer* binding (placeholder)
            q_region_offset,           // 7  Q region byte offset within shard (0)
            0,                         // 8  Q L1 write offset (reader = 0)
            read_kv_heads,             // 9
            per_core_out_kv_heads,     // 10
            per_core_in_kv_heads,      // 11
            remote_kv_head_start_idx,  // 12
            kv_x,                      // 13
            kv_y,                      // 14
            0,                         // 15 k/v buffer base -> Buffer* binding (placeholder)
            k_region_offset,           // 16 K region byte offset within shard
            k_num_tiles,               // 17
            num_cores_x,               // 18
        };
        reader_runtime_args.insert(reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
        reader_runtime_args.insert(reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());

        remote_q_read += per_risc0_out_q_heads;
        q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
        q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
        remote_q_head_start_idx = (remote_q_head_start_idx + per_risc0_out_q_heads) % per_core_in_q_heads;

        // Reader gets the args as built above (risc0 values); writer gets the same vector with the
        // risc1 q values patched, and (for kv cores) the V region offset over slot 16.  Slots 6/15
        // stay 0 here — the input bases flow through the tensor::in_q / tensor::in_kv bindings instead.
        std::vector<uint32_t> writer_runtime_args = reader_runtime_args;

        writer_runtime_args[1] = per_risc1_out_q_heads;
        writer_runtime_args[3] = remote_q_head_start_idx;
        writer_runtime_args[4] = q_x;
        writer_runtime_args[5] = q_y;
        writer_runtime_args[8] = per_risc0_out_q_heads * head_size;  // risc1 writes Q after risc0's heads

        if (per_risc1_out_q_heads > 0) {
            remote_q_read += per_risc1_out_q_heads;
            q_y = (remote_q_read / per_core_in_q_heads) / num_cores_x;
            q_x = (remote_q_read / per_core_in_q_heads) % num_cores_x;
            remote_q_head_start_idx = (per_risc1_out_q_heads + remote_q_head_start_idx) % per_core_in_q_heads;
        }

        if (read_kv_heads) {
            // Writer reads V where reader read K: same clean buffer base (slot 15), but the V region
            // sits after K within the shard.
            writer_runtime_args[16] = v_region_offset;
            remote_kv_read += per_core_out_kv_heads;
            kv_y = (remote_kv_read / per_core_in_kv_heads) / num_cores_x;
            kv_x = (remote_kv_read / per_core_in_kv_heads) % num_cores_x;
            remote_kv_head_start_idx = (remote_kv_head_start_idx + per_core_out_kv_heads) % per_core_in_kv_heads;
        }

        result.push_back({core, std::move(reader_runtime_args), std::move(writer_runtime_args)});
    }

    return result;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts NlpCreateHeadsDeviceOperation::Sharded::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor_q;
    auto& output = tensor_return_value;
    IDevice* device = input_tensor.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    Tensor& q_out_tensor = std::get<0>(output);
    Tensor& k_out_tensor = std::get<1>(output);
    Tensor& v_out_tensor = std::get<2>(output);

    auto q_shard_spec = q_out_tensor.shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    uint32_t q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;

    auto k_shard_spec = k_out_tensor.shard_spec().value();
    uint32_t k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    auto v_shard_spec = v_out_tensor.shard_spec().value();
    uint32_t v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    // Inputs (Case 2, raw base via get_bank_base_address): clean Q base always; the KV base is the KV tensor
    // when present, else the fused Q tensor (mirrors the legacy q_input_buffer / kv_input_buffer selection).
    const bool has_kv = tensor_args.input_tensor_kv.has_value();
    const Tensor& in_q_tensor = input_tensor;
    const Tensor& in_kv_tensor = has_kv ? tensor_args.input_tensor_kv.value() : input_tensor;

    ProgramSpec spec;
    spec.name = "nlp_create_qkv_heads_sharded";

    // Tensor parameters. Inputs are Case 2 (raw base); outputs back the borrowed output DFBs.
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = QKV_SH_IN_Q, .spec = in_q_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = QKV_SH_IN_KV, .spec = in_kv_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = QKV_SH_Q_OUT, .spec = q_out_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = QKV_SH_K_OUT, .spec = k_out_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(TensorParameter{.unique_id = QKV_SH_V_OUT, .spec = v_out_tensor.tensor_spec()});

    // Borrowed-memory DFBs (legacy c_16/c_17/c_18 with .buffer = output.buffer()). The reader-config and
    // writer-config instances both run on q_cores, so all three DFBs derive a q_cores node set.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = QKV_SH_Q_DFB,
        .entry_size = single_tile_size,
        .num_entries = q_num_tiles,
        .data_format_metadata = cb_data_format,
        .borrowed_from = QKV_SH_Q_OUT,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = QKV_SH_K_DFB,
        .entry_size = single_tile_size,
        .num_entries = k_num_tiles,
        .data_format_metadata = cb_data_format,
        .borrowed_from = QKV_SH_K_OUT,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = QKV_SH_V_DFB,
        .entry_size = single_tile_size,
        .num_entries = v_num_tiles,
        .data_format_metadata = cb_data_format,
        .borrowed_from = QKV_SH_V_OUT,
    });

    const std::filesystem::path sharded_src{
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp"};

    // Named RTA schema (identical for both instances). The legacy base-address slots (6 = Q input,
    // 15 = K/V input) become the tensor bindings below; the per-remote-core NoC coordinate arrays
    // (legacy slots 19+) become runtime varargs.
    const std::vector<std::string> rta_names = {
        "head_size",
        "num_q_heads",
        "num_q_heads_per_core",
        "remote_q_head_start_idx",
        "start_q_x",
        "start_q_y",
        "q_region_offset",
        "q_offset",
        "read_kv_heads",
        "num_kv_heads",
        "num_kv_heads_per_core",
        "remote_kv_head_start_idx",
        "start_kv_x",
        "start_kv_y",
        "kv_region_offset",
        "num_kv_tiles",
        "num_x"};

    // Per-core runtime args (legacy computation, unchanged). Each entry's flat vector still carries the
    // slot-6/15 base-address placeholders (now unused) and the NoC coord arrays from slot 19 on.
    auto per_core_args = build_sharded_core_args(operation_attributes, tensor_args, tensor_return_value);
    const uint32_t num_varargs =
        per_core_args.empty() ? 0u : static_cast<uint32_t>(per_core_args.front().reader_args.size() - 19);

    // Reader-config instance: Q output PRODUCER (c_16, 1P+1C dual-instance), K output self-loop (c_17).
    KernelSpec reader{
        .unique_id = QKV_SH_READER,
        .source = sharded_src,
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    reader.dfb_bindings = {
        DFBBinding{.dfb_spec_name = QKV_SH_Q_DFB, .accessor_name = "q_out", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = QKV_SH_K_DFB, .accessor_name = "kv_out", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = QKV_SH_K_DFB, .accessor_name = "kv_out", .endpoint_type = DFBEndpointType::CONSUMER},
    };
    reader.tensor_bindings = {
        TensorBinding{.tensor_parameter_name = QKV_SH_IN_Q, .accessor_name = "in_q"},
        TensorBinding{.tensor_parameter_name = QKV_SH_IN_KV, .accessor_name = "in_kv"}};
    reader.runtime_arg_schema.runtime_arg_names = rta_names;
    reader.advanced_options.num_runtime_varargs = num_varargs;

    // Writer-config instance: Q output CONSUMER (c_16), V output self-loop (c_18).
    KernelSpec writer{
        .unique_id = QKV_SH_WRITER,
        .source = sharded_src,
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };
    writer.dfb_bindings = {
        DFBBinding{.dfb_spec_name = QKV_SH_Q_DFB, .accessor_name = "q_out", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = QKV_SH_V_DFB, .accessor_name = "kv_out", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = QKV_SH_V_DFB, .accessor_name = "kv_out", .endpoint_type = DFBEndpointType::CONSUMER},
    };
    writer.tensor_bindings = reader.tensor_bindings;  // same in_q / in_kv Case 2 bindings
    writer.runtime_arg_schema.runtime_arg_names = rta_names;
    writer.advanced_options.num_runtime_varargs = num_varargs;

    spec.kernels = {reader, writer};
    spec.work_units = {
        WorkUnitSpec{.name = "sh_main", .kernels = {QKV_SH_READER, QKV_SH_WRITER}, .target_nodes = q_cores}};

    // ---- ProgramRunArgs (per-core named RTAs + NoC-coord varargs) ----
    // Named-arg slot map into build_sharded_core_args' flat vectors. Slots 6 and 15 (base addresses) are
    // omitted here — they flow through the tensor bindings instead.
    static const std::pair<const char*, int> kSlotMap[] = {
        {"head_size", 0},
        {"num_q_heads", 1},
        {"num_q_heads_per_core", 2},
        {"remote_q_head_start_idx", 3},
        {"start_q_x", 4},
        {"start_q_y", 5},
        {"q_region_offset", 7},
        {"q_offset", 8},
        {"read_kv_heads", 9},
        {"num_kv_heads", 10},
        {"num_kv_heads_per_core", 11},
        {"remote_kv_head_start_idx", 12},
        {"start_kv_x", 13},
        {"start_kv_y", 14},
        {"kv_region_offset", 16},
        {"num_kv_tiles", 17},
        {"num_x", 18}};

    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = QKV_SH_READER};
    KernelRunArgs writer_run{.kernel = QKV_SH_WRITER};

    for (auto& e : per_core_args) {
        for (const auto& [name, slot] : kSlotMap) {
            reader_run.runtime_arg_values[name][e.core] = e.reader_args[slot];
            writer_run.runtime_arg_values[name][e.core] = e.writer_args[slot];
        }
        reader_run.advanced_options.runtime_varargs[e.core] =
            std::vector<uint32_t>(e.reader_args.begin() + 19, e.reader_args.end());
        writer_run.advanced_options.runtime_varargs[e.core] =
            std::vector<uint32_t>(e.writer_args.begin() + 19, e.writer_args.end());
    }

    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args.emplace(QKV_SH_IN_Q, TensorArgument{in_q_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(QKV_SH_IN_KV, TensorArgument{in_kv_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(QKV_SH_Q_OUT, TensorArgument{q_out_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(QKV_SH_K_OUT, TensorArgument{k_out_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(QKV_SH_V_OUT, TensorArgument{v_out_tensor.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::experimental::transformer
