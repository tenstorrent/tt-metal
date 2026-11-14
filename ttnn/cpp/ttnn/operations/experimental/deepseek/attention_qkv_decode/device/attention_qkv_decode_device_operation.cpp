// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attention_qkv_decode_device_operation.hpp"

#include <array>
#include <optional>

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::attention_qkv_decode {

using namespace tt::tt_metal;
using tt::tt_metal::experimental::PrefetcherPipe;
using tt::tt_metal::experimental::SenderCoreType;

namespace {

void validate_weight(const Tensor& w, const char* name, uint32_t k, uint32_t shard_width) {
    TT_FATAL(
        w.layout() == Layout::TILE && w.buffer()->buffer_type() == BufferType::DRAM && w.nd_shard_spec().has_value(),
        "attention_qkv_decode: {} must be a TILE weight ND-sharded in DRAM",
        name);
    const auto& shape = w.logical_shape();
    const auto& shard = w.nd_shard_spec()->shard_shape;
    TT_FATAL(shape[-2] == k, "attention_qkv_decode: {} must have K = {} rows, got {}", name, k, shape);
    TT_FATAL(
        shard[-2] == k && shard[-1] == shard_width,
        "attention_qkv_decode: {} must be sharded one [{}, {}] slab per core, got shard {}",
        name,
        k,
        shard_width,
        shard);
}

void validate_gamma(const Tensor& gamma, const char* name, uint32_t width, const CoreRangeSet& cores) {
    TT_FATAL(
        gamma.layout() == Layout::TILE && gamma.dtype() == DataType::BFLOAT16 &&
            gamma.tensor_spec().tile().get_height() == 1 &&
            gamma.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            gamma.buffer()->buffer_type() == BufferType::L1 && gamma.logical_shape()[-1] == width &&
            gamma.shard_spec()->shape[1] == tt::constants::TILE_WIDTH && gamma.shard_spec()->grid == cores,
        "attention_qkv_decode: {} must be a bfloat16 1x32 TILE [1, {}] WIDTH_SHARDED one tile per core of {}",
        name,
        width,
        cores.str());
}

}  // namespace

AttentionQkvDecodeDeviceOperation::program_factory_t AttentionQkvDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void AttentionQkvDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& t) {
    for (const Tensor* x :
         {&t.tokens,
          &t.w_q_a,
          &t.w_kv,
          &t.w_q_b,
          &t.q_a_gamma,
          &t.kv_gamma,
          &t.cos,
          &t.sin,
          &t.kv_cache,
          &t.cache_pos}) {
        TT_FATAL(x->storage_type() == StorageType::DEVICE, "attention_qkv_decode: inputs must be on device");
    }
    auto* device = t.tokens.device();
    TT_FATAL(device->arch() == tt::ARCH::BLACKHOLE, "attention_qkv_decode: custom_mm is Blackhole-only");
    const auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(
        grid.x >= kGridX && grid.y >= kGridY,
        "attention_qkv_decode: needs a {}x{} worker grid, got {}x{}",
        kGridX,
        kGridY,
        grid.x,
        grid.y);

    const auto& tokens = t.tokens;
    TT_FATAL(
        tokens.layout() == Layout::ROW_MAJOR && tokens.dtype() == DataType::BFLOAT16 &&
            tokens.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "attention_qkv_decode: tokens must be the ROW_MAJOR bfloat16 HEIGHT_SHARDED replica of all_gather_for_matmul");
    const auto& tokens_shard = tokens.shard_spec().value();
    const uint32_t k = tokens.logical_shape()[-1];
    TT_FATAL(
        tokens_shard.shape[0] == 1 && tokens_shard.shape[1] == k && k % tt::constants::TILE_WIDTH == 0,
        "attention_qkv_decode: tokens must hold one [1, K] row per core, got shard {}",
        tokens_shard.shape);
    const CoreRangeSet proj_cores(std::vector<CoreRange>{
        CoreRange({0, kQbCoresY}, {kRowCoresX - 1, kGridY - 1}), CoreRange({kRowCoresX, 0}, {kGridX - 1, kGridY - 1})});
    TT_FATAL(
        tokens_shard.grid.contains(proj_cores),
        "attention_qkv_decode: the tokens replica must cover the q_a and kv cores {}, got {}",
        proj_cores.str(),
        tokens_shard.grid.str());

    const uint32_t q_lora = t.w_q_a.logical_shape()[-1];
    const uint32_t head_dim = t.w_kv.logical_shape()[-1];
    const uint32_t num_qa_cores = (kGridX - kRowCoresX) * kGridY;
    const uint32_t num_kv_cores = kRowCoresX * (kGridY - kQbCoresY);
    TT_FATAL(
        q_lora == num_qa_cores * tt::constants::TILE_WIDTH,
        "attention_qkv_decode: q_a must have {} columns (one tile per q_a core), got {}",
        num_qa_cores * tt::constants::TILE_WIDTH,
        q_lora);
    TT_FATAL(
        head_dim == num_kv_cores * tt::constants::TILE_WIDTH,
        "attention_qkv_decode: kv must have {} columns (one tile per kv core), got {}",
        num_kv_cores * tt::constants::TILE_WIDTH,
        head_dim);
    validate_weight(t.w_q_a, "w_q_a", k, tt::constants::TILE_WIDTH);
    validate_weight(t.w_kv, "w_kv", k, tt::constants::TILE_WIDTH);
    const uint32_t n_q = t.w_q_b.logical_shape()[-1];
    TT_FATAL(
        n_q % (kNumQbCores * tt::constants::TILE_WIDTH) == 0 && n_q % head_dim == 0,
        "attention_qkv_decode: q_b's width {} must split into whole heads of {} and over {} cores",
        n_q,
        head_dim,
        kNumQbCores);
    const uint32_t nc_tiles = n_q / kNumQbCores / tt::constants::TILE_WIDTH;
    const uint32_t head_tiles = head_dim / tt::constants::TILE_WIDTH;
    TT_FATAL(
        nc_tiles >= 2 && nc_tiles <= 16 && head_tiles % nc_tiles == 0,
        "attention_qkv_decode: each q_b core must hold 2..16 tiles of one head, got {} of a {}-tile head",
        nc_tiles,
        head_tiles);
    validate_weight(t.w_q_b, "w_q_b", q_lora, nc_tiles * tt::constants::TILE_WIDTH);

    validate_gamma(t.q_a_gamma, "q_a_gamma", q_lora, weight_role_cores(WeightRole::QA));
    validate_gamma(t.kv_gamma, "kv_gamma", head_dim, weight_role_cores(WeightRole::KV));

    TT_FATAL(
        t.w_comp.has_value() == t.comp_output.has_value(),
        "attention_qkv_decode: w_comp and comp_output must be given together");
    if (t.w_comp.has_value()) {
        const Tensor& w_comp = *t.w_comp;
        const Tensor& comp_output = *t.comp_output;
        TT_FATAL(
            w_comp.storage_type() == StorageType::DEVICE && comp_output.storage_type() == StorageType::DEVICE,
            "attention_qkv_decode: w_comp and comp_output must be on device");
        validate_weight(w_comp, "w_comp", k, tt::constants::TILE_WIDTH);
        const uint32_t n_comp = w_comp.logical_shape()[-1];
        TT_FATAL(
            n_comp % tt::constants::TILE_WIDTH == 0 && n_comp / tt::constants::TILE_WIDTH <= kNumQbCores,
            "attention_qkv_decode: w_comp's {} columns must be at most {} tiles, one per q_b core",
            n_comp,
            kNumQbCores);
        TT_FATAL(
            comp_output.layout() == Layout::ROW_MAJOR && comp_output.dtype() == DataType::BFLOAT16 &&
                !comp_output.is_sharded() && comp_output.buffer()->buffer_type() == BufferType::DRAM &&
                comp_output.logical_shape()[-1] == n_comp && comp_output.logical_shape().volume() == n_comp,
            "attention_qkv_decode: comp_output must be a DRAM-interleaved ROW_MAJOR bfloat16 [1, 1, 1, {}], got {}",
            n_comp,
            comp_output.logical_shape());
        TT_FATAL(
            tokens_shard.grid.contains(weight_role_cores(WeightRole::QB)),
            "attention_qkv_decode: with w_comp the tokens replica must also cover the q_b cores {}, got {}",
            weight_role_cores(WeightRole::QB).str(),
            tokens_shard.grid.str());
    }

    if (!args.prefetcher_pipes.empty()) {
        const uint32_t k_tiles = k / tt::constants::TILE_WIDTH;
        const std::array<uint32_t, kNumWeightRoles> slab_bytes = {
            k_tiles * tt::tile_size(datatype_to_dataformat_converter(t.w_q_a.dtype())),
            k_tiles * tt::tile_size(datatype_to_dataformat_converter(t.w_kv.dtype())),
            (q_lora / tt::constants::TILE_WIDTH) * nc_tiles *
                tt::tile_size(datatype_to_dataformat_converter(t.w_q_b.dtype())),
        };
        std::array<CoreRangeSet, kNumWeightRoles> covered;
        const auto pipes = ttnn::prefetcher_pipe_refs(args.prefetcher_pipes);
        for (size_t p = 0; p < pipes.size(); ++p) {
            const PrefetcherPipe& pipe = pipes[p];
            TT_FATAL(
                pipe.sender_core_type() == SenderCoreType::Dram,
                "attention_qkv_decode: prefetcher pipe {} must have a DRAM (tensor prefetcher) sender",
                p);
            uint32_t role = 0;
            while (role < kNumWeightRoles &&
                   !weight_role_cores(static_cast<WeightRole>(role)).contains(pipe.receiver_cores())) {
                ++role;
            }
            TT_FATAL(
                role < kNumWeightRoles,
                "attention_qkv_decode: prefetcher pipe {}'s receivers {} lie in none of the q_a, kv or q_b cores",
                p,
                pipe.receiver_cores().str());
            TT_FATAL(
                pipe.ring_size() >= slab_bytes[role],
                "attention_qkv_decode: prefetcher pipe {} holds {} B per core, less than its {} B weight slab",
                p,
                pipe.ring_size(),
                slab_bytes[role]);
            covered[role] = covered[role].merge(pipe.receiver_cores());
        }
        for (uint32_t role = 0; role < kNumWeightRoles; ++role) {
            const CoreRangeSet cores = weight_role_cores(static_cast<WeightRole>(role));
            TT_FATAL(
                covered[role].num_cores() == cores.num_cores(),
                "attention_qkv_decode: the prefetcher pipes must reach every core of {}, got {}",
                cores.str(),
                covered[role].str());
        }
    }

    const uint32_t rope_dim = t.cos.logical_shape()[-1];
    for (const Tensor* x : {&t.cos, &t.sin}) {
        TT_FATAL(
            x->layout() == Layout::TILE && x->dtype() == DataType::BFLOAT16 &&
                x->tensor_spec().tile().get_height() == tt::constants::TILE_HEIGHT && !x->is_sharded() &&
                x->logical_shape()[-1] == rope_dim && x->logical_shape()[-2] == 1,
            "attention_qkv_decode: cos / sin must be interleaved bfloat16 32x32 TILE [1, 1, 1, rope_dim]");
    }
    TT_FATAL(
        rope_dim == 2 * tt::constants::TILE_WIDTH && nc_tiles >= rope_dim / tt::constants::TILE_WIDTH,
        "attention_qkv_decode: rope_dim must be 64 (the trailing two tiles of a head), got {}",
        rope_dim);

    const auto& cache = t.kv_cache;
    TT_FATAL(
        cache.layout() == Layout::TILE && cache.dtype() == DataType::BFLOAT16 && !cache.is_sharded() &&
            cache.tensor_spec().tile().get_height() == tt::constants::TILE_HEIGHT &&
            cache.logical_shape()[-1] == head_dim,
        "attention_qkv_decode: kv_cache must be an interleaved bfloat16 32x32 TILE [.., rows, {}]",
        head_dim);
    TT_FATAL(
        t.cache_pos.dtype() == DataType::INT32 && t.cache_pos.layout() == Layout::ROW_MAJOR &&
            !t.cache_pos.is_sharded(),
        "attention_qkv_decode: cache_pos must be an interleaved ROW_MAJOR int32 [1]");

    const auto& out_mem = args.output_mem_config;
    TT_FATAL(
        out_mem.buffer_type() == BufferType::L1 || out_mem.buffer_type() == BufferType::DRAM,
        "attention_qkv_decode: output memory config must be L1 or DRAM");
}

AttentionQkvDecodeDeviceOperation::spec_return_value_t AttentionQkvDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& t) {
    const uint32_t head_dim = t.w_kv.logical_shape()[-1];
    const uint32_t heads = t.w_q_b.logical_shape()[-1] / head_dim;
    return TensorSpec(
        ttnn::Shape({1, 1, heads, head_dim}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), args.output_mem_config));
}

AttentionQkvDecodeDeviceOperation::tensor_return_value_t AttentionQkvDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& t) {
    return ttnn::create_device_tensor(compute_output_specs(args, t), t.tokens.device());
}

}  // namespace ttnn::operations::experimental::deepseek::attention_qkv_decode

namespace ttnn::prim {

ttnn::Tensor attention_qkv_decode(
    const ttnn::Tensor& tokens,
    const ttnn::Tensor& w_q_a,
    const ttnn::Tensor& w_kv,
    const ttnn::Tensor& w_q_b,
    const ttnn::Tensor& q_a_gamma,
    const ttnn::Tensor& kv_gamma,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& kv_cache,
    const ttnn::Tensor& cache_pos,
    float epsilon,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const ttnn::PrefetcherPipeList& prefetcher_pipes,
    const std::optional<ttnn::Tensor>& w_comp,
    const std::optional<ttnn::Tensor>& comp_output) {
    using OperationType =
        ttnn::operations::experimental::deepseek::attention_qkv_decode::AttentionQkvDecodeDeviceOperation;
    using namespace ttnn::operations::experimental::deepseek::attention_qkv_decode;
    const uint32_t head_dim = w_kv.logical_shape()[-1];
    const uint32_t heads = w_q_b.logical_shape()[-1] / head_dim;
    // flash_mqa_decode's own layout for its q: every head on the first core.
    const tt::tt_metal::MemoryConfig default_mem(
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec(
            CoreRangeSet(CoreRange({0, 0}, {0, 0})), {heads, head_dim}, tt::tt_metal::ShardOrientation::ROW_MAJOR));
    auto attrs = OperationType::operation_attributes_t{
        .epsilon = epsilon,
        .output_mem_config = memory_config.value_or(default_mem),
        .prefetcher_pipes = prefetcher_pipes,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .tokens = tokens,
        .w_q_a = w_q_a,
        .w_kv = w_kv,
        .w_q_b = w_q_b,
        .q_a_gamma = q_a_gamma,
        .kv_gamma = kv_gamma,
        .cos = cos,
        .sin = sin,
        .kv_cache = kv_cache,
        .cache_pos = cache_pos,
        .w_comp = w_comp,
        .comp_output = comp_output,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
