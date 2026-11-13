// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_device_operation.hpp"

#include <algorithm>
#include <array>
#include <set>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "d2d_routes.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

FusedExpertsDeviceOperation::program_factory_t FusedExpertsDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    if (operation_attributes.all_reduce.has_value()) {
        return MultiCoreAllReduce{};
    }
    return MultiCore{};
}

void FusedExpertsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.input_tensor;

    TT_FATAL(x.storage_type() == StorageType::DEVICE, "fused_experts: input_tensor must be on device");
    TT_FATAL(
        x.layout() == tt::tt_metal::Layout::TILE || x.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "fused_experts: input_tensor must be TILE or ROW_MAJOR, got {}",
        x.layout());

    // Two ROW_MAJOR shapes are accepted. Interleaved (the decode path that came off DRAM): one token
    // row is broadcast to every core by the {1,0} input sender, several rows are read by every core's
    // writer. Replicated onto every compute core in L1 (the `all_gather_for_matmul` replica that
    // matmul_decode also consumes in place): each core already holds the rows, so there is nothing
    // to read or broadcast -- see fused_experts_input_is_replicated.
    const bool input_replicated = fused_experts_input_is_replicated(x);
    const uint32_t batch = fused_experts_input_rows(x);
    if (x.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        TT_FATAL(
            batch >= 1 && batch <= tt::constants::TILE_HEIGHT,
            "fused_experts: ROW_MAJOR input rows must share one tile row, so there must be 1 to {}, got {}",
            tt::constants::TILE_HEIGHT,
            batch);
        TT_FATAL(
            x.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "fused_experts: ROW_MAJOR input must be BFLOAT16 (its rows are 32-element bf16 faces), got {}",
            x.dtype());
        TT_FATAL(
            input_replicated || batch == 1 || !x.memory_config().is_sharded(),
            "fused_experts: a several-row ROW_MAJOR input must be interleaved or the L1 HEIGHT_SHARDED replica, "
            "got {}",
            x.memory_config());
    }
    if (input_replicated) {
        const auto& in_shard = x.memory_config().shard_spec().value();
        TT_FATAL(
            in_shard.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR,
            "fused_experts: replicated input requires ROW_MAJOR shard orientation, got {}",
            in_shard.orientation);
        TT_FATAL(
            in_shard.shape[0] == batch,
            "fused_experts: replicated input shard height ({}) must equal the token row count ({})",
            in_shard.shape[0],
            batch);
        TT_FATAL(
            static_cast<uint32_t>(x.logical_shape()[-1]) == in_shard.shape[1],
            "fused_experts: replicated input shard width ({}) must equal the hidden dim ({}): the whole row "
            "is replicated on every core",
            in_shard.shape[1],
            x.logical_shape()[-1]);
        // A one-row shard is handed to the compute as k_tiles 1x32 tiles by aliasing cb_input over the
        // buffer, and several rows are restriped 32 columns at a time, so either way a row has to be a
        // whole number of tile widths and carry no padding.
        TT_FATAL(
            in_shard.shape[1] % tt::constants::TILE_WIDTH == 0,
            "fused_experts: replicated input row width ({}) must be divisible by the 1x32 tile width ({})",
            in_shard.shape[1],
            tt::constants::TILE_WIDTH);
        TT_FATAL(
            x.padded_shape()[-1] == x.logical_shape()[-1],
            "fused_experts: replicated input padded width ({}) must equal the logical width ({}) so the "
            "replica is one contiguous row per core",
            x.padded_shape()[-1],
            x.logical_shape()[-1]);
    }

    const uint32_t num_weights_arg = static_cast<uint32_t>(tensor_args.gate_up_weights.size());

    // Exactly one selection source: the router's ids, the ranking score row the op top-k's itself, or
    // the router weight the op runs itself. More than one would be ambiguous (which one wins?) and
    // none leaves the op no way to know which experts to run.
    const bool run_router = tensor_args.router_weight.has_value();
    TT_FATAL(
        static_cast<int>(tensor_args.routing_indices.has_value()) +
                static_cast<int>(tensor_args.ranking_scores.has_value()) + static_cast<int>(run_router) ==
            1,
        "fused_experts: exactly one of routing_indices (the router's selected ids), ranking_scores (an E-wide row "
        "to top-k on device) or router_weight (the router, run inside the op) must be provided");
    TT_FATAL(
        tensor_args.routing_scores.has_value() != run_router,
        "fused_experts: routing_scores is required with routing_indices / ranking_scores and must be absent with "
        "router_weight (the op computes the scores itself)");
    TT_FATAL(
        tensor_args.router_bias.has_value() == run_router,
        "fused_experts: router_weight and router_bias must be given together");

    if (tensor_args.routing_scores.has_value()) {
        const auto& scores = *tensor_args.routing_scores;
        TT_FATAL(scores.storage_type() == StorageType::DEVICE, "fused_experts: routing_scores must be on device");

        // Scores may be TILE (prefill) or ROW_MAJOR (decode), where each token's LinearDecode stick is
        // a linear [1, E] page and is indexed as b * page_stride + e*2 rather than tilized.
        TT_FATAL(
            scores.layout() == tt::tt_metal::Layout::TILE || scores.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "fused_experts: routing_scores must be TILE or ROW_MAJOR, got {}",
            scores.layout());
        TT_FATAL(
            scores.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "fused_experts: routing_scores must be BFLOAT16, got {}",
            scores.dtype());
        // The leader kernel reads TILE scores with a hand-rolled `tile_elem_offset` that bakes in
        // face_r_dim == 16 and the (16x16 face, 4 faces per tile) 32x32 layout, so require 32x32
        // explicitly. ROW_MAJOR decode scores are a linear stick and skip this check.
        if (scores.layout() == tt::tt_metal::Layout::TILE) {
            TT_FATAL(
                scores.tensor_spec().tile().get_height() == tt::constants::TILE_HEIGHT &&
                    scores.tensor_spec().tile().get_width() == tt::constants::TILE_WIDTH,
                "fused_experts: TILE routing_scores must use the standard {}x{} tile (leader reads them "
                "with a 16x16-face-baked offset function); got {}x{}",
                tt::constants::TILE_HEIGHT,
                tt::constants::TILE_WIDTH,
                scores.tensor_spec().tile().get_height(),
                scores.tensor_spec().tile().get_width());
        }
        // Pages are read by page id, so both selection tensors must be interleaved rather than
        // sharded.
        TT_FATAL(
            !scores.memory_config().is_sharded(),
            "fused_experts: routing_scores must be interleaved (read by page id)");

        TT_FATAL(
            static_cast<uint32_t>(scores.logical_shape()[-1]) == num_weights_arg,
            "fused_experts: routing_scores last dim ({}) must equal the number of experts ({})",
            scores.logical_shape()[-1],
            num_weights_arg);
        TT_FATAL(
            static_cast<uint32_t>(scores.logical_shape()[-2]) == batch,
            "fused_experts: routing_scores must have one row per token ({}), got {}",
            batch,
            scores.logical_shape()[-2]);

        if (tensor_args.routing_indices.has_value()) {
            const auto& idx = *tensor_args.routing_indices;
            TT_FATAL(idx.storage_type() == StorageType::DEVICE, "fused_experts: routing_indices must be on device");

            // Ids stay TILE: topk / embedding emit them that way and the kernel walks 16x16 faces to
            // reach a token's row.
            TT_FATAL(
                idx.layout() == tt::tt_metal::Layout::TILE,
                "fused_experts: routing_indices must be TILE layout (the topk index output, unmodified)");
            // UINT16 is what ttnn.topk emits. BFLOAT16 is accepted because ttnn.embedding only
            // gathers from a bfloat16 table, and that is how a table-driven router (the hash
            // router's frozen tid2eid) hands over its ids; every expert id below 256 is exactly
            // representable in bf16, so nothing is lost.
            TT_FATAL(
                idx.dtype() == tt::tt_metal::DataType::UINT16 || idx.dtype() == tt::tt_metal::DataType::BFLOAT16,
                "fused_experts: routing_indices must be UINT16 (a topk index output) or BFLOAT16 (an "
                "embedding gather), got {}",
                idx.dtype());
            TT_FATAL(
                idx.dtype() != tt::tt_metal::DataType::BFLOAT16 || num_weights_arg <= 256,
                "fused_experts: BFLOAT16 routing_indices need num_experts <= 256 so every id is exact "
                "in bf16, got {}",
                num_weights_arg);
            // Unlike x_tok, routing ids are NOT sized to the input's tile: the leader kernel
            // (compute_expert_ids.cpp) reads them byte-wise via a hand-rolled tile_elem_offset that
            // bakes in face_r_dim == 16 and the (16x16 face, 4 faces per tile) 32x32 layout. A tiny
            // routing-id tile would compute wrong byte offsets and read garbage silently, so require
            // 32x32 explicitly. (Regenerating them at 32x32 costs nothing -- the tile carries at most
            // B*top_k <= 32*16 elements either way, and cb_routing is dominated by the O(E) selection
            // scratch, not the routing tile itself.)
            TT_FATAL(
                idx.tensor_spec().tile().get_height() == tt::constants::TILE_HEIGHT &&
                    idx.tensor_spec().tile().get_width() == tt::constants::TILE_WIDTH,
                "fused_experts: routing_indices must use the standard {}x{} tile (leader reads them "
                "with a 16x16-face-baked offset function); got {}x{}",
                tt::constants::TILE_HEIGHT,
                tt::constants::TILE_WIDTH,
                idx.tensor_spec().tile().get_height(),
                idx.tensor_spec().tile().get_width());
            TT_FATAL(
                !idx.memory_config().is_sharded(),
                "fused_experts: routing_indices must be interleaved (read by page id)");
            // The ids sit in the first `top_k` columns of a token's row, so they must all land in the
            // first 16-wide face of the tile -- the only face the kernel reads for them.
            TT_FATAL(
                attributes.top_k <= 16,
                "fused_experts: top_k ({}) must be <= 16 (the ids must fit one 16-wide tile face)",
                attributes.top_k);
            TT_FATAL(
                static_cast<uint32_t>(idx.logical_shape()[-1]) == attributes.top_k,
                "fused_experts: routing_indices last dim ({}) must equal top_k ({})",
                idx.logical_shape()[-1],
                attributes.top_k);
            TT_FATAL(
                static_cast<uint32_t>(idx.logical_shape()[-2]) == batch,
                "fused_experts: routing_indices must have one row per token ({}), got {}",
                batch,
                idx.logical_shape()[-2]);
        } else {
            const auto& ranking = *tensor_args.ranking_scores;
            TT_FATAL(ranking.storage_type() == StorageType::DEVICE, "fused_experts: ranking_scores must be on device");
            TT_FATAL(
                ranking.layout() == tt::tt_metal::Layout::TILE || ranking.layout() == tt::tt_metal::Layout::ROW_MAJOR,
                "fused_experts: ranking_scores must be TILE or ROW_MAJOR, got {}",
                ranking.layout());
            TT_FATAL(
                ranking.dtype() == tt::tt_metal::DataType::BFLOAT16,
                "fused_experts: ranking_scores must be BFLOAT16, got {}",
                ranking.dtype());
            if (ranking.layout() == tt::tt_metal::Layout::TILE) {
                TT_FATAL(
                    ranking.tensor_spec().tile().get_height() == tt::constants::TILE_HEIGHT &&
                        ranking.tensor_spec().tile().get_width() == tt::constants::TILE_WIDTH,
                    "fused_experts: TILE ranking_scores must use the standard {}x{} tile (leader reads them "
                    "with a 16x16-face-baked offset function); got {}x{}",
                    tt::constants::TILE_HEIGHT,
                    tt::constants::TILE_WIDTH,
                    ranking.tensor_spec().tile().get_height(),
                    ranking.tensor_spec().tile().get_width());
            }
            TT_FATAL(
                !ranking.memory_config().is_sharded(),
                "fused_experts: ranking_scores must be interleaved (read by page id)");
            // The leader keeps the running top-k in its own registers/stack (one (score, id) pair per
            // selected expert), so the ranking path is not bounded by the 16-wide tile face the id
            // path is; 32 keeps those arrays small on the RISC-V stack.
            TT_FATAL(
                attributes.top_k <= 32,
                "fused_experts: top_k ({}) must be <= 32 with ranking_scores (the leader's running top-k "
                "lives on the RISC-V stack)",
                attributes.top_k);
            TT_FATAL(
                static_cast<uint32_t>(ranking.logical_shape()[-1]) == num_weights_arg,
                "fused_experts: ranking_scores last dim ({}) must equal the number of experts ({})",
                ranking.logical_shape()[-1],
                num_weights_arg);
            TT_FATAL(
                static_cast<uint32_t>(ranking.logical_shape()[-2]) == batch,
                "fused_experts: ranking_scores must have one row per token ({}), got {}",
                batch,
                ranking.logical_shape()[-2]);
        }
    }

    if (run_router) {
        const auto& gate = *tensor_args.router_weight;
        const auto& bias = *tensor_args.router_bias;
        const uint32_t hidden_dim = static_cast<uint32_t>(x.logical_shape()[-1]);
        // The router cores read the activation from the same one-row cb_input the expert cores use: the
        // replica it is aliased over, or the {1,0} broadcast's copy.
        TT_FATAL(
            batch == 1,
            "fused_experts: router_weight runs on one token row, got {} rows ({})",
            batch,
            x.memory_config());
        TT_FATAL(gate.storage_type() == StorageType::DEVICE, "fused_experts: router_weight must be on device");
        TT_FATAL(gate.layout() == tt::tt_metal::Layout::TILE, "fused_experts: router_weight must be TILE layout");
        TT_FATAL(
            gate.buffer()->buffer_type() == tt::tt_metal::BufferType::L1 ||
                gate.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "fused_experts: router_weight must be ND-sharded in L1 (over the router cores) or DRAM");
        TT_FATAL(
            num_weights_arg % tt::constants::TILE_WIDTH == 0,
            "fused_experts: router_weight needs a whole number of 32-expert tile columns, got {} experts",
            num_weights_arg);
        if (attributes.persistent.has_value()) {
            // The persistent op's router is sharded by expert over its chips: each chip's router_weight is its own
            // [hidden, 32] tile column of the gate (see FusedPersistentParams), K-split over its partial cores.
            TT_FATAL(
                static_cast<uint32_t>(gate.logical_shape()[-2]) == hidden_dim &&
                    static_cast<uint32_t>(gate.logical_shape()[-1]) == tt::constants::TILE_WIDTH &&
                    num_weights_arg == tt::constants::TILE_WIDTH * x.device()->shape().mesh_size(),
                "fused_experts: the persistent op's router_weight is each chip's [{}, {}] column of the gate, {} "
                "experts over {} chips; got [{}, {}]",
                hidden_dim,
                tt::constants::TILE_WIDTH,
                num_weights_arg,
                x.device()->shape().mesh_size(),
                gate.logical_shape()[-2],
                gate.logical_shape()[-1]);
        } else {
            TT_FATAL(
                static_cast<uint32_t>(gate.logical_shape()[-2]) == hidden_dim &&
                    static_cast<uint32_t>(gate.logical_shape()[-1]) == num_weights_arg,
                "fused_experts: router_weight must be [{}, {}] (hidden x experts), got [{}, {}]",
                hidden_dim,
                num_weights_arg,
                gate.logical_shape()[-2],
                gate.logical_shape()[-1]);
            const auto& nd = gate.memory_config().nd_shard_spec();
            TT_FATAL(
                nd.has_value() && static_cast<uint32_t>(nd->shard_shape[-2]) == hidden_dim &&
                    static_cast<uint32_t>(nd->shard_shape[-1]) == tt::constants::TILE_WIDTH,
                "fused_experts: router_weight must be ND-sharded [{}, {}] (one 32-expert tile column per shard)",
                hidden_dim,
                tt::constants::TILE_WIDTH);
        }
        TT_FATAL(bias.storage_type() == StorageType::DEVICE, "fused_experts: router_bias must be on device");
        TT_FATAL(
            bias.layout() == tt::tt_metal::Layout::ROW_MAJOR && bias.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "fused_experts: router_bias must be ROW_MAJOR BFLOAT16, got {} {}",
            bias.layout(),
            bias.dtype());
        TT_FATAL(
            bias.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM && !bias.memory_config().is_sharded(),
            "fused_experts: router_bias must be DRAM interleaved (read as one page)");
        TT_FATAL(
            static_cast<uint32_t>(bias.logical_shape()[-1]) == num_weights_arg &&
                bias.logical_shape().volume() == num_weights_arg,
            "fused_experts: router_bias must be one row of {} experts, got {}",
            num_weights_arg,
            bias.logical_shape());
        // Each router core ranks at least one 32-expert tile and hands the leader its k best.
        TT_FATAL(
            attributes.top_k <= tt::constants::TILE_WIDTH,
            "fused_experts: top_k ({}) must be <= {} with router_weight",
            attributes.top_k,
            tt::constants::TILE_WIDTH);
    }

    TT_FATAL(
        attributes.top_k > 0 && attributes.top_k <= num_weights_arg,
        "fused_experts: top_k ({}) must be in [1, {}] (with ranking_scores or router_weight it must be passed "
        "explicitly: there is no id tensor to infer it from)",
        attributes.top_k,
        num_weights_arg);

    TT_FATAL(
        tensor_args.gate_up_weights.size() == tensor_args.down_weights.size(),
        "fused_experts: gate_up_weights ({}) and down_weights ({}) must have the same length",
        tensor_args.gate_up_weights.size(),
        tensor_args.down_weights.size());

    const uint32_t num_weights = num_weights_arg;
    TT_FATAL(num_weights > 0, "fused_experts: need at least one expert");

    // The op takes all `num_weights` experts and runs the gate_up matmul only for the
    // `num_experts` routing-selected ("hit") experts; the caller must pass the actual hit count,
    // i.e. the size of the *union* of the token rows' selections, because every hit expert is
    // evaluated for every row and rows that did not select it contribute through a zero weight.
    // That size is data dependent -- it grows with the batch, and shrinks if a table-driven router
    // names an expert twice for one token -- so it cannot be checked more tightly here than
    // against the expert count. A bound below the actual union drops the surplus (see the kernel).
    TT_FATAL(
        attributes.num_experts > 0 && attributes.num_experts <= num_weights,
        "fused_experts: num_experts ({}) must be in [1, {}]",
        attributes.num_experts,
        num_weights);

    // Experts run in blocks of this many, and only a block's activations are held in L1 at once, so
    // this -- not num_experts -- is what has to fit. `invoke` has already resolved 0 and clamped it
    // to num_experts, so anything outside the range means it was set directly on the attributes.
    TT_FATAL(
        attributes.experts_block_size > 0 && attributes.experts_block_size <= attributes.num_experts,
        "fused_experts: experts_block_size ({}) must be in [1, num_experts ({})]",
        attributes.experts_block_size,
        attributes.num_experts);

    // One DRAM weight shard per core of the serial grid (8 rows of H/512 columns: 8x8 at H == 4096);
    // compute uses 96 cores when num_experts == 6. Each shard owns 64 columns of the H output row.
    constexpr uint32_t TILE_DIM = 32;
    constexpr uint32_t kDownShardCols = 2u * TILE_DIM;
    constexpr uint32_t kGridRows = 8;
    const uint32_t hidden = static_cast<uint32_t>(x.logical_shape()[-1]);
    TT_FATAL(
        hidden % (kDownShardCols * kGridRows) == 0,
        "fused_experts: hidden dim ({}) must fill whole {}-row grid columns of {}-column shards",
        hidden,
        kGridRows,
        kDownShardCols);
    const uint32_t num_weight_shards = hidden / kDownShardCols;

    // gate_up weights must be DRAM ND-sharded so that each shard is exactly one I-tile of gate
    // plus its paired up tile (read in a single NoC read). The SwiGLU I dim is one 32-column
    // tile per shard, so a shard holds [gate_32 | up_32] and the weight is permuted on the
    // host into matching per-shard [gate | up] blocks. At I == 2048 that is 64 shards; TP's
    // smaller local I yields fewer shards, which the 16-core groups still cover 1-for-1.
    const uint32_t i_tiles = attributes.intermediate_size / TILE_DIM;
    const uint32_t swiglu_tiles_per_core = std::max<uint32_t>(1u, i_tiles / num_weight_shards);
    const uint32_t kColsPerCore = 2u * TILE_DIM * swiglu_tiles_per_core;
    // Phase 1 holds gate and up for this core's slice in DST at once, which fits 4 fp32 tiles.
    TT_FATAL(
        2u * swiglu_tiles_per_core <= 4u,
        "fused_experts: intermediate_size ({}) gives {} SwiGLU tiles per core, but the gate+up "
        "SwiGLU pass holds 2x that in DST (max 4 fp32 tiles)",
        attributes.intermediate_size,
        swiglu_tiles_per_core);
    for (uint32_t e = 0; e < num_weights; ++e) {
        const auto& w = tensor_args.gate_up_weights[e];
        TT_FATAL(
            w.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "fused_experts: gate_up_weights[{}] must be in DRAM",
            e);
        const auto& nd = w.memory_config().nd_shard_spec();
        TT_FATAL(nd.has_value(), "fused_experts: gate_up_weights[{}] must be ND-sharded (one shard per core)", e);
        const auto& shard_shape = nd->shard_shape;
        TT_FATAL(
            static_cast<uint32_t>(shard_shape[-1]) == kColsPerCore,
            "fused_experts: gate_up_weights[{}] shard last dim ({}) must be {} (one core's [gate | up] slice)",
            e,
            shard_shape[-1],
            kColsPerCore);
        // A shard spans the full K dim, or an equal whole-tile piece of it (K-shards, read back to back).
        const uint32_t k_rows = static_cast<uint32_t>(w.logical_shape()[-2]);
        const uint32_t shard_rows = static_cast<uint32_t>(shard_shape[-2]);
        TT_FATAL(
            shard_rows % TILE_DIM == 0 && k_rows % shard_rows == 0,
            "fused_experts: gate_up_weights[{}] shard rows ({}) must be a whole-tile divisor of the K dim ({})",
            e,
            shard_rows,
            k_rows);
        TT_FATAL(
            shard_rows == static_cast<uint32_t>(
                              tensor_args.gate_up_weights.front().memory_config().nd_shard_spec()->shard_shape[-2]),
            "fused_experts: gate_up_weights[{}] shard rows ({}) must match gate_up_weights[0]'s -- the readers share "
            "one layout",
            e,
            shard_rows);
    }

    // down weights must be DRAM ND-sharded so each shard is exactly one core's [I, 64] column
    // slice (read in a single NoC read). Each shard spans the full I (contraction) dim and one
    // core's 64-column H output slice; H/64 shards cover the output H dim.
    const uint32_t down_shard_cols = kDownShardCols;
    for (uint32_t e = 0; e < num_weights; ++e) {
        const auto& w = tensor_args.down_weights[e];
        TT_FATAL(
            w.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "fused_experts: down_weights[{}] must be in DRAM",
            e);
        const auto& nd = w.memory_config().nd_shard_spec();
        TT_FATAL(nd.has_value(), "fused_experts: down_weights[{}] must be ND-sharded (one shard per core)", e);
        const auto& shard_shape = nd->shard_shape;
        TT_FATAL(
            static_cast<uint32_t>(w.logical_shape()[-2]) == attributes.intermediate_size,
            "fused_experts: down_weights[{}] K dim ({}) must equal intermediate_size ({})",
            e,
            w.logical_shape()[-2],
            attributes.intermediate_size);
        TT_FATAL(
            static_cast<uint32_t>(w.logical_shape()[-1]) == hidden,
            "fused_experts: down_weights[{}] output dim ({}) must equal hidden ({})",
            e,
            w.logical_shape()[-1],
            hidden);
        TT_FATAL(
            static_cast<uint32_t>(shard_shape[-1]) == down_shard_cols,
            "fused_experts: down_weights[{}] shard last dim ({}) must be {} (one core's 64-column H slice)",
            e,
            shard_shape[-1],
            down_shard_cols);
        TT_FATAL(
            static_cast<uint32_t>(shard_shape[-2]) == static_cast<uint32_t>(w.logical_shape()[-2]),
            "fused_experts: down_weights[{}] shard must span the full I dim ({} rows), got {}",
            e,
            w.logical_shape()[-2],
            shard_shape[-2]);
    }

    // Token rows (batch): activations are [1, 1, B, H] -- the B tokens of a batched decode step (or
    // of a short prefill chunk) are packed into dim -2, each with its own routing-weight row, and
    // all B are computed in one op.
    //
    // The token-row-shaped pipeline (input, output, cb_act, cb_out, cb_mm, cb_rscalar, cb_down_out,
    // cb_acc, cb_wtmp) all track the input's tile height, so B is capped at that tile height rather
    // than a fixed 32. Weights and routing stay 32x32. The batch/seq dims of a [1, B, S, H]
    // activation must still be folded into dim -2 (B*S tokens) by the caller: batching pays off here
    // only because ONE fetch of an expert's weights serves every token, which requires every token's
    // activation to be resident on every core simultaneously. Tokens sharing one tile row is what
    // keeps the tile counts, the DST budget, the L1 footprint and the matmul cost identical to the
    // single-token case; tokens spread over separate tile rows (e.g. an unfolded [1, B, 1, H]) would
    // multiply both the resident activation (k_tiles tiles) and the gathered activation block
    // (num_experts * i_tiles tiles, already the dominant L1 consumer) by B.
    const auto input_tile = fused_experts_compute_tile(x);
    const uint32_t input_tile_h = input_tile.get_height();
    const uint32_t input_tile_w = input_tile.get_width();
    TT_FATAL(
        input_tile_w == tt::constants::TILE_WIDTH,
        "fused_experts: input_tensor tile width ({}) must be {} -- the token-row tile only varies "
        "in height; width is fixed by the 16x16 face layout the routing / weight kernels use",
        input_tile_w,
        tt::constants::TILE_WIDTH);
    // Only tile heights whose face layout is well-defined (num_faces along H is a whole number of
    // face rows): height <= 16 (one face row) or height == 32 (two face rows). Sizes in between
    // would leave a partial second face row that the kernel-side layout math does not handle.
    TT_FATAL(
        input_tile_h == tt::constants::TILE_HEIGHT || input_tile_h <= tt::constants::FACE_HEIGHT,
        "fused_experts: input_tensor tile height ({}) must be either {} or <= {} (partial second "
        "face row is not supported)",
        input_tile_h,
        tt::constants::TILE_HEIGHT,
        tt::constants::FACE_HEIGHT);
    TT_FATAL(
        batch >= 1 && batch <= input_tile_h,
        "fused_experts: token rows (input_tensor dim -2, or the shard height for a replicated ROW_MAJOR "
        "input) must be in [1, tile height {}], got {}",
        input_tile_h,
        batch);
    const auto& x_shape = x.logical_shape();
    for (int d = 0; d + 2 < static_cast<int>(x_shape.rank()); ++d) {
        TT_FATAL(
            x_shape[d] == 1,
            "fused_experts: input_tensor dim {} must be 1 (got {}); the tokens of a [1, B, S, H] "
            "activation must be folded into dim -2 as [1, 1, B*S, H] so they share one tile row",
            d,
            x_shape[d]);
    }
    // Matmul contraction dim: input H must match gate_up K (rows).
    TT_FATAL(
        static_cast<uint32_t>(x.logical_shape()[-1]) ==
            static_cast<uint32_t>(tensor_args.gate_up_weights.front().logical_shape()[-2]),
        "fused_experts: input hidden dim ({}) must equal gate_up K ({})",
        x.logical_shape()[-1],
        tensor_args.gate_up_weights.front().logical_shape()[-2]);

    // SwiGLU splits the gate_up output (2I) into gate/up halves of size I each.
    const uint32_t two_intermediate = static_cast<uint32_t>(tensor_args.gate_up_weights.front().logical_shape()[-1]);
    TT_FATAL(
        two_intermediate == 2u * attributes.intermediate_size,
        "fused_experts: gate_up output dim ({}) must equal 2 * intermediate_size ({})",
        two_intermediate,
        2u * attributes.intermediate_size);

    TT_FATAL(
        tensor_args.shared_gate_up_weights.has_value() == tensor_args.shared_down_weights.has_value(),
        "fused_experts: shared_gate_up_weights and shared_down_weights must be given together");
    if (tensor_args.shared_gate_up_weights.has_value()) {
        const auto& gu = *tensor_args.shared_gate_up_weights;
        const auto& dn = *tensor_args.shared_down_weights;
        for (const Tensor* w : {&gu, &dn}) {
            TT_FATAL(
                w->storage_type() == StorageType::DEVICE, "fused_experts: shared expert weights must be on device");
            TT_FATAL(
                w->layout() == tt::tt_metal::Layout::TILE, "fused_experts: shared expert weights must be TILE layout");
            TT_FATAL(
                w->buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
                "fused_experts: shared expert weights must be in DRAM");
            TT_FATAL(
                w->memory_config().nd_shard_spec().has_value(),
                "fused_experts: shared expert weights must be ND-sharded");
        }
        TT_FATAL(
            static_cast<uint32_t>(gu.logical_shape()[-2]) == hidden,
            "fused_experts: shared_gate_up_weights K ({}) must equal hidden ({})",
            gu.logical_shape()[-2],
            hidden);
        const uint32_t shared_two_i = static_cast<uint32_t>(gu.logical_shape()[-1]);
        TT_FATAL(
            shared_two_i > 0 && shared_two_i % (2u * TILE_DIM) == 0,
            "fused_experts: shared_gate_up_weights width ({}) must be a positive multiple of {} ([gate | up] tile "
            "pairs)",
            shared_two_i,
            2u * TILE_DIM);
        const auto& gu_shard = gu.memory_config().nd_shard_spec()->shard_shape;
        TT_FATAL(
            static_cast<uint32_t>(gu_shard[-2]) == hidden && static_cast<uint32_t>(gu_shard[-1]) == 2u * TILE_DIM,
            "fused_experts: shared_gate_up_weights shard must be [{}, {}] (one I-tile's [gate | up]), got [{}, {}]",
            hidden,
            2u * TILE_DIM,
            gu_shard[-2],
            gu_shard[-1]);
        TT_FATAL(
            static_cast<uint32_t>(dn.logical_shape()[-2]) == shared_two_i / 2u &&
                static_cast<uint32_t>(dn.logical_shape()[-1]) == hidden,
            "fused_experts: shared_down_weights must be [{}, {}], got [{}, {}]",
            shared_two_i / 2u,
            hidden,
            dn.logical_shape()[-2],
            dn.logical_shape()[-1]);
        const auto& dn_shard = dn.memory_config().nd_shard_spec()->shard_shape;
        TT_FATAL(
            static_cast<uint32_t>(dn_shard[-2]) == TILE_DIM && static_cast<uint32_t>(dn_shard[-1]) == kDownShardCols,
            "fused_experts: shared_down_weights shard must be [{}, {}] (one I-tile of one H-slice), got [{}, {}]",
            TILE_DIM,
            kDownShardCols,
            dn_shard[-2],
            dn_shard[-1]);
    }

    if (attributes.all_reduce.has_value()) {
        const auto& ar = *attributes.all_reduce;
        const auto& mem = attributes.output_memory_config;
        const auto grid = x.device()->compute_with_storage_grid_size();
        const std::array<uint32_t, 2> ar_shard_shape{1, hidden / kDownCoresPerExpert};
        TT_FATAL(ar.ring_size > 1, "fused_experts: all_reduce needs more than one chip, got {}", ar.ring_size);
        if (!ar.cluster_axis.has_value()) {
            // One pass over a 2D mesh: a 2D fabric multicast reaches every chip, rows and columns at
            // once, which a 1D fabric's single-axis lines cannot.
            const auto& mesh = *x.device();
            TT_FATAL(
                mesh.shape().dims() == 2 && ar.ring_size == mesh.shape().mesh_size(),
                "fused_experts: an all_reduce over both mesh axes needs the input on all {} chips of a 2D mesh, "
                "got {} chips of a {}D mesh",
                mesh.shape().mesh_size(),
                ar.ring_size,
                mesh.shape().dims());
            TT_FATAL(
                tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig()),
                "fused_experts: an all_reduce over both mesh axes needs a 2D fabric");
            TT_FATAL(
                ttnn::ccl::is_axis_straight(mesh, 0) && ttnn::ccl::is_axis_straight(mesh, 1),
                "fused_experts: an all_reduce over both mesh axes needs both axes wired straight");
        }
        TT_FATAL(
            ar.num_links >= 1 && ar.num_links <= kDownCoresPerExpert,
            "fused_experts: all_reduce num_links must be in [1, {}], got {}",
            kDownCoresPerExpert,
            ar.num_links);
        TT_FATAL(batch == 1, "fused_experts: all_reduce runs on one token row only, got {}", batch);
        TT_FATAL(
            x.layout() == tt::tt_metal::Layout::ROW_MAJOR,
            "fused_experts: all_reduce needs a ROW_MAJOR input, got {}",
            x.layout());
        TT_FATAL(
            mem.buffer_type() == tt::tt_metal::BufferType::L1 &&
                mem.memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED &&
                mem.shard_spec().has_value() &&
                mem.shard_spec()->grid ==
                    tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {1, kGridY - 1})) &&
                mem.shard_spec()->orientation == tt::tt_metal::ShardOrientation::COL_MAJOR &&
                mem.shard_spec()->shape == ar_shard_shape,
            "fused_experts: all_reduce needs the L1 WIDTH_SHARDED [1, H/16] output on (0,0)-(1,7), COL_MAJOR, got {}",
            mem);
        // The 6-expert path (num_experts == 6 on a >= 12-wide grid) is the only one with that output.
        TT_FATAL(
            attributes.num_experts == kParallelExperts && grid.x >= 12,
            "fused_experts: all_reduce needs the 6-expert path (num_experts {} on a {}-wide grid)",
            attributes.num_experts,
            grid.x);
    }

    TT_FATAL(
        attributes.persistent.has_value() == tensor_args.layer_table.has_value(),
        "fused_experts: the persistent op's parameters and its layer_table come together");
    if (attributes.persistent.has_value()) {
        const auto& persistent = *attributes.persistent;
        const auto& table = *tensor_args.layer_table;
        const auto& mesh = *x.device();
        TT_FATAL(
            attributes.all_reduce.has_value() && !attributes.all_reduce->cluster_axis.has_value(),
            "fused_experts: the persistent op reduces over the whole mesh, so it needs all_reduce over both axes");
        TT_FATAL(
            input_replicated && batch == 1,
            "fused_experts: the persistent op takes its token row into the one-row L1 replica, got {}",
            x.memory_config());
        TT_FATAL(
            tensor_args.router_weight.has_value(),
            "fused_experts: the persistent op runs the router (each layer's gate column from the layer table)");
        TT_FATAL(
            tensor_args.shared_gate_up_weights.has_value(),
            "fused_experts: the persistent op needs the fused shared expert");
        TT_FATAL(persistent.num_layers >= 1, "fused_experts: the persistent op needs at least one layer");
        TT_FATAL(mesh.shape().dims() == 2, "fused_experts: the persistent op runs on a 2D mesh, got {}", mesh.shape());
        const uint32_t table_width = 16u + 2u * static_cast<uint32_t>(tensor_args.gate_up_weights.size());
        TT_FATAL(
            table.dtype() == tt::tt_metal::DataType::UINT32 && table.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
                table.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM &&
                !table.memory_config().is_sharded() && table.logical_shape()[-1] == table_width,
            "fused_experts: layer_table must be [num_layers, {}] UINT32 ROW_MAJOR DRAM interleaved, got {} {} {} {}",
            table_width,
            table.logical_shape(),
            table.dtype(),
            table.layout(),
            table.memory_config());
    }

    // TODO: validate per-expert weight shapes against H / 2I / I, dtypes, and tile alignment.
}

void FusedExpertsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

tt::tt_metal::operation::Hash FusedExpertsDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // CRITICAL: the per-expert gate_up / down weight DRAM base addresses are baked into the
    // dataflow kernels as COMPILE-TIME args (see `append_addrs_ct` in the program factory), so a
    // program compiled for one set of expert weights is only valid for those exact buffers. The
    // default device-op hash keys solely on tensor specs (shape / dtype / layout / memory-config)
    // plus the scalar attributes -- all identical from one MoE layer to the next -- so it would
    // return a stale cached program holding the *previous* layer's (by then possibly freed) weight
    // addresses, making the matmuls read garbage DRAM (observed as ~1e37 / inf outputs). Fold every
    // weight buffer address into the hash so a different set of weight tensors misses the program
    // cache and recompiles with the correct baked-in addresses.
    std::vector<uint32_t> weight_addresses;
    weight_addresses.reserve(tensor_args.gate_up_weights.size() + tensor_args.down_weights.size());
    for (const auto& w : tensor_args.gate_up_weights) {
        weight_addresses.push_back(static_cast<uint32_t>(w.buffer()->address()));
    }
    for (const auto& w : tensor_args.down_weights) {
        weight_addresses.push_back(static_cast<uint32_t>(w.buffer()->address()));
    }
    // top_k / scaling / eps are compile-time args of the routing kernel, so they have to key the
    // program alongside the weight addresses. So does which selection source is in play: with
    // `ranking_scores` the leader is compiled with a different routing body and cb_routing is laid
    // out differently (a second score region when the ranking row is a separate tensor). That is
    // keyed by the two optional specs plus the two explicit flags below -- the tensors alone cannot
    // say whether the ranking row IS the score row, since two tensors with the same spec hash the
    // same.
    // Input tile height is baked into every token-row-shaped CB's tile descriptor and into the
    // routing-scalar layout CT args, so tiny-tile programs must not alias a 32x32 program in the
    // cache. Fold the input tile's H and W explicitly (the default spec-only hash keys on tensor
    // spec, but not on the tile shape field within it). ROW_MAJOR scores switch the leader's
    // score addressing (linear stick vs 32x32 faces), which is already in the scores tensor spec.
    const auto input_tile = fused_experts_compute_tile(tensor_args.input_tensor);
    auto hash = tt::tt_metal::operation::hash_operation<FusedExpertsDeviceOperation>(
        attributes.num_experts,
        attributes.intermediate_size,
        attributes.swiglu_limit,
        attributes.top_k,
        attributes.routed_scaling_factor,
        attributes.routing_eps,
        // Both of these are baked into the kernels as compile-time args (block size and the
        // two-hub split), so they have to key the program as well.
        attributes.experts_block_size,
        attributes.two_hub_gather,
        attributes.output_memory_config,
        tensor_args.input_tensor,
        tensor_args.routing_indices,
        tensor_args.routing_scores,
        tensor_args.ranking_scores,
        // Which selection source is in play, and whether the ranking row IS the score row (the
        // aliased read). The tensor specs alone cannot express the latter: two tensors with the same
        // shape / dtype / layout / memory config hash the same, so an aliased program and a
        // separate-buffer program would collide even though their kernels differ -- the aliased one
        // skips the second read and scans the score region, the other reads a region of its own, and
        // their cb_routing layouts differ. Reusing one for the other selects on the wrong row.
        !tensor_args.routing_indices.has_value(),
        tensor_args.ranking_scores.has_value() && tensor_args.routing_scores.has_value() &&
            tensor_args.ranking_scores->buffer() == tensor_args.routing_scores->buffer(),
        tensor_args.gate_up_weights.front(),
        tensor_args.down_weights.front(),
        weight_addresses,
        input_tile.get_height(),
        input_tile.get_width(),
        tensor_args.routing_scores.has_value() &&
            tensor_args.routing_scores->layout() == tt::tt_metal::Layout::ROW_MAJOR,
        // The shared expert's addresses are runtime args, so only its specs and clamp key the program.
        tensor_args.shared_gate_up_weights,
        tensor_args.shared_down_weights,
        attributes.shared_swiglu_limit,
        // The same goes for the router's weight and bias.
        tensor_args.router_weight,
        tensor_args.router_bias,
        attributes.all_reduce.has_value(),
        attributes.all_reduce.has_value() ? attributes.all_reduce->cluster_axis : std::nullopt,
        attributes.all_reduce.has_value() ? attributes.all_reduce->ring_size : 0u,
        attributes.all_reduce.has_value() ? attributes.all_reduce->num_links : 0u,
        attributes.all_reduce.has_value() ? static_cast<uint32_t>(attributes.all_reduce->topology) : 0u,
        // The persistent op's socket addresses, output route and layer count are compile-time args, and
        // its control core places kernels.
        attributes.persistent.has_value(),
        attributes.persistent.has_value() ? attributes.persistent->num_layers : 0u,
        attributes.persistent.has_value() ? attributes.persistent->socket_core.x : 0u,
        attributes.persistent.has_value() ? attributes.persistent->socket_core.y : 0u,
        attributes.persistent.has_value() ? attributes.persistent->input_socket_config_addresses
                                          : std::vector<uint32_t>{},
        attributes.persistent.has_value() ? attributes.persistent->partials_addresses : std::vector<uint32_t>{},
        attributes.persistent.has_value() ? attributes.persistent->sub_parts : 0u,
        attributes.persistent.has_value() ? attributes.persistent->sum_core_noc_xs
                                          : std::vector<std::vector<uint32_t>>{},
        attributes.persistent.has_value() ? attributes.persistent->sum_core_noc_ys
                                          : std::vector<std::vector<uint32_t>>{},
        attributes.persistent.has_value() ? attributes.persistent->partials_route_args
                                          : std::vector<std::vector<std::vector<uint32_t>>>{},
        attributes.persistent.has_value() ? attributes.persistent->partials_connection_mesh_ids
                                          : std::vector<std::vector<uint32_t>>{},
        attributes.persistent.has_value() ? attributes.persistent->partials_connection_chip_ids
                                          : std::vector<std::vector<uint32_t>>{},
        attributes.persistent.has_value() ? attributes.persistent->partials_connection_index
                                          : std::vector<std::vector<uint32_t>>{},
        tensor_args.layer_table);
    return hash;
}

FusedExpertsDeviceOperation::spec_return_value_t FusedExpertsDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // The output is, per token row, the routing-weighted sum of every selected expert's down
    // matmul result:
    //   act       = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit),
    //               where [gate, up] = x @ gate_up_w[hit_ids[i]];
    //   output[b] = sum_i routing_weights[b, hit_ids[i]] * (act[b] @ down_w[hit_ids[i]]).
    // Shape [1, B, H] (the B token rows, padded to the input's tile height in TILE layout),
    // BFLOAT16. H is the hidden size (== down weight output dim == input hidden dim).
    //
    // The output inherits the input tensor's page config so its tile height matches the input's
    // (rather than defaulting to 32x32). At B <= 16 that leaves the output stored as a single tile
    // row too, so downstream ops can keep the same tiny tile without an intervening retile.
    const auto& input_tensor = tensor_args.input_tensor;
    // The row count, not dim -2: a replicated input carries `num_cores * rows` there.
    const uint32_t batch = fused_experts_input_rows(input_tensor);
    const uint32_t hidden = static_cast<uint32_t>(input_tensor.logical_shape()[-1]);
    const ttnn::Shape output_shape({1, batch, hidden});
    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            input_tensor.tensor_spec().page_config(),
            attributes.output_memory_config));
}

FusedExpertsDeviceOperation::tensor_return_value_t FusedExpertsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

std::tuple<FusedExpertsDeviceOperation::operation_attributes_t, FusedExpertsDeviceOperation::tensor_args_t>
FusedExpertsDeviceOperation::invoke(
    const Tensor& input_tensor,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    uint32_t num_experts,
    uint32_t intermediate_size,
    float swiglu_limit,
    uint32_t top_k,
    float routed_scaling_factor,
    float routing_eps,
    uint32_t experts_block_size,
    bool two_hub_gather,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& routing_scores,
    const std::optional<Tensor>& routing_indices,
    const std::optional<Tensor>& ranking_scores,
    const std::optional<Tensor>& shared_gate_up_weights,
    const std::optional<Tensor>& shared_down_weights,
    std::optional<float> shared_swiglu_limit,
    const std::optional<Tensor>& router_weight,
    const std::optional<Tensor>& router_bias,
    const std::optional<std::vector<uint32_t>>& all_reduce_cluster_axes,
    uint32_t all_reduce_num_links,
    tt::tt_fabric::Topology all_reduce_topology,
    const std::optional<Tensor>& persistent_layer_table,
    const std::optional<tt::tt_metal::CoreCoord>& persistent_socket_core,
    const std::optional<std::vector<tt::tt_metal::distributed::MeshSocket>>& persistent_input_sockets,
    const std::optional<std::vector<Tensor>>& persistent_partials) {
    operation_attributes_t attributes{
        .num_experts = num_experts,
        .intermediate_size = intermediate_size,
        .swiglu_limit = swiglu_limit,
        .shared_swiglu_limit = shared_swiglu_limit.value_or(swiglu_limit),
        // 0 means "no blocking": resolve it here so the attribute (and the program hash) always
        // carries the block size the kernels are actually compiled for.
        .experts_block_size = experts_block_size == 0 ? num_experts : std::min(experts_block_size, num_experts),
        // k defaults to the index tensor's width, so a caller that already shaped the router's
        // output correctly does not have to restate it. The ranking path has no id tensor to read
        // it from, so there it must be given (validation rejects the resulting 0).
        .top_k = top_k == 0 && routing_indices.has_value() ? static_cast<uint32_t>(routing_indices->logical_shape()[-1])
                                                           : top_k,
        .routed_scaling_factor = routed_scaling_factor,
        .routing_eps = routing_eps,
        .two_hub_gather = two_hub_gather,
        // A replicated input is L1 HEIGHT_SHARDED over the compute grid; the [1, B, H] result has no
        // meaningful shard spec of its own (the writer addresses it as one row of pages), so it
        // defaults to DRAM interleaved rather than inheriting the replica's layout.
        .output_memory_config = memory_config.value_or(
            fused_experts_input_is_replicated(input_tensor) ? ttnn::DRAM_MEMORY_CONFIG : input_tensor.memory_config()),
    };
    if (all_reduce_cluster_axes.has_value()) {
        // Only the axes the mesh spans more than one chip along reduce anything; both of a 2D mesh
        // reduce over every chip at once, which is what an unset cluster_axis means.
        const auto mesh_shape = input_tensor.device()->shape();
        std::vector<uint32_t> axes;
        for (const uint32_t axis : *all_reduce_cluster_axes) {
            TT_FATAL(
                axis < mesh_shape.dims(),
                "fused_experts: all_reduce axis {} is out of range for a {}D mesh",
                axis,
                mesh_shape.dims());
            if (mesh_shape[axis] > 1 && std::find(axes.begin(), axes.end(), axis) == axes.end()) {
                axes.push_back(axis);
            }
        }
        TT_FATAL(!axes.empty(), "fused_experts: the all_reduce axes span a single chip of a {} mesh", mesh_shape);
        const std::optional<uint32_t> cluster_axis = axes.size() == 1 ? std::optional<uint32_t>(axes[0]) : std::nullopt;
        attributes.all_reduce = FusedAllReduceParams{
            .cluster_axis = cluster_axis,
            .ring_size = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis),
            .num_links = all_reduce_num_links,
            .topology = all_reduce_topology,
        };
    }
    if (persistent_layer_table.has_value()) {
        using tt::tt_metal::distributed::SocketEndpoint;
        TT_FATAL(
            persistent_socket_core.has_value() && persistent_input_sockets.has_value() &&
                persistent_partials.has_value() && !persistent_input_sockets->empty() &&
                persistent_input_sockets->size() == persistent_partials->size() &&
                persistent_input_sockets->size() <= kMaxPersistentRequesters && attributes.all_reduce.has_value(),
            "fused_experts: the persistent op needs the socket core, the all-reduce, and 1 to {} pairs of input "
            "sockets and partials buffers along with persistent_layer_table",
            kMaxPersistentRequesters);
        const auto& socket_core = *persistent_socket_core;
        auto* mesh = input_tensor.device();
        const uint32_t num_links = attributes.all_reduce->num_links;
        const uint32_t row_bytes = static_cast<uint32_t>(input_tensor.logical_shape()[-1]) * 2u;
        const uint32_t num_chips = static_cast<uint32_t>(mesh->shape().mesh_size());
        TT_FATAL(
            row_bytes % num_links == 0, "fused_experts: a {} B row does not split into {} parts", row_bytes, num_links);
        FusedPersistentParams persistent{
            .num_layers = static_cast<uint32_t>(persistent_layer_table->logical_shape()[-2]),
            .socket_core = socket_core,
            .row_bytes = row_bytes,
            .num_chips = num_chips,
        };
        // Per chip of the op, in its linear index (its slot) order.
        std::vector<ttnn::MeshCoordinate> chips(num_chips, ttnn::MeshCoordinate(0, 0));
        for (const auto& coord : ttnn::MeshCoordinateRange(mesh->shape())) {
            chips.at(ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, coord, std::nullopt)) = coord;
        }
        persistent.partials_route_args.resize(num_chips);
        persistent.partials_connection_mesh_ids.resize(num_chips);
        persistent.partials_connection_chip_ids.resize(num_chips);
        persistent.partials_connection_index.resize(num_chips);
        std::vector<std::vector<tt::tt_fabric::FabricNodeId>> connections(num_chips);
        for (size_t k = 0; k < persistent_input_sockets->size(); ++k) {
            const auto& input_socket = (*persistent_input_sockets)[k];
            const auto& partials = (*persistent_partials)[k];
            TT_FATAL(
                input_socket.get_socket_endpoint_type() == SocketEndpoint::RECEIVER &&
                    input_socket.get_mesh_device() == mesh,
                "fused_experts: requester {} must receive on its input socket on the op's mesh",
                k);
            // Every chip's control core receives the rows.
            std::set<ttnn::MeshCoordinate> receivers;
            for (const auto& c : input_socket.get_config().socket_connection_config) {
                TT_FATAL(
                    c.receiver_core.core_coord == socket_core,
                    "fused_experts: requester {}'s input socket must receive on the socket core {}, not {}",
                    k,
                    socket_core.str(),
                    c.receiver_core.core_coord.str());
                receivers.insert(c.receiver_core.device_coord);
            }
            TT_FATAL(
                receivers.size() == mesh->shape().mesh_size(),
                "fused_experts: requester {}'s input socket must reach every chip of the {} mesh, it reaches {}",
                k,
                mesh->shape(),
                receivers.size());
            // The partials buffer: on every chip of the requester, one L1 shard per sum core -- num_chips slots of
            // that core's columns of a row (the row split evenly over the shards, in order), then a flag word per
            // chip. A link's part of a row covers sub_parts consecutive shards.
            TT_FATAL(
                partials.storage_type() == StorageType::DEVICE && partials.device() != mesh &&
                    partials.memory_config().buffer_type() == tt::tt_metal::BufferType::L1 &&
                    partials.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                "fused_experts: requester {}'s partials must be an L1 HEIGHT_SHARDED tensor on its own mesh",
                k);
            const auto& shard = partials.shard_spec().value();
            const auto sum_cores = corerange_to_cores(shard.grid, std::nullopt, /*row_wise=*/true);
            const uint32_t shard_bytes = partials.buffer()->aligned_size_per_bank();
            const auto num_parts = static_cast<uint32_t>(sum_cores.size());
            TT_FATAL(
                num_parts % num_links == 0 && row_bytes % num_parts == 0 &&
                    (persistent.sub_parts == 0 || persistent.sub_parts == num_parts / num_links),
                "fused_experts: requester {}'s partials have {} shards, which must split the {} links' parts evenly, "
                "alike for every requester",
                k,
                num_parts,
                num_links);
            persistent.sub_parts = num_parts / num_links;
            const uint32_t sub_bytes = row_bytes / num_parts;
            TT_FATAL(
                shard_bytes >= num_chips * (sub_bytes + 4u),
                "fused_experts: requester {}'s partials shards need at least {} B, got {} B",
                k,
                num_chips * (sub_bytes + 4u),
                shard_bytes);
            auto* requester = partials.device();
            persistent.partials_addresses.push_back(static_cast<uint32_t>(partials.buffer()->address()));
            std::vector<uint32_t> xs;
            std::vector<uint32_t> ys;
            for (const auto& core : sum_cores) {
                const auto noc = requester->worker_core_from_logical_core(core);
                xs.push_back(static_cast<uint32_t>(noc.x));
                ys.push_back(static_cast<uint32_t>(noc.y));
            }
            persistent.sum_core_noc_xs.push_back(std::move(xs));
            persistent.sum_core_noc_ys.push_back(std::move(ys));
            // Every chip of the op sends its partial to every chip of the requester.
            std::vector<tt::tt_fabric::FabricNodeId> requester_chips;
            for (const auto& coord : ttnn::MeshCoordinateRange(requester->shape())) {
                requester_chips.push_back(requester->get_fabric_node_id(coord));
            }
            for (uint32_t chip = 0; chip < num_chips; ++chip) {
                const auto routes = d2d_routes(mesh->get_fabric_node_id(chips[chip]), requester_chips);
                TT_FATAL(
                    routes.size() <= kMaxPersistentRoutes,
                    "fused_experts: {}'s partials to requester {} take {} sends, more than the kernel's {}",
                    chips[chip],
                    k,
                    routes.size(),
                    kMaxPersistentRoutes);
                // A requester's sends all leave in one direction (d2d_routes); each direction gets one connection.
                auto& chip_connections = connections[chip];
                auto it = std::find(chip_connections.begin(), chip_connections.end(), routes.front().next);
                if (it == chip_connections.end()) {
                    chip_connections.push_back(routes.front().next);
                    it = chip_connections.end() - 1;
                }
                persistent.partials_connection_index[chip].push_back(
                    static_cast<uint32_t>(it - chip_connections.begin()));
                std::vector<uint32_t> route_args;
                for (const auto& route : routes) {
                    append_d2d_route_args(route, route_args);
                }
                persistent.partials_route_args[chip].push_back(std::move(route_args));
            }
            persistent.input_socket_config_addresses.push_back(
                static_cast<uint32_t>(input_socket.get_config_buffer_address()));
        }
        for (uint32_t chip = 0; chip < num_chips; ++chip) {
            TT_FATAL(
                connections[chip].size() <= kMaxPersistentConnections,
                "fused_experts: {}'s partials leave it in {} directions, more than the kernel's {}",
                chips[chip],
                connections[chip].size(),
                kMaxPersistentConnections);
            for (const auto& connection : connections[chip]) {
                persistent.partials_connection_mesh_ids[chip].push_back(*connection.mesh_id);
                persistent.partials_connection_chip_ids[chip].push_back(connection.chip_id);
            }
        }
        attributes.persistent = std::move(persistent);
    }
    tensor_args_t tensor_args{
        .input_tensor = input_tensor,
        .routing_indices = routing_indices,
        .routing_scores = routing_scores,
        .ranking_scores = ranking_scores,
        .gate_up_weights = gate_up_weights,
        .down_weights = down_weights,
        .shared_gate_up_weights = shared_gate_up_weights,
        .shared_down_weights = shared_down_weights,
        .router_weight = router_weight,
        .router_bias = router_bias,
        .layer_table = persistent_layer_table,
    };
    return {std::move(attributes), std::move(tensor_args)};
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

namespace ttnn::prim {
ttnn::operations::experimental::deepseek::moe::fused_experts::FusedExpertsDeviceOperation::tensor_return_value_t
fused_experts(
    const Tensor& input_tensor,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    uint32_t num_experts,
    uint32_t intermediate_size,
    float swiglu_limit,
    uint32_t top_k,
    float routed_scaling_factor,
    float routing_eps,
    uint32_t experts_block_size,
    bool two_hub_gather,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& routing_scores,
    const std::optional<Tensor>& routing_indices,
    const std::optional<Tensor>& ranking_scores,
    const std::optional<Tensor>& shared_gate_up_weights,
    const std::optional<Tensor>& shared_down_weights,
    std::optional<float> shared_swiglu_limit,
    const std::optional<Tensor>& router_weight,
    const std::optional<Tensor>& router_bias,
    const std::optional<std::vector<uint32_t>>& all_reduce_cluster_axes,
    uint32_t all_reduce_num_links,
    tt::tt_fabric::Topology all_reduce_topology,
    const std::optional<Tensor>& persistent_layer_table,
    const std::optional<tt::tt_metal::CoreCoord>& persistent_socket_core,
    const std::optional<std::vector<tt::tt_metal::distributed::MeshSocket>>& persistent_input_sockets,
    const std::optional<std::vector<Tensor>>& persistent_partials) {
    using OperationType = ttnn::operations::experimental::deepseek::moe::fused_experts::FusedExpertsDeviceOperation;
    auto [operation_attributes, tensor_args] = OperationType::invoke(
        input_tensor,
        gate_up_weights,
        down_weights,
        num_experts,
        intermediate_size,
        swiglu_limit,
        top_k,
        routed_scaling_factor,
        routing_eps,
        experts_block_size,
        two_hub_gather,
        memory_config,
        routing_scores,
        routing_indices,
        ranking_scores,
        shared_gate_up_weights,
        shared_down_weights,
        shared_swiglu_limit,
        router_weight,
        router_bias,
        all_reduce_cluster_axes,
        all_reduce_num_links,
        all_reduce_topology,
        persistent_layer_table,
        persistent_socket_core,
        persistent_input_sockets,
        persistent_partials);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
