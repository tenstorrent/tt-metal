// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_plan.hpp"

#include <algorithm>
#include <cstdint>

#include <fmt/format.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_chain_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_ksplit.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"

namespace ttnn::prim::ring_joint_plan {

RingJointInputParams resolve_ring_joint_input_params(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    RingJointInputParams joint_input_params;
    joint_input_params.has_joint_tensors = tensor_args.joint_q.has_value();
    joint_input_params.joint_is_sharded = tensor_args.joint_is_sharded();

    if (joint_input_params.has_joint_tensors) {
        joint_input_params.joint_q = &tensor_args.joint_q.value();
        joint_input_params.joint_k = &tensor_args.joint_k.value();
        joint_input_params.joint_v =
            tensor_args.joint_v.has_value() ? &tensor_args.joint_v.value() : joint_input_params.joint_k;
    }

    if (joint_input_params.joint_is_sharded) {
        joint_input_params.gathered_joint_k = &tensor_args.gathered_joint_k.value();
        joint_input_params.gathered_joint_v = &tensor_args.gathered_joint_v.value();
        // Per-shard physical length is the (padded, tile-aligned) joint Q shard on this device.
        // The padded total L is that shard times the ring; logical_l carries the true token count,
        // which may be smaller than L when the joint prompt does not fill the last shard. This
        // mirrors how the spatial path keeps logical_n separate from the padded gathered length.
        joint_input_params.L_local = joint_input_params.joint_q->logical_shape()[2];
        joint_input_params.L =
            joint_input_params.L_local * static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
        joint_input_params.logical_l = static_cast<uint32_t>(args.logical_l);
    } else if (joint_input_params.has_joint_tensors) {
        joint_input_params.L = joint_input_params.joint_q->logical_shape()[2];
        joint_input_params.L_local = joint_input_params.L;
        joint_input_params.logical_l = joint_input_params.L;
    }

    return joint_input_params;
}

bool kernel_chunked(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    return tensor_args.is_chunked(args.kv_stripe_split) && !args.is_cross;
}

std::pair<std::vector<Tensor>, std::vector<Tensor>> fused_gather_kv_tensors(const RingJointSDPAInputs& tensor_args) {
    std::vector<Tensor> inputs = {tensor_args.input_k};
    std::vector<Tensor> outputs = {tensor_args.gathered_k};
    if (tensor_args.input_v.has_value()) {
        inputs.push_back(tensor_args.input_v.value());
        outputs.push_back(tensor_args.gathered_v.value_or(tensor_args.gathered_k));
    }
    return {inputs, outputs};
}

std::string single_q_chunk_blocker(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args, const KSplitShape& shape) {
    const auto& q_shape = tensor_args.input_q.logical_shape();
    const uint32_t B = q_shape[0];
    const uint32_t L = resolve_ring_joint_input_params(args, tensor_args).L;
    const uint32_t Sq_chunk_t = args.get_q_chunk_size() / tt::constants::TILE_HEIGHT;
    if (args.is_balanced) {
        return "is_balanced is set";
    }
    if (!shape.use_streaming_compute) {
        return "fp32_dest_acc_en is set";
    }
    if (B != 1) {
        return fmt::format("the batch is {}", B);
    }
    if (L != 0) {
        return fmt::format("there is a joint sequence of {} rows", L);
    }
    if (shape.max_q_per_core > 1) {
        return fmt::format(
            "{} Q chunks share {} cores", B * q_shape[1] * shape.num_q_chunks, shape.grid.x * shape.grid.y);
    }
    if (Sq_chunk_t % shape.writer_out_row_group_h != 0) {
        return fmt::format(
            "the {}-tile Q chunk is not a whole number of {}-tile row groups", Sq_chunk_t, shape.writer_out_row_group_h);
    }
    return {};
}

KSplitShape derive_ksplit(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    const auto& q_shape = tensor_args.input_q.logical_shape();
    const uint32_t B = q_shape[0];
    const uint32_t NH = q_shape[1];
    const uint32_t NHK = tensor_args.gathered_k.logical_shape()[1];
    const uint32_t NHV = tensor_args.v_num_heads();
    const bool v_shares_k_buffer = tensor_args.v_shares_k_buffer();
    const uint32_t q_chunk_size = args.get_q_chunk_size();
    const uint32_t Sq_chunk_t = q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t Sk_chunk_t = args.get_k_chunk_size() / tt::constants::TILE_HEIGHT;
    const uint32_t vDHt = tensor_args.v_head_dim(args.latent_v_head_dim) / tt::constants::TILE_WIDTH;
    const RingJointInputParams joint = resolve_ring_joint_input_params(args, tensor_args);
    const uint32_t sliding_window_size = args.sliding_window_size.value_or(0);
    const bool has_sliding_window = sliding_window_size > 0;
    const bool chunked = kernel_chunked(args, tensor_args);
    const bool gqa_grouped_kv = ring_joint::is_gqa_grouped_kv_head_mode(v_shares_k_buffer, NH, NHK, NHV);
    const bool k_uses_batch_chain = ring_joint::uses_shared_k_batch_chain(gqa_grouped_kv, NHK);
    const bool build_kv_chains = !has_sliding_window && B == 1;

    KSplitShape shape;
    shape.grid = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                 : tensor_args.input_q.device()->compute_with_storage_grid_size();
    shape.num_q_chunks =
        tt::div_up(static_cast<uint32_t>(q_shape[2]), q_chunk_size) + tt::div_up(joint.L_local, q_chunk_size);
    const uint32_t all_heads_num_q_chunks = B * NH * shape.num_q_chunks;
    shape.max_q_per_core = tt::div_up(all_heads_num_q_chunks, static_cast<uint32_t>(shape.grid.x * shape.grid.y));
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tensor_args.input_q.device()->arch(), args.compute_kernel_config);
    shape.use_streaming_compute = !fp32_dest_acc_en;

    // Keep the host phase-2 layout at one column for in-place latent V. The compute helper
    // independently batches strided K^T rows into DST, including a short final batch.
    const uint32_t dst_size = ttnn::get_dest_reg_count(args.compute_kernel_config);
    const auto [out_subblock_h, out_subblock_w] = detail::determine_largest_subblock_size(
        Sq_chunk_t,
        vDHt,
        dst_size,
        /*max_subblock_h=*/shape.use_streaming_compute ? 2 : UINT32_MAX,
        /*max_subblock_w=*/kt_inplace_v_enabled(v_shares_k_buffer, Sq_chunk_t) ? 1u : UINT32_MAX);
    shape.out_subblock_h = out_subblock_h;
    shape.out_subblock_w = out_subblock_w;
    // Streaming compute may widen the QKT@V row group beyond the host matmul subblock
    // height for odd Q chunks. The writer must drain cb_out with the same row-group
    // cadence that compute pushes, otherwise deferred-save rows can be popped and
    // reused before the matching grouped write has safely landed.
    shape.writer_out_row_group_h =
        shape.use_streaming_compute
            ? ttnn::transformer::sdpa::streaming_qktv_h(out_subblock_h, out_subblock_w, dst_size, Sq_chunk_t)
            : out_subblock_h;

    // K split: when the (head, Q chunk) units leave the grid idle, the rows are divided into bands that each hold
    // every unit once and attend to a slice of every ring iteration's K chunks; the last band merges. Bands are whole
    // rows so each row still consumes one K sequence through its multicast. See kernels/ring_joint_ksplit.hpp.
    const uint32_t ksplit_requested = args.program_config.has_value() ? args.program_config->k_splits : 1;
    if (ksplit_requested > 1 && chunked) {
        const std::string blocker = single_q_chunk_blocker(args, tensor_args, shape);
        TT_FATAL(blocker.empty(), "k_splits={} cannot be applied: {}. Set k_splits=1.", ksplit_requested, blocker);
        // Separate V beside a shared K: bands stay on their K slice through the row multicast, which needs chains and
        // at least two columns (TT_FATAL after the multicast pass).
        TT_FATAL(
            has_sliding_window || (gqa_grouped_kv && NHK == 1) ||
                (k_uses_batch_chain && !v_shares_k_buffer && build_kv_chains && shape.grid.x >= 2),
            "k_splits={} cannot be applied: dense attention needs one grouped K/V head (there are {}), or a shared K "
            "with a separate V on chains. Set k_splits=1.",
            ksplit_requested,
            NHK);
        // Split KV slices packed source streams, not one source's local K chunks.
        TT_FATAL(
            args.kv_stripe_split == 1,
            "k_splits={} cannot be applied: kv_stripe_split is {}. Set k_splits=1.",
            ksplit_requested,
            args.kv_stripe_split);
        shape.rows_per_split = tt::div_up(all_heads_num_q_chunks, static_cast<uint32_t>(shape.grid.x));
        uint32_t max_ksplit =
            std::min(static_cast<uint32_t>(shape.grid.y) / shape.rows_per_split, ring_joint::kKSplitMaxCount);
        std::string sliding_limit;
        if (has_sliding_window) {
            // Every sliding band needs two K chunks: the oldest halo chunk can mask a Q chunk's later rows entirely.
            const uint32_t sliding_cap = std::max(
                1u,
                ring_joint::chunked_sliding_halo_tile_rows(
                    sliding_window_size, tt::constants::TILE_HEIGHT, Sk_chunk_t) /
                    Sk_chunk_t / 2);
            max_ksplit = std::min(max_ksplit, sliding_cap);
            sliding_limit = fmt::format(", sliding window {} allows {}", sliding_window_size, sliding_cap);
        }
        TT_FATAL(
            ksplit_requested <= max_ksplit,
            "k_splits={}, but at most {} can be applied: {} (head, Q chunk) units take {} rows per band of the "
            "{}x{} grid; the kernel caps it at {}{}.",
            ksplit_requested,
            max_ksplit,
            all_heads_num_q_chunks,
            shape.rows_per_split,
            shape.grid.x,
            shape.grid.y,
            ring_joint::kKSplitMaxCount,
            sliding_limit);
        shape.count = ksplit_requested;
    }
    return shape;
}

bool plain_gathered_slices(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    const auto& ag = args.all_gather_operation_attributes;
    const RingJointInputParams joint = resolve_ring_joint_input_params(args, tensor_args);
    const Tensor& input_k = tensor_args.input_k;
    const Tensor& input_v = tensor_args.input_v.has_value() ? tensor_args.input_v.value() : input_k;
    if (!(ag.ring_size > 2 && kernel_chunked(args, tensor_args) && !args.has_sliding_window() &&
          !joint.joint_is_sharded && joint.L == 0 && !args.is_balanced && args.kv_stripe_split == 1 &&
          !tensor_args.v_shares_k_buffer() && input_v.padded_shape()[2] == input_k.padded_shape()[2])) {
        return false;
    }
    const auto [inputs, outputs] = fused_gather_kv_tensors(tensor_args);
    return ring_attention_all_gather_async_detail::uses_output_bank_owned_schedule(inputs, outputs, ag.dim);
}

bool slice_readiness_enabled(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    return args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Ring &&
           plain_gathered_slices(args, tensor_args);
}

double floater_compute_to_transfer(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args, const KSplitShape& ksplit) {
    const auto bytes_per_element = [](const Tensor& tensor) {
        return static_cast<double>(tt::tt_metal::tile_size(tensor.dtype())) /
               static_cast<double>(tt::constants::TILE_HW);
    };
    const auto& q_shape = tensor_args.input_q.logical_shape();
    // The rows the scheduled Q chunks compute: num_q_chunks whole chunks, so a ragged last chunk counts in full.
    const uint32_t q_rows = ksplit.num_q_chunks * args.get_q_chunk_size();
    // k_expand_weight: [1, heads, no-PE width, latent width].
    const auto& wuk_shape = tensor_args.k_expand_weight->logical_shape();
    const uint32_t v_dim = tensor_args.v_head_dim(args.latent_v_head_dim);
    double bytes_per_token = tensor_args.gathered_k.padded_shape()[3] * bytes_per_element(tensor_args.gathered_k);
    if (tensor_args.gathered_v.has_value()) {
        bytes_per_token +=
            static_cast<double>(tensor_args.v_num_heads()) * v_dim * bytes_per_element(tensor_args.gathered_v.value());
    }
    const tt::tt_metal::MathFidelity fidelity =
        args.program_config->matmul_math_fidelity.value_or(ttnn::get_math_fidelity(args.compute_kernel_config));
    return ring_joint::floater_compute_to_transfer({
        .heads = static_cast<uint32_t>(q_shape[1]),
        .q_rows = q_rows,
        .qk_dim = static_cast<uint32_t>(q_shape[3]),
        .v_dim = v_dim,
        .expand_in = static_cast<uint32_t>(wuk_shape[3]),
        .expand_out = static_cast<uint32_t>(wuk_shape[2]),
        .fidelity_multiplier =
            static_cast<uint32_t>(tt::tt_metal::operation::OpPerformanceModel::fidelity_multiplier(fidelity)),
        .bytes_per_token = bytes_per_token,
        .cores = static_cast<uint32_t>(ksplit.grid.x * ksplit.grid.y),
        // A line's end devices receive every slice from one side.
        .ring_links = (args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Ring ? 2u : 1u) *
                      args.all_gather_operation_attributes.num_links,
    });
}

std::optional<ring_joint::FloaterSchedule> floater_eligibility(
    const RingJointSDPAParams& args,
    const RingJointSDPAInputs& tensor_args,
    const KSplitShape& ksplit,
    bool refuse_unusable_schedule) {
    const auto& ag = args.all_gather_operation_attributes;
    const uint32_t NH = tensor_args.input_q.logical_shape()[1];
    // A schedule given in the config that the call cannot use is refused rather than dropped, so a tuning run never
    // times the program without floaters by mistake. Only after validation: before it, a malformed schedule is no
    // schedule, and the program factory's call reports it.
    const bool requested = args.program_config.has_value() && args.program_config->floater_schedule.has_value();
    const std::optional<ring_joint::FloaterSchedule> requested_schedule =
        requested && (refuse_unusable_schedule ||
                      ring_joint::floater_schedule_values_well_formed(args.program_config->floater_schedule.value()))
            ? ring_joint::floater_schedule_from_values(args.program_config->floater_schedule.value())
            : std::nullopt;
    const auto refuse = [&](const char* reason) {
        TT_FATAL(
            !refuse_unusable_schedule || !requested_schedule.has_value(),
            "floater_schedule is set, but this call cannot run floaters: {}",
            reason);
    };
    if (!(tensor_args.has_k_expand() && ksplit.count > 1 && !args.has_sliding_window() &&
          !tensor_args.attention_sink.has_value() && ksplit.num_q_chunks == 2 && args.program_config.has_value() &&
          ring_joint::floater_layout_supported(NH, ksplit.count) && ksplit.grid == CoreCoord(11, 10) &&
          ag.ring_size == 8 && args.ccl_core_grid_offset == CoreCoord(ksplit.grid.x, 0) && ag.num_links == 2 &&
          ag.core_allocation_strategy == ttnn::ccl::CoreAllocationStrategy::COL_MAJOR)) {
        refuse(
            "floaters need expanded K, a K split, no sliding window or attention sink, two Q chunks per head, 24 heads "
            "x 2 bands or 16 x 3, an 11x10 grid and an 8-chip ring with 2 gather links");
        return std::nullopt;
    }
    const std::optional<ring_joint::FloaterSchedule> schedule =
        requested ? requested_schedule
        : floater_compute_to_transfer(args, tensor_args, ksplit) >= ring_joint::kFloaterMinComputeToTransfer
            ? ring_joint::floater_layout_schedule(NH, ksplit.count)
            : std::nullopt;
    if (!schedule.has_value()) {
        return std::nullopt;
    }
    if (!plain_gathered_slices(args, tensor_args)) {
        refuse("floaters cut their pieces from the plain chunked dense gather's slices");
        return std::nullopt;
    }
    return schedule;
}

bool runs_floaters(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    return floater_eligibility(args, tensor_args, derive_ksplit(args, tensor_args), /*refuse_unusable_schedule=*/false)
        .has_value();
}

}  // namespace ttnn::prim::ring_joint_plan
