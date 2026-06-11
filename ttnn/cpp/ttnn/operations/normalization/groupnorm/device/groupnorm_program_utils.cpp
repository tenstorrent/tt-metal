// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_program_utils.hpp"

#include <limits>
#include <algorithm>

#include <tt-metalium/device.hpp>                // tt::tt_metal::IDevice
#include <tt-metalium/tt_backend_api_types.hpp>  // tt::DataFormat, tt::tile_size
#include <ttnn/tensor/types.hpp>                 // tt::tt_metal::datatype_to_dataformat_converter

namespace ttnn::prim {

int get_max_subblock(uint32_t n, uint32_t max_subblock_w) {
    if (n <= max_subblock_w) {
        return n;
    }

    for (int quotient = max_subblock_w; quotient > 1; --quotient) {
        if (n % quotient == 0) {
            return quotient;
        }
    }
    return 1;
}

bool is_rectangle_grid(const std::vector<CoreCoord>& core_coords) {
    if (core_coords.empty()) {
        return true;
    }

    int min_x = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max();
    int max_y = std::numeric_limits<int>::min();

    for (const auto& coord : core_coords) {
        min_x = std::min(min_x, static_cast<int>(coord.x));
        max_x = std::max(max_x, static_cast<int>(coord.x));
        min_y = std::min(min_y, static_cast<int>(coord.y));
        max_y = std::max(max_y, static_cast<int>(coord.y));
    }

    return ((max_x - min_x + 1) * (max_y - min_y + 1)) == static_cast<int>(core_coords.size());
}

void split_and_form_rectangle_grids(
    std::vector<CoreCoord>& group,
    std::vector<CoreCoord>& mcast_group_first,
    std::vector<CoreCoord>& mcast_group_mid,
    std::vector<CoreCoord>& mcast_group_last) {
    size_t remove_front = 0;
    size_t remove_back = 0;
    size_t min_total_removal = group.size();

    for (size_t front = 0; front <= group.size(); ++front) {
        for (size_t back = 0; front + back <= group.size(); ++back) {
            if (is_rectangle_grid(std::vector<CoreCoord>(group.begin() + front, group.end() - back))) {
                size_t total_removal = front + back;
                if (total_removal < min_total_removal) {
                    min_total_removal = total_removal;
                    remove_front = front;
                    remove_back = back;
                }
            }
        }
    }

    // Pop and push the front outliers
    for (size_t i = 0; i < remove_front; ++i) {
        mcast_group_first.push_back(mcast_group_mid.front());
        mcast_group_mid.erase(mcast_group_mid.begin());
    }

    // Pop and push the back outliers
    for (size_t i = 0; i < remove_back; ++i) {
        mcast_group_last.push_back(mcast_group_mid.back());
        mcast_group_mid.pop_back();
    }
}

std::pair<uint32_t, uint32_t> find_max_tile_span(uint32_t W, uint32_t group_size, uint32_t tile_width) {
    uint32_t current_position = 0;
    uint32_t max_tile_span = 0;
    uint32_t num_groups_before_start_again_at_tile_beginning = static_cast<uint32_t>(-1);
    bool calc_num_groups_before_start_again_at_tile_beginning = true;

    while (current_position < W) {
        uint32_t group_end = current_position + group_size;
        uint32_t start_tile = current_position / tile_width;
        uint32_t end_tile = (group_end - 1) / tile_width;
        uint32_t current_tile_span = end_tile - start_tile + 1;

        max_tile_span = std::max(max_tile_span, current_tile_span);

        current_position = group_end;

        if (current_position % tile_width == 0 && calc_num_groups_before_start_again_at_tile_beginning) {
            num_groups_before_start_again_at_tile_beginning = current_position / group_size;
            calc_num_groups_before_start_again_at_tile_beginning = false;
        }
    }

    return {max_tile_span, num_groups_before_start_again_at_tile_beginning};
}

uint32_t GroupNormShardedStaticCbSizes::total(
    bool with_negative_mask,
    bool untilize_out,
    bool has_gamma,
    bool has_beta,
    bool has_input_mask,
    bool reader_repack_output,
    bool use_welford) const {
    uint32_t t = 0;
    t += in_CB_size;  // c_1 tilized input
    // Non-negative-mask path keeps a second copy for untilize-out (c_30); negative-mask path
    // replaces it with the (smaller) negative-mask CB (c_14).
    t += with_negative_mask ? in_negative_mask_CB_size : (untilize_out ? in_CB_size : 0u);
    t += in2_CB_size;  // c_2 scaler
    t += in3_CB_size;  // c_3 eps
    if (!use_welford) {
        t += in2_CB_size;  // c_4 scaler-c
    }
    if (has_gamma) {
        t += in5_CB_size;  // c_5
    }
    if (has_beta) {
        t += in6_CB_size;  // c_6
    }
    if (has_input_mask) {
        t += in_mask_CB_size;  // c_7
    }
    if (reader_repack_output) {
        t += repack_CB_size;  // c_11/c_12
    }
    t += x_CB_size;           // c_13
    t += ex_partial_CB_size;  // c_8
    if (!use_welford) {
        t += single_tile_size;  // c_10 ex_external
    }
    t += ex_global_CB_size;  // c_9/c_15
    t += ex2pe_CB_size;      // c_17
    t += single_tile_size;   // c_26 ones
    return t;
}

GroupNormShardedStaticCbSizes compute_sharded_gn_static_cb_sizes(
    const ttnn::Tensor& input,
    std::optional<tt::tt_metal::DataType> gamma_dtype,
    std::optional<tt::tt_metal::DataType> beta_dtype,
    bool use_welford,
    uint32_t num_groups) {
    // Mirror the factory's intermediate derivations exactly (see
    // groupnorm_sharded_program_factory.cpp) so the sizes are byte-identical to what it
    // allocates. find_max_tile_span uses the default tile_width to match the factory.
    const auto& shard_spec = input.shard_spec().value();
    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    const uint32_t tile_width = input.tensor_spec().tile().get_width();

    const uint32_t per_core_M = shard_spec.shape[0];
    const uint32_t per_core_N = shard_spec.shape[1];
    const uint32_t per_core_Mt = per_core_M / tile_height;
    const uint32_t per_core_Nt = (per_core_N + tile_width - 1) / tile_width;

    const auto& shape = input.padded_shape();
    const uint32_t num_batches = shape[0];
    const uint32_t W = shape[3];
    const uint32_t H = shape[2] * num_batches;
    const uint32_t group_size = W / num_groups;

    const uint32_t num_shards_r = H / per_core_M;
    const uint32_t num_batches_per_core = num_batches > num_shards_r ? num_batches / num_shards_r : 1;
    const uint32_t num_shards_c = W / per_core_N;
    const uint32_t num_groups_per_core = num_groups > num_shards_c ? num_groups / num_shards_c : 1;

    const auto [block_wt, num_groups_per_reset] = find_max_tile_span(per_core_N, group_size);
    (void)num_groups_per_reset;
    const uint32_t block_ht = per_core_Mt / num_batches_per_core;
    const uint32_t interm_block_tiles = block_ht * block_wt;
    const uint32_t in0_block_tiles = per_core_Nt * per_core_Mt;

    // Tile sizes by data format. The op fixes the compute (intermediate) format to bf16, and
    // the input/negative masks are bf16; gamma/beta share one CB format (gamma's, overridden by
    // beta's when beta is present), matching the factory's gamma_beta_cb_data_format.
    const tt::DataFormat in_fmt = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    constexpr tt::DataFormat bf16_fmt = tt::DataFormat::Float16_b;
    const uint32_t in_ts = tt::tile_size(in_fmt);
    const uint32_t ts = tt::tile_size(bf16_fmt);

    tt::DataFormat gamma_beta_fmt = bf16_fmt;
    if (gamma_dtype.has_value()) {
        gamma_beta_fmt = tt::tt_metal::datatype_to_dataformat_converter(*gamma_dtype);
    }
    if (beta_dtype.has_value()) {
        gamma_beta_fmt = tt::tt_metal::datatype_to_dataformat_converter(*beta_dtype);
    }
    const uint32_t gamma_beta_ts = tt::tile_size(gamma_beta_fmt);

    GroupNormShardedStaticCbSizes s;
    s.in_CB_size = in0_block_tiles * in_ts;
    s.in2_CB_size = ts * (use_welford ? 3u : 1u);
    s.in3_CB_size = ts;
    s.in5_CB_size = per_core_Nt * gamma_beta_ts;
    s.in6_CB_size = per_core_Nt * gamma_beta_ts;
    s.in_mask_CB_size = block_wt * ts * (use_welford ? num_groups_per_core : 2u);
    s.in_negative_mask_CB_size = block_wt * ts * 2u;
    s.repack_CB_size = per_core_Nt * in_ts * 2u;
    s.x_CB_size = ts * (use_welford ? 1u : interm_block_tiles);
    s.ex_partial_CB_size = ts * (use_welford ? 2u : 1u);
    s.ex_global_CB_size = s.ex_partial_CB_size * (use_welford ? num_groups_per_core : 1u);
    s.ex2pe_CB_size = use_welford ? ts * num_groups_per_core : s.ex_partial_CB_size;
    s.single_tile_size = ts;
    return s;
}

uint32_t compute_sharded_gn_l1_footprint(
    const ttnn::Tensor& input,
    const ttnn::TensorSpec& output_spec,
    const tt::tt_metal::IDevice& device,
    std::optional<tt::tt_metal::DataType> gamma_dtype,
    std::optional<tt::tt_metal::DataType> beta_dtype,
    bool has_input_mask,
    bool with_negative_mask,
    bool use_welford,
    bool inplace,
    uint32_t num_groups) {
    const auto cb = compute_sharded_gn_static_cb_sizes(input, gamma_dtype, beta_dtype, use_welford, num_groups);

    // Flags decide which CBs are actually allocated, matching the factory's conditionals.
    const uint32_t tile_width = input.tensor_spec().tile().get_width();
    const bool reader_repack_output = (input.shard_spec().value().shape[1] % tile_width) != 0;
    const bool untilize_out = output_spec.layout() == tt::tt_metal::Layout::ROW_MAJOR;

    uint32_t footprint = cb.total(
        with_negative_mask,
        untilize_out,
        gamma_dtype.has_value(),
        beta_dtype.has_value(),
        has_input_mask,
        reader_repack_output,
        use_welford);

    // Beyond the static CB region added above, the output buffer is the only other new L1 the op
    // allocates — and only when it lands in L1 and isn't computed in place. Its size is exact:
    // taken from the output spec's own per-bank computation, not estimated.
    const bool output_allocates_l1 =
        !inplace && output_spec.memory_config().buffer_type() == tt::tt_metal::BufferType::L1;
    if (output_allocates_l1) {
        footprint += static_cast<uint32_t>(output_spec.compute_consumed_memory_bytes_per_bank(device));
    }
    return footprint;
}

}  // namespace ttnn::prim
