// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose.hpp"

#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "device/transpose_device_operation.hpp"
#include "device/transpose_utils.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"

#include <tt-metalium/hal.hpp>

#include <algorithm>
#include <array>

namespace ttnn::operations::experimental::quasar::transpose_op {

using ttnn::operations::data_movement::has_subaligned_shard_row;

namespace detail {

using namespace tt::tt_metal::experimental;
using namespace tt;
using tt::tt_metal::BufferType;
using ttnn::operations::experimental::quasar::transpose_op::adjust_shard_spec_to_shape;
using ttnn::operations::experimental::quasar::transpose_op::is_native_transpose_sharding;

// to_memory_config hangs the device moving RM data into or out of a shard wider than the tensor row
// when the shard row isn't L1 aligned, in L1 and DRAM alike.
inline void check_stageable_shard_row(
    const MemoryConfig& mc, uint32_t tensor_width, uint32_t element_size, const char* side) {
    if (!mc.shard_spec().has_value()) {
        return;
    }
    const uint32_t shard_width = mc.shard_spec()->shape[1];
    TT_FATAL(
        shard_width <= tensor_width || (shard_width * element_size) % tt::tt_metal::hal::get_l1_alignment() == 0,
        "Row-major {} shard width {} exceeds the tensor width {} and its row of {} B is not a multiple of the {} B "
        "L1 alignment",
        side,
        shard_width,
        tensor_width,
        shard_width * element_size,
        tt::tt_metal::hal::get_l1_alignment());
}

inline Tensor unshard_for_staging(const Tensor& t) {
    check_stageable_shard_row(t.memory_config(), t.logical_shape()[-1], t.element_size(), "input");
    return ttnn::to_memory_config(t, MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1), std::nullopt);
}

// `normalized_mc` is output_mc as the output tensor will carry it (an ND config normalized to a legacy
// spec), when they differ.
inline Tensor reshard_staged_output(
    const Tensor& staged,
    const MemoryConfig& output_mc,
    const std::optional<MemoryConfig>& normalized_mc = std::nullopt) {
    check_stageable_shard_row(
        normalized_mc.value_or(output_mc), staged.logical_shape()[-1], staged.element_size(), "output");
    return ttnn::to_memory_config(staged, output_mc, std::nullopt);
}

// On a sharded 4D input of any layout, ttnn::permute builds these orders from data_movement transposes
// (data_movement/permute/permute.cpp), whose sharded intermediates can have sub-aligned RM shard rows.
inline bool permute_chains_transposes(const Tensor& a, const ttsl::SmallVector<int64_t>& dims) {
    if (!a.is_sharded() || dims.size() != 4) {
        return false;
    }
    static constexpr std::array<std::array<int64_t, 4>, 6> chained_orders = {
        {{0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}}};
    return std::any_of(chained_orders.begin(), chained_orders.end(), [&](const auto& order) {
        return std::equal(order.begin(), order.end(), dims.begin());
    });
}

// For RM data, ttnn::permute rejects an explicit output whose shard row isn't aligned (in any sharded
// layout), lets prim::permute write sub-aligned B/W shards for a default, specless or ND output, and on
// a B/W-sharded input may chain transposes through sub-aligned intermediates. Permute an interleaved
// copy instead, into L1 interleaved and resharded to the requested output when its row is sub-aligned.
inline Tensor permute_with_subaligned_output(
    const Tensor& a,
    const ttsl::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& output_mem_config,
    float pad_value) {
    if (a.layout() != Layout::ROW_MAJOR) {
        return ttnn::permute(a, dims, output_mem_config, pad_value);
    }
    const auto& input_mc = a.memory_config();

    // ttnn::permute's check on an explicit RM shard spec, which applies to every sharded layout.
    const auto permute_rejects = [&](const std::optional<MemoryConfig>& mc) {
        if (!mc.has_value() || !mc->is_sharded() || !mc->shard_spec().has_value()) {
            return false;
        }
        const uint32_t alignment = mc->buffer_type() == BufferType::DRAM ? tt::tt_metal::hal::get_dram_alignment()
                                                                         : tt::tt_metal::hal::get_l1_alignment();
        return (mc->shard_spec()->shape[1] * a.element_size()) % alignment != 0;
    };

    // The config to request from the staged path, and the same config as the output will carry it.
    std::optional<MemoryConfig> requested_mc = output_mem_config;
    std::optional<MemoryConfig> normalized_mc = output_mem_config;
    bool subaligned_output = false;
    const bool chained = permute_chains_transposes(a, dims);
    if (output_mem_config.has_value() && output_mem_config->is_sharded() &&
        output_mem_config->shard_spec().has_value()) {
        subaligned_output = permute_rejects(output_mem_config);
    } else if (
        !output_mem_config.has_value() && chained && input_mc.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        // The chain's transposes fall back to L1 interleaved for a B/W input's default output whenever they
        // can't rescale its shard spec, which is the usual case; prim::permute's derived spec would instead
        // split the permuted rows into a handful of bytes per shard.
        requested_mc = normalized_mc = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
    } else if (output_mem_config.has_value() ? output_mem_config->is_sharded() : a.is_sharded()) {
        // Same effective config permute hands prim::permute for a default, specless or ND output.
        const auto effective_mc = output_mem_config.value_or(
            input_mc.created_with_nd_shard_spec() ? MemoryConfig(input_mc.buffer_type(), input_mc.nd_shard_spec())
                                                  : MemoryConfig(input_mc.memory_layout(), input_mc.buffer_type()));
        ttsl::SmallVector<uint32_t> prim_dims(dims.begin(), dims.end());
        normalized_mc = ttnn::operations::data_movement::PermuteDeviceOperation::compute_output_specs(
                            {.dims = prim_dims, .output_mem_config = effective_mc, .pad_value = pad_value},
                            {.input_tensor = a, .optional_output_tensor = std::nullopt})
                            .memory_config();
        subaligned_output = has_subaligned_shard_row(*normalized_mc, a.element_size());
        // A caller's ND config stays ND; a default or specless output takes the derived spec.
        const bool explicit_nd = output_mem_config.has_value() && output_mem_config->nd_shard_spec().has_value();
        requested_mc = explicit_nd ? output_mem_config : normalized_mc;
    }

    // Any sharded layout: a HEIGHT-sharded input's first WH step can produce sub-aligned WIDTH shards.
    if (!subaligned_output && !chained) {
        return ttnn::permute(a, dims, output_mem_config, pad_value);
    }
    const auto interleaved_l1 = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
    const Tensor x = a.is_sharded() ? unshard_for_staging(a) : a;
    if (!subaligned_output && !permute_rejects(requested_mc)) {
        return ttnn::permute(x, dims, requested_mc, pad_value);
    }
    const Tensor staged = ttnn::permute(x, dims, interleaved_l1, pad_value);
    return reshard_staged_output(staged, *requested_mc, normalized_mc);
}

inline Tensor transpose_(
    const Tensor& a,
    ttnn::prim::qsr::TransposeOpDim transpose_dim,
    const std::optional<MemoryConfig>& output_mem_config,
    float pad_value = 0.0f) {
    MemoryConfig output_mem_constructed;
    if (!output_mem_config.has_value() ||
        (output_mem_config.value().is_sharded() && !output_mem_config.value().shard_spec().has_value())) {
        // Native sharded subset: derive output shard_spec from input's. Otherwise fall back to L1
        // interleaved (the interleaved factories handle non-native via TensorAccessor).
        const bool native = is_native_transpose_sharding(a.tensor_spec());
        if (a.is_sharded() && native) {
            // Seed: honor user's requested sharded layout (spec gets synthesized below); else
            // inherit input config so downstream can promote (e.g. N=C=1 → WIDTH_SHARDED).
            const bool user_requested_layout = output_mem_config.has_value() && output_mem_config.value().is_sharded();
            output_mem_constructed = user_requested_layout ? output_mem_config.value() : a.memory_config();
            // If shard geometry can't scale to a valid output shard, hand back a shard-spec-less
            // sharded MemoryConfig (device op synthesizes via generate_transpose_shard_spec) when
            // the user requested sharded; otherwise fall back to L1 interleaved.
            const auto shard_derivation_fallback = [&]() {
                if (user_requested_layout) {
                    output_mem_constructed = MemoryConfig(
                        output_mem_config.value().memory_layout(), output_mem_config.value().buffer_type());
                } else {
                    output_mem_constructed = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
                }
            };
            const auto& input_padded_shape = a.padded_shape();
            if (transpose_dim == ttnn::prim::qsr::TransposeOpDim::WH) {
                const uint32_t W = input_padded_shape[3], C = input_padded_shape[1], N = input_padded_shape[0];
                auto shard_spec = a.shard_spec().value();
                // N=C=1 height-sharded-with-full-width → WIDTH_SHARDED (skip if user requested layout).
                const bool can_promote_to_width_sharded =
                    !user_requested_layout && N == 1 && C == 1 && shard_spec.shape[1] == W;
                if (can_promote_to_width_sharded) {
                    std::swap(shard_spec.shape[0], shard_spec.shape[1]);
                    if (a.layout() == Layout::TILE && (shard_spec.shape[0] % tt::constants::TILE_HEIGHT != 0 ||
                                                       shard_spec.shape[1] % tt::constants::TILE_WIDTH != 0)) {
                        shard_derivation_fallback();
                    } else {
                        output_mem_constructed = MemoryConfig(
                            TensorMemoryLayout::WIDTH_SHARDED, output_mem_constructed.buffer_type(), shard_spec);
                    }
                } else {
                    auto output_padded_shape = input_padded_shape;
                    std::swap(output_padded_shape[2], output_padded_shape[3]);
                    auto adjusted = adjust_shard_spec_to_shape(shard_spec, input_padded_shape, output_padded_shape);
                    if (!adjusted.has_value() ||
                        (a.layout() == Layout::TILE && (adjusted->shape[0] % tt::constants::TILE_HEIGHT != 0 ||
                                                        adjusted->shape[1] % tt::constants::TILE_WIDTH != 0))) {
                        shard_derivation_fallback();
                    } else {
                        output_mem_constructed = MemoryConfig(
                            output_mem_constructed.memory_layout(),
                            output_mem_constructed.buffer_type(),
                            std::move(adjusted));
                    }
                }
            } else if (transpose_dim == ttnn::prim::qsr::TransposeOpDim::HC && a.layout() == Layout::TILE) {
                auto shard_spec = a.shard_spec().value();
                // HC TILE padded-shape contract: dim[1] = logical H, dim[2] = round_up(logical C, TILE_HEIGHT).
                auto output_padded_shape = input_padded_shape;
                output_padded_shape[1] = a.logical_shape()[2];
                output_padded_shape[2] = tt::round_up(a.logical_shape()[1], tt::constants::TILE_HEIGHT);
                auto adjusted = adjust_shard_spec_to_shape(shard_spec, input_padded_shape, output_padded_shape);
                if (!adjusted.has_value() || adjusted->shape[0] % tt::constants::TILE_HEIGHT != 0 ||
                    adjusted->shape[1] % tt::constants::TILE_WIDTH != 0) {
                    shard_derivation_fallback();
                } else {
                    output_mem_constructed = MemoryConfig(
                        output_mem_constructed.memory_layout(),
                        output_mem_constructed.buffer_type(),
                        std::move(adjusted));
                }
            }
        } else if (output_mem_config.has_value()) {
            // User-requested sharded output (no spec): honor the layout; device op will synthesize
            // the spec. Must precede the non-native fallback below so it isn't overridden.
            output_mem_constructed = output_mem_config.value();
        } else if (a.is_sharded()) {
            // Non-native sharded input → mirror input's memory_layout (no shard_spec, device op
            // synthesizes via adjust_shard_spec_to_shape / generate_transpose_shard_spec)
            output_mem_constructed = MemoryConfig(a.memory_config().memory_layout(), a.memory_config().buffer_type());
        } else {
            output_mem_constructed = a.memory_config();
        }
    } else {
        output_mem_constructed = output_mem_config.value();
    }

    // noc_async_write_sharded mis-splits rows of a B/W-sharded RM output whose shard row isn't a
    // multiple of the buffer alignment, and each prim rescales or synthesizes its own spec for a
    // specless config. Ask the prim for the spec it will write; if it's sub-aligned, write L1
    // interleaved and reshard to it.
    const auto launch_prim = [&](const auto& compute_output_spec, const auto& run) -> ttnn::Tensor {
        if (a.layout() == Layout::ROW_MAJOR && output_mem_constructed.is_sharded()) {
            const auto output_mc = compute_output_spec().memory_config();
            if (has_subaligned_shard_row(output_mc, a.element_size())) {
                const auto staged = run(MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1));
                return reshard_staged_output(staged, output_mc);
            }
        }
        return run(output_mem_constructed);
    };

    auto prim_permute = [&](const ttnn::Tensor& input, const ttsl::SmallVector<uint32_t>& dims) -> ttnn::Tensor {
        return launch_prim(
            [&] {
                return ttnn::operations::data_movement::PermuteDeviceOperation::compute_output_specs(
                    {.dims = dims, .output_mem_config = output_mem_constructed, .pad_value = pad_value},
                    {.input_tensor = input, .optional_output_tensor = std::nullopt});
            },
            [&](const MemoryConfig& mc) { return ttnn::prim::permute(input, dims, mc, std::nullopt, pad_value); });
    };

    bool interleaved_rm = !a.is_sharded() && a.layout() == Layout::ROW_MAJOR;
    switch (transpose_dim) {
        case ttnn::prim::qsr::TransposeOpDim::HC:
            if (interleaved_rm) {
                return prim_permute(a, ttsl::SmallVector<uint32_t>{0, 2, 1, 3});
            }
            break;
        case ttnn::prim::qsr::TransposeOpDim::NH:
            return permute_with_subaligned_output(
                a, ttsl::SmallVector<int64_t>({2, 1, 0, 3}), output_mem_config, pad_value);
        case ttnn::prim::qsr::TransposeOpDim::NW:
            return permute_with_subaligned_output(
                a, ttsl::SmallVector<int64_t>({3, 1, 2, 0}), output_mem_config, pad_value);
        case ttnn::prim::qsr::TransposeOpDim::CW:
            return permute_with_subaligned_output(
                a, ttsl::SmallVector<int64_t>({0, 3, 2, 1}), output_mem_config, pad_value);
        case ttnn::prim::qsr::TransposeOpDim::CN:
            if (interleaved_rm) {
                return prim_permute(a, ttsl::SmallVector<uint32_t>({1, 0, 2, 3}));
            }
            break;
        case ttnn::prim::qsr::TransposeOpDim::WH:
            if (interleaved_rm) {
                return prim_permute(a, ttsl::SmallVector<uint32_t>({0, 1, 3, 2}));
            }
            if (a.layout() == Layout::ROW_MAJOR) {
                // Only compute the RM WH CB-vs-L1 budget when actually on the RM WH path:
                // the allocator query and padded-shape arithmetic are wasted work otherwise.
                const uint64_t Wt = tt::div_up(a.logical_shape()[3], tt::constants::TILE_WIDTH);
                const uint64_t Ht = tt::div_up(a.logical_shape()[2], tt::constants::TILE_HEIGHT);
                const uint64_t tile_size = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(a.dtype()));
                const uint64_t cb_size_for_rm = (2 * Wt + 2 * Ht + Ht * Wt) * tile_size;
                auto* device = a.device();
                auto lowest_address = device->lowest_occupied_compute_l1_address();
                uint32_t max_l1_space =
                    lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
                max_l1_space =
                    max_l1_space - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
                if (cb_size_for_rm > max_l1_space) {
                    return prim_permute(a, ttsl::SmallVector<uint32_t>({0, 1, 3, 2}));
                }
            }
            break;
        default: break;
    }
    return launch_prim(
        [&] {
            return ttnn::prim::qsr::TransposeDeviceOperation::compute_output_specs(
                {.dim = transpose_dim, .output_mem_config = output_mem_constructed, .pad_value = pad_value},
                {.input = a});
        },
        [&](const MemoryConfig& mc) { return ttnn::prim::qsr::transpose(a, transpose_dim, mc, pad_value); });
}

ttnn::Tensor transpose_nd(
    const ttnn::Tensor& input_tensor,
    uint32_t dim1,
    uint32_t dim2,
    const std::optional<MemoryConfig>& memory_config_arg,
    float pad_value = 0.0f) {
    const auto rank = input_tensor.logical_shape().rank();
    ttsl::SmallVector<int64_t> permutation;
    permutation.reserve(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        permutation.push_back(i);
    }
    std::swap(permutation[dim1], permutation[dim2]);
    return permute_with_subaligned_output(input_tensor, permutation, memory_config_arg, pad_value);
}

inline bool is_block_or_width_sharded_mc(const tt::tt_metal::MemoryConfig& mc) {
    return mc.is_sharded() && (mc.memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
                               mc.memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED);
}

}  // namespace detail

ttnn::Tensor transpose_impl(
    const ttnn::Tensor& input_tensor,
    int64_t dim1,
    int64_t dim2,
    const std::optional<MemoryConfig>& memory_config_arg,
    float pad_value = 0.0f) {
    {
        // An identity transpose that keeps the input's placement returns the input untouched, before any
        // staging or rank-based delegation can copy it.
        const auto& logical = input_tensor.logical_shape();
        const uint32_t n1 = logical.get_normalized_index(dim1);
        const uint32_t n2 = logical.get_normalized_index(dim2);
        const auto& padded = input_tensor.padded_shape();
        if ((n1 == n2 || (padded[n1] == 1 && padded[n2] == 1)) &&
            (!memory_config_arg.has_value() || memory_config_arg.value() == input_tensor.memory_config())) {
            return input_tensor;
        }
    }
    {
        // Irregular RM block/width sharded hits a pages_per_shard misread in noc_async_*_sharded, and
        // so does a shard row that isn't a multiple of the buffer alignment (the helpers use the
        // aligned page size as both the page stride and the per-page payload). Unshard until the
        // kernel-side helpers handle those geometries.
        const bool rm = input_tensor.layout() == Layout::ROW_MAJOR;
        const bool in_sharded = detail::is_block_or_width_sharded_mc(input_tensor.memory_config());
        const auto& input_logical = input_tensor.logical_shape();
        const bool irregular_hw = input_logical.rank() >= 2 && (input_logical[-1] % tt::constants::TILE_WIDTH != 0 ||
                                                                input_logical[-2] % tt::constants::TILE_HEIGHT != 0);
        const bool subaligned_row = has_subaligned_shard_row(input_tensor.memory_config(), input_tensor.element_size());
        const uint32_t n1 = input_logical.get_normalized_index(dim1);
        const uint32_t n2 = input_logical.get_normalized_index(dim2);
        if (rm && in_sharded && (irregular_hw || subaligned_row)) {
            // Snapshot orientation before the L1-interleaved staging hop strips it.
            std::optional<tt::tt_metal::ShardOrientation> input_orientation_hint;
            if (input_tensor.shard_spec().has_value()) {
                input_orientation_hint = input_tensor.shard_spec()->orientation;
            }
            const auto interleaved_l1 =
                tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1);
            auto resolved_mc = memory_config_arg;
            if (memory_config_arg.has_value() && memory_config_arg->is_sharded() &&
                !memory_config_arg->shard_spec().has_value() && !memory_config_arg->nd_shard_spec().has_value()) {
                // Logical, not padded: a partially populated last input shard pads the input, but the
                // staged transpose writes an unpadded RM output.
                ttsl::SmallVector<uint32_t> out_shape_vec(input_logical.cbegin(), input_logical.cend());
                std::swap(out_shape_vec[n1], out_shape_vec[n2]);
                auto spec = generate_transpose_shard_spec(
                    input_tensor,
                    ttnn::Shape(std::move(out_shape_vec)),
                    memory_config_arg->memory_layout(),
                    input_orientation_hint,
                    OutputTiling::RowMajor,
                    memory_config_arg->buffer_type());
                const uint32_t alignment = memory_config_arg->buffer_type() == tt::tt_metal::BufferType::DRAM
                                               ? tt::tt_metal::hal::get_dram_alignment()
                                               : tt::tt_metal::hal::get_l1_alignment();
                const bool subaligned_synth_row = (spec.shape[1] * input_tensor.element_size()) % alignment != 0;
                auto synthesized_mc = tt::tt_metal::MemoryConfig(
                    memory_config_arg->memory_layout(), memory_config_arg->buffer_type(), std::move(spec));
                if (subaligned_synth_row) {
                    // ttnn::permute (NH/NW/CW, rank > 4) rejects an explicit sub-aligned RM shard row in any
                    // sharded layout, and a specless request would lose the input's orientation. Transpose
                    // interleaved and reshard to the synthesized spec instead.
                    Tensor x = detail::unshard_for_staging(input_tensor);
                    const Tensor staged = transpose_impl(x, dim1, dim2, interleaved_l1, pad_value);
                    return detail::reshard_staged_output(staged, synthesized_mc);
                }
                resolved_mc = std::move(synthesized_mc);
            }
            Tensor x = detail::unshard_for_staging(input_tensor);
            return transpose_impl(x, dim1, dim2, resolved_mc, pad_value);
        }
    }
    const auto& input_shape = input_tensor.logical_shape();
    uint32_t normalized_dim1 = input_shape.get_normalized_index(dim1);
    uint32_t normalized_dim2 = input_shape.get_normalized_index(dim2);

    Tensor input_unsqueezed = input_tensor;
    uint32_t initial_rank = input_shape.rank();
    if (initial_rank < 4) {
        input_unsqueezed = ttnn::unsqueeze_to_4D(input_tensor);
        uint32_t rank_diff = 4 - initial_rank;
        normalized_dim1 += rank_diff;
        normalized_dim2 += rank_diff;
    } else if (initial_rank > 4) {
        return detail::transpose_nd(input_tensor, normalized_dim1, normalized_dim2, memory_config_arg, pad_value);
    }

    bool wh = (normalized_dim1 == 2 && normalized_dim2 == 3) || (normalized_dim2 == 2 && normalized_dim1 == 3);
    bool cn = (normalized_dim1 == 0 && normalized_dim2 == 1) || (normalized_dim2 == 0 && normalized_dim1 == 1);
    bool bfloat8_supported = cn || wh;
    bool typecast = input_unsqueezed.dtype() == DataType::BFLOAT8_B and !bfloat8_supported;
    Tensor input_typecasted = typecast ? ttnn::typecast(input_unsqueezed, DataType::BFLOAT16) : input_unsqueezed;

    TT_FATAL(normalized_dim1 <= 3, "dimension has to be 0-3 only corresponding to N,C,H,W");
    TT_FATAL(normalized_dim2 <= 3, "dimension has to be 0-3 only corresponding to N,C,H,W");

    Tensor output;
    if ((normalized_dim1 == normalized_dim2) || (input_typecasted.padded_shape()[normalized_dim1] == 1 &&
                                                 input_typecasted.padded_shape()[normalized_dim2] == 1)) {
        if (memory_config_arg.has_value() && input_typecasted.memory_config() != memory_config_arg.value()) {
            output = ttnn::clone(
                input_typecasted,
                std::nullopt,
                memory_config_arg.value_or(input_typecasted.memory_config()),
                std::nullopt);
        } else {
            output = input_typecasted;
        }
    } else {
        if (normalized_dim1 > normalized_dim2) {
            std::swap(normalized_dim1, normalized_dim2);
        }

        ttnn::prim::qsr::TransposeOpDim transpose_dim = ttnn::prim::qsr::TransposeOpDim::NW;

        if (normalized_dim2 == 3 && normalized_dim1 == 0) {
            transpose_dim = ttnn::prim::qsr::TransposeOpDim::NW;
        } else if (normalized_dim2 == 3 && normalized_dim1 == 1) {
            transpose_dim = ttnn::prim::qsr::TransposeOpDim::CW;
        } else if (normalized_dim2 == 3 && normalized_dim1 == 2) {
            transpose_dim = ttnn::prim::qsr::TransposeOpDim::WH;
        } else if (normalized_dim2 == 2 && normalized_dim1 == 0) {
            transpose_dim = ttnn::prim::qsr::TransposeOpDim::NH;
        } else if (normalized_dim2 == 2 && normalized_dim1 == 1) {
            transpose_dim = ttnn::prim::qsr::TransposeOpDim::HC;
        } else if (normalized_dim2 == 1 && normalized_dim1 == 0) {
            transpose_dim = ttnn::prim::qsr::TransposeOpDim::CN;
        } else {
            TT_ASSERT(false, "Unsupported transpose dims");
        }
        output = detail::transpose_(input_typecasted, transpose_dim, memory_config_arg, pad_value);
    }
    output = initial_rank < 4u ? ttnn::squeeze_from_4D(output, initial_rank) : output;
    return typecast ? ttnn::typecast(output, DataType::BFLOAT8_B) : output;
}

}  // namespace ttnn::operations::experimental::quasar::transpose_op

namespace ttnn::operations::experimental::quasar {

ttnn::Tensor transpose(
    const ttnn::Tensor& input_tensor,
    int64_t dim1,
    int64_t dim2,
    const std::optional<MemoryConfig>& memory_config,
    float pad_value) {
    return transpose_op::transpose_impl(input_tensor, dim1, dim2, memory_config, pad_value);
}

ttnn::Tensor transpose(const ttnn::Tensor& input_tensor, int64_t dim1, int64_t dim2, float pad_value) {
    return transpose(input_tensor, dim1, dim2, std::nullopt, pad_value);
}

}  // namespace ttnn::operations::experimental::quasar
