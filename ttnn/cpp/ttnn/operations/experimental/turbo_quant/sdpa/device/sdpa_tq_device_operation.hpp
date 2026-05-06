// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::turbo_quant {

struct SDPATQDeviceOperation {
    struct operation_attributes_t {
        float scale;
        std::vector<float> centroids;  // centroid values (size = 2^bits)
        bool pre_rescaled;             // true: BFP4 values are centroid×norm, skip gather+norm
        tt::tt_metal::MemoryConfig output_mem_config;
        // Tier 2A: max number of cores assigned per (batch, q_head) tuple. When >1,
        // the chunk loop is split across this many "worker" cores per tuple and the
        // partial (max, sum, out) state is merged via cross-core reduce. The actual
        // K used at runtime is min(this, num_cores_grid / (B * NQH)). 1 = legacy
        // behavior (one core per tuple, full chunk loop, no reduce).
        uint32_t num_cores_per_head = 1;

        // Sliding-window hybrid: when true, compute kernel packs LSE = max + log(sum)
        // to cb_lse_out (c_3) and writer writes it to a second output tensor. Used
        // by the host-level online softmax combine that merges old-positions TQ
        // SDPA with recent-positions standard SDPA. See LSE_COMBINE_DESIGN.md.
        // Mutually exclusive with num_cores_per_head > 1 (cb_lse_out aliases the
        // Tier 2A reducer's cb_merge_new_max).
        bool return_lse = false;

        // Sliding-window fused hybrid (Phase 1, plumbing only): when > 0, the
        // kernel runs a single fused SDPA over both the TQ cache (chunks fully
        // in [0, cur_pos - recent_window)) and a BFP8 ring buffer (chunks
        // covering the most recent `recent_window` positions). When 0 (default),
        // legacy behavior — single source determined by k_indices alone.
        // tensor_args_t.k_ring / v_ring / ring_page_table must be set when
        // recent_window > 0. See SLIDING_WINDOW_DESIGN.md.
        // Reader and compute kernel still ignore the ring inputs in Phase 1;
        // Phase 2/3 wires the per-chunk source branch.
        uint32_t recent_window = 0;
    };

    struct tensor_args_t {
        const Tensor& q;           // [B, NQH, 1, DH] BF16
        const Tensor& k_indices;   // [B, NKH, Sk, DH] BFP4 paged
        const Tensor& k_norms;     // [B, NKH, Sk, 1] BF16
        const Tensor& v_indices;   // [B, NKH, Sk, vDH] BFP4 paged
        const Tensor& v_norms;     // [B, NKH, Sk, 1] BF16
        const Tensor& page_table;  // [B, max_pages] Int32
        const Tensor& cur_pos;     // [B] Int32

        // Hybrid-mode ring inputs. Required iff
        // operation_attributes_t::recent_window > 0; ignored otherwise.
        // k_ring / v_ring are paged BFP8 caches over the most recent W tokens
        // (W = recent_window). ring_page_table maps logical block i → physical
        // block i (identity for now). See SLIDING_WINDOW_DESIGN.md.
        std::optional<Tensor> k_ring = std::nullopt;
        std::optional<Tensor> v_ring = std::nullopt;
        std::optional<Tensor> ring_page_table = std::nullopt;
    };

    // Returns either {out} or {out, lse} depending on return_lse. A vector lets
    // us keep callers that don't ask for LSE on a simple `outs[0]` while the
    // hybrid path picks up `outs[1]` for the combine.
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    struct MultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
            std::size_t grid_size_x;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<MultiCore>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::turbo_quant

namespace ttnn::prim {

std::vector<Tensor> turbo_quant_sdpa_decode(
    const Tensor& q,
    const Tensor& k_indices,
    const Tensor& k_norms,
    const Tensor& v_indices,
    const Tensor& v_norms,
    const Tensor& page_table,
    const Tensor& cur_pos,
    const std::vector<float>& centroids,
    float scale,
    bool pre_rescaled = false,
    uint32_t num_cores_per_head = 1,
    bool return_lse = false,
    uint32_t recent_window = 0,
    const std::optional<Tensor>& k_ring = std::nullopt,
    const std::optional<Tensor>& v_ring = std::nullopt,
    const std::optional<Tensor>& ring_page_table = std::nullopt);

}  // namespace ttnn::prim
