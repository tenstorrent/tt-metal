// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Bring-up fork option kv_lanes (deepseek_v41_flash_d_p PB.3A, packed prefill): sparse_sdpa where each device reads
// its OWN kv tensor, one of up to 8 same-shaped tensors picked by the device's lane (lane_of_chip, indexed by the
// row-major chip index of the mesh coordinate). The program is SparseSDPAOperation's, one per device (the per-device
// descriptor and cache-hit override differ only in the kv address); every other input is shared.

#include "sparse_sdpa_device_operation.hpp"
#include <vector>

namespace ttnn::prim::bringup {

// PB.7B kv_lane_rows: words a device in the row map ([n, (end, lane) x KV_ROWS_MAX]); the kernels' per-core runtime
// args after (tok_start, tok_count) are [n, (token end, kv address) x KV_ROWS_MAX] (sparse_sdpa_common.hpp).
constexpr uint32_t KV_ROWS_MAX = 8;
constexpr uint32_t KV_ROWS_WORDS = 1 + 2 * KV_ROWS_MAX;

struct SparseSDPALanesParams {
    SparseSDPAParams base;
    std::vector<uint32_t> lane_of_chip;  // row-major chip index -> index into kv_lanes. Hashed.
    std::vector<uint32_t> lane_rows;     // PB.7B: per device KV_ROWS_WORDS (empty: off). Runtime: only "on" hashed.
};

struct SparseSDPALanesInputs {
    Tensor q;
    Tensor kv;  // kv_lanes[0] (the spec / validation stand-in)
    Tensor indices;
    std::optional<Tensor> attention_sink;
    std::optional<Tensor> kv_window;
    std::vector<Tensor> kv_lanes;
};

struct SparseSDPALanesOperation {
    using operation_attributes_t = SparseSDPALanesParams;
    using tensor_args_t = SparseSDPALanesInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SparseSDPALanesProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& attrs,
            const tensor_args_t& t,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);

        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<SparseSDPALanesProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

Tensor sparse_sdpa_lanes(
    const Tensor& q,
    const std::vector<Tensor>& kv_lanes,
    const std::vector<uint32_t>& lane_of_chip,
    const std::vector<uint32_t>& lane_rows,
    const Tensor& indices,
    const SparseSDPAParams& base,
    const std::optional<Tensor>& attention_sink,
    const std::optional<Tensor>& kv_window);

}  // namespace ttnn::prim::bringup
