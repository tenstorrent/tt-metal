// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The flat routed expert overlapped with combine_fabric2d in one program per chip (the flat expert's counterpart of
// hybrid_routed_expert_moe's combine overlap, #58093): combine runs on grid rows 0-1 (its senders next to the eth
// cores, its untilizers and collector), the flat expert on the rows below them (MIMO_FL_ROWS), and every flat y writer
// reports each local expert to combine's collector once its rows have landed (se_cmb_done.hpp), so combine takes an
// expert while the next ones are still being computed.
//
// y is row-major bf16 by default (y_row_major): combine's readers read its rows directly, so combine needs no
// untilizers and only grid row 0 (its senders and collector), and the flat expert takes rows 1-9. With bfp8 y tiles
// combine's untilizers read them as they read the hybrid's, on rows 0-1. Combine walks the experts in slot order (one
// pass); the flat expert's arena spans the whole grid and also holds combine's L1.
#pragma once

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "flat_routed_expert_program_factory.hpp"
#include "flat_routed_expert_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/hybrid_routed_expert_ffn/device/combine/combine_fabric2d_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::flat_routed_expert {

struct FlatCombineOverlapParams {
    FlatRoutedExpertConfig flat;  // with cmb_rt set
    tt::tt_metal::distributed::MeshDevice* device = nullptr;
    uint32_t experts_per_chip = 0;
    uint32_t num_experts_per_tok = 0;
    uint32_t seq_len_per_chip = 0;
    uint32_t axis = 0;
    uint32_t num_links = 2;
    // the flat expert's grid rectangle (logical): combine multicasts `go` over it
    tt::tt_metal::CoreRange flat_cores{tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{0, 0}};
    // its cores outside that rectangle, on combine's rows (MIMO_FL_XDOWN): `go` reaches them by unicast
    std::vector<tt::tt_metal::CoreCoord> extra_cores;
    // y writers reporting to the collector
    uint32_t writers = 0;
    std::optional<tt::tt_metal::GlobalSemaphore> fwd_arrived;
    std::optional<tt::tt_metal::GlobalSemaphore> final_arrived;
    std::optional<tt::tt_metal::GlobalSemaphore> expert_go;
    // every buffer address the program bakes in (combine's are compile-time arguments): the cache key, so a cache hit
    // never needs a patch
    std::vector<uint32_t> addrs;
    // the flat config's fields, for the cache key (the nested config is not reflected)
    std::vector<uint32_t> flat_key;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "flat_key",
        "experts_per_chip",
        "num_experts_per_tok",
        "seq_len_per_chip",
        "axis",
        "num_links",
        "writers",
        "addrs");
    auto attribute_values() const {
        return std::forward_as_tuple(
            flat_key, experts_per_chip, num_experts_per_tok, seq_len_per_chip, axis, num_links, writers, addrs);
    }
};

struct FlatCombineOverlapInputs {
    Tensor x;
    Tensor counts;
    Tensor regions;
    Tensor global_expert_ids;
    Tensor gate_up_weights;
    Tensor down_weights;
    std::optional<Tensor> reader_down_weights;
    Tensor done_words;
    Tensor arena;  // over every worker core: the flat expert's buffers on its rows, combine's on rows 0-1
    Tensor words;
    Tensor y;  // the flat expert's output [rows, H], what combine reads: bf16 ROW_MAJOR (y_row_major) or bfp8 TILE
    Tensor dispatched_metadata;
    Tensor expert_offsets;
    Tensor global_expert_idx_table;  // the full (groups, extent, experts_per_chip) table, replicated
    Tensor output;                   // combine's output [1, 1, seq, top-k, H] bf16 ROW_MAJOR
};

struct FlatCombineOverlapSharedVariables {
    // combine's forwarding buffer and semaphores live as long as the cached workload
    std::shared_ptr<tt::tt_metal::WorkloadDescriptor> combine;
};

struct FlatCombineOverlapFactory {
    using shared_variables_t = FlatCombineOverlapSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const FlatCombineOverlapParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const FlatCombineOverlapInputs& tensor_args,
        Tensor& tensor_return_value);

    // every address is in the cache key: nothing to patch
    static void override_runtime_arguments(
        cached_mesh_workload_t&, const FlatCombineOverlapParams&, const FlatCombineOverlapInputs&, Tensor&) {}
};

struct FlatCombineOverlapDeviceOperation {
    using operation_attributes_t = FlatCombineOverlapParams;
    using tensor_args_t = FlatCombineOverlapInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<FlatCombineOverlapFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& t) {
        return t.output.tensor_spec();
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t& t) {
        return t.output;
    }
};

}  // namespace ttnn::operations::bringup::flat_routed_expert

namespace ttnn::prim::bringup {
ttnn::Tensor flat_combine_overlap(
    const ttnn::operations::bringup::flat_routed_expert::FlatCombineOverlapParams& params,
    const ttnn::operations::bringup::flat_routed_expert::FlatCombineOverlapInputs& inputs);
}  // namespace ttnn::prim::bringup
