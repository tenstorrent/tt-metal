// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::operations::experimental::quasar {

// Output-dtype rule: FLOAT32/UINT16 pass through; every other input dtype collapses to BFLOAT16 on RM output.
tt::tt_metal::DataType fold_output_dtype(tt::tt_metal::DataType input_dtype);

// SRC0/SRC1 entries per C-tile; > 1 lets untilize fill one group while the writer drains the
// previous. Factory + L1 capacity predicate scale by this so a depth change stays in sync.
inline constexpr uint32_t kFoldSrcCbDepthPerCTile = 2;

// Bytes the tile-native writer's per-super-block RM scratch needs (one output row of contiguous sticks).
uint64_t tile_native_fold_scratch_bytes(const Tensor& input_tensor, uint32_t stride_h, uint32_t stride_w);

// One source of truth for the tile-native gate: returns nullopt when supported, otherwise a short
// reason string. validate_fold FATALs on the reason; the composite consults the boolean shim below.
std::optional<std::string> tile_native_fold_rejection_reason(
    const Tensor& input_tensor, uint32_t stride_h, uint32_t stride_w);

inline bool is_tile_native_fold_supported(const Tensor& t, uint32_t stride_h, uint32_t stride_w) {
    return !tile_native_fold_rejection_reason(t, stride_h, stride_w).has_value();
}

struct Fold {
    struct operation_attributes_t {
        uint32_t stride_h{};
        uint32_t stride_w{};
        bool is_sharded{};
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct MultiCore {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    struct MultiCoreDRAMFold {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<MultiCore, MultiCoreDRAMFold>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::quasar

namespace ttnn::prim::qsr {
ttnn::operations::experimental::quasar::Fold::tensor_return_value_t fold(
    const ttnn::Tensor& input_tensor, uint32_t stride_h, uint32_t stride_w);
}  // namespace ttnn::prim::qsr
