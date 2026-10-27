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
#include "ttnn/operations/data_movement/common/fold_common.hpp"

namespace ttnn::operations::experimental::quasar {

// Re-export shared tile-native-fold gate symbols; single source in data_movement/common/fold_common.hpp.
using ttnn::operations::data_movement::fold_common::fold_output_dtype;
using ttnn::operations::data_movement::fold_common::is_tile_native_fold_supported;
using ttnn::operations::data_movement::fold_common::kFoldL1CodeStackReserveBytes;
using ttnn::operations::data_movement::fold_common::kFoldSrcCbDepthPerCTile;
using ttnn::operations::data_movement::fold_common::tile_native_fold_rejection_reason;
using ttnn::operations::data_movement::fold_common::tile_native_fold_scratch_bytes;

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
