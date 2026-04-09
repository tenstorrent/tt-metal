// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::turbo_quant {

enum class TurboQuantOpType : uint8_t {
    BUCKETIZE,         // y_hat → indices
    GATHER_CENTROIDS,  // indices → centroid values
};

struct TurboQuantDeviceOperation {
    struct operation_attributes_t {
        TurboQuantOpType op_type;
        // For BUCKETIZE:  inner boundary values  (size = 2^bits − 1)
        // For GATHER_CENTROIDS: centroid values   (size = 2^bits)
        std::vector<float> params;
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Single multi-core program factory for both op types.
    struct MultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
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

// Fused bucketize: normalised rotated values → integer bucket indices (as BF16).
Tensor turbo_quant_bucketize(const Tensor& input_tensor, const std::vector<float>& boundaries);

// Fused gather centroids: integer indices (as BF16) → centroid values.
Tensor turbo_quant_gather_centroids(const Tensor& input_tensor, const std::vector<float>& centroids);

}  // namespace ttnn::prim
