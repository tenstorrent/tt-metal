// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct ChunkedGatedDeltaDeviceOperation {
    struct operation_attributes_t {
        uint32_t total_num_heads;
        uint32_t seq_len;
        uint32_t dim_k;
        uint32_t dim_v;
    };

    struct tensor_args_t {
        const Tensor& g_exp;
        const Tensor& factor;
        const Tensor& bktv;
        const Tensor& state;
    };

    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SingleCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Multi-core variant: parallelizes the per-head recurrence loop across the
    // storage-grid cores. Each core processes a contiguous slice of the global
    // head dimension; the per-head logic is identical to ``SingleCore``.
    struct MultiCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<SingleCore, MultiCore>;

    static operation_attributes_t compute_operation_attributes(const tensor_args_t& tensor_args);

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
ttnn::experimental::prim::ChunkedGatedDeltaDeviceOperation::tensor_return_value_t chunked_gated_delta(
    const Tensor& g_exp, const Tensor& factor, const Tensor& bktv, const Tensor& state);
}  // namespace ttnn::prim
