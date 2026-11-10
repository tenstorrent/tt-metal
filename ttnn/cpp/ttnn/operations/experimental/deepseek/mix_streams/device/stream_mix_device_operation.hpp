// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "stream_mix_device_operation_types.hpp"
#include "stream_mix_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::mix_streams {

struct StreamMixDeviceOperation {
    using operation_attributes_t = StreamMixParams;
    using tensor_args_t = StreamMixInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = StreamMixTensorReturn;
    using program_factory_t = std::variant<StreamMixProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::mix_streams

namespace ttnn::prim {

// Hyper-connection stream mixing over stream-major streams [1, hc, T, D] (no per-token tile padding):
//   weights = comb [1, T, hc, hc]: out[i] = sum_j comb[t, j, i] * streams[j] + post[t, i] * sublayer_out
//   weights = pre  [1, T, 1, hc]:  out[0] = sum_j pre[t, 0, j] * streams[j]      (the collapse)
// Returns [1, n_out, T, D].
Tensor stream_mix(
    const Tensor& streams,
    const Tensor& weights,
    const std::optional<Tensor>& post = std::nullopt,
    const std::optional<Tensor>& sublayer_out = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::prim
