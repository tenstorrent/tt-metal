// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "mix_streams_device_operation_types.hpp"
#include "mix_streams_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::mix_streams {

struct MixStreamsDeviceOperation {
    using operation_attributes_t = MixStreamsParams;
    using tensor_args_t = MixStreamsInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = MixStreamsTensorReturn;
    using program_factory_t = std::variant<MixStreamsProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// True when the inputs satisfy the fused kernel's preconditions (see
// ``validate_tensors`` in the .cpp for the individual constraints). Lets the
// composite ``mix_streams`` entry point fall back to the eager op sequence for
// shapes the single-kernel path does not cover.
bool is_fusable(const Tensor& post, const Tensor& comb, const Tensor& sublayer_out, const Tensor& streams);

}  // namespace ttnn::operations::experimental::deepseek::mix_streams

namespace ttnn::prim {

// Fused hyper-connection stream-mixing step, as a single device op:
//   out[t, m, n] = post[t, m] * sublayer_out[t, n] + sum_k comb[t, k, m] * streams[t, k, n]
// where t indexes the B*S flattened tokens, m/k the hc streams and n the hidden dim.
// Returns the new residual-stream stack [B, S, hc, D].
Tensor mix_streams(
    const Tensor& post,
    const Tensor& comb,
    const Tensor& sublayer_out,
    const Tensor& streams,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::prim
