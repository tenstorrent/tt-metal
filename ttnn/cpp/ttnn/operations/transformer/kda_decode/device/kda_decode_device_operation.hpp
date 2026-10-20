// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include "kda_decode_device_operation_types.hpp"
#include "kda_decode_program_factory.hpp"

namespace ttnn::prim {

// Device operation returning {o, recurrent (aliased), conv_state (aliased)}.
struct KdaDecodeDeviceOperation {
    using operation_attributes_t = KdaDecodeParams;
    using tensor_args_t = KdaDecodeInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<KdaDecodeProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> kda_decode(
    const Tensor& qkv,
    const Tensor& gates,
    const Tensor& conv_state,
    const Tensor& recurrent,
    const Tensor& taps,
    const Tensor& chan_consts,
    const Tensor& o_norm_w,
    const Tensor& selectors,
    uint32_t num_heads,
    float gate_lower_bound,
    float l2_eps,
    float norm_eps,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
