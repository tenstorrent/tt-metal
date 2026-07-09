// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "neighbor_pad_conv3d_device_operation_types.hpp"
#include "neighbor_pad_conv3d_program_factory.hpp"

namespace ttnn::experimental::prim {

struct NpConv3dDeviceOperation {
    using operation_attributes_t = NpConv3dParams;
    using tensor_args_t = NpConv3dInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<NpConv3dMeshWorkloadFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Primitive entry point: runs fused NeighborPad (fabric-only H-halo) + Conv3d in a single program.
// halo_buffer must be a pre-allocated compact DRAM buffer sized for
//   2 * outer_dim_size * np_padding_h * W sticks (+ W-halo section when np_padding_w > 0).
Tensor neighbor_pad_conv3d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const Tensor& halo_buffer,
    const ttnn::experimental::prim::NpConv3dParams& params);

}  // namespace ttnn::prim
