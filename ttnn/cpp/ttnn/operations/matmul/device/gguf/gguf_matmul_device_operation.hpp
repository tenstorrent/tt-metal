// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/operations/matmul/device/gguf/gguf_matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/gguf/gguf_matmul_program_factory.hpp"

namespace ttnn::prim {

// input [..., M, K] (BFLOAT16) x weight [K, N] (a GGUF dtype) -> [..., M, N] BFLOAT16, M <= 32, on Blackhole. The
// weight's records are decoded on the SFPU and multiplied from DEST (api/compute/experimental/gguf_decode.h,
// dest_srca_mm.h); ttnn::matmul and ttnn::linear run it for every 32-row block when the weight is GGUF.
struct GgufMatmulDeviceOperation {
    using operation_attributes_t = GgufMatmulParams;
    using tensor_args_t = GgufMatmulInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<GgufMatmulProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor gguf_matmul(const Tensor& input, const Tensor& weight, const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim
