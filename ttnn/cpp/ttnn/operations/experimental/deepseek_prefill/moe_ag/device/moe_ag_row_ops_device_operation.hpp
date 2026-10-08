// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

#include "moe_ag_row_ops_device_operation_types.hpp"

namespace ttnn::prim {

// The all-gather MoE block's row adds / untilizes: single-output descriptor ops over the whole worker grid.
#define MOE_AG_ROW_OP(NAME)                                                                                      \
    struct NAME##DeviceOperation {                                                                               \
        using operation_attributes_t = NAME##Params;                                                             \
        using tensor_args_t = NAME##Inputs;                                                                      \
        using spec_return_value_t = tt::tt_metal::TensorSpec;                                                    \
        using tensor_return_value_t = Tensor;                                                                    \
        struct ProgramFactory {                                                                                  \
            static tt::tt_metal::ProgramDescriptor create_descriptor(                                            \
                const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);                    \
        };                                                                                                       \
        using program_factory_t = std::variant<ProgramFactory>;                                                  \
        static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);         \
        static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);    \
        static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&); \
    };

MOE_AG_ROW_OP(MoeAgSumRowsTiled)
MOE_AG_ROW_OP(MoeAgAddRows)
MOE_AG_ROW_OP(MoeAgUntilizeActive)
MOE_AG_ROW_OP(MoeAgUntilizeX)

#undef MOE_AG_ROW_OP

Tensor moe_ag_sum_rows_tiled(
    const Tensor& src,
    uint32_t num_rows,
    uint32_t num_blocks,
    uint32_t block_stride,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

Tensor moe_ag_add_rows(
    const Tensor& a,
    const Tensor& b,
    const Tensor& chip_info,
    uint32_t num_rows,
    uint32_t a_offset = 0,
    uint32_t b_offset = 0,
    bool info_offset = false,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

Tensor moe_ag_untilize_active(
    const Tensor& y,
    const Tensor& counts,
    const Tensor& regions,
    const Tensor& local_slot_map,
    uint32_t experts_per_chip,
    uint32_t tiles_per_block = 32,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

Tensor moe_ag_untilize_x(const Tensor& x, const std::optional<Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
