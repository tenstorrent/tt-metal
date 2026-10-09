// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "concat_device_operation_types.hpp"
#include "concat_program_factory.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim::qsr {

// Quasar (Metal 2.0) copy of ttnn::prim::ConcatDeviceOperation.
//
// Only the generic TensorAccessor-based factory is ported. The original's other factories are not:
// the zero-copy L1-sharded ones (S2S tiled / RM / multi, S2I, block-sharded) alias input shards as
// circular buffers, and the tiled-unaligned one is a legacy ProgramDescriptor factory with
// untilize/retilize compute kernels. The host op (concat.cpp) routes everything to this factory.
struct ConcatDeviceOperation {
    using operation_attributes_t = ConcatParams;
    using tensor_args_t = ConcatInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<ConcatProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

ConcatDeviceOperation::tensor_return_value_t concat(
    const std::vector<Tensor>& input_tensors,
    std::int64_t dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::prim::qsr
