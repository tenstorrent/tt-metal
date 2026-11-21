// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation_types.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"

#include <optional>
#include <variant>
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim::qsr {

struct SliceDeviceOperation {
    using operation_attributes_t = SliceParams;
    using tensor_args_t = SliceInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<SliceProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Port of tt-metal e517beb3f41 (#47602): custom hash to avoid false program-cache hits on colliding shapes.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};

SliceDeviceOperation::tensor_return_value_t slice(
    const Tensor& input,
    const ttnn::Shape& slice_start,
    const ttnn::Shape& slice_end,
    const ttnn::Shape& step,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool use_tensor_args,
    std::optional<Tensor> start_tensor = std::nullopt,
    std::optional<Tensor> end_tensor = std::nullopt,
    const std::optional<uint32_t>& slice_dim = std::nullopt,
    const std::optional<uint32_t>& num_devices = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

// True when the slice kernel can address `memory_config` of a tensor of `layout` by page id: any TILE
// tensor, and a ROW_MAJOR tensor whose page is a whole row (interleaved or height sharded). A ROW_MAJOR
// block / width sharded tensor splits a row over clusters; quasar::slice unshards it first.
bool slice_addresses_whole_pages(tt::tt_metal::Layout layout, const tt::tt_metal::MemoryConfig& memory_config);

// True when a TILE slice cannot move whole tiles (its last two dims do not start on a tile boundary or are
// strided) and the kernel assembles every output tile from input elements instead.
bool slice_gathers_tile_elements(const SliceParams& args, const Tensor& input);

// Dtypes the tile element gather moves: plain 16-bit / 32-bit elements. Block formats have no addressable
// element; quasar::slice sends those through ROW_MAJOR.
bool slice_tile_gather_supports(tt::tt_metal::DataType dtype);

}  // namespace ttnn::prim::qsr
