// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefix_scan_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "prefix_scan_program_factory.hpp"

namespace ttnn::operations::experimental::ssm {

void PrefixScan::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    TT_FATAL(input_tensors.size() == 3, "Expected 3 input tensors (A, Bx, H)");

    const auto& a = input_tensors[0];
    const auto& bx = input_tensors[1];
    TT_FATAL(a.dtype() == bx.dtype(), "Expected input tensors to have the same data type");
    TT_FATAL(a.layout() == Layout::TILE && bx.layout() == Layout::TILE, "Expected input tensors to be tile layout");
    TT_FATAL(a.get_legacy_shape() == bx.get_legacy_shape(), "Expected input tensors to have the same shape");

    const auto& shape = a.get_legacy_shape();
    TT_FATAL(shape.rank() == 4, "Expected input tensors to be rank 4");
    TT_FATAL(shape[0] == 1 && shape[1] == 1, "Dimension 0 and 1 should be size 1");
    TT_FATAL(shape[2] >= tt::constants::TILE_HEIGHT && shape[2] % tt::constants::TILE_HEIGHT == 0,
             "Sequence length should be a multiple of 32");

    const auto& h = input_tensors.at(2);
    TT_FATAL(h.dtype() == DataType::BFLOAT16, "Expected initial hidden state to be bfloat16");
    TT_FATAL(h.layout() == Layout::ROW_MAJOR, "Expected initial hidden state to be row-major");

    TT_FATAL(a.is_sharded() && bx.is_sharded() && h.is_sharded(), "Expected input tensors to be sharded");
    TT_FATAL(a.shard_spec().has_value() && bx.shard_spec().has_value() && h.shard_spec().has_value(),
             "Expected input tensors to be sharded");
    TT_FATAL(a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
             "Expected A tensor to be row major orientation");
    TT_FATAL(bx.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
             "Expected Bx tensor to be row major orientation");
    TT_FATAL(h.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
             "Expected h tensor to be row major orientation");
}

std::vector<tt::tt_metal::LegacyShape> PrefixScan::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& a = input_tensors.at(0);
    return {a.get_legacy_shape()};
}

std::vector<Tensor> PrefixScan::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->dtype, Layout::TILE, this->memory_config);
}

operation::ProgramWithCallbacks PrefixScan::create_program(const std::vector<Tensor>& input_tensors,
                                                           std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    const auto& bx = input_tensors.at(1);
    const auto& h = input_tensors.at(2);
    auto& output = output_tensors.at(0);
    auto device_compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    return detail::multi_core_ssm_prefix_scan(a, bx, h, output, math_fidelity, device_compute_with_storage_grid_size);
}
}  // namespace ttnn::operations::experimental::ssm
