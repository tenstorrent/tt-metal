// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chronological_topology_device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
namespace ttnn::experimental::prim {
ChronologyOperation::program_factory_t ChronologyOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ChronologyFactory{};
}
void ChronologyOperation::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& in) {
    TT_FATAL(a.sp_size > 0 && a.local_rows > 0 && a.local_rows % 32 == 0, "chronology: invalid partition geometry");
    TT_FATAL(a.batch_heads > 0 && a.key_dim > 0 && a.value_dim > 0, "chronology: invalid state geometry");
    for (const auto* t : {&in.start, &in.rank}) {
        kda_factory_detail::check_allocated_device_tensor(*t, "chronology", "control");
        kda_factory_detail::check_layout(*t, Layout::ROW_MAJOR, "chronology", "control");
        kda_factory_detail::check_dtype(*t, DataType::UINT32, "chronology", "control");
        kda_factory_detail::check_interleaved(*t, "chronology", "control");
        TT_FATAL(t->logical_shape().volume() == 1, "chronology: start and rank must be scalar tensors");
    }
    kda_factory_detail::check_same_device(in.start, in.rank, "chronology", "rank");
}
ChronologyOperation::spec_return_value_t ChronologyOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    return {tt::tt_metal::TensorSpec(
        Shape({8 + 2 * a.sp_size, 8}),
        tt::tt_metal::TensorLayout(
            DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG))};
}
ChronologyOperation::tensor_return_value_t ChronologyOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(a, in)[0], in.start.device())};
}
}  // namespace ttnn::experimental::prim
