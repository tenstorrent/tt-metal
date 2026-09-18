// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chronological_selections_device_operation.hpp"
#include "kernels/chronology.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
namespace ttnn::experimental::prim {
ChronologicalSelectionsOperation::program_factory_t ChronologicalSelectionsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ChronologicalSelectionsFactory{};
}
void ChronologicalSelectionsOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& in) {
    kda_factory_detail::check_actual_start(in.actual_start, in.actual_start, "chronological_selections");
    if (in.actual_end) {
        kda_factory_detail::check_actual_start(in.actual_start, *in.actual_end, "chronological_selections");
    }
    TT_FATAL(
        a.sequence_parallel_axis < in.actual_start.device()->shape().dims() && a.local_rows > 0 &&
            a.local_rows % 32 == 0,
        "chronological_selections: invalid partition geometry");
    TT_FATAL(a.batch_heads > 0 && a.key_dim > 0 && a.value_dim > 0, "chronological_selections: invalid state geometry");
}
ChronologicalSelectionsOperation::spec_return_value_t ChronologicalSelectionsOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return {tt::tt_metal::TensorSpec(
        Shape(
            {kda_chronology::selection::record_count(in.actual_start.device()->shape()[a.sequence_parallel_axis]),
             kda_chronology::selection::record_width}),
        tt::tt_metal::TensorLayout(
            DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG))};
}
ChronologicalSelectionsOperation::tensor_return_value_t ChronologicalSelectionsOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(a, in)[0], in.actual_start.device())};
}
}  // namespace ttnn::experimental::prim
