// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "data_movement_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded_nanobind.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/reshard_nanobind.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved_nanobind.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_nanobind.hpp"
#include "ttnn/operations/data_movement/chunk/chunk_nanobind.hpp"
#include "ttnn/operations/data_movement/clone/clone_nanobind.hpp"
#include "ttnn/operations/data_movement/concat/concat_nanobind.hpp"
#include "ttnn/operations/data_movement/copy/copy_nanobind.hpp"
#include "ttnn/operations/data_movement/expand/expand_nanobind.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad_nanobind.hpp"
#include "ttnn/operations/data_movement/fill_rm/fill_rm_nanobind.hpp"
#include "ttnn/operations/data_movement/fold/fold_nanobind.hpp"
#include "ttnn/operations/data_movement/indexed_fill/indexed_fill_nanobind.hpp"
#include "ttnn/operations/data_movement/moe_expert_token_remap/moe_expert_token_remap_nanobind.hpp"
#include "ttnn/operations/data_movement/moe_routing_remap/moe_routing_remap_nanobind.hpp"
#include "ttnn/operations/data_movement/move/move_nanobind.hpp"
#include "ttnn/operations/data_movement/non_zero_indices/non_zero_indices_nanobind.hpp"
#include "ttnn/operations/data_movement/pad/pad_nanobind.hpp"
#include "ttnn/operations/data_movement/permute/permute_nanobind.hpp"
#include "ttnn/operations/data_movement/repeat/repeat_nanobind.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave_nanobind.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape_nanobind.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_nanobind.hpp"
#include "ttnn/operations/data_movement/roll/roll_nanobind.hpp"
#include "ttnn/operations/data_movement/view/view_nanobind.hpp"
#include "ttnn/operations/data_movement/scatter/scatter_nanobind.hpp"
#include "ttnn/operations/data_movement/scatter/tosa_scatter_nanobind.hpp"
#include "ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/interleaved_to_sharded_partial_nanobind.hpp"
#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/sharded_to_interleaved_partial_nanobind.hpp"
#include "ttnn/operations/data_movement/slice/slice_nanobind.hpp"
#include "ttnn/operations/data_movement/split/split_nanobind.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze_nanobind.hpp"
#include "ttnn/operations/data_movement/stack/stack_nanobind.hpp"
#include "ttnn/operations/data_movement/tilize/tilize_nanobind.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding_nanobind.hpp"
#include "ttnn/operations/data_movement/transpose/transpose_nanobind.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze_nanobind.hpp"
#include "ttnn/operations/data_movement/untilize/untilize_nanobind.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding_nanobind.hpp"
#include "ttnn/operations/data_movement/sort/sort_nanobind.hpp"
#include "ttnn/operations/data_movement/gather/gather_nanobind.hpp"
#include "ttnn/operations/data_movement/gather/tosa/gather_tosa_nanobind.hpp"

namespace ttnn::operations::data_movement {

void py_module(nb::module_& mod) {
    bind_fill_pad(mod);
    bind_fill_rm(mod);
    bind_fold_operation(mod);
    bind_non_zero_indices(mod);
    clone::bind_clone_operation(mod);
    detail::bind_concat(mod);
    detail::bind_indexed_fill(mod);
    detail::bind_pad(mod);
    detail::bind_permute(mod);
    detail::bind_repeat_interleave(mod);
    detail::bind_slice(mod);
    detail::bind_split(mod);
    detail::bind_tilize(mod);
    detail::bind_tilize_with_val_padding(mod);
    detail::bind_tilize_with_zero_padding(mod);
    detail::bind_transpose(mod);
    detail::bind_untilize(mod);
    detail::bind_untilize_with_unpadding(mod);
    detail::bind_scatter(mod);
    detail::bind_scatter_add(mod);
    detail::bind_tosa_scatter(mod);
    detail::bind_assign(mod);
    detail::bind_bcast(mod);
    detail::bind_copy(mod);
    detail::bind_moe_expert_token_remap(mod);
    detail::bind_moe_routing_remap(mod);
    detail::bind_move(mod);
    bind_chunk(mod);
    bind_expand(mod);
    bind_interleaved_to_sharded(mod);
    bind_interleaved_to_sharded_partial(mod);
    bind_repeat(mod);
    bind_reshape_enum(mod);
    bind_reshape(mod);
    bind_reshape_view(mod);
    bind_view(mod);
    bind_reshard(mod);
    bind_sharded_to_interleaved(mod);
    bind_sharded_to_interleaved_partial(mod);
    bind_squeeze(mod);
    bind_stack(mod);
    bind_unsqueeze(mod);
    bind_roll(mod);
    detail::bind_sort_operation(mod);
    detail::bind_gather_operation(mod);
    detail::bind_gather_tosa_operation(mod);
}
}  // namespace ttnn::operations::data_movement
