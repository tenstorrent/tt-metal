// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "data_movement_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded_pybind.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/reshard_pybind.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved_pybind.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_pybind.hpp"
#include "ttnn/operations/data_movement/chunk/chunk_pybind.hpp"
#include "ttnn/operations/data_movement/clone/clone_pybind.hpp"
#include "ttnn/operations/data_movement/concat/concat_pybind.hpp"
#include "ttnn/operations/data_movement/copy/copy_pybind.hpp"
#include "ttnn/operations/data_movement/expand/expand_pybind.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad_pybind.hpp"
#include "ttnn/operations/data_movement/fill_rm/fill_rm_pybind.hpp"
#include "ttnn/operations/data_movement/fold/fold_pybind.hpp"
#include "ttnn/operations/data_movement/indexed_fill/indexed_fill_pybind.hpp"
#include "ttnn/operations/data_movement/move/move_pybind.hpp"
#include "ttnn/operations/data_movement/non_zero_indices/non_zero_indices_pybind.hpp"
#include "ttnn/operations/data_movement/pad/pad_pybind.hpp"
#include "ttnn/operations/data_movement/permute/permute_pybind.hpp"
#include "ttnn/operations/data_movement/repeat/repeat_pybind.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave_pybind.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape_pybind.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_pybind.hpp"
#include "ttnn/operations/data_movement/roll/roll_pybind.hpp"
#include "ttnn/operations/data_movement/view/view_pybind.hpp"
#include "ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/interleaved_to_sharded_partial_pybind.hpp"
#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/sharded_to_interleaved_partial_pybind.hpp"
#include "ttnn/operations/data_movement/slice/slice_pybind.hpp"
#include "ttnn/operations/data_movement/split/split_pybind.hpp"
#include "ttnn/operations/data_movement/squeeze/squeeze_pybind.hpp"
#include "ttnn/operations/data_movement/stack/stack_pybind.hpp"
#include "ttnn/operations/data_movement/tilize/tilize_pybind.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding_pybind.hpp"
#include "ttnn/operations/data_movement/transpose/transpose_pybind.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze_pybind.hpp"
#include "ttnn/operations/data_movement/untilize/untilize_pybind.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding_pybind.hpp"

namespace ttnn::operations::data_movement {

void py_module(py::module& module) {
    bind_fill_pad(module);
    bind_fill_rm(module);
    bind_fold_operation(module);
    bind_non_zero_indices(module);
    clone::bind_clone_operation(module);
    detail::bind_concat(module);
    detail::bind_indexed_fill(module);
    detail::bind_pad(module);
    detail::bind_permute(module);
    detail::bind_repeat_interleave(module);
    detail::bind_slice(module);
    detail::bind_split(module);
    detail::bind_tilize(module);
    detail::bind_tilize_with_val_padding(module);
    detail::bind_tilize_with_zero_padding(module);
    detail::bind_transpose(module);
    detail::bind_untilize(module);
    detail::bind_untilize_with_unpadding(module);
    detail::py_bind_assign(module);
    detail::py_bind_bcast(module);
    detail::py_bind_copy(module);
    detail::py_bind_move(module);
    py_bind_chunk(module);
    py_bind_expand(module);
    py_bind_interleaved_to_sharded(module);
    py_bind_interleaved_to_sharded_partial(module);
    py_bind_repeat(module);
    py_bind_reshape(module);
    py_bind_reshape_view(module);
    py_bind_view(module);
    py_bind_reshard(module);
    py_bind_sharded_to_interleaved(module);
    py_bind_sharded_to_interleaved_partial(module);
    py_bind_squeeze(module);
    py_bind_stack(module);
    py_bind_unsqueeze(module);
    py_bind_roll(module);
}
}  // namespace ttnn::operations::data_movement
