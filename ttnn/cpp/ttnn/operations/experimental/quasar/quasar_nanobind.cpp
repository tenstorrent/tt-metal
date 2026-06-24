// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "quasar_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/quasar/pad/pad_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/tilize/tilize_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/move/move_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/untilize_with_unpadding/untilize_with_unpadding_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/slice/slice_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/transpose/transpose_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/reshard/reshard_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/pool_generic/generic_pools_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/conv2d/conv2d_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/matmul/matmul_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/binary/binary_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/fold/fold_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/to_memory_config/to_memory_config_nanobind.hpp"

namespace ttnn::operations::experimental::quasar {

void bind_quasar(nb::module_& mod) {
    auto m_quasar = mod.def_submodule("quasar", "Quasar (metal 2.0) operations");

    // Data-movement ops (host namespace ttnn::operations::experimental::quasar::detail).
    detail::bind_pad(m_quasar);
    detail::bind_tilize(m_quasar);
    detail::bind_move(m_quasar);
    detail::bind_untilize_with_unpadding(m_quasar);
    detail::bind_slice(m_quasar);
    detail::bind_slice_descriptor(m_quasar);
    detail::bind_transpose(m_quasar);
    detail::bind_reshard(m_quasar);

    // conv2d.
    detail::bind_conv2d(m_quasar);

    // pool (host namespace ttnn::operations::pool::quasar).
    ttnn::operations::pool::quasar::py_module(m_quasar);

    // matmul (its own py_module binds matmul/linear/addmm/sparse_matmul + program-config classes).
    matmul::py_module(m_quasar);

    // binary front-end (add/subtract/multiply/... -> quasar binary_ng device op).
    binary::py_module(m_quasar);

    // fold (compositional data-movement op).
    detail::bind_fold_operation(m_quasar);

    // to_memory_config (dispatches to quasar reshard / interleaved_to_sharded / sharded_to_interleaved).
    detail::bind_to_memory_config(m_quasar);

    // NOTE: halo and binary_ng have no python binding (internal device backends).
}

}  // namespace ttnn::operations::experimental::quasar
