// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Python bindings of the derived bring-up ops (INDEX.md): ttnn._ttnn.operations.bringup, re-exported as
// ttnn.bringup. fork_op.py adds one declaration and one call per fork between the markers.

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// BEGIN FORKED OPS (fork_op.py)
namespace ttnn::operations::bringup::indexer_score::detail {
void bind_indexer_score(nb::module_& mod);
}
namespace ttnn::operations::bringup {
void bind_sdpa(nb::module_& mod);
}
namespace ttnn::operations::bringup::offset_cumsum::detail {
void bind_experimental_offset_cumsum_operation(nb::module_& mod);
}
namespace ttnn::operations::bringup::detail {
void bind_combine(nb::module_& mod);
}
namespace ttnn::operations::bringup::detail {
void bind_dispatch(nb::module_& mod);
}
namespace ttnn::operations::bringup::detail {
void bind_unified_routed_expert_ffn(nb::module_& mod);
}
namespace ttnn::operations::bringup::rms_norm_ttnn::detail {
void bind_rms_norm_ttnn(nb::module_& mod);
}
namespace ttnn::operations::bringup::mhc_post_ttnn::detail {
void bind_mhc_post_ttnn(nb::module_& mod);
}
// END FORKED OPS (fork_op.py)

namespace ttnn::bringup {

void py_module(nb::module_& mod) {
    // BEGIN FORKED OPS (fork_op.py)
    ::ttnn::operations::bringup::indexer_score::detail::bind_indexer_score(mod);
    ::ttnn::operations::bringup::bind_sdpa(mod);
    ::ttnn::operations::bringup::offset_cumsum::detail::bind_experimental_offset_cumsum_operation(mod);
    ::ttnn::operations::bringup::detail::bind_combine(mod);
    ::ttnn::operations::bringup::detail::bind_dispatch(mod);
    ::ttnn::operations::bringup::detail::bind_unified_routed_expert_ffn(mod);
    ::ttnn::operations::bringup::rms_norm_ttnn::detail::bind_rms_norm_ttnn(mod);
    ::ttnn::operations::bringup::mhc_post_ttnn::detail::bind_mhc_post_ttnn(mod);
    // END FORKED OPS (fork_op.py)
}

}  // namespace ttnn::bringup
