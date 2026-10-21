// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "relay_affine_prefix_nanobind.hpp"

#include "relay_affine_prefix.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace ttnn::operations::experimental::kda::relay_affine_prefix::detail {

void bind_relay_affine_prefix(nb::module_& mod) {
    ttnn::bind_function<"relay_affine_prefix", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        The sequence-parallel prefix as a relay around the SP ring: rank ``p`` (chronological position, from
        ``actual_start``) takes its entry state ``E_p`` from its predecessor, computes
        ``E_{p+1} = A_p E_p + B_p`` from its packed ``[A | B]`` summary and forwards it, one head per core and
        one tile column at a time; the last rank sends the final state on to the first. ``cores`` lists a core
        per head, then a sender core per link. The landing buffers (FP32 L1, sharded one head per head core) and
        the two global ``semaphores`` (entry and final arrival) must outlive the call: a peer writes them before
        this rank's call starts. Calls must be separated by an SP-line barrier.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: FP32 ``[B*H, K, K]`` entry states and the final state (valid on the
            first and last chronological ranks).
        )doc",
        &ttnn::experimental::kda::relay_affine_prefix,
        nb::arg("packed").noconvert(),
        nb::arg("initial_state").noconvert(),
        nb::arg("actual_start").noconvert(),
        nb::arg("entry_landing").noconvert(),
        nb::arg("state_staging").noconvert(),
        nb::arg("final_landing").noconvert(),
        nb::arg("semaphores"),
        nb::kw_only(),
        nb::arg("local_rows"),
        nb::arg("num_links"),
        nb::arg("cores"),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("entry_memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::relay_affine_prefix::detail
