// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sp_matmul_schedule_test_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/core_coord.hpp>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/ccl/sp_matmul_fusion_common/sp_matmul_fusion_common.hpp"
#include "ttnn/operations/experimental/ccl/sp_matmul_fusion_common/sp_matmul_schedule_test.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::ccl {

void bind_sp_matmul_schedule_test(nb::module_& mod) {
    ttnn::bind_function<"sp_matmul_schedule_test", "ttnn.experimental.">(
        mod,
        R"doc(
        Test-only: runs the 2D-mcast matmul on `input_tensor` [B*T,1,Ms,K] x `weight_tensor` with the sequence-parallel
        slice schedule mechanism (MatmulFusedOpSignaler SP_REDUCE_SCATTER, no CCL). Iteration j of the matmul batch
        loop reads sub-batch `schedule_words[j] & 0xFF` and writes sub-batch `(schedule_words[j] >> 8) & 0xFF`; both
        must be permutations of 0..B*T-1. Default (ag_mode=False): SP_REDUCE_SCATTER signaler, wait/local bits must
        be 0. ag_mode=True: SP_ALL_GATHER signaler with the input itself as the alternate (local) in0 buffer; is_local
        and wait_dir may be set, wait_count must be 0. Result equals ttnn.matmul with the same program config
        (fuse_batch=False).
        )doc",
        &ttnn::experimental::sp_matmul_schedule_test,
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("schedule_words"),
        nb::arg("transpose_b"),
        nb::arg("program_config"),
        nb::kw_only(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("ag_mode") = false);

    ttnn::bind_function<"sp_sub_batched_view", "ttnn.experimental.">(
        mod,
        R"doc(
        Metadata-only reshape [B,1,S,X] -> [B*num_slices,1,S/num_slices,X] of a tiled device tensor (same allocation);
        sub-batch b*num_slices+t is dim-2 slice t of batch b.
        )doc",
        &ttnn::experimental::ccl::sub_batched_view,
        nb::arg("input_tensor"),
        nb::arg("num_slices"));

    ttnn::bind_function<"sp_matmul_program_config", "ttnn.experimental.">(
        mod,
        R"doc(
        Derived MatmulMultiCoreReuseMultiCastProgramConfig (fuse_batch=False) for one sequence-parallel sub-batched
        matmul: `input_tensor` is the [B*T,1,S/T,K] view, `grid` the matmul core rectangle at (0,0).
        )doc",
        &ttnn::experimental::ccl::sp_matmul_program_config,
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("grid"),
        nb::arg("transpose_b"),
        nb::arg("compute_kernel_config"));
}

}  // namespace ttnn::operations::experimental::ccl
