// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include "interleaved_to_sharded.hpp"
#include "e09_sharding/Stage2/exercise_cpp/sharded_add.hpp"
#include "ttnn-nanobind/decorators.hpp"

namespace nb = nanobind;

NB_MODULE(_e09_exercise, mod) {
    mod.doc() = "E09 Exercise: Interleaved to Sharded + Sharded Add";

    ttnn::bind_registered_operation(
        mod,
        ttnn::s09s1_interleaved_to_sharded,
        R"doc(s09s1_interleaved_to_sharded(input: ttnn.Tensor, shard_strategy: int) -> ttnn.Tensor

        Convert a DRAM INTERLEAVED tensor to L1 sharded (HEIGHT, WIDTH, or BLOCK).
        shard_strategy: pass ttnn.TensorMemoryLayout.<TYPE>.value
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::s09s1_interleaved_to_sharded)& self,
               const ttnn::Tensor& input,
               uint32_t shard_strategy) -> ttnn::Tensor { return self(input, shard_strategy); },
            nb::arg("input"),
            nb::arg("shard_strategy")});

    ttnn::bind_registered_operation(
        mod,
        ttnn::s09s2_sharded_add,
        R"doc(s09s2_sharded_add(input_a: ttnn.Tensor, input_b: ttnn.Tensor) -> ttnn.Tensor

        Elementwise add of two L1 SHARDED tensors. Both inputs must have
        the same shape and sharding config. Output is L1 SHARDED.
        )doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::s09s2_sharded_add)& self, const ttnn::Tensor& input_a, const ttnn::Tensor& input_b)
                -> ttnn::Tensor { return self(input_a, input_b); },
            nb::arg("input_a"),
            nb::arg("input_b")});
}
