// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "transformer_nanobind.hpp"

#include <cstddef>
#include <optional>
#include <tt_stl/reflection.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "attention_softmax/attention_softmax_nanobind.hpp"
#include "concatenate_heads/concatenate_heads_nanobind.hpp"
#include "sdpa/sdpa_nanobind.hpp"
#include "sdpa_config.hpp"
#include "sdpa_decode/sdpa_decode_nanobind.hpp"
#include "sdpa_windowed/sdpa_windowed_nanobind.hpp"
#include "split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_nanobind.hpp"

namespace ttnn::operations::transformer {

void py_module(nb::module_& mod) {
    nb::class_<SDPAProgramConfig>(mod, "SDPAProgramConfig")
        .def(
            nb::init<CoreCoord, std::optional<CoreRangeSet>, std::size_t, std::size_t, std::optional<bool>>(),
            nb::kw_only(),
            nb::arg("compute_with_storage_grid_size"),
            nb::arg("sub_core_grids") = nb::none(),
            nb::arg("q_chunk_size").noconvert(),
            nb::arg("k_chunk_size").noconvert(),
            nb::arg("exp_approx_mode") = nb::none())
        .def_rw("compute_with_storage_grid_size", &SDPAProgramConfig::compute_with_storage_grid_size)
        .def_rw("sub_core_grids", &SDPAProgramConfig::sub_core_grids)
        .def_rw("q_chunk_size", &SDPAProgramConfig::q_chunk_size)
        .def_rw("k_chunk_size", &SDPAProgramConfig::k_chunk_size)
        .def_rw("exp_approx_mode", &SDPAProgramConfig::exp_approx_mode)
        .def("__repr__", [](const SDPAProgramConfig& config) {
            return fmt::format(
                "SDPAProgramConfig(compute_with_storage_grid_size={}, sub_core_grids={}, q_chunk_size={}, k_chunk_size={}, exp_approx_mode={})",
                config.compute_with_storage_grid_size,
                config.sub_core_grids,
                config.q_chunk_size,
                config.k_chunk_size,
                config.exp_approx_mode);
        });

    bind_attention_softmax(mod);
    bind_concatenate_heads(mod);
    bind_split_query_key_value_and_split_heads(mod);

    bind_sdpa(mod);
    bind_sdpa_decode(mod);
    bind_sdpa_windowed(mod);
}

}  // namespace ttnn::operations::transformer
