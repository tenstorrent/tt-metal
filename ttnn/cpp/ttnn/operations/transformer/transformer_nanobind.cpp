// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    nb::enum_<RingProxyCase>(mod, "RingProxyCase")
        .value("NONE", RingProxyCase::None)
        .value("DIAG", RingProxyCase::Diag)
        .value("UP", RingProxyCase::Up)
        .value("DOWN", RingProxyCase::Down);

    nb::class_<SDPAProgramConfig>(mod, "SDPAProgramConfig")
        .def(
            nb::init<
                CoreCoord,
                std::optional<CoreRangeSet>,
                std::size_t,
                std::size_t,
                std::optional<bool>,
                uint32_t,
                RingProxyCase>(),
            nb::kw_only(),
            nb::arg("compute_with_storage_grid_size"),
            nb::arg("sub_core_grids") = nb::none(),
            nb::arg("q_chunk_size").noconvert(),
            nb::arg("k_chunk_size").noconvert(),
            nb::arg("exp_approx_mode") = nb::none(),
            nb::arg("max_cores_per_head_batch") = 16,
            nb::arg("ring_proxy_case") = RingProxyCase::None)
        .def_rw("compute_with_storage_grid_size", &SDPAProgramConfig::compute_with_storage_grid_size)
        .def_rw("sub_core_grids", &SDPAProgramConfig::sub_core_grids)
        .def_rw("q_chunk_size", &SDPAProgramConfig::q_chunk_size)
        .def_rw("k_chunk_size", &SDPAProgramConfig::k_chunk_size)
        .def_rw("exp_approx_mode", &SDPAProgramConfig::exp_approx_mode)
        .def_rw("max_cores_per_head_batch", &SDPAProgramConfig::max_cores_per_head_batch)
        .def_rw("ring_proxy_case", &SDPAProgramConfig::ring_proxy_case)
        .def("__repr__", [](const SDPAProgramConfig& config) {
            return fmt::format(
                "SDPAProgramConfig(compute_with_storage_grid_size={}, sub_core_grids={}, q_chunk_size={}, "
                "k_chunk_size={}, exp_approx_mode={}, max_cores_per_head_batch={}, ring_proxy_case={})",
                config.compute_with_storage_grid_size,
                config.sub_core_grids,
                config.q_chunk_size,
                config.k_chunk_size,
                config.exp_approx_mode,
                config.max_cores_per_head_batch,
                static_cast<uint8_t>(config.ring_proxy_case));
        });

    bind_attention_softmax(mod);
    bind_concatenate_heads(mod);
    bind_split_query_key_value_and_split_heads(mod);

    bind_sdpa(mod);
    bind_sdpa_decode(mod);
    bind_sdpa_windowed(mod);
}

}  // namespace ttnn::operations::transformer
