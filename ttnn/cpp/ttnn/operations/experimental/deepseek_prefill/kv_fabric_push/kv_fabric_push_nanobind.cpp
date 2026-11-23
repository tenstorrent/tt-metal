// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_fabric_push_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "kv_fabric_push.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::detail {

namespace nb = nanobind;

void bind_kv_fabric_push(nb::module_& mod) {
    using ttnn::experimental::KvFabricPush;
    using ttnn::experimental::KvPushCache;
    using ttnn::experimental::KvPushLaunch;
    using ttnn::experimental::KvPushLayer;
    using ttnn::experimental::KvPushRing;

    nb::class_<KvPushCache>(mod, "KvPushCache", "Where the KV push reads one cache of a compressed layer.")
        .def(nb::init<>())
        .def_rw("source", &KvPushCache::source)
        .def_rw("first_page", &KvPushCache::first_page)
        .def_rw("stripe_rows", &KvPushCache::stripe_rows)
        .def_rw("owners", &KvPushCache::owners)
        .def_rw("owners_are_chips", &KvPushCache::owners_are_chips);

    nb::class_<KvPushLayer>(mod, "KvPushLayer", "A compressed source layer the KV push writes.")
        .def(nb::init<>())
        .def_rw("ratio", &KvPushLayer::ratio)
        .def_rw("first_entry", &KvPushLayer::first_entry)
        .def_rw("kv", &KvPushLayer::kv)
        .def_rw("keys", &KvPushLayer::keys)
        .def_rw("key_tile_cols", &KvPushLayer::key_tile_cols)
        .def_rw("blocks", &KvPushLayer::blocks)
        .def_rw("pools", &KvPushLayer::pools);

    nb::class_<KvPushRing>(mod, "KvPushRing", "A window ring the last KV push launch writes.")
        .def(nb::init<>())
        .def_rw("destination", &KvPushRing::destination)
        .def_rw("addr", &KvPushRing::addr)
        .def_rw("first_source_page", &KvPushRing::first_source_page)
        .def_rw("pages", &KvPushRing::pages)
        .def_rw("first_page", &KvPushRing::first_page);

    nb::class_<KvPushLaunch>(mod, "KvPushLaunch", "What one KV push launch moved, and its host seconds.")
        .def_ro("bytes", &KvPushLaunch::bytes)
        .def_ro("pages", &KvPushLaunch::pages)
        .def_ro("runs", &KvPushLaunch::runs)
        .def_ro("plan_seconds", &KvPushLaunch::plan_seconds)
        .def_ro("enqueue_seconds", &KvPushLaunch::enqueue_seconds)
        .def_ro("stream_seconds", &KvPushLaunch::stream_seconds);

    nb::class_<KvFabricPush>(
        mod,
        "KvFabricPush",
        R"doc(
        Writes a prefill's compressed caches (and, last, the window rings) into another mesh's DRAM over TT-Fabric, a
        launch at a time as the prefill's chunks complete them. A launch enqueues one program a chip on the mesh's
        command queue and streams the chip's runs to it through an H2D socket (DEVICE_PULL) on core (0, 0).

        Args:
            mesh_device (MeshDevice): the sending mesh.
            fifo_bytes (int): each chip's socket FIFO in pinned host memory (a multiple of 512).
        )doc")
        .def(
            nb::init<const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>&, uint32_t>(),
            nb::arg("mesh_device"),
            nb::arg("fifo_bytes"))
        .def(
            "begin",
            &KvFabricPush::begin,
            nb::arg("sources"),
            nb::arg("dst_mesh_id"),
            nb::arg("dst_chips"),
            nb::arg("dst_banks"),
            nb::arg("block"),
            nb::arg("layers"),
            R"doc(
            A new request: the source tensors the layers name, the destination mesh, each destination chip's fabric
            chip id and DRAM banks ((NoC x, NoC y, offset) each, flattened), the decode cache block in entries, and
            the layers.
            )doc")
        .def(
            "push",
            &KvFabricPush::push,
            nb::arg("end"),
            nb::arg("last"),
            nb::arg("rings") = std::nullopt,
            nb::arg("ring_runs") = std::vector<KvPushRing>{},
            R"doc(
            Enqueue a launch pushing every entry complete at position ``end`` not pushed yet (``last``: every key
            block too, the last one partial, and the ``rings`` source's ``ring_runs``). Returns a KvPushLaunch.
            )doc")
        .def_prop_ro("total_bytes", &KvFabricPush::total_bytes);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::detail
