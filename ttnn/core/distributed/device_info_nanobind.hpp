// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/info.hpp>
#include <tt-metalium/experimental/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

// Python bindings for the experimental device query API (tt-metalium/experimental/mesh_device.hpp), exposed as methods
// on MeshDevice so call sites already look like the graduated API; only the tags are experimental.
// C++ stays template-based: each tag gets one InfoTag object that carries the type-erased calls for it, and the tags
// come from experimental::info::all_tags, so adding a property needs no change here.
namespace ttnn::distributed::device_info {

namespace nb = nanobind;

// Python face of one property tag. Compared and hashed by name, so copies of a tag are interchangeable.
struct InfoTag {
    std::string name;
    std::function<nb::object(tt::tt_metal::distributed::MeshDevice&)> uniform;
    std::function<nb::object(tt::tt_metal::distributed::MeshDevice&, const tt::tt_metal::distributed::MeshCoordinate&)>
        at;
    std::function<nb::object(tt::tt_metal::distributed::MeshDevice&)> per_device;
};

template <class Tag>
InfoTag make_info_tag() {
    namespace query = tt::tt_metal::experimental::mesh_device;
    using tt::tt_metal::distributed::MeshCoordinate;
    using tt::tt_metal::distributed::MeshCoordinateRange;
    using tt::tt_metal::distributed::MeshDevice;

    return InfoTag{
        std::string(Tag::name),
        [](MeshDevice& mesh) -> nb::object { return nb::cast(query::get_info<Tag>(mesh)); },
        [](MeshDevice& mesh, const MeshCoordinate& coord) -> nb::object {
            return nb::cast(query::get_info<Tag>(mesh, coord));
        },
        [](MeshDevice& mesh) -> nb::object {
            const auto values = query::get_info_per_device<Tag>(mesh);
            nb::dict result;
            for (const MeshCoordinate& coord : MeshCoordinateRange(values.shape())) {
                if (values.is_local(coord)) {
                    result[nb::cast(coord)] = nb::cast(values.at(coord).value());
                }
            }
            return result;
        }};
}

template <class... Tags>
void bind_tags(
    nb::module_& info_module, std::vector<InfoTag>& all, tt::tt_metal::experimental::info::tag_list<Tags...> /*tags*/) {
    (
        [&] {
            InfoTag tag = make_info_tag<Tags>();
            const std::string name = tag.name;
            all.push_back(tag);
            info_module.attr(name.c_str()) = nb::cast(std::move(tag));
        }(),
        ...);
}

// Adds the tags to the experimental module as `info`, and `get_info` / `get_info_per_device` to MeshDevice.
inline void bind_device_info(
    nb::module_& m_experimental, nb::class_<tt::tt_metal::distributed::MeshDevice>& nb_mesh_device) {
    using tt::tt_metal::distributed::MeshCoordinate;
    using tt::tt_metal::distributed::MeshDevice;

    auto m_info = m_experimental.def_submodule("info", "Tags naming the device properties that get_info can query.");

    nb::class_<InfoTag>(m_info, "InfoTag", "A device property that get_info can query. Experimental API; may change.")
        .def_prop_ro("name", [](const InfoTag& tag) { return tag.name; })
        .def("__repr__", [](const InfoTag& tag) { return "InfoTag(" + tag.name + ")"; })
        .def(
            "__eq__", [](const InfoTag& a, const InfoTag& b) { return a.name == b.name; }, nb::is_operator())
        .def("__hash__", [](const InfoTag& tag) { return std::hash<std::string>{}(tag.name); });

    std::vector<InfoTag> all;
    bind_tags(m_info, all, tt::tt_metal::experimental::info::all_tags{});
    m_info.def(
        "all_tags",
        [all]() { return all; },
        R"doc(
            Every property tag that get_info can query.

            Returns:
                List[InfoTag]: The tags, in declaration order.
        )doc");

    nb_mesh_device.def(
        "get_info",
        [](MeshDevice& mesh, const InfoTag& info, const std::optional<MeshCoordinate>& coord) -> nb::object {
            return coord ? info.at(mesh, *coord) : info.uniform(mesh);
        },
        nb::arg("info"),
        nb::arg("coord") = nb::none(),
        R"doc(
            Query a device property.

            Without ``coord``, returns the value for the whole mesh, which must be the same on every local
            device. With ``coord``, returns the value for the device at that coordinate.

            Args:
                info (InfoTag): The property, for example ``ttnn.experimental.info.l1_alignment``. The tags are
                    experimental and may change.
                coord (MeshCoordinate, optional): Query one device instead of the whole mesh.

            Returns:
                The value of the property. Its type depends on ``info``.

            Raises:
                RuntimeError: If ``coord`` is outside the mesh's shape or names a device this rank does not
                    drive, if the mesh has no local devices, or if the local devices report different values.
        )doc");

    nb_mesh_device.def(
        "get_info_per_device",
        [](MeshDevice& mesh, const InfoTag& info) -> nb::object { return info.per_device(mesh); },
        nb::arg("info"),
        R"doc(
            Query a device property for every device this rank drives.

            Args:
                info (InfoTag): The property, for example ``ttnn.experimental.info.l1_alignment``. The tags are
                    experimental and may change.

            Returns:
                Dict[MeshCoordinate, Any]: The value for each local device. Devices owned by other ranks are not
                included; use ``get_view().is_local(coord)`` to tell which ones those are.
        )doc");
}

}  // namespace ttnn::distributed::device_info
