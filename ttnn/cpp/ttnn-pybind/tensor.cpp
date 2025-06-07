// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "ttnn-pybind/json_class.hpp"
#include "ttnn-pybind/export_enum.hpp"

#include <tt-metalium/host_buffer.hpp>
#include "ttnn/tensor/serialization.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::tensor {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

void tensor_mem_config_module_types(py::module& m_tensor) {
    export_enum<Layout>(m_tensor);
    export_enum<DataType>(m_tensor);
    export_enum<StorageType>(m_tensor);
    export_enum<MathFidelity>(m_tensor);
    export_enum<TensorMemoryLayout>(m_tensor);
    export_enum<ShardOrientation>(m_tensor);
    export_enum<ShardMode>(m_tensor);

    py::enum_<tt::tt_metal::BufferType>(m_tensor, "BufferType")
        .value("DRAM", BufferType::DRAM)
        .value("L1", BufferType::L1)
        .value("L1_SMALL", BufferType::L1_SMALL)
        .value("TRACE", BufferType::TRACE);

    tt_serializable_class<tt::tt_metal::CoreCoord>(m_tensor, "CoreCoord", R"doc(
        Class defining core coordinate
    )doc");

    py::class_<Tile>(m_tensor, "Tile", R"doc(
        Class defining tile dims
    )doc");

    py::class_<ttnn::TensorSpec>(m_tensor, "TensorSpec", R"doc(
        Class defining the specification of Tensor
    )doc");

    tt_serializable_class<MemoryConfig>(m_tensor, "MemoryConfig", R"doc(
        Class defining memory configuration for storing tensor data on TT Accelerator device.
        There are eight DRAM memory banks on TT Accelerator device, indexed as 0, 1, 2, ..., 7.
    )doc");

    tt_serializable_class<tt::tt_metal::ShardSpec>(m_tensor, "ShardSpec", R"doc(
        Class defining the specs required for sharding.
    )doc");

    tt_serializable_class<tt::tt_metal::NdShardSpec>(m_tensor, "NdShardSpec", R"doc(
        Class defining the specs required for ND sharding.
        Currently, the support for ND sharding is experimental and may not work with all of the tensor operations.
    )doc");

    tt_serializable_class<tt::tt_metal::CoreRange>(m_tensor, "CoreRange", R"doc(
        Class defining a range of cores)doc");

    tt_serializable_class<tt::tt_metal::CoreRangeSet>(m_tensor, "CoreRangeSet", R"doc(
        Class defining a set of CoreRanges required for sharding)doc");

    py::class_<tt::tt_metal::HostBuffer>(m_tensor, "HostBuffer", py::buffer_protocol())
        .def("__getitem__", [](const HostBuffer& self, std::size_t index) { return self.view_bytes()[index]; })
        .def("__len__", [](const HostBuffer& self) { return self.view_bytes().size(); })
        .def(
            "__iter__",
            [](const HostBuffer& self) {
                return py::make_iterator(self.view_bytes().begin(), self.view_bytes().end());
            },
            py::keep_alive<0, 1>())
        .def_buffer([](HostBuffer& self) -> py::buffer_info {
            return py::buffer_info(
                self.view_bytes().data(),                       /* Pointer to buffer */
                sizeof(std::byte),                              /* Size of one scalar */
                py::format_descriptor<unsigned char>::format(), /* Python struct-style format descriptor */
                1,                                              /* Number of dimensions */
                {self.view_bytes().size()},                     /* Buffer dimensions */
                {sizeof(std::byte)}                             /* Strides (in bytes) for each index */
            );
        });
}

void tensor_mem_config_module(py::module& m_tensor) {
    auto py_core_coord = static_cast<py::class_<CoreCoord>>(m_tensor.attr("CoreCoord"));
    py_core_coord.def(py::init<std::size_t, std::size_t>())
        .def(py::init<>([](std::tuple<std::size_t, std::size_t> core_coord) {
            return CoreCoord(std::get<0>(core_coord), std::get<1>(core_coord));
        }))
        .def("__repr__", [](const CoreCoord& self) -> std::string { return self.str(); })
        .def_readonly("x", &CoreCoord::x)
        .def_readonly("y", &CoreCoord::y);
    py::implicitly_convertible<std::tuple<std::size_t, std::size_t>, CoreCoord>();

    auto py_tile = static_cast<py::class_<Tile>>(m_tensor.attr("Tile"));
    py_tile
        .def(py::init<const std::array<uint32_t, 2>&, bool>(), py::arg("tile_shape"), py::arg("transpose_tile") = false)
        .def(py::init<>([](const std::array<uint32_t, 2>& tile_shape, bool transpose_tile = false) {
            return Tile{tile_shape, transpose_tile};
        }))
        .def(
            "__repr__",
            [](const Tile& self) {
                return fmt::format("Tile with shape: [{}, {}]", self.get_tile_shape()[0], self.get_tile_shape()[1]);
            })
        .def_readonly("tile_shape", &Tile::tile_shape)
        .def_readonly("face_shape", &Tile::face_shape)
        .def_readonly("num_faces", &Tile::num_faces)
        .def_readonly("partial_face", &Tile::partial_face)
        .def_readonly("narrow_tile", &Tile::narrow_tile)
        .def_readonly("transpose_within_face", &Tile::transpose_within_face)
        .def_readonly("transpose_of_faces", &Tile::transpose_of_faces);

    auto pyTensorSpec = static_cast<py::class_<TensorSpec>>(m_tensor.attr("TensorSpec"));
    pyTensorSpec.def("shape", &TensorSpec::logical_shape, "Logical shape of a tensor")
        .def("layout", &TensorSpec::layout, "Layout of a tensor")
        .def("dtype", &TensorSpec::data_type, "Dtype of a tensor");

    auto pyMemoryConfig = static_cast<py::class_<MemoryConfig>>(m_tensor.attr("MemoryConfig"));
    pyMemoryConfig
        .def(
            py::init<>(
                [](TensorMemoryLayout memory_layout, BufferType buffer_type, std::optional<ShardSpec> shard_spec) {
                    return MemoryConfig{memory_layout, buffer_type, std::move(shard_spec)};
                }),
            py::arg("memory_layout") = TensorMemoryLayout::INTERLEAVED,
            py::arg("buffer_type") = BufferType::DRAM,
            py::arg("shard_spec") = std::nullopt,
            R"doc(
                Create MemoryConfig class.
                If interleaved is set to True, tensor data will be interleaved across multiple DRAM banks on TT Accelerator device.
                Otherwise, tensor data will be stored in a DRAM bank selected by dram_channel (valid values are 0, 1, ..., 7).

                Example of creating MemoryConfig specifying that tensor data should be stored in DRAM bank 3.

                .. code-block:: python

                    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.SINGLE_BANK)
            )doc")
        .def(
            py::init<>([](BufferType buffer_type, NdShardSpec nd_shard_spec) {
                return MemoryConfig{buffer_type, std::move(nd_shard_spec)};
            }),
            py::arg("buffer_type"),
            py::arg("nd_shard_spec"),
            R"doc(
                Create MemoryConfig class.
                This constructor is used to create MemoryConfig for ND sharded tensors.
                Currently, the support for ND sharding is experimental and may not work with all of the tensor operations.

                Example of creating MemoryConfig for ND sharded tensors.

                .. code-block:: python

                    mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape([1, 1, 1, 1]), ttnn.CoreRangeSet([ttnn.CoreCoord(0, 0)])))
            )doc")
        .def(
            "__hash__",
            [](const MemoryConfig& memory_config) -> tt::stl::hash::hash_t {
                return tt::stl::hash::detail::hash_object(memory_config);
            })
        .def("is_sharded", &MemoryConfig::is_sharded, "Whether tensor data is sharded across multiple cores in L1")
        .def(
            "with_shard_spec",
            &MemoryConfig::with_shard_spec,
            "Returns a new MemoryConfig with the shard spec set to the given value")
        .def_property_readonly(
            "interleaved",
            [](const MemoryConfig& memory_config) {
                return memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED;
            },
            "Whether tensor data is interleaved across multiple DRAM channels")
        .def_property_readonly(
            "buffer_type", &MemoryConfig::buffer_type, "Buffer type to store tensor data. Can be DRAM or L1")
        .def_property_readonly("memory_layout", &MemoryConfig::memory_layout, "Memory layout of tensor data.")
        .def_property_readonly("shard_spec", &MemoryConfig::shard_spec, "Memory layout of tensor data.")
        .def_property_readonly("nd_shard_spec", &MemoryConfig::nd_shard_spec, "ND shard spec of tensor data.")
        .def(py::self == py::self)
        .def(py::self != py::self);

    m_tensor.def(
        "dump_memory_config",
        py::overload_cast<const std::string&, const MemoryConfig&>(&dump_memory_config),
        R"doc(
            Dump memory config to file
        )doc");
    m_tensor.def(
        "load_memory_config",
        py::overload_cast<const std::string&>(&load_memory_config),
        R"doc(
            Load memory config to file
        )doc");

    auto pyCoreRange = static_cast<py::class_<CoreRange>>(m_tensor.attr("CoreRange"));
    pyCoreRange.def(py::init<>([](const CoreCoord& start, const CoreCoord& end) { return CoreRange{start, end}; }))
        .def_readonly("start", &CoreRange::start_coord)
        .def_readonly("end", &CoreRange::end_coord)
        .def("grid_size", &CoreRange::grid_size);

    auto pyCoreRangeSet = static_cast<py::class_<CoreRangeSet>>(m_tensor.attr("CoreRangeSet"));
    pyCoreRangeSet.def(py::init<>([](const std::set<CoreRange>& core_ranges) { return CoreRangeSet(core_ranges); }))
        .def(py::init<>([](const std::vector<CoreRange>& core_ranges) {
            return CoreRangeSet(tt::stl::Span<const CoreRange>(core_ranges));
        }))
        .def(
            "bounding_box",
            &CoreRangeSet::bounding_box,
            "Returns a CoreRange i.e. bounding box covering all the core ranges in the CoreRangeSet")
        .def("num_cores", &CoreRangeSet::num_cores, "Returns total number of cores in the CoreRangeSet")
        .def("subtract", &CoreRangeSet::subtract, "Subtract common CoreRanges from current i.e. it returns A - (AnB)")
        .def("ranges", &CoreRangeSet::ranges, "Returns the core ranges in the CoreRangeSet");

    auto pyShardSpec = static_cast<py::class_<ShardSpec>>(m_tensor.attr("ShardSpec"));
    pyShardSpec
        .def(
            py::init<>([](const CoreRangeSet& core_sets,
                          const std::array<uint32_t, 2>& shard_shape,
                          const ShardOrientation& shard_orientation,
                          const ShardMode& shard_mode) {
                return ShardSpec(core_sets, shard_shape, shard_orientation, shard_mode);
            }),
            py::arg("grid"),
            py::arg("shard_shape"),
            py::arg("shard_orientation"),
            py::arg("shard_mode") = ShardMode::PHYSICAL)
        .def(
            py::init<>([](const CoreRangeSet& core_sets,
                          const std::array<uint32_t, 2>& shard_shape,
                          const std::array<uint32_t, 2>& physical_shard_shape,
                          const ShardOrientation& shard_orientation) {
                return ShardSpec(core_sets, shard_shape, physical_shard_shape, shard_orientation);
            }),
            py::arg("grid"),
            py::arg("shard_shape"),
            py::arg("physical_shard_shape"),
            py::arg("shard_orientation"))
        .def_readwrite("shape", &ShardSpec::shape, "Shape of shard.")
        .def_readwrite("grid", &ShardSpec::grid, "Grid to layout shards.")
        .def_readwrite("orientation", &ShardSpec::orientation, "Orientation of cores to read shards")
        .def_readwrite("mode", &ShardSpec::mode, "Treat shard shape as physical (default) or logical")
        .def("num_cores", &ShardSpec::num_cores, "Number of cores")
        .def(py::self == py::self)
        .def(py::self != py::self);

    auto pyNdShardSpec = static_cast<py::class_<NdShardSpec>>(m_tensor.attr("NdShardSpec"));
    pyNdShardSpec
        .def(
            py::init<>([](const ttnn::Shape& shard_shape,
                          const CoreRangeSet& grid,
                          const ShardOrientation& orientation) { return NdShardSpec(shard_shape, grid, orientation); }),
            py::arg("shard_shape"),
            py::arg("grid"),
            py::arg("orientation") = ShardOrientation::ROW_MAJOR)
        .def_readwrite("shard_shape", &NdShardSpec::shard_shape, "Shape of shard.")
        .def_readwrite("grid", &NdShardSpec::grid, "Grid to layout shards.")
        .def_readwrite("orientation", &NdShardSpec::orientation, "Orientation of cores to distribute shards")
        .def(
            "num_cores", [](const NdShardSpec& self) { return self.grid.num_cores(); }, "Number of cores")
        .def(py::self == py::self)
        .def(py::self != py::self);

    m_tensor.def(
        "dump_tensor",
        &dump_tensor,
        py::arg("filename"),
        py::arg("tensor"),
        py::arg("strategy") = std::unordered_map<std::string, std::string>{},
        R"doc(
            Dump tensor to file
        )doc");

    m_tensor.def(
        "load_tensor",
        py::overload_cast<const std::string&, MeshDevice*>(&load_tensor),
        py::arg("file_name"),
        py::arg("device") = nullptr,
        R"doc(Load tensor to file)doc");
}

}  // namespace ttnn::tensor
