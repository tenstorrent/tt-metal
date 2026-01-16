// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bfloat_dtype_traits.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn-nanobind/json_class.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/serialization.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_buffer.hpp>

using namespace tt::tt_metal;

// NOLINTBEGIN(bugprone-unused-raii)

namespace ttnn::tensor {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

void tensor_mem_config_module_types(nb::module_& m_tensor) {
    export_enum<Layout>(m_tensor);
    export_enum<DataType>(m_tensor);
    export_enum<StorageType>(m_tensor);
    export_enum<MathFidelity>(m_tensor);
    export_enum<TensorMemoryLayout>(m_tensor);
    export_enum<ShardOrientation>(m_tensor);
    // export_enum<ShardMode>(m_tensor);

    nb::enum_<tt::tt_metal::BufferType>(m_tensor, "BufferType")
        .value("DRAM", BufferType::DRAM)
        .value("L1", BufferType::L1)
        .value("L1_SMALL", BufferType::L1_SMALL)
        .value("TRACE", BufferType::TRACE);

    nb::enum_<TensorSpec::ShardShapeAlignment>(m_tensor, "ShardShapeAlignment")
        .value(
            "NONE",
            TensorSpec::ShardShapeAlignment::NONE,
            "No shard shape alignment will be performed. If the shard shape is not following the alignment "
            "requirements, an exception will be thrown.")
        .value(
            "REQUIRED",
            TensorSpec::ShardShapeAlignment::REQUIRED,
            "Shard shape will be automatically aligned to the minimum required alignment. The Required alignment may "
            "cause higher memory usage and lower read/write performance for some use cases.")
        .value(
            "RECOMMENDED",
            TensorSpec::ShardShapeAlignment::RECOMMENDED,
            "Shard shape will be automatically aligned to the recommended alignment, trying to achieve optimal "
            "performance and memory usage.");

    nb::enum_<ShardDistributionStrategy>(m_tensor, "ShardDistributionStrategy")
        .value(
            "ROUND_ROBIN_1D",
            ShardDistributionStrategy::ROUND_ROBIN_1D,
            "Distribute each shard to each of the cores in a linearized list in a round-robin manner.")
        .value(
            "GRID_2D",
            ShardDistributionStrategy::GRID_2D,
            "Distribute a 2D grid of shards to a 2D grid of cores with one to one mapping.");

    tt_serializable_class<tt::tt_metal::CoreCoord>(m_tensor, "CoreCoord", R"doc(
        Class defining core coordinate
    )doc");

    nb::class_<Tile>(m_tensor, "Tile", R"doc(
        Class defining tile dims
    )doc");

    nb::class_<ttnn::TensorSpec>(m_tensor, "TensorSpec", R"doc(
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

    // the buffer_protocol was dropped in nanobind in favor of ndarray.
    // So either have to write a caster or figure out how to map HostBuffer to ndarray

    // nb::ndarray<uint8_t, nb::shape<-1>, nb::device::cpu, nb::c_contig>

    // note: ndarray has several gotchas. See for more information:
    // https://github.com/wjakob/nanobind/blob/master/docs/ndarray.rst
    nb::class_<tt::tt_metal::HostBuffer>(m_tensor, "HostBuffer")
        .def(nb::init<>())
        //.def(nb::init<const std::shared_ptr<std::vector<std::byte>>>()) // broke unity build
        .def(nb::init<std::vector<std::byte>&&>())
        .def(nb::init<const std::vector<std::byte>&>())
        .def("__getitem__", [](const HostBuffer& self, std::size_t index) { return self.view_bytes()[index]; })
        .def("__len__", [](const HostBuffer& self) { return self.view_bytes().size(); })
        .def(
            "__iter__",
            [](const HostBuffer& self) {
                return nb::make_iterator<nb::rv_policy::reference_internal>(
                    nb::type<tt::tt_metal::HostBuffer>(),
                    "iterator",
                    self.view_bytes().begin(),
                    self.view_bytes().end());
            },
            nb::keep_alive<0, 1>())
        .def(
            "__array__",
            [](HostBuffer& self) {
                return nb::ndarray<uint8_t, nb::array_api, nb::device::cpu, nb::shape<-1>, nb::c_contig>(
                    self.view_bytes().data(), {self.view_bytes().size()});
            },
            nb::rv_policy::reference_internal)
        .def("__dlpack_device__", [](nb::handle) { return std::make_pair(nb::device::cpu::value, 0); })
        .def("__dlpack__", [](nb::pointer_and_handle<tt::tt_metal::HostBuffer> self, const nb::kwargs& kwargs) {
            using array_api_t = nb::ndarray<uint8_t, nb::array_api, nb::device::cpu, nb::shape<-1>, nb::c_contig>;
            nb::object aa = nb::cast(
                array_api_t(self.p->view_bytes().data(), {self.p->view_bytes().size()}),
                nb::rv_policy::reference_internal,  // nb::rv_policy::none?
                self.h);
            return aa.attr("__dlpack__")(**kwargs);
        });
}

void tensor_mem_config_module(nb::module_& m_tensor) {
    auto py_core_coord = static_cast<nb::class_<CoreCoord>>(m_tensor.attr("CoreCoord"));
    py_core_coord.def(nb::init<std::size_t, std::size_t>())
        .def(
            "__init__",
            [](CoreCoord* t, std::tuple<std::size_t, std::size_t> core_coord) {
                new (t) CoreCoord(std::get<0>(core_coord), std::get<1>(core_coord));
            })
        .def(
            "__init__",
            [](CoreCoord* t, nb::detail::tuple<nb::int_, nb::int_> core_coord) {
                new (t) CoreCoord(
                    static_cast<std::size_t>(core_coord.get<0>()), static_cast<std::size_t>(core_coord.get<1>()));
            })
        .def("__repr__", [](const CoreCoord& self) -> std::string { return self.str(); })
        .def_ro("x", &CoreCoord::x)
        .def_ro("y", &CoreCoord::y);
    nb::implicitly_convertible<std::tuple<std::size_t, std::size_t>, CoreCoord>();
    nb::implicitly_convertible<nb::detail::tuple<nb::int_, nb::int_>, CoreCoord>();

    auto py_tile = static_cast<nb::class_<Tile>>(m_tensor.attr("Tile"));
    py_tile
        .def(nb::init<const std::array<uint32_t, 2>&, bool>(), nb::arg("tile_shape"), nb::arg("transpose_tile") = false)
        .def(
            "__init__",
            [](Tile* t, const std::array<uint32_t, 2>& tile_shape, bool transpose_tile = false) {
                new (t) Tile{tile_shape, transpose_tile};
            })
        .def(
            "__repr__",
            [](const Tile& self) {
                return fmt::format("Tile with shape: [{}, {}]", self.get_tile_shape()[0], self.get_tile_shape()[1]);
            })
        .def_prop_ro("tile_shape", &Tile::get_tile_shape)
        .def_prop_ro("face_shape", &Tile::get_face_shape)
        .def_prop_ro("num_faces", &Tile::get_num_faces)
        .def_prop_ro("partial_face", &Tile::get_partial_face)
        .def_prop_ro("narrow_tile", &Tile::get_narrow_tile)
        .def_prop_ro("transpose_within_face", &Tile::get_transpose_within_face)
        .def_prop_ro("transpose_of_faces", &Tile::get_transpose_of_faces)
        .def(
            "get_tile_size",
            [](const Tile& self, DataType dtype) {
                return self.get_tile_size(datatype_to_dataformat_converter(dtype));
            },
            nb::arg("dtype"),
            "Get tile size in bytes for the given data type")
        .def(nb::self == nb::self);

    auto pyTensorSpec = static_cast<nb::class_<TensorSpec>>(m_tensor.attr("TensorSpec"));
    pyTensorSpec
        .def(
            "__init__",
            [](TensorSpec* t,
               const ttnn::Shape& shape,
               DataType dtype,
               Layout layout,
               BufferType buffer_type,
               const std::optional<Tile>& tile) {
                new (t) TensorSpec(
                    shape,
                    TensorLayout(
                        dtype, PageConfig(layout, tile), MemoryConfig(TensorMemoryLayout::INTERLEAVED, buffer_type)));
            },
            nb::arg("shape"),
            nb::arg("dtype"),
            nb::arg("layout"),
            nb::arg("buffer_type") = BufferType::DRAM,
            nb::arg("tile") = nb::none(),
            R"doc(
                Create TensorSpec class.
                This constructor is used to create TensorSpec for tensors that are not sharded.
            )doc")
        .def(
            "__init__",
            [](TensorSpec* t,
               const ttnn::Shape& shape,
               DataType dtype,
               Layout layout,
               TensorMemoryLayout memory_layout,
               const std::optional<ShardSpec>& shard_spec,
               BufferType buffer_type,
               const std::optional<Tile>& tile) {
                new (t) TensorSpec(
                    shape,
                    TensorLayout(
                        dtype, PageConfig(layout, tile), MemoryConfig(memory_layout, buffer_type, shard_spec)));
            },
            nb::arg("shape"),
            nb::arg("dtype"),
            nb::arg("layout"),
            nb::arg("memory_layout"),
            nb::arg("shard_spec") = nb::none(),
            nb::arg("buffer_type") = BufferType::DRAM,
            nb::arg("tile") = nb::none(),
            R"doc(
                Create TensorSpec class.
                This constructor is used to create TensorSpec for tensors that are sharded.
            )doc")
        .def(
            "__init__",
            [](TensorSpec* t,
               const ttnn::Shape& shape,
               DataType dtype,
               Layout layout,
               const NdShardSpec& nd_shard_spec,
               BufferType buffer_type,
               const std::optional<Tile>& tile) {
                new (t) TensorSpec(
                    shape, TensorLayout(dtype, PageConfig(layout, tile), MemoryConfig(buffer_type, nd_shard_spec)));
            },
            nb::arg("shape"),
            nb::arg("dtype"),
            nb::arg("layout"),
            nb::arg("nd_shard_spec"),
            nb::arg("buffer_type") = BufferType::DRAM,
            nb::arg("tile") = nb::none(),
            R"doc(
                Create TensorSpec class.
                This constructor is used to create TensorSpec for ND sharded tensors.
                Currently, the support for ND sharding is experimental and may not work with all of the tensor operations.
            )doc")
        .def(
            "sharded_across_dims",
            [](const TensorSpec& self,
               const std::vector<int32_t>& dims,
               CoreRangeSet grid,
               ShardOrientation orientation) {
                return self.sharded_across_dims(tt::stl::Span<const int32_t>(dims), std::move(grid), orientation);
            },
            nb::arg("dims"),
            nb::arg("grid"),
            nb::arg("orientation") = ShardOrientation::ROW_MAJOR,
            R"doc(
                Shards TensorSpec across the specified dimensions.
                This would result in the shard shape to be minimal (typically 1 or tile size) in the sharded dimensions.
                Currently, the support for ND sharding is experimental and may not work with all of the tensor operations.
            )doc")
        .def(
            "sharded_across_dims_except",
            [](const TensorSpec& self,
               const std::vector<int32_t>& dims,
               CoreRangeSet grid,
               ShardOrientation orientation) {
                return self.sharded_across_dims_except(
                    tt::stl::Span<const int32_t>(dims), std::move(grid), orientation);
            },
            nb::arg("dims"),
            nb::arg("grid"),
            nb::arg("orientation") = ShardOrientation::ROW_MAJOR,
            R"doc(
                Shards TensorSpec across all dimensions except for the specified ones.
                This would result in the shard shape to be minimal (typically 1 or tile size) in all dimensions except for the specified ones.
                Currently, the support for ND sharding is experimental and may not work with all of the tensor operations.
            )doc")
        .def(
            "height_sharded",
            &TensorSpec::height_sharded,
            nb::arg("grid"),
            nb::arg("orientation") = ShardOrientation::ROW_MAJOR,
            R"doc(
                Performs 2D height sharding for TensorSpec.
                This flattens the tensor into a 2D shape and splits it along the height to achieve as close to equal distribution as possible, while maintaining just 1 shard per core.
            )doc")
        .def(
            "width_sharded",
            &TensorSpec::width_sharded,
            nb::arg("grid"),
            nb::arg("orientation") = ShardOrientation::ROW_MAJOR,
            R"doc(
                Performs 2D width sharding for TensorSpec.
                This flattens the tensor into a 2D shape and splits it along the width to achieve as close to equal distribution as possible, while maintaining just 1 shard per core.
            )doc")
        .def(
            "block_sharded",
            &TensorSpec::block_sharded,
            nb::arg("grid"),
            nb::arg("orientation") = ShardOrientation::ROW_MAJOR,
            R"doc(
                Performs 2D block sharding for TensorSpec.
                This flattens the tensor into a 2D shape and splits it into 2D contiguous blocks, putting each block onto the corresponding core in the 2D grid.
            )doc")
        .def(
            "block_sharded",
            [](const TensorSpec& self, const CoreRangeSet& grid, ShardOrientation orientation) {
                TT_FATAL(grid.ranges().size() == 1, "Block sharding requires a single CoreRange");
                return self.block_sharded(grid.ranges()[0], orientation);
            },
            nb::arg("grid"),
            nb::arg("orientation") = ShardOrientation::ROW_MAJOR,
            R"doc(
                Performs 2D block sharding for TensorSpec.
                This flattens the tensor into a 2D shape and splits it into 2D contiguous blocks, putting each block onto the corresponding core in the 2D grid.
            )doc")
        .def(
            "sharded",
            [](const TensorSpec& self,
               const Shape& shard_shape,
               const CoreRangeSet& grid,
               TensorSpec::ShardShapeAlignment shard_alignment,
               ShardOrientation orientation,
               ShardDistributionStrategy shard_distribution_strategy) {
                return self.sharded(shard_shape, grid, shard_alignment, orientation, shard_distribution_strategy);
            },
            nb::arg("shard_shape"),
            nb::arg("grid"),
            nb::arg("shard_alignment") = TensorSpec::ShardShapeAlignment::RECOMMENDED,
            nb::arg("orientation") = ShardOrientation::ROW_MAJOR,
            nb::arg("shard_distribution_strategy") = ShardDistributionStrategy::ROUND_ROBIN_1D,
            R"doc(
                Performs arbitrary sharding for TensorSpec using the specified shard shape, grid, shard shape alignment, and other optional parameters.
                Currently, the support for ND sharding is experimental and may not work with all of the tensor operations.
            )doc")
        .def(
            "sharded",
            [](const TensorSpec& self,
               const NdShardSpec& nd_shard_spec,
               TensorSpec::ShardShapeAlignment shard_alignment) {
                return self.sharded(nd_shard_spec, shard_alignment);
            },
            nb::arg("nd_shard_spec"),
            nb::arg("shard_alignment") = TensorSpec::ShardShapeAlignment::RECOMMENDED,
            R"doc(
                Performs arbitrary sharding for TensorSpec using the specified shard spec and shard shape alignment.
                Currently, the support for ND sharding is experimental and may not work with all of the tensor operations.
            )doc")
        .def_prop_ro("shape", &TensorSpec::logical_shape, "Logical shape of a tensor")
        .def_prop_ro("layout", &TensorSpec::layout, "Layout of a tensor")
        .def_prop_ro("dtype", &TensorSpec::data_type, "Dtype of a tensor")
        .def_prop_ro("tile", &TensorSpec::tile, "Tile of a tensor")
        .def_prop_ro("memory_config", &TensorSpec::memory_config, "Memory config of a tensor")
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    auto pyMemoryConfig = static_cast<nb::class_<MemoryConfig>>(m_tensor.attr("MemoryConfig"));
    pyMemoryConfig
        .def(
            "__init__",
            [](MemoryConfig* t,
               TensorMemoryLayout memory_layout,
               BufferType buffer_type,
               std::optional<ShardSpec> shard_spec) {
                new (t) MemoryConfig(memory_layout, buffer_type, std::move(shard_spec));
            },
            nb::arg("memory_layout") = TensorMemoryLayout::INTERLEAVED,
            nb::arg("buffer_type") = BufferType::DRAM,
            nb::arg("shard_spec") = nb::none(),
            R"doc(
                Create MemoryConfig class.
                If interleaved is set to True, tensor data will be interleaved across multiple DRAM banks on TT Accelerator device.
                Otherwise, tensor data will be stored in a DRAM bank selected by dram_channel (valid values are 0, 1, ..., 7).

                Example of creating MemoryConfig specifying that tensor data should be stored in DRAM bank 3.

                .. code-block:: python

                    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED)
            )doc")
        .def(
            "__init__",
            [](MemoryConfig* t, BufferType buffer_type, NdShardSpec nd_shard_spec) {
                new (t) MemoryConfig(buffer_type, std::move(nd_shard_spec));
            },
            nb::arg("buffer_type"),
            nb::arg("nd_shard_spec"),
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
        .def_prop_ro(
            "interleaved",
            [](const MemoryConfig& memory_config) {
                return memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED;
            },
            "Whether tensor data is interleaved across multiple DRAM channels")
        .def_prop_ro("buffer_type", &MemoryConfig::buffer_type, "Buffer type to store tensor data. Can be DRAM or L1")
        .def_prop_ro("memory_layout", &MemoryConfig::memory_layout, "Memory layout of tensor data.")
        .def_prop_ro("shard_spec", &MemoryConfig::shard_spec, "Memory layout of tensor data.")
        .def_prop_ro("nd_shard_spec", &MemoryConfig::nd_shard_spec, "ND shard spec of tensor data.")
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    auto pyCoreRange = static_cast<nb::class_<CoreRange>>(m_tensor.attr("CoreRange"));
    pyCoreRange
        .def(
            "__init__",
            [](CoreRange* t, const CoreCoord& start, const CoreCoord& end) { new (t) CoreRange{start, end}; },
            nb::arg("start"),
            nb::arg("end"))
        .def_ro("start", &CoreRange::start_coord)
        .def_ro("end", &CoreRange::end_coord)
        .def("grid_size", &CoreRange::grid_size);

    auto pyCoreRangeSet = static_cast<nb::class_<CoreRangeSet>>(m_tensor.attr("CoreRangeSet"));
    pyCoreRangeSet
        .def(
            "__init__",
            [](CoreRangeSet* t, const std::vector<CoreRange>& core_ranges) {
                new (t) CoreRangeSet(tt::stl::Span<const CoreRange>(core_ranges));
            },
            nb::arg("core_ranges"))
        .def(
            "__init__",
            [](CoreRangeSet* t, const std::set<CoreRange>& core_ranges) { new (t) CoreRangeSet(core_ranges); },
            nb::arg("core_ranges"))
        .def(
            "bounding_box",
            &CoreRangeSet::bounding_box,
            "Returns a CoreRange i.e. bounding box covering all the core ranges in the CoreRangeSet")
        .def("num_cores", &CoreRangeSet::num_cores, "Returns total number of cores in the CoreRangeSet")
        .def("size", &CoreRangeSet::size, "Returns number of core ranges in the CoreRangeSet")
        .def("empty", &CoreRangeSet::empty, "Returns true if the CoreRangeSet has no core ranges")
        .def("subtract", &CoreRangeSet::subtract, "Subtract common CoreRanges from current i.e. it returns A - (AnB)")
        .def("ranges", &CoreRangeSet::ranges, "Returns the core ranges in the CoreRangeSet")
        .def(
            "contains",
            nb::overload_cast<const CoreCoord&>(&CoreRangeSet::contains, nb::const_),
            nb::arg("core"),
            "Check if a core coordinate is contained in this CoreRangeSet")
        .def(
            "merge",
            &CoreRangeSet::merge<CoreRangeSet>,
            nb::arg("other"),
            "Merge this CoreRangeSet with another CoreRangeSet and return the result")
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    m_tensor.def(
        "corerange_to_cores",
        &tt::tt_metal::corerange_to_cores,
        nb::arg("core_range_set"),
        nb::arg("max_cores") = nb::none(),
        nb::arg("row_wise") = false,
        "Convert a CoreRangeSet to a vector of CoreCoords");

    auto pyShardSpec = static_cast<nb::class_<ShardSpec>>(m_tensor.attr("ShardSpec"));
    pyShardSpec
        .def(
            "__init__",
            [](ShardSpec* t,
               const CoreRangeSet& core_sets,
               const std::array<uint32_t, 2>& shard_shape,
               const ShardOrientation& shard_orientation) {
                new (t) ShardSpec(core_sets, shard_shape, shard_orientation);
            },
            nb::arg("grid"),
            nb::arg("shard_shape"),
            nb::arg("shard_orientation"))
        .def_rw("shape", &ShardSpec::shape, "Shape of shard.")
        .def_rw("grid", &ShardSpec::grid, "Grid to layout shards.")
        .def_rw("orientation", &ShardSpec::orientation, "Orientation of cores to read shards")
        .def("num_cores", &ShardSpec::num_cores, "Number of cores")
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    auto pyNdShardSpec = static_cast<nb::class_<NdShardSpec>>(m_tensor.attr("NdShardSpec"));
    pyNdShardSpec
        .def(
            "__init__",
            [](NdShardSpec* t,
               const ttnn::Shape& shard_shape,
               const CoreRangeSet& grid,
               const ShardOrientation& orientation,
               ShardDistributionStrategy shard_distribution_strategy) {
                new (t) NdShardSpec(shard_shape, grid, orientation, shard_distribution_strategy);
            },
            nb::arg("shard_shape"),
            nb::arg("grid"),
            nb::arg("orientation") = ShardOrientation::ROW_MAJOR,
            nb::arg("shard_distribution_strategy") = ShardDistributionStrategy::ROUND_ROBIN_1D)
        .def_rw("shard_shape", &NdShardSpec::shard_shape, "Shape of shard.")
        .def_rw("grid", &NdShardSpec::grid, "Grid to layout shards.")
        .def_rw("orientation", &NdShardSpec::orientation, "Orientation of cores to distribute shards")
        .def_rw(
            "shard_distribution_strategy", &NdShardSpec::shard_distribution_strategy, "Strategy to distribute shards")
        .def(
            "num_cores", [](const NdShardSpec& self) { return self.grid.num_cores(); }, "Number of cores")
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    m_tensor
        .def(
            "dump_tensor_flatbuffer",
            &dump_tensor_flatbuffer,
            nb::arg("filename"),
            nb::arg("tensor"),
            R"doc(
                Dump tensor to file using FlatBuffer format with inline file storage.
            )doc")
        .def(
            "load_tensor_flatbuffer",
            nb::overload_cast<const std::string&, MeshDevice*>(&load_tensor_flatbuffer),
            nb::arg("file_name"),
            nb::arg("device") = nullptr,
            R"doc(
                Load tensor to file using FlatBuffer format with inline file storage.
            )doc");
}

}  // namespace ttnn::tensor

// NOLINTEND(bugprone-unused-raii)
