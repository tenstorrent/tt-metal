// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings_tensor.hpp"
#include "ttnn/cpp/pybind11/json_class.hpp"

#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/serialization.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_lib_bindings.hpp"
#include "tt_lib_bindings_tensor_impl.hpp"
#include "type_caster.hpp"

namespace py = pybind11;
namespace tt::tt_metal {

namespace detail {
template <class T>
struct DataTypeToFormatType {
    using type = T;
};

template <>
struct DataTypeToFormatType<bfloat16> {
    using type = uint16_t;
};

template <class CppType>
void implement_buffer_protocol(py::module& m_tensor, std::string_view name) {
    auto py_buffer_t = static_cast<py::class_<CppType>>(m_tensor.attr(name.data()));
    using DataType = typename CppType::value_type;

    py_buffer_t.def("__getitem__", [](const CppType& self, std::size_t index) { return self[index]; })
        .def("__len__", [](const CppType& self) { return self.size(); })
        .def(
            "__iter__",
            [](const CppType& self) { return py::make_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>())
        .def_buffer([](CppType& self) -> py::buffer_info {
            using FormatType = typename DataTypeToFormatType<DataType>::type;
            return py::buffer_info(
                self.begin(),                                /* Pointer to buffer */
                sizeof(DataType),                            /* Size of one scalar */
                py::format_descriptor<FormatType>::format(), /* Python struct-style format descriptor */
                1,                                           /* Number of dimensions */
                {self.size()},                               /* Buffer dimensions */
                {sizeof(DataType)}                           /* Strides (in bytes) for each index */
            );
        });
};

}  // namespace detail

void TensorModuleTypes(py::module& m_tensor) {
    detail::export_enum<Layout>(m_tensor);
    detail::export_enum<DataType>(m_tensor);
    detail::export_enum<StorageType>(m_tensor);
    detail::export_enum<MathFidelity>(m_tensor);
    detail::export_enum<TensorMemoryLayout>(m_tensor);
    detail::export_enum<ShardOrientation>(m_tensor);

    py::enum_<BufferType>(m_tensor, "BufferType")
        .value("DRAM", BufferType::DRAM)
        .value("L1", BufferType::L1)
        .value("L1_SMALL", BufferType::L1_SMALL);

    tt_serializable_class<CoreCoord>(m_tensor, "CoreCoord", R"doc(
        Class defining core coordinate
    )doc");

    py::class_<Shape>(m_tensor, "Shape", R"doc(
        Class defining tensor shape
    )doc");

    tt_serializable_class<MemoryConfig>(m_tensor, "MemoryConfig", R"doc(
        Class defining memory configuration for storing tensor data on TT Accelerator device.
        There are eight DRAM memory banks on TT Accelerator device, indexed as 0, 1, 2, ..., 7.
    )doc");

    tt_serializable_class<ShardSpec>(m_tensor, "ShardSpec", R"doc(
        Class defining the specs required for sharding.
    )doc");

    tt_serializable_class<CoreRange>(m_tensor, "CoreRange", R"doc(
        Class defining a range of cores)doc");

    tt_serializable_class<CoreRangeSet>(m_tensor, "CoreRangeSet", R"doc(
        Class defining a set of CoreRanges required for sharding)doc");

    py::class_<owned_buffer::Buffer<uint8_t>>(m_tensor, "owned_buffer_for_uint8_t", py::buffer_protocol());
    py::class_<owned_buffer::Buffer<uint16_t>>(m_tensor, "owned_buffer_for_uint16_t", py::buffer_protocol());
    py::class_<owned_buffer::Buffer<int32_t>>(m_tensor, "owned_buffer_for_int32_t", py::buffer_protocol());
    py::class_<owned_buffer::Buffer<uint32_t>>(m_tensor, "owned_buffer_for_uint32_t", py::buffer_protocol());
    py::class_<owned_buffer::Buffer<float>>(m_tensor, "owned_buffer_for_float32_t", py::buffer_protocol());
    py::class_<owned_buffer::Buffer<bfloat16>>(m_tensor, "owned_buffer_for_bfloat16_t", py::buffer_protocol());
    py::class_<borrowed_buffer::Buffer<std::uint8_t>>(m_tensor, "borrowed_buffer_for_uint8_t", py::buffer_protocol());
    py::class_<borrowed_buffer::Buffer<std::uint16_t>>(m_tensor, "borrowed_buffer_for_uint16_t", py::buffer_protocol());
    py::class_<borrowed_buffer::Buffer<std::int32_t>>(m_tensor, "borrowed_buffer_for_int32_t", py::buffer_protocol());
    py::class_<borrowed_buffer::Buffer<std::uint32_t>>(m_tensor, "borrowed_buffer_for_uint32_t", py::buffer_protocol());
    py::class_<borrowed_buffer::Buffer<float>>(m_tensor, "borrowed_buffer_for_float32_t", py::buffer_protocol());
    py::class_<borrowed_buffer::Buffer<bfloat16>>(m_tensor, "borrowed_buffer_for_bfloat16_t", py::buffer_protocol());

    detail::TensorModulePyTensorTypes(m_tensor);
}

void TensorModule(py::module& m_tensor) {
    auto py_core_coord = static_cast<py::class_<CoreCoord>>(m_tensor.attr("CoreCoord"));
    py_core_coord.def(py::init<std::size_t, std::size_t>())
        .def(py::init<>([](std::tuple<std::size_t, std::size_t> core_coord) {
            return CoreCoord(std::get<0>(core_coord), std::get<1>(core_coord));
        }))
        .def("__repr__", [](const CoreCoord& self) -> std::string { return self.str(); })
        .def_readonly("x", &CoreCoord::x)
        .def_readonly("y", &CoreCoord::y);
    py::implicitly_convertible<std::tuple<std::size_t, std::size_t>, CoreCoord>();

    auto py_shape = static_cast<py::class_<Shape>>(m_tensor.attr("Shape"));
    py_shape.def(py::init<std::array<uint32_t, 4>>())
        .def(
            py::init(
                [](const std::vector<uint32_t>& shape,
                   const std::optional<std::vector<uint32_t>>& padded_shape) -> Shape {
                    if (padded_shape.has_value()) {
                        return Shape{shape, padded_shape.value()};
                    } else {
                        return Shape{shape};
                    }
                }),
            py::arg("shape"),
            py::arg("padded_shape") = std::nullopt)
        .def("__len__", [](const Shape& self) { return self.rank(); })
        .def("__eq__", [](const Shape& self, const Shape& other) { return self == other; })
        .def("__eq__", [](const Shape& self, const std::vector<uint32_t>& other) { return self == Shape{other}; })
        .def("__eq__", [](const Shape& self, const std::array<uint32_t, 4>& other) { return self == Shape{other}; })
        .def("__eq__", [](const Shape& self, const py::none) { return false; })
        .def("__getitem__", [](const Shape& self, const std::int64_t index) { return self[index]; })
        .def(
            "__getitem__",
            [](const Shape& self, const py::slice slice) {
                size_t start = 0, stop = 0, step = 0, slicelength = 0;
                if (!slice.compute(self.rank(), &start, &stop, &step, &slicelength)) {
                    throw std::runtime_error("Invalid slice");
                }

                std::vector<uint32_t> output;
                for (auto index = start; index < stop; index += step) {
                    output.push_back(self[index]);
                }
                return Shape{output};
            })
        .def(
            "__iter__",
            [](const Shape& self) { return py::make_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>())
        .def("__repr__", [](const Shape& self) { return fmt::format("{}", self); })
        .def("without_padding", [](const Shape& self) -> Shape { return self.without_padding(); });

    py::implicitly_convertible<std::vector<uint32_t>, Shape>();

    auto pyMemoryConfig = static_cast<py::class_<MemoryConfig>>(m_tensor.attr("MemoryConfig"));
    pyMemoryConfig
        .def(
            py::init<>(
                [](TensorMemoryLayout memory_layout, BufferType buffer_type, std::optional<ShardSpec> shard_spec) {
                    return MemoryConfig{
                        .memory_layout = memory_layout, .buffer_type = buffer_type, .shard_spec = shard_spec};
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

                    mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.SINGLE_BANK)
            )doc")
        .def(
            "__hash__",
            [](const MemoryConfig& memory_config) -> tt::stl::hash::hash_t {
                return tt::stl::hash::detail::hash_object(memory_config);
            })
        .def("is_sharded", &MemoryConfig::is_sharded, "Whether tensor data is sharded across multiple cores in L1")
        .def_property_readonly(
            "interleaved",
            [](const MemoryConfig& memory_config) {
                return memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED;
            },
            "Whether tensor data is interleaved across multiple DRAM channels")
        .def_readonly("buffer_type", &MemoryConfig::buffer_type, "Buffer type to store tensor data. Can be DRAM or L1")
        .def_readonly("memory_layout", &MemoryConfig::memory_layout, "Memory layout of tensor data.")
        .def_readwrite("shard_spec", &MemoryConfig::shard_spec, "Memory layout of tensor data.")
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
        .def(
            "bounding_box",
            &CoreRangeSet::bounding_box,
            "Returns a CoreRange i.e. bounding box covering all the core ranges in the CoreRangeSet")
        .def("num_cores", &CoreRangeSet::num_cores, "Returns total number of cores in the CoreRangeSet");


    auto pyShardSpec = static_cast<py::class_<ShardSpec>>(m_tensor.attr("ShardSpec"));
    pyShardSpec
        .def(py::init<>([](const CoreRangeSet& core_sets,
                           const std::array<uint32_t, 2>& shard_shape,
                           const ShardOrientation& shard_orientation,
                           const bool& halo) { return ShardSpec(core_sets, shard_shape, shard_orientation, halo); }))
        .def_readwrite("shape", &ShardSpec::shape, "Shape of shard.")
        .def_readwrite("grid", &ShardSpec::grid, "Grid to layout shards.")
        .def_readwrite("orientation", &ShardSpec::orientation, "Orientation of cores to read shards")
        .def("num_cores", &ShardSpec::num_cores, "Number of cores")
        .def(py::self == py::self)
        .def(py::self != py::self);

    detail::implement_buffer_protocol<owned_buffer::Buffer<uint8_t>>(m_tensor, "owned_buffer_for_uint8_t");
    detail::implement_buffer_protocol<owned_buffer::Buffer<uint16_t>>(m_tensor, "owned_buffer_for_uint16_t");
    detail::implement_buffer_protocol<owned_buffer::Buffer<int32_t>>(m_tensor, "owned_buffer_for_int32_t");
    detail::implement_buffer_protocol<owned_buffer::Buffer<uint32_t>>(m_tensor, "owned_buffer_for_uint32_t");
    detail::implement_buffer_protocol<owned_buffer::Buffer<float>>(m_tensor, "owned_buffer_for_float32_t");
    detail::implement_buffer_protocol<owned_buffer::Buffer<bfloat16>>(m_tensor, "owned_buffer_for_bfloat16_t");
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<std::uint8_t>>(m_tensor, "borrowed_buffer_for_uint8_t");
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<std::uint16_t>>(m_tensor, "borrowed_buffer_for_uint16_t");
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<std::int32_t>>(m_tensor, "borrowed_buffer_for_int32_t");
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<std::uint32_t>>(m_tensor, "borrowed_buffer_for_uint32_t");
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<float>>(m_tensor, "borrowed_buffer_for_float32_t");
    detail::implement_buffer_protocol<borrowed_buffer::Buffer<bfloat16>>(m_tensor, "borrowed_buffer_for_bfloat16_t");

    // TMs
    m_tensor.def(
        "convert_conv_weight_tensor_to_tiled_layout",
        &convert_conv_weight_tensor_to_tiled_layout,
        py::arg("conv_weight_tensor").noconvert(),
        py::arg("in1_block_h"),
        py::arg("in1_block_w"),
        py::arg("output_dtype").noconvert() = std::nullopt,
        R"doc(
        Converts convolution weights to 2d matrix tiled layout on host
        Returns a new tensor with the converted layout.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def(
        "convert_conv_weight_tensor_to_special_padding_tiled_layout",
        &convert_conv_weight_tensor_to_special_padding_tiled_layout,
        py::arg("conv_weight_tensor").noconvert(),
        py::arg("in1_block_h"),
        py::arg("in1_block_w"),
        py::arg("output_dtype").noconvert() = std::nullopt,
        R"doc(
       Converts convolution weights to 2d matrix tiled layout on host with special block height padding
       Returns a new tensor with the converted layout.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");

    m_tensor.def(
        "convert_conv_weight_tensor_to_grouped_layout",
        &convert_conv_weight_tensor_to_grouped_layout,
        py::arg("conv_weight_tensor").noconvert(),
        py::arg("num_groups"),
        py::arg("output_dtype").noconvert() = std::nullopt,
        R"doc(
        Converts convolution weights to grouped layout with padded zeros
        Returns a new tensor with the converted layout.

        +----------+----------------------+-----------+-------------+----------+
        | Argument | Description          | Data type | Valid range | Required |
        +==========+======================+===========+=============+==========+
        | a        | Input tensor         | Tensor    |             | Yes      |
        +----------+----------------------+-----------+-------------+----------+
    )doc");

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
        static_cast<Tensor (*)(const std::string&, Device*)>(&load_tensor<Device*>),
        py::arg("file_name"),
        py::arg("device") = nullptr,
        R"doc(Load tensor to file)doc");
    m_tensor.def(
        "load_tensor",
        static_cast<Tensor (*)(const std::string&, MeshDevice*)>(&load_tensor<MeshDevice*>),
        py::arg("file_name"),
        py::arg("device") = nullptr,
        R"doc(Load tensor to file)doc");

    m_tensor.def(
        "allocate_tensor_on_device",
        py::overload_cast<const ttnn::Shape&, DataType, Layout, Device*, const MemoryConfig&>(
            &allocate_tensor_on_device),
        py::arg("shape"),
        py::arg("dtype"),
        py::arg("layout"),
        py::arg("device"),
        py::arg("memory_config") = MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED},
        R"doc(
            Allocate a tensor with specified attributes on a device.
        )doc");

    m_tensor.def(
        "allocate_tensor_on_device",
        py::overload_cast<const ttnn::Shape&, DataType, Layout, MeshDevice*, const MemoryConfig&>(
            &allocate_tensor_on_device),
        py::arg("shape"),
        py::arg("dtype"),
        py::arg("layout"),
        py::arg("device"),
        py::arg("memory_config") = MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED},
        R"doc(
            Allocate a tensor with specified attributes on a device.
        )doc");

    m_tensor.def(
        "write_tensor",
        py::overload_cast<Tensor, Tensor, uint8_t>(&write_tensor),
        py::arg("host_tensor"),
        py::arg("device_tensor"),
        py::arg("cq_id") = 0,
        R"doc(
            Copy a host tensor to its equivalent tensor on a device.
        )doc");

    detail::TensorModulePyTensor(m_tensor);
}

}  // namespace tt::tt_metal
