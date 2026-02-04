// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "program_descriptors.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn-nanobind/small_vector_caster.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/mesh_program_descriptor.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "ttnn/tensor/tensor_utils.hpp"

NB_MAKE_OPAQUE(std::vector<UnpackToDestMode>);
NB_MAKE_OPAQUE(std::vector<uint32_t>);

namespace ttnn::program_descriptors {

using CoreCoord = tt::tt_metal::CoreCoord;

// Helper class to enable Python syntax: rtargs[x][y] = [arg1, arg2, ...]
// This translates to: rtargs.push_back({CoreCoord(x, y), {arg1, arg2, ...}})
class RuntimeArgsColProxy {
public:
    RuntimeArgsColProxy(tt::tt_metal::KernelDescriptor::RuntimeArgs& args, size_t x) : args_(args), x_(x) {}

    void set_item(size_t y, const std::vector<uint32_t>& values) {
        CoreCoord target(x_, y);
        for (auto& [coord, vec] : args_) {
            if (coord == target) {
                vec = values;  // Update existing
                return;
            }
        }
        args_.push_back({target, values});  // Append if not found
    }

    std::vector<uint32_t>& get_item(size_t y) {
        CoreCoord target(x_, y);
        for (auto& [coord, values] : args_) {
            if (coord == target) {
                return values;
            }
        }
        throw std::out_of_range(
            "No runtime args found for CoreCoord(" + std::to_string(x_) + ", " + std::to_string(y) + ")");
    }

    void extend_item(size_t y, const std::vector<uint32_t>& values) {
        std::vector<uint32_t>& target_vec = get_item(y);
        target_vec.insert(target_vec.end(), values.begin(), values.end());
    }

    void append_item(size_t y, uint32_t value) {
        std::vector<uint32_t>& target_vec = get_item(y);
        target_vec.push_back(value);
    }

private:
    tt::tt_metal::KernelDescriptor::RuntimeArgs& args_;
    size_t x_;
};

// Wrapper class that provides 2D indexing syntax for RuntimeArgs
class RuntimeArgsWrapper {
public:
    RuntimeArgsWrapper() = default;

    RuntimeArgsColProxy get_col(size_t x) { return RuntimeArgsColProxy(args_, x); }

    void append(const CoreCoord& coord, const std::vector<uint32_t>& values) { args_.push_back({coord, values}); }

    tt::tt_metal::KernelDescriptor::RuntimeArgs& get() { return args_; }
    const tt::tt_metal::KernelDescriptor::RuntimeArgs& get() const { return args_; }

    size_t size() const { return args_.size(); }

    std::pair<CoreCoord, std::vector<uint32_t>>& at(size_t idx) { return args_.at(idx); }

    void clear() { args_.clear(); }

private:
    tt::tt_metal::KernelDescriptor::RuntimeArgs args_;
};

// View into existing RuntimeArgs (does not own data) - for accessing kernel_desc.runtime_args
class RuntimeArgsView {
public:
    explicit RuntimeArgsView(tt::tt_metal::KernelDescriptor::RuntimeArgs& args) : args_(args) {}
    RuntimeArgsColProxy get_col(size_t x) { return RuntimeArgsColProxy(args_, x); }
    size_t size() const { return args_.size(); }
    tt::tt_metal::KernelDescriptor::RuntimeArgs& get_ref() { return args_; }

private:
    tt::tt_metal::KernelDescriptor::RuntimeArgs& args_;
};

void py_module_types(nb::module_& mod) {
    nb::bind_vector<std::vector<uint32_t>>(mod, "VectorUInt32");

    // Bind RuntimeArgs helper classes for Python 2D indexing syntax: rtargs[x][y] = [args]
    nb::class_<RuntimeArgsColProxy>(mod, "RuntimeArgsColProxy", R"pbdoc(
        Proxy class for getting/setting runtime args at a specific x-coordinate.
        Used internally to enable rtargs[x][y] = [args] syntax.
    )pbdoc")
        .def(
            "__setitem__",
            &RuntimeArgsColProxy::set_item,
            nb::arg("y"),
            nb::arg("values"),
            R"pbdoc(
                Set runtime args for a specific core coordinate (upsert).

                Args:
                    y: Y coordinate of the core
                    values: List of runtime argument values
            )pbdoc")
        .def(
            "__getitem__",
            &RuntimeArgsColProxy::get_item,
            nb::arg("y"),
            nb::rv_policy::reference_internal,
            R"pbdoc(
                Get runtime args for a specific y core coordinate.
                Returns mutable reference to the runtime args.
            )pbdoc")
        .def(
            "extend",
            &RuntimeArgsColProxy::extend_item,
            nb::arg("y"),
            nb::arg("values"),
            R"pbdoc(
                Extend runtime args for a specific core coordinate.
            )pbdoc")
        .def(
            "append",
            &RuntimeArgsColProxy::append_item,
            nb::arg("y"),
            nb::arg("value"),
            R"pbdoc(
                Append a value to runtime args for a specific core coordinate.
            )pbdoc");

    nb::class_<RuntimeArgsWrapper>(mod, "RuntimeArgs", R"pbdoc(
        Wrapper for kernel runtime arguments that supports 2D indexing.

        Enables Python syntax: rtargs[x][y] = [arg1, arg2, ...]
        This translates to storing runtime args for CoreCoord(x, y).

        Matches the legacy API convention where runtime_args[i][j] is for core(i, j).

        Example:
            >>> rtargs = ttnn.RuntimeArgs()
            >>> rtargs[0][0] = [1, 2, 3]  # Args for core (0, 0)
            >>> rtargs[0][1] = [4, 5, 6]  # Args for core (0, 1)
            >>> rtargs[1][0] = [7, 8, 9]  # Args for core (1, 0)
            >>> kernel_desc.runtime_args = rtargs
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
            Create an empty RuntimeArgs container.
        )pbdoc")
        .def(
            "__getitem__",
            &RuntimeArgsWrapper::get_col,
            nb::arg("x"),
            R"pbdoc(
                Get a column proxy for setting args at a specific x-coordinate.

                Args:
                    x: X coordinate

                Returns:
                    RuntimeArgsColProxy for setting y values
            )pbdoc")
        .def(
            "append",
            &RuntimeArgsWrapper::append,
            nb::arg("coord"),
            nb::arg("values"),
            R"pbdoc(
                Append runtime args for a specific core coordinate.

                Args:
                    coord: CoreCoord specifying the core
                    values: List of runtime argument values
            )pbdoc")
        .def("__len__", &RuntimeArgsWrapper::size)
        .def(
            "get",
            [](RuntimeArgsWrapper& self, size_t idx) { return self.at(idx); },
            nb::arg("idx"),
            R"pbdoc(
                Get runtime args entry by index.

                Args:
                    idx: Index into the list of (CoreCoord, args) pairs

                Returns:
                    Tuple of (CoreCoord, list of args)
            )pbdoc")
        .def(
            "to_list",
            [](RuntimeArgsWrapper& self) { return self.get(); },
            R"pbdoc(
                Get all runtime args as a list of (CoreCoord, args) pairs.

                Returns:
                    List of (CoreCoord, list of args) tuples
            )pbdoc")
        .def("clear", &RuntimeArgsWrapper::clear, "Clear all runtime args")
        .def(
            "__iter__",
            [](RuntimeArgsWrapper& self) {
                return nb::make_iterator(
                    nb::type<RuntimeArgsWrapper>(), "iterator", self.get().begin(), self.get().end());
            },
            nb::keep_alive<0, 1>(),
            "Iterate over (CoreCoord, args) pairs");

    // Bind RuntimeArgsView for accessing existing runtime_args on KernelDescriptor
    nb::class_<RuntimeArgsView>(mod, "RuntimeArgsView")
        .def("__getitem__", &RuntimeArgsView::get_col, nb::arg("x"), nb::keep_alive<0, 1>())
        .def("__len__", &RuntimeArgsView::size);

    // Bind TileDescriptor first
    nb::class_<tt::tt_metal::TileDescriptor>(mod, "TileDescriptor", R"pbdoc(
        Descriptor for tile dimensions.

        Defines the height and width of a tile, which can be standard (32x32)
        or tiny tiles (e.g., 16x32 for certain operations).
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
            Default constructor for TileDescriptor (32x32 tile).
        )pbdoc")
        .def(
            nb::init<uint32_t, uint32_t, bool>(),
            nb::arg("height"),
            nb::arg("width"),
            nb::arg("transpose") = false,
            R"pbdoc(
                Initialize a TileDescriptor with custom dimensions.

                Args:
                    height: Height of the tile in elements
                    width: Width of the tile in elements
                    transpose: Whether the tile is transposed
            )pbdoc")
        .def(
            nb::init<const tt::tt_metal::Tile&>(),
            nb::arg("tile"),
            R"pbdoc(
                Initialize a TileDescriptor from a Tile object.

                Args:
                    tile: Tile object to create descriptor from
            )pbdoc")
        .def_rw("height", &tt::tt_metal::TileDescriptor::height, "Height of the tile in elements")
        .def_rw("width", &tt::tt_metal::TileDescriptor::width, "Width of the tile in elements")
        .def_rw("transpose", &tt::tt_metal::TileDescriptor::transpose, "Whether the tile is transposed");

    // Bind CBDescriptor and related types
    nb::class_<tt::tt_metal::CBFormatDescriptor>(mod, "CBFormatDescriptor", R"pbdoc(
        Descriptor for command buffer format configuration.

        Defines the format settings for sections of the command buffer,
        including buffer index, data format, and page size.
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
            Default constructor for CBFormatDescriptor.
        )pbdoc")
        .def(
            nb::init<uint8_t, tt::DataFormat, uint32_t, std::optional<tt::tt_metal::TileDescriptor>>(),
            nb::arg("buffer_index"),
            nb::arg("data_format"),
            nb::arg("page_size"),
            nb::arg("tile") = nb::none(),
            R"pbdoc(
                Initialize a CBFormatDescriptor with buffer index, data format and page size.

                Args:
                    buffer_index: Index of the buffer within the command buffer
                    data_format: Format of the data in the buffer
                    page_size: Size of a page in bytes
                    tile: Optional tile descriptor for custom tile dimensions (defaults to None)
            )pbdoc")
        .def(
            "__init__",
            [](tt::tt_metal::CBFormatDescriptor* t,
               uint8_t buffer_index,
               ttnn::DataType data_type,
               uint32_t page_size,
               std::optional<tt::tt_metal::TileDescriptor> tile) {
                // DataType to DataFormat conversion
                tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(data_type);
                new (t) tt::tt_metal::CBFormatDescriptor(buffer_index, data_format, page_size, tile);
            },
            nb::arg("buffer_index"),
            nb::arg("data_format"),
            nb::arg("page_size"),
            nb::arg("tile") = nb::none(),
            R"pbdoc(
                Initialize a CBFormatDescriptor with buffer index, TTNN data type, page size, and optional tile descriptor.

                This constructor automatically converts TTNN DataType to TT-Metal DataFormat.

                Args:
                    buffer_index: Index of the buffer within the command buffer
                    data_format: TTNN data type to be converted to TT-Metal data format
                    page_size: Size of a page in bytes
                    tile: Optional tile descriptor for custom tile dimensions (defaults to None)
            )pbdoc")
        .def_rw(
            "buffer_index",
            &tt::tt_metal::CBFormatDescriptor::buffer_index,
            "Index of the buffer within the command buffer")
        .def_rw("data_format", &tt::tt_metal::CBFormatDescriptor::data_format, "Format of the data in the buffer")
        .def_rw("page_size", &tt::tt_metal::CBFormatDescriptor::page_size, "Size of a page in bytes")
        .def_rw("tile", &tt::tt_metal::CBFormatDescriptor::tile, "Optional tile descriptor for custom tile dimensions");

    nb::class_<tt::tt_metal::CBDescriptor>(mod, "CBDescriptor", R"pbdoc(
        Circular Buffer Descriptor.

        Describes the structure and configuration of a command buffer,
        including its size, core ranges, and format descriptors.
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
            Default constructor for CBDescriptor.
        )pbdoc")
        .def(
            nb::init<uint32_t, CoreRangeSet, tt::tt_metal::CBDescriptor::FormatDescriptors>(),
            nb::arg("total_size"),
            nb::arg("core_ranges"),
            nb::arg("format_descriptors"),
            R"pbdoc(
                Initialize a CBDescriptor with total size, core ranges, and format descriptors.

                Args:
                    total_size: Total size of the command buffer in bytes
                    core_ranges: Set of core ranges where the command buffer is applicable
                    format_descriptors: Collection of format descriptors for different sections of the buffer
            )pbdoc")
        .def_rw("total_size", &tt::tt_metal::CBDescriptor::total_size, "Total size of the command buffer in bytes")
        .def_rw(
            "core_ranges",
            &tt::tt_metal::CBDescriptor::core_ranges,
            "Set of core ranges where the command buffer is applicable")
        .def_rw(
            "format_descriptors",
            &tt::tt_metal::CBDescriptor::format_descriptors,
            "Collection of format descriptors for different sections of the buffer");

    // Helper function for creating CBDescriptor from sharded tensor
    mod.def(
        "cb_descriptor_from_sharded_tensor",
        &tt::tt_metal::cb_descriptor_from_sharded_tensor,
        nb::arg("cb_index"),
        nb::arg("tensor"),
        R"pbdoc(
            Create a CBDescriptor from a sharded tensor.

            This function simplifies CB creation for sharded tensors by automatically deriving
            all CB configuration fields from the tensor's shard specification.

            Args:
                cb_index: The circular buffer index (CB ID)
                tensor: A sharded tensor to derive CB configuration from

            Returns:
                CBDescriptor with all fields (total_size, core_ranges, format_descriptors, buffer)
                automatically populated from the tensor

            Example:
                >>> # Assuming device_input_tensor is a sharded tensor
                >>> cb_desc = ttnn.cb_descriptor_from_sharded_tensor(
                ...     0,
                ...     device_input_tensor
                ... )
                >>> # Use cb_desc in ProgramDescriptor
                >>> program_desc = ttnn.ProgramDescriptor()
                >>> program_desc.cbs = [cb_desc]

            Note:
                The tensor must be sharded (have a shard specification), otherwise this will raise an error.
        )pbdoc");

    // Bind KernelDescriptor related types
    nb::class_<tt::tt_metal::ReaderConfigDescriptor>(mod, "ReaderConfigDescriptor", R"pbdoc(
        Configuration descriptor for reader components in a kernel.

        Defines how data should be read during kernel execution.
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
        Default constructor for ReaderConfigDescriptor.
    )pbdoc");

    nb::class_<tt::tt_metal::WriterConfigDescriptor>(mod, "WriterConfigDescriptor", R"pbdoc(
        Configuration descriptor for writer components in a kernel.

        Defines how data should be written during kernel execution.
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
        Default constructor for WriterConfigDescriptor.
    )pbdoc");

    nb::class_<tt::tt_metal::DataMovementConfigDescriptor>(mod, "DataMovementConfigDescriptor", R"pbdoc(
        Configuration descriptor for data movement operations.

        Controls processor selection, NOC routing, and NOC mode for data movement kernels.
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
            Default constructor for DataMovementConfigDescriptor.
        )pbdoc")
        .def(
            nb::init<tt::tt_metal::DataMovementProcessor, tt::tt_metal::NOC, tt::tt_metal::NOC_MODE>(),
            nb::arg("processor") = tt::tt_metal::DataMovementProcessor::RISCV_0,
            nb::arg("noc") = tt::tt_metal::NOC::RISCV_0_default,
            nb::arg("noc_mode") = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            R"pbdoc(
                Constructor for DataMovementConfigDescriptor with parameters.

                Args:
                    processor: Data movement processor to use (default: RISCV_0)
                    noc: Network-on-chip to use (default: RISCV_0_default)
                    noc_mode: NOC mode for data movement (default: DM_DEDICATED_NOC)
            )pbdoc")
        .def_rw("processor", &tt::tt_metal::DataMovementConfigDescriptor::processor, "Data movement processor to use")
        .def_rw("noc", &tt::tt_metal::DataMovementConfigDescriptor::noc, "Network-on-chip to use")
        .def_rw("noc_mode", &tt::tt_metal::DataMovementConfigDescriptor::noc_mode, "NOC mode for data movement");

    export_enum<UnpackToDestMode>(mod, "UnpackToDestMode");

    // nanobind bind_vector docs:
    // the item accessor __getitem__ copies the accessed element by default.
    // Consequently, writes to elements may not propagate in the expected way.
    nb::bind_vector<std::vector<UnpackToDestMode>>(mod, "VectorUnpackToDestMode");

    nb::class_<tt::tt_metal::ComputeConfigDescriptor>(mod, "ComputeConfigDescriptor", R"pbdoc(
        Configuration descriptor for compute operations.

        Controls various aspects of computation precision, synchronization,
        and numerical behavior during kernel execution.
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
            Default constructor for ComputeConfigDescriptor.
        )pbdoc")
        .def(
            "__init__",
            [](tt::tt_metal::ComputeConfigDescriptor* t,
               MathFidelity math_fidelity,
               bool math_approx_mode,
               bool fp32_dest_acc_en,
               bool dst_full_sync_en,
               bool bfp8_pack_precise) {
                new (t) tt::tt_metal::ComputeConfigDescriptor{
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en,
                    .bfp8_pack_precise = bfp8_pack_precise,
                    .math_approx_mode = math_approx_mode};
            },
            nb::arg("math_fidelity") = nb::cast(MathFidelity::HiFi4),
            nb::arg("math_approx_mode") = false,
            nb::arg("fp32_dest_acc_en") = false,
            nb::arg("dst_full_sync_en") = false,
            nb::arg("bfp8_pack_precise") = false,
            R"pbdoc(
                Constructor for ComputeConfigDescriptor with parameters.

                Args:
                    math_fidelity: Mathematical precision level (default: HiFi4)
                    math_approx_mode: Enable approximation mode (default: False)
                    fp32_dest_acc_en: Enable FP32 destination accumulation (default: False)
                    dst_full_sync_en: Enable full destination synchronization (default: False)
                    bfp8_pack_precise: Enable precise BFP8 packing (default: False)
            )pbdoc")
        .def_rw(
            "math_fidelity",
            &tt::tt_metal::ComputeConfigDescriptor::math_fidelity,
            "Controls mathematical precision during computation")
        .def_rw(
            "fp32_dest_acc_en",
            &tt::tt_metal::ComputeConfigDescriptor::fp32_dest_acc_en,
            "Enable FP32 destination accumulation")
        .def_rw(
            "dst_full_sync_en",
            &tt::tt_metal::ComputeConfigDescriptor::dst_full_sync_en,
            "Enable full synchronization for destinations")
        .def_rw(
            "unpack_to_dest_mode",
            &tt::tt_metal::ComputeConfigDescriptor::unpack_to_dest_mode,
            "Mode for unpacking to destination")
        .def_rw(
            "bfp8_pack_precise",
            &tt::tt_metal::ComputeConfigDescriptor::bfp8_pack_precise,
            "Use precise packing for BFP8 format")
        .def_rw(
            "math_approx_mode",
            &tt::tt_metal::ComputeConfigDescriptor::math_approx_mode,
            "Approximation mode for mathematical operations");

    // TODO_NANOBIND: do we still need this?
    // export_enum<tt::tt_metal::KernelDescriptor::SourceType>(mod, "SourceType");

    auto kernel_descriptor_class = nb::class_<tt::tt_metal::KernelDescriptor>(mod, "KernelDescriptor", R"pbdoc(
        Descriptor for a computational kernel.

        Contains all the information needed to compile and execute a kernel,
        including source code, compilation options, runtime arguments, and configuration.
    )pbdoc");

    // Bind SourceType as a nested enum within KernelDescriptor
    nb::enum_<tt::tt_metal::KernelDescriptor::SourceType>(kernel_descriptor_class, "SourceType", R"pbdoc(
        Source type for kernel source code.

        Defines whether the kernel source is provided as a file path or inline source code.
    )pbdoc")
        .value("FILE_PATH", tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH, "Kernel source is a file path")
        .value("SOURCE_CODE", tt::tt_metal::KernelDescriptor::SourceType::SOURCE_CODE, "Kernel source is inline code");

    kernel_descriptor_class
        .def(nb::init<>(), R"pbdoc(
            Default constructor for KernelDescriptor.
        )pbdoc")
        .def(
            nb::init<
                const std::string&,
                tt::tt_metal::KernelDescriptor::SourceType,
                CoreRangeSet,
                tt::tt_metal::KernelDescriptor::CompileTimeArgs,
                tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs,
                tt::tt_metal::KernelDescriptor::Defines,
                tt::tt_metal::KernelDescriptor::RuntimeArgs,
                tt::tt_metal::KernelDescriptor::CommonRuntimeArgs,
                std::optional<tt::tt_metal::KernelBuildOptLevel>,
                tt::tt_metal::KernelDescriptor::ConfigDescriptor>(),
            nb::arg("kernel_source"),
            nb::arg("source_type") = nb::cast(tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH),
            nb::arg("core_ranges"),
            nb::arg("compile_time_args") = nb::cast(tt::tt_metal::KernelDescriptor::CompileTimeArgs()),
            nb::arg("named_compile_time_args") = nb::cast(tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs()),
            nb::arg("defines") = nb::cast(tt::tt_metal::KernelDescriptor::Defines()),
            nb::arg("runtime_args") = nb::cast(tt::tt_metal::KernelDescriptor::RuntimeArgs()),
            nb::arg("common_runtime_args") = tt::tt_metal::KernelDescriptor::CommonRuntimeArgs(),
            nb::arg("opt_level") = nb::none(),
            nb::arg("config"),
            R"pbdoc(
                Initialize a KernelDescriptor with complete configuration.

                Args:
                    kernel_source: Path to kernel source file or inline kernel source code
                    source_type: Type of source (FILE_PATH or INLINE)
                    core_ranges: Set of core ranges where the kernel will execute
                    compile_time_args: Arguments provided at compile time
                    named_compile_time_args: Named arguments provided at compile time
                    defines: Preprocessor definitions for kernel compilation
                    runtime_args: Arguments provided at runtime
                    common_runtime_args: Common runtime arguments shared across kernels
                    opt_level: Optimization level for kernel compilation
                    config: Configuration descriptor for the kernel
            )pbdoc")
        .def_rw(
            "kernel_source",
            &tt::tt_metal::KernelDescriptor::kernel_source,
            "Path to kernel source file or inline kernel source code")
        .def_rw("source_type", &tt::tt_metal::KernelDescriptor::source_type, "Type of source (FILE_PATH or INLINE)")
        .def_rw(
            "core_ranges",
            &tt::tt_metal::KernelDescriptor::core_ranges,
            "Set of core ranges where the kernel will execute")
        .def_rw(
            "compile_time_args",
            &tt::tt_metal::KernelDescriptor::compile_time_args,
            "Arguments provided at compile time")
        .def_rw(
            "named_compile_time_args",
            &tt::tt_metal::KernelDescriptor::named_compile_time_args,
            "Named arguments provided at compile time")
        .def_rw("defines", &tt::tt_metal::KernelDescriptor::defines, "Preprocessor definitions for kernel compilation")
        .def_prop_rw(
            "runtime_args",
            [](tt::tt_metal::KernelDescriptor& self) { return RuntimeArgsView(self.runtime_args); },
            [](tt::tt_metal::KernelDescriptor& self, const nb::object& value) {
                // Accept RuntimeArgsWrapper, RuntimeArgsView, or the raw RuntimeArgs type
                if (nb::isinstance<RuntimeArgsWrapper>(value)) {
                    self.runtime_args = nb::cast<RuntimeArgsWrapper&>(value).get();
                } else if (nb::isinstance<RuntimeArgsView>(value)) {
                    // Copy from view (though unusual to assign a view)
                    self.runtime_args = nb::cast<RuntimeArgsView&>(value).get_ref();
                } else {
                    self.runtime_args = nb::cast<tt::tt_metal::KernelDescriptor::RuntimeArgs>(value);
                }
            },
            nb::keep_alive<0, 1>(),  // Keep KernelDescriptor alive while RuntimeArgsView exists
            R"pbdoc(
                Runtime arguments for the kernel.

                Returns a RuntimeArgsView that supports 2D indexing:
                    >>> args = kernel_desc.runtime_args[x][y]  # Get args for core (x, y)
                    >>> args.append(42)  # Modify in place

                Can also be set using:
                1. A RuntimeArgs wrapper with 2D indexing: rtargs[i][j] = [args]
                2. A list of (CoreCoord, args) pairs directly

                Example using RuntimeArgs wrapper:
                    >>> rtargs = ttnn.RuntimeArgs()
                    >>> rtargs[0][0] = [1, 2, 3]
                    >>> kernel_desc.runtime_args = rtargs

                Example using direct list:
                    >>> kernel_desc.runtime_args = [(ttnn.CoreCoord(0, 0), [1, 2, 3])]
            )pbdoc")
        .def_rw(
            "common_runtime_args",
            &tt::tt_metal::KernelDescriptor::common_runtime_args,
            "Common runtime arguments shared across all cores")
        .def_rw("config", &tt::tt_metal::KernelDescriptor::config, "Configuration descriptor for the kernel");

    // Bind SemaphoreDescriptor
    nb::class_<tt::tt_metal::SemaphoreDescriptor>(mod, "SemaphoreDescriptor", R"pbdoc(
        Descriptor for synchronization semaphores.

        Used for coordinating execution between different kernels and operations.
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
        Default constructor for SemaphoreDescriptor.
    )pbdoc")
        .def(
            nb::init<uint32_t, tt::CoreType, CoreRangeSet, uint32_t>(),
            nb::arg("id"),
            nb::arg("core_type") = nb::cast(tt::CoreType::WORKER),
            nb::arg("core_ranges"),
            nb::arg("initial_value"),
            R"pbdoc(
                Initialize a SemaphoreDescriptor with id, core type, core ranges, and initial value.
            )pbdoc")
        .def_ro("id", &tt::tt_metal::SemaphoreDescriptor::id, "Semaphore ID")
        .def_rw("core_type", &tt::tt_metal::SemaphoreDescriptor::core_type, "Type of core for the semaphore")
        .def_rw("core_ranges", &tt::tt_metal::SemaphoreDescriptor::core_ranges, "Core ranges for the semaphore")
        .def_rw("initial_value", &tt::tt_metal::SemaphoreDescriptor::initial_value, "Initial value for the semaphore");

    nb::class_<tt::tt_metal::ProgramDescriptor>(mod, "ProgramDescriptor", R"pbdoc(
        Descriptor for a complete program.

        A program is a collection of kernels, semaphores, and command buffers
        that work together to perform a computation task on the hardware.
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
            Default constructor for ProgramDescriptor.
        )pbdoc")
        .def(
            nb::init<
                tt::tt_metal::ProgramDescriptor::KernelDescriptors,
                tt::tt_metal::ProgramDescriptor::SemaphoreDescriptors,
                tt::tt_metal::ProgramDescriptor::CBDescriptors>(),
            nb::arg("kernels") = nb::cast(tt::tt_metal::ProgramDescriptor::KernelDescriptors()),
            nb::arg("semaphores") = nb::cast(tt::tt_metal::ProgramDescriptor::SemaphoreDescriptors()),
            nb::arg("cbs") = nb::cast(tt::tt_metal::ProgramDescriptor::CBDescriptors()),
            R"pbdoc(
                Initialize a ProgramDescriptor with kernels, semaphores, and command buffers.

                Args:
                    kernels: Collection of kernel descriptors
                    semaphores: Collection of semaphore descriptors
                    cbs: Collection of command buffer descriptors
            )pbdoc")
        .def_rw("kernels", &tt::tt_metal::ProgramDescriptor::kernels, "Collection of kernel descriptors")
        .def_rw("semaphores", &tt::tt_metal::ProgramDescriptor::semaphores, "Collection of semaphore descriptors")
        .def_rw("cbs", &tt::tt_metal::ProgramDescriptor::cbs, "Collection of command buffer descriptors");

    nb::class_<tt::tt_metal::experimental::MeshProgramDescriptor>(mod, "MeshProgramDescriptor", R"pbdoc(
        Descriptor for a mesh program.

        A mesh program is a collection of ProgramDescriptors, one for each device in the mesh.
        This behaves like a list of (MeshCoordinateRange, ProgramDescriptor) pairs with dict-like access.
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
            Default constructor. Creates an empty MeshProgramDescriptor.
        )pbdoc")
        .def(
            "__init__",
            [](tt::tt_metal::experimental::MeshProgramDescriptor* self, const nb::dict& mesh_programs) {
                new (self) tt::tt_metal::experimental::MeshProgramDescriptor();
                for (const auto& [key_obj, value_obj] : mesh_programs) {
                    auto key = nb::cast<tt::tt_metal::distributed::MeshCoordinateRange>(key_obj);
                    auto value = nb::cast<tt::tt_metal::ProgramDescriptor>(value_obj);
                    self->mesh_programs.emplace_back(key, value);
                }
            },
            nb::arg("mesh_programs"),
            R"pbdoc(
            Constructor that initializes from a Python dict.

            Args:
                mesh_programs: Dictionary mapping MeshCoordinateRange to ProgramDescriptor

            Example:
                desc = ttnn.MeshProgramDescriptor()
                desc[range] = program_descriptor
                # Positional argument:
                desc = ttnn.MeshProgramDescriptor({range1: prog1, range2: prog2})
                # Keyword argument:
                desc = ttnn.MeshProgramDescriptor(mesh_programs={range1: prog1, range2: prog2})
        )pbdoc")
        .def(
            "__getitem__",
            [](const tt::tt_metal::experimental::MeshProgramDescriptor& self,
               const tt::tt_metal::distributed::MeshCoordinateRange& key) {
                for (const auto& [k, v] : self.mesh_programs) {
                    if (k == key) {
                        return v;
                    }
                }
                throw std::runtime_error("MeshCoordinateRange not found in MeshProgramDescriptor");
            },
            nb::arg("key"),
            R"pbdoc(
                Get the ProgramDescriptor for a given MeshCoordinateRange.
            )pbdoc")
        .def(
            "__setitem__",
            [](tt::tt_metal::experimental::MeshProgramDescriptor& self,
               const tt::tt_metal::distributed::MeshCoordinateRange& key,
               const tt::tt_metal::ProgramDescriptor& value) { self.mesh_programs.emplace_back(key, value); },
            nb::arg("key"),
            nb::arg("value"),
            R"pbdoc(
                Add a ProgramDescriptor for a given MeshCoordinateRange.
            )pbdoc")
        .def(
            "__contains__",
            [](const tt::tt_metal::experimental::MeshProgramDescriptor& self,
               const tt::tt_metal::distributed::MeshCoordinateRange& key) {
                for (const auto& [k, v] : self.mesh_programs) {
                    if (k == key) {
                        return true;
                    }
                }
                return false;
            },
            nb::arg("key"),
            R"pbdoc(
                Check if a MeshCoordinateRange exists in the MeshProgramDescriptor.
            )pbdoc")
        .def(
            "__iter__",
            [](const tt::tt_metal::experimental::MeshProgramDescriptor& self) {
                return nb::make_iterator<nb::rv_policy::reference_internal>(
                    nb::type<tt::tt_metal::experimental::MeshProgramDescriptor>(),
                    "iterator",
                    self.mesh_programs.begin(),
                    self.mesh_programs.end());
            },
            nb::keep_alive<0, 1>(),
            R"pbdoc(
                Iterate over (MeshCoordinateRange, ProgramDescriptor) pairs.
            )pbdoc");

    // TODO_NANOBIND: AFFECTS BEHAVIOR
    [[maybe_unused]]
    auto e_CBIndex = export_enum<tt::CBIndex>(mod, "CBIndex");
    // e_CBIndex
    //     .def("__init__",
    //          [](CBIndex* t, nb::int_ i) {
    //                 ;
    //          });
    //     .def(nb::init<>())
    //     .def(nb::self == nb::self);
    // nb::implicitly_convertible<nb::int_, tt::CBIndex>();
}

}  // namespace ttnn::program_descriptors
