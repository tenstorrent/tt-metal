// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_descriptors.hpp"

#include <optional>
#include <string>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn-pybind/export_enum.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::program_descriptors {

void py_module_types(py::module& module) {
    // Bind CBDescriptor and related types
    py::class_<tt::tt_metal::CBFormatDescriptor>(module, "CBFormatDescriptor", R"pbdoc(
        Descriptor for command buffer format configuration.

        Defines the format settings for sections of the command buffer,
        including buffer index, data format, and page size.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Default constructor for CBFormatDescriptor.
        )pbdoc")
        .def(
            py::init<uint8_t, tt::DataFormat, uint32_t>(),
            py::arg("buffer_index"),
            py::arg("data_format"),
            py::arg("page_size"),
            R"pbdoc(
                Initialize a CBFormatDescriptor with buffer index, data format and page size.

                Args:
                    buffer_index: Index of the buffer within the command buffer
                    data_format: Format of the data in the buffer
                    page_size: Size of a page in bytes
            )pbdoc")
        .def(
            py::init([](uint8_t buffer_index, ttnn::DataType data_type, uint32_t page_size) {
                // DataType to DataFormat conversion
                tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(data_type);
                return tt::tt_metal::CBFormatDescriptor(buffer_index, data_format, page_size);
            }),
            py::arg("buffer_index"),
            py::arg("data_format"),
            py::arg("page_size"),
            R"pbdoc(
                Initialize a CBFormatDescriptor with buffer index, TTNN data type and page size.

                This constructor automatically converts TTNN DataType to TT-Metal DataFormat.

                Args:
                    buffer_index: Index of the buffer within the command buffer
                    data_format: TTNN data type to be converted to TT-Metal data format
                    page_size: Size of a page in bytes
            )pbdoc")
        .def_readwrite(
            "buffer_index",
            &tt::tt_metal::CBFormatDescriptor::buffer_index,
            "Index of the buffer within the command buffer")
        .def_readwrite(
            "data_format", &tt::tt_metal::CBFormatDescriptor::data_format, "Format of the data in the buffer")
        .def_readwrite("page_size", &tt::tt_metal::CBFormatDescriptor::page_size, "Size of a page in bytes");

    py::class_<tt::tt_metal::CBDescriptor>(module, "CBDescriptor", R"pbdoc(
        Command Buffer Descriptor.

        Describes the structure and configuration of a command buffer,
        including its size, core ranges, and format descriptors.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Default constructor for CBDescriptor.
        )pbdoc")
        .def(
            py::init<uint32_t, CoreRangeSet, tt::tt_metal::CBDescriptor::FormatDescriptors>(),
            py::arg("total_size"),
            py::arg("core_ranges"),
            py::arg("format_descriptors"),
            R"pbdoc(
                Initialize a CBDescriptor with total size, core ranges, and format descriptors.

                Args:
                    total_size: Total size of the command buffer in bytes
                    core_ranges: Set of core ranges where the command buffer is applicable
                    format_descriptors: Collection of format descriptors for different sections of the buffer
            )pbdoc")
        .def_readwrite(
            "total_size", &tt::tt_metal::CBDescriptor::total_size, "Total size of the command buffer in bytes")
        .def_readwrite(
            "core_ranges",
            &tt::tt_metal::CBDescriptor::core_ranges,
            "Set of core ranges where the command buffer is applicable")
        .def_readwrite(
            "format_descriptors",
            &tt::tt_metal::CBDescriptor::format_descriptors,
            "Collection of format descriptors for different sections of the buffer");

    // Bind KernelDescriptor related types
    py::class_<tt::tt_metal::ReaderConfigDescriptor>(module, "ReaderConfigDescriptor", R"pbdoc(
        Configuration descriptor for reader components in a kernel.

        Defines how data should be read during kernel execution.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
        Default constructor for ReaderConfigDescriptor.
    )pbdoc");

    py::class_<tt::tt_metal::WriterConfigDescriptor>(module, "WriterConfigDescriptor", R"pbdoc(
        Configuration descriptor for writer components in a kernel.

        Defines how data should be written during kernel execution.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
        Default constructor for WriterConfigDescriptor.
    )pbdoc");

    export_enum<UnpackToDestMode>(module, "UnpackToDestMode");
    py::bind_vector<std::vector<UnpackToDestMode>>(module, "VectorUnpackToDestMode");

    py::class_<tt::tt_metal::ComputeConfigDescriptor>(module, "ComputeConfigDescriptor", R"pbdoc(
        Configuration descriptor for compute operations.

        Controls various aspects of computation precision, synchronization,
        and numerical behavior during kernel execution.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Default constructor for ComputeConfigDescriptor.
        )pbdoc")
        .def_readwrite(
            "math_fidelity",
            &tt::tt_metal::ComputeConfigDescriptor::math_fidelity,
            "Controls mathematical precision during computation")
        .def_readwrite(
            "fp32_dest_acc_en",
            &tt::tt_metal::ComputeConfigDescriptor::fp32_dest_acc_en,
            "Enable FP32 destination accumulation")
        .def_readwrite(
            "dst_full_sync_en",
            &tt::tt_metal::ComputeConfigDescriptor::dst_full_sync_en,
            "Enable full synchronization for destinations")
        .def_readwrite(
            "unpack_to_dest_mode",
            &tt::tt_metal::ComputeConfigDescriptor::unpack_to_dest_mode,
            "Mode for unpacking to destination")
        .def_readwrite(
            "bfp8_pack_precise",
            &tt::tt_metal::ComputeConfigDescriptor::bfp8_pack_precise,
            "Use precise packing for BFP8 format")
        .def_readwrite(
            "math_approx_mode",
            &tt::tt_metal::ComputeConfigDescriptor::math_approx_mode,
            "Approximation mode for mathematical operations");

    export_enum<tt::tt_metal::KernelDescriptor::SourceType>(module, "SourceType");
    py::class_<tt::tt_metal::KernelDescriptor>(module, "KernelDescriptor", R"pbdoc(
        Descriptor for a computational kernel.

        Contains all the information needed to compile and execute a kernel,
        including source code, compilation options, runtime arguments, and configuration.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Default constructor for KernelDescriptor.
        )pbdoc")
        .def(
            py::init<
                const std::string&,
                tt::tt_metal::KernelDescriptor::SourceType,
                CoreRangeSet,
                tt::tt_metal::KernelDescriptor::CompileTimeArgs,
                tt::tt_metal::KernelDescriptor::Defines,
                tt::tt_metal::KernelDescriptor::RuntimeArgs,
                tt::tt_metal::KernelDescriptor::CommonRuntimeArgs,
                std::optional<tt::tt_metal::KernelBuildOptLevel>,
                tt::tt_metal::KernelDescriptor::ConfigDescriptor>(),
            py::arg("kernel_source"),
            py::arg("source_type") = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH,
            py::arg("core_ranges"),
            py::arg("compile_time_args"),
            py::arg("defines") = tt::tt_metal::KernelDescriptor::Defines(),
            py::arg("runtime_args"),
            py::arg("common_runtime_args") = tt::tt_metal::KernelDescriptor::CommonRuntimeArgs(),
            py::arg("opt_level") = std::nullopt,
            py::arg("config"),
            R"pbdoc(
                Initialize a KernelDescriptor with complete configuration.

                Args:
                    kernel_source: Path to kernel source file or inline kernel source code
                    source_type: Type of source (FILE_PATH or INLINE)
                    core_ranges: Set of core ranges where the kernel will execute
                    compile_time_args: Arguments provided at compile time
                    defines: Preprocessor definitions for kernel compilation
                    runtime_args: Arguments provided at runtime
                    common_runtime_args: Common runtime arguments shared across kernels
                    opt_level: Optimization level for kernel compilation
                    config: Configuration descriptor for the kernel
            )pbdoc")
        .def_readwrite(
            "kernel_source",
            &tt::tt_metal::KernelDescriptor::kernel_source,
            "Path to kernel source file or inline kernel source code")
        .def_readwrite(
            "source_type", &tt::tt_metal::KernelDescriptor::source_type, "Type of source (FILE_PATH or INLINE)")
        .def_readwrite(
            "core_ranges",
            &tt::tt_metal::KernelDescriptor::core_ranges,
            "Set of core ranges where the kernel will execute")
        .def_readwrite(
            "compile_time_args",
            &tt::tt_metal::KernelDescriptor::compile_time_args,
            "Arguments provided at compile time")
        .def_readwrite(
            "defines", &tt::tt_metal::KernelDescriptor::defines, "Preprocessor definitions for kernel compilation")
        .def_readwrite("runtime_args", &tt::tt_metal::KernelDescriptor::runtime_args, "Arguments provided at runtime")
        .def_readwrite("config", &tt::tt_metal::KernelDescriptor::config, "Configuration descriptor for the kernel");

    // Bind SemaphoreDescriptor
    py::class_<tt::tt_metal::SemaphoreDescriptor>(module, "SemaphoreDescriptor", R"pbdoc(
        Descriptor for synchronization semaphores.

        Used for coordinating execution between different kernels and operations.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
        Default constructor for SemaphoreDescriptor.
    )pbdoc");

    py::class_<tt::tt_metal::ProgramDescriptor>(module, "ProgramDescriptor", R"pbdoc(
        Descriptor for a complete program.

        A program is a collection of kernels, semaphores, and command buffers
        that work together to perform a computation task on the hardware.
    )pbdoc")
        .def(py::init<>(), R"pbdoc(
            Default constructor for ProgramDescriptor.
        )pbdoc")
        .def(
            py::init<
                tt::tt_metal::ProgramDescriptor::KernelDescriptors,
                tt::tt_metal::ProgramDescriptor::SemaphoreDescriptors,
                tt::tt_metal::ProgramDescriptor::CBDescriptors>(),
            py::arg("kernels"),
            py::arg("semaphores"),
            py::arg("cbs"),
            R"pbdoc(
                Initialize a ProgramDescriptor with kernels, semaphores, and command buffers.

                Args:
                    kernels: Collection of kernel descriptors
                    semaphores: Collection of semaphore descriptors
                    cbs: Collection of command buffer descriptors
            )pbdoc")
        .def_readwrite("kernels", &tt::tt_metal::ProgramDescriptor::kernels, "Collection of kernel descriptors")
        .def_readwrite(
            "semaphores", &tt::tt_metal::ProgramDescriptor::semaphores, "Collection of semaphore descriptors")
        .def_readwrite("cbs", &tt::tt_metal::ProgramDescriptor::cbs, "Collection of command buffer descriptors");

    export_enum<tt::CBIndex>(module, "CBIndex");
    py::implicitly_convertible<py::int_, tt::CBIndex>();
}

}  // namespace ttnn::program_descriptors
