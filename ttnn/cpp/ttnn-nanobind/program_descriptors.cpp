// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-nanobind/program_descriptors.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include <tt-metalium/program_descriptors.hpp>

NB_MAKE_OPAQUE(std::vector<UnpackToDestMode>);

namespace ttnn::program_descriptors {

void py_module_types(nb::module_& mod) {
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
            nb::init<uint8_t, tt::DataFormat, uint32_t>(),
            nb::arg("buffer_index"),
            nb::arg("data_format"),
            nb::arg("page_size"),
            R"pbdoc(
                Initialize a CBFormatDescriptor with buffer index, data format and page size.

                Args:
                    buffer_index: Index of the buffer within the command buffer
                    data_format: Format of the data in the buffer
                    page_size: Size of a page in bytes
            )pbdoc")
        .def(
            "__init__",
            [](tt::tt_metal::CBFormatDescriptor* t,
               uint8_t buffer_index,
               ttnn::DataType data_type,
               uint32_t page_size) {
                // DataType to DataFormat conversion
                tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(data_type);
                new (t) tt::tt_metal::CBFormatDescriptor(buffer_index, data_format, page_size);
            },
            nb::arg("buffer_index"),
            nb::arg("data_format"),
            nb::arg("page_size"),
            R"pbdoc(
                Initialize a CBFormatDescriptor with buffer index, TTNN data type and page size.

                This constructor automatically converts TTNN DataType to TT-Metal DataFormat.

                Args:
                    buffer_index: Index of the buffer within the command buffer
                    data_format: TTNN data type to be converted to TT-Metal data format
                    page_size: Size of a page in bytes
            )pbdoc")
        .def_rw(
            "buffer_index",
            &tt::tt_metal::CBFormatDescriptor::buffer_index,
            "Index of the buffer within the command buffer")
        .def_rw("data_format", &tt::tt_metal::CBFormatDescriptor::data_format, "Format of the data in the buffer")
        .def_rw("page_size", &tt::tt_metal::CBFormatDescriptor::page_size, "Size of a page in bytes");

    nb::class_<tt::tt_metal::CBDescriptor>(mod, "CBDescriptor", R"pbdoc(
        Command Buffer Descriptor.

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

    export_enum<tt::tt_metal::KernelDescriptor::SourceType>(mod, "SourceType");

    nb::class_<tt::tt_metal::KernelDescriptor>(mod, "KernelDescriptor", R"pbdoc(
        Descriptor for a computational kernel.

        Contains all the information needed to compile and execute a kernel,
        including source code, compilation options, runtime arguments, and configuration.
    )pbdoc");

    // Bind SourceType as a nested enum within KernelDescriptor
    py::enum_<tt::tt_metal::KernelDescriptor::SourceType>(kernel_descriptor_class, "SourceType", R"pbdoc(
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
                tt::tt_metal::KernelDescriptor::Defines,
                tt::tt_metal::KernelDescriptor::RuntimeArgs,
                tt::tt_metal::KernelDescriptor::CommonRuntimeArgs,
                std::optional<tt::tt_metal::KernelBuildOptLevel>,
                tt::tt_metal::KernelDescriptor::ConfigDescriptor>(),
            nb::arg("kernel_source"),
            nb::arg("source_type") = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH,
            nb::arg("core_ranges"),
            nb::arg("compile_time_args"),
            nb::arg("defines") = tt::tt_metal::KernelDescriptor::Defines(),
            nb::arg("runtime_args"),
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
        .def_rw("defines", &tt::tt_metal::KernelDescriptor::defines, "Preprocessor definitions for kernel compilation")
        .def_rw("runtime_args", &tt::tt_metal::KernelDescriptor::runtime_args, "Arguments provided at runtime")
        .def_rw("config", &tt::tt_metal::KernelDescriptor::config, "Configuration descriptor for the kernel");

    // needed to set SemaphoreDescriptor CoreType default
    // nb::module_::import_("types");
    // nb::module_::import_("ttnn.types");

    // Bind SemaphoreDescriptor
    nb::class_<tt::tt_metal::SemaphoreDescriptor>(mod, "SemaphoreDescriptor", R"pbdoc(
        Descriptor for synchronization semaphores.

        Used for coordinating execution between different kernels and operations.
    )pbdoc")
        .def(nb::init<>(), R"pbdoc(
        Default constructor for SemaphoreDescriptor.
    )pbdoc")
        .def(  // TODO_NANOBIND: FIX. AFFECTS BEHAVIOR!
            nb::init<CoreType, CoreRangeSet, uint32_t>(),
            nb::arg("core_type"),  //= nb::cast(CoreType::WORKER), // TODO_NANOBIND causes segfault when import ttnn???
            nb::arg("core_ranges"),
            nb::arg("initial_value"),
            R"pbdoc(
                Initialize a SemaphoreDescriptor with core type, core ranges, and initial value.
            )pbdoc")
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
            nb::arg("kernels"),
            nb::arg("semaphores"),
            nb::arg("cbs"),
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
