// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include "program_descriptors.hpp"
#include "ttnn-pybind/export_enum.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::program_descriptors {

void py_module_types(py::module& module) {
    // Bind CBDescriptor and related types
    py::class_<tt::tt_metal::CBFormatDescriptor>(module, "CBFormatDescriptor")
        .def(py::init<>())
        .def(
            py::init<uint8_t, tt::DataFormat, uint32_t>(),
            py::arg("buffer_index"),
            py::arg("data_format"),
            py::arg("page_size"))
        .def(
            py::init([](uint8_t buffer_index, ttnn::DataType data_type, uint32_t page_size) {
                // DataType to DataFormat conversion
                tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(data_type);
                return tt::tt_metal::CBFormatDescriptor(buffer_index, data_format, page_size);
            }),
            py::arg("buffer_index"),
            py::arg("data_format"),
            py::arg("page_size"))
        .def_readwrite("buffer_index", &tt::tt_metal::CBFormatDescriptor::buffer_index)
        .def_readwrite("data_format", &tt::tt_metal::CBFormatDescriptor::data_format)
        .def_readwrite("page_size", &tt::tt_metal::CBFormatDescriptor::page_size);

    py::class_<tt::tt_metal::CBDescriptor>(module, "CBDescriptor")
        .def(py::init<>())
        .def(
            py::init<uint32_t, CoreRangeSet, tt::tt_metal::CBDescriptor::FormatDescriptors>(),
            py::arg("total_size"),
            py::arg("core_ranges"),
            py::arg("format_descriptors"))
        .def_readwrite("total_size", &tt::tt_metal::CBDescriptor::total_size)
        .def_readwrite("core_ranges", &tt::tt_metal::CBDescriptor::core_ranges)
        .def_readwrite("format_descriptors", &tt::tt_metal::CBDescriptor::format_descriptors);

    // Bind KernelDescriptor related types
    py::class_<tt::tt_metal::ReaderConfigDescriptor>(module, "ReaderConfigDescriptor").def(py::init<>());

    py::class_<tt::tt_metal::WriterConfigDescriptor>(module, "WriterConfigDescriptor").def(py::init<>());

    export_enum<UnpackToDestMode>(module, "UnpackToDestMode");
    py::bind_vector<std::vector<UnpackToDestMode>>(module, "VectorUnpackToDestMode");

    py::class_<tt::tt_metal::ComputeConfigDescriptor>(module, "ComputeConfigDescriptor")
        .def(py::init<>())
        .def_readwrite("math_fidelity", &tt::tt_metal::ComputeConfigDescriptor::math_fidelity)
        .def_readwrite("fp32_dest_acc_en", &tt::tt_metal::ComputeConfigDescriptor::fp32_dest_acc_en)
        .def_readwrite("dst_full_sync_en", &tt::tt_metal::ComputeConfigDescriptor::dst_full_sync_en)
        .def_readwrite("unpack_to_dest_mode", &tt::tt_metal::ComputeConfigDescriptor::unpack_to_dest_mode)
        .def_readwrite("bfp8_pack_precise", &tt::tt_metal::ComputeConfigDescriptor::bfp8_pack_precise)
        .def_readwrite("math_approx_mode", &tt::tt_metal::ComputeConfigDescriptor::math_approx_mode);

    export_enum<tt::tt_metal::KernelDescriptor::SourceType>(module, "SourceType");
    py::class_<tt::tt_metal::KernelDescriptor>(module, "KernelDescriptor")
        .def(py::init<>())
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
            py::arg("config"))
        .def_readwrite("kernel_source", &tt::tt_metal::KernelDescriptor::kernel_source)
        .def_readwrite("source_type", &tt::tt_metal::KernelDescriptor::source_type)
        .def_readwrite("core_ranges", &tt::tt_metal::KernelDescriptor::core_ranges)
        .def_readwrite("compile_time_args", &tt::tt_metal::KernelDescriptor::compile_time_args)
        .def_readwrite("defines", &tt::tt_metal::KernelDescriptor::defines)
        .def_readwrite("runtime_args", &tt::tt_metal::KernelDescriptor::runtime_args)
        .def_readwrite("config", &tt::tt_metal::KernelDescriptor::config);

    // Bind SemaphoreDescriptor - don't care about this yet
    py::class_<tt::tt_metal::SemaphoreDescriptor>(module, "SemaphoreDescriptor").def(py::init<>());

    py::class_<tt::tt_metal::ProgramDescriptor>(module, "ProgramDescriptor")
        .def(py::init<>())
        .def(
            py::init<
                tt::tt_metal::ProgramDescriptor::KernelDescriptors,
                tt::tt_metal::ProgramDescriptor::SemaphoreDescriptors,
                tt::tt_metal::ProgramDescriptor::CBDescriptors>(),
            py::arg("kernels"),
            py::arg("semaphores"),
            py::arg("cbs"))
        .def_readwrite("kernels", &tt::tt_metal::ProgramDescriptor::kernels)
        .def_readwrite("semaphores", &tt::tt_metal::ProgramDescriptor::semaphores)
        .def_readwrite("cbs", &tt::tt_metal::ProgramDescriptor::cbs);

    export_enum<tt::CBIndex>(module, "CBIndex");
    py::implicitly_convertible<py::int_, tt::CBIndex>();
}

}  // namespace ttnn::program_descriptors
