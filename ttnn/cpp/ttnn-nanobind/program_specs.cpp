// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_specs.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn-nanobind/metal2_casters.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::program_specs {

namespace m2 = tt::tt_metal::experimental;

void py_module_types(nb::module_& mod) {
    export_enum<tt::tt_metal::UnpackMode>(mod, "UnpackMode");
    export_enum<tt::tt_metal::Precision>(mod, "Precision");
    export_enum<tt::tt_metal::KernelBuildOptLevel>(mod, "KernelBuildOptLevel");
    export_enum<m2::DFBEndpointType>(mod, "DFBEndpointType");
    export_enum<m2::DFBAccessPattern>(mod, "DFBAccessPattern");

    // ---------------------------------------------------------------- tensor parameters

    nb::class_<m2::TensorSpecRelaxations>(mod, "TensorSpecRelaxations")
        .def(
            "__init__",
            [](m2::TensorSpecRelaxations* self, bool match_padded_shape_only, bool dynamic_tensor_shape) {
                new (self) m2::TensorSpecRelaxations{
                    .match_padded_shape_only = match_padded_shape_only, .dynamic_tensor_shape = dynamic_tensor_shape};
            },
            nb::arg("match_padded_shape_only") = false,
            nb::arg("dynamic_tensor_shape") = false)
        .def_rw("match_padded_shape_only", &m2::TensorSpecRelaxations::match_padded_shape_only)
        .def_rw("dynamic_tensor_shape", &m2::TensorSpecRelaxations::dynamic_tensor_shape);

    nb::class_<m2::TensorParameter>(mod, "TensorParameter")
        .def(
            "__init__",
            [](m2::TensorParameter* self,
               m2::TensorParamName unique_id,
               const tt::tt_metal::TensorSpec& spec,
               m2::TensorSpecRelaxations relaxations) {
                new (self)
                    m2::TensorParameter{.unique_id = std::move(unique_id), .spec = spec, .relaxations = relaxations};
            },
            nb::arg("unique_id"),
            nb::arg("spec"),
            nb::arg("relaxations") = m2::TensorSpecRelaxations{})
        .def_rw("unique_id", &m2::TensorParameter::unique_id)
        .def_rw("spec", &m2::TensorParameter::spec)
        .def_rw("relaxations", &m2::TensorParameter::relaxations);

    // ---------------------------------------------------------------- dataflow buffers

    nb::class_<m2::DFBAdvancedOptions>(mod, "DFBAdvancedOptions")
        .def(
            "__init__",
            [](m2::DFBAdvancedOptions* self,
               std::vector<m2::DFBSpecName> alias_with,
               bool allow_instance_multi_binding) {
                new (self) m2::DFBAdvancedOptions{
                    .alias_with = std::move(alias_with), .allow_instance_multi_binding = allow_instance_multi_binding};
            },
            nb::arg("alias_with") = std::vector<m2::DFBSpecName>{},
            nb::arg("allow_instance_multi_binding") = false)
        .def_rw("alias_with", &m2::DFBAdvancedOptions::alias_with)
        .def_rw("allow_instance_multi_binding", &m2::DFBAdvancedOptions::allow_instance_multi_binding);

    nb::class_<m2::DataflowBufferSpec>(mod, "DataflowBufferSpec")
        .def(
            // data_format is taken as a ttnn.DataType (as CBFormatDescriptor does) and converted.
            "__init__",
            [](m2::DataflowBufferSpec* self,
               m2::DFBSpecName unique_id,
               uint32_t entry_size,
               uint32_t num_entries,
               std::optional<ttnn::DataType> data_format,
               std::optional<tt::tt_metal::Tile> tile_format,
               std::optional<tt::tt_metal::FaceGeometry> unpack_face_geometry,
               std::optional<m2::TensorParamName> borrowed_from,
               m2::DFBAdvancedOptions advanced_options) {
                std::optional<tt::DataFormat> df;
                if (data_format.has_value()) {
                    df = tt::tt_metal::datatype_to_dataformat_converter(*data_format);
                }
                new (self) m2::DataflowBufferSpec{
                    .unique_id = std::move(unique_id),
                    .entry_size = entry_size,
                    .num_entries = num_entries,
                    .data_format_metadata = df,
                    .tile_format_metadata = tile_format,
                    .unpack_face_geometry_metadata = unpack_face_geometry,
                    .borrowed_from = std::move(borrowed_from),
                    .advanced_options = std::move(advanced_options)};
            },
            nb::arg("unique_id"),
            nb::arg("entry_size"),
            nb::arg("num_entries"),
            nb::arg("data_format") = nb::none(),
            nb::arg("tile_format") = nb::none(),
            nb::arg("unpack_face_geometry") = nb::none(),
            nb::arg("borrowed_from") = nb::none(),
            nb::arg("advanced_options") = m2::DFBAdvancedOptions{})
        .def_rw("unique_id", &m2::DataflowBufferSpec::unique_id)
        .def_rw("entry_size", &m2::DataflowBufferSpec::entry_size)
        .def_rw("num_entries", &m2::DataflowBufferSpec::num_entries)
        // tt::DataFormat has no Python binding; surface it as its integer value.
        .def_prop_ro(
            "data_format_metadata",
            [](const m2::DataflowBufferSpec& self) -> std::optional<uint32_t> {
                if (!self.data_format_metadata.has_value()) {
                    return std::nullopt;
                }
                return static_cast<uint32_t>(*self.data_format_metadata);
            })
        .def_rw("tile_format_metadata", &m2::DataflowBufferSpec::tile_format_metadata)
        .def_rw("borrowed_from", &m2::DataflowBufferSpec::borrowed_from)
        .def_rw("advanced_options", &m2::DataflowBufferSpec::advanced_options);

    // ---------------------------------------------------------------- semaphores, scratchpads

    nb::class_<m2::SemaphoreSpec>(mod, "SemaphoreSpec")
        .def(
            "__init__",
            [](m2::SemaphoreSpec* self, m2::SemaphoreSpecName unique_id, m2::Nodes target_nodes) {
                new (self)
                    m2::SemaphoreSpec{.unique_id = std::move(unique_id), .target_nodes = std::move(target_nodes)};
            },
            nb::arg("unique_id"),
            nb::arg("target_nodes"))
        .def_rw("unique_id", &m2::SemaphoreSpec::unique_id)
        .def_rw("target_nodes", &m2::SemaphoreSpec::target_nodes);

    nb::class_<m2::ScratchpadSpec>(mod, "ScratchpadSpec")
        .def(
            "__init__",
            [](m2::ScratchpadSpec* self, m2::ScratchpadSpecName unique_id, uint32_t size_per_node) {
                new (self) m2::ScratchpadSpec{.unique_id = std::move(unique_id), .size_per_node = size_per_node};
            },
            nb::arg("unique_id"),
            nb::arg("size_per_node"))
        .def_rw("unique_id", &m2::ScratchpadSpec::unique_id)
        .def_rw("size_per_node", &m2::ScratchpadSpec::size_per_node);

    // ---------------------------------------------------------------- hardware configs

    nb::class_<m2::DataMovementGen1Config>(mod, "DataMovementGen1Config")
        .def(
            "__init__",
            [](m2::DataMovementGen1Config* self,
               tt::tt_metal::DataMovementProcessor processor,
               tt::tt_metal::NOC noc,
               tt::tt_metal::NOC_MODE noc_mode) {
                new (self) m2::DataMovementGen1Config{.processor = processor, .noc = noc, .noc_mode = noc_mode};
            },
            nb::arg("processor"),
            nb::arg("noc"),
            nb::arg("noc_mode") = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC)
        .def_rw("processor", &m2::DataMovementGen1Config::processor)
        .def_rw("noc", &m2::DataMovementGen1Config::noc)
        .def_rw("noc_mode", &m2::DataMovementGen1Config::noc_mode);

    nb::class_<m2::DataMovementGen2Config>(mod, "DataMovementGen2Config")
        .def(
            "__init__",
            [](m2::DataMovementGen2Config* self,
               std::vector<m2::DFBSpecName> disable_dfb_implicit_sync_for,
               bool disable_dfb_implicit_sync_for_all) {
                new (self) m2::DataMovementGen2Config{
                    .disable_dfb_implicit_sync_for = std::move(disable_dfb_implicit_sync_for),
                    .disable_dfb_implicit_sync_for_all = disable_dfb_implicit_sync_for_all};
            },
            nb::arg("disable_dfb_implicit_sync_for") = std::vector<m2::DFBSpecName>{},
            nb::arg("disable_dfb_implicit_sync_for_all") = false);

    nb::class_<m2::ComputeGen1Config>(mod, "ComputeGen1Config")
        .def(
            "__init__",
            [](m2::ComputeGen1Config* self,
               MathFidelity fpu_math_fidelity,
               tt::tt_metal::Precision sfpu_precision_mode,
               tt::tt_metal::Precision bfp_pack_precision_mode,
               bool enable_32_bit_dest,
               bool double_buffer_dest,
               m2::ComputeUnpackModes unpack_modes) {
                new (self) m2::ComputeGen1Config{
                    .fpu_math_fidelity = fpu_math_fidelity,
                    .sfpu_precision_mode = sfpu_precision_mode,
                    .bfp_pack_precision_mode = bfp_pack_precision_mode,
                    .enable_32_bit_dest = enable_32_bit_dest,
                    .double_buffer_dest = double_buffer_dest,
                    .unpack_modes = std::move(unpack_modes)};
            },
            nb::arg("fpu_math_fidelity") = MathFidelity::HiFi4,
            nb::arg("sfpu_precision_mode") = tt::tt_metal::Precision::Precise,
            nb::arg("bfp_pack_precision_mode") = tt::tt_metal::Precision::Approximate,
            nb::arg("enable_32_bit_dest") = false,
            nb::arg("double_buffer_dest") = true,
            nb::arg("unpack_modes") = m2::ComputeUnpackModes{})
        .def_rw("fpu_math_fidelity", &m2::ComputeGen1Config::fpu_math_fidelity)
        .def_rw("sfpu_precision_mode", &m2::ComputeGen1Config::sfpu_precision_mode)
        .def_rw("bfp_pack_precision_mode", &m2::ComputeGen1Config::bfp_pack_precision_mode)
        .def_rw("enable_32_bit_dest", &m2::ComputeGen1Config::enable_32_bit_dest)
        .def_rw("double_buffer_dest", &m2::ComputeGen1Config::double_buffer_dest)
        .def_rw("unpack_modes", &m2::ComputeGen1Config::unpack_modes);

    nb::class_<m2::ComputeGen2Config>(mod, "ComputeGen2Config")
        .def(
            "__init__",
            [](m2::ComputeGen2Config* self,
               MathFidelity fpu_math_fidelity,
               tt::tt_metal::Precision sfpu_precision_mode,
               bool enable_32_bit_dest,
               bool double_buffer_dest,
               m2::ComputeUnpackModes unpack_modes,
               bool enable_2x_src_register) {
                new (self) m2::ComputeGen2Config{
                    .fpu_math_fidelity = fpu_math_fidelity,
                    .sfpu_precision_mode = sfpu_precision_mode,
                    .enable_32_bit_dest = enable_32_bit_dest,
                    .double_buffer_dest = double_buffer_dest,
                    .unpack_modes = std::move(unpack_modes),
                    .enable_2x_src_register = enable_2x_src_register};
            },
            nb::arg("fpu_math_fidelity") = MathFidelity::HiFi4,
            nb::arg("sfpu_precision_mode") = tt::tt_metal::Precision::Precise,
            nb::arg("enable_32_bit_dest") = false,
            nb::arg("double_buffer_dest") = true,
            nb::arg("unpack_modes") = m2::ComputeUnpackModes{},
            nb::arg("enable_2x_src_register") = false);

    mod.def("create_reader_dm_config", &m2::CreateReaderGen1DataMovementConfig);
    mod.def("create_writer_dm_config", &m2::CreateWriterGen1DataMovementConfig);

    // ---------------------------------------------------------------- kernels

    nb::class_<m2::KernelSpec::SourceCode>(mod, "SourceCode")
        .def(
            "__init__",
            [](m2::KernelSpec::SourceCode* self, std::string code) {
                new (self) m2::KernelSpec::SourceCode{.code = std::move(code)};
            },
            nb::arg("code"))
        .def_rw("code", &m2::KernelSpec::SourceCode::code);

    nb::class_<m2::KernelSpec::CompilerOptions>(mod, "CompilerOptions")
        .def(
            "__init__",
            [](m2::KernelSpec::CompilerOptions* self,
               std::vector<std::filesystem::path> include_paths,
               m2::KernelSpec::CompilerOptions::Defines defines,
               tt::tt_metal::KernelBuildOptLevel opt_level) {
                new (self) m2::KernelSpec::CompilerOptions{
                    .include_paths = std::move(include_paths), .defines = std::move(defines), .opt_level = opt_level};
            },
            nb::arg("include_paths") = std::vector<std::filesystem::path>{},
            nb::arg("defines") = m2::KernelSpec::CompilerOptions::Defines{},
            nb::arg("opt_level") = tt::tt_metal::KernelBuildOptLevel::O2)
        .def_rw("include_paths", &m2::KernelSpec::CompilerOptions::include_paths)
        .def_rw("defines", &m2::KernelSpec::CompilerOptions::defines)
        .def_rw("opt_level", &m2::KernelSpec::CompilerOptions::opt_level);

    nb::class_<m2::DFBBinding>(mod, "DFBBinding")
        .def(
            "__init__",
            [](m2::DFBBinding* self,
               m2::DFBSpecName dfb_spec_name,
               std::string accessor_name,
               m2::DFBEndpointType endpoint_type,
               m2::DFBAccessPattern access_pattern) {
                new (self) m2::DFBBinding{
                    .dfb_spec_name = std::move(dfb_spec_name),
                    .accessor_name = std::move(accessor_name),
                    .endpoint_type = endpoint_type,
                    .access_pattern = access_pattern};
            },
            nb::arg("dfb_spec_name"),
            nb::arg("accessor_name"),
            nb::arg("endpoint_type"),
            nb::arg("access_pattern") = m2::DFBAccessPattern::STRIDED)
        .def_rw("dfb_spec_name", &m2::DFBBinding::dfb_spec_name)
        .def_rw("accessor_name", &m2::DFBBinding::accessor_name)
        .def_rw("endpoint_type", &m2::DFBBinding::endpoint_type)
        .def_rw("access_pattern", &m2::DFBBinding::access_pattern);

    mod.def("producer_of", &m2::ProducerOf, nb::arg("dfb_spec_name"), nb::arg("accessor_name"));
    mod.def("consumer_of", &m2::ConsumerOf, nb::arg("dfb_spec_name"), nb::arg("accessor_name"));
    mod.def("strided_consumer_of", &m2::StridedConsumerOf, nb::arg("dfb_spec_name"), nb::arg("accessor_name"));
    mod.def("all_consumer_of", &m2::AllConsumerOf, nb::arg("dfb_spec_name"), nb::arg("accessor_name"));

    nb::class_<m2::SemaphoreBinding>(mod, "SemaphoreBinding")
        .def(
            "__init__",
            [](m2::SemaphoreBinding* self, m2::SemaphoreSpecName semaphore_spec_name, std::string accessor_name) {
                new (self) m2::SemaphoreBinding{
                    .semaphore_spec_name = std::move(semaphore_spec_name), .accessor_name = std::move(accessor_name)};
            },
            nb::arg("semaphore_spec_name"),
            nb::arg("accessor_name"))
        .def_rw("semaphore_spec_name", &m2::SemaphoreBinding::semaphore_spec_name)
        .def_rw("accessor_name", &m2::SemaphoreBinding::accessor_name);

    nb::class_<m2::ScratchpadBinding>(mod, "ScratchpadBinding")
        .def(
            "__init__",
            [](m2::ScratchpadBinding* self, m2::ScratchpadSpecName scratchpad_spec_name, std::string accessor_name) {
                new (self) m2::ScratchpadBinding{
                    .scratchpad_spec_name = std::move(scratchpad_spec_name), .accessor_name = std::move(accessor_name)};
            },
            nb::arg("scratchpad_spec_name"),
            nb::arg("accessor_name"))
        .def_rw("scratchpad_spec_name", &m2::ScratchpadBinding::scratchpad_spec_name)
        .def_rw("accessor_name", &m2::ScratchpadBinding::accessor_name);

    nb::class_<m2::TensorBinding>(mod, "TensorBinding")
        .def(
            "__init__",
            [](m2::TensorBinding* self, m2::TensorParamName tensor_parameter_name, std::string accessor_name) {
                new (self) m2::TensorBinding{
                    .tensor_parameter_name = std::move(tensor_parameter_name),
                    .accessor_name = std::move(accessor_name)};
            },
            nb::arg("tensor_parameter_name"),
            nb::arg("accessor_name"))
        .def_rw("tensor_parameter_name", &m2::TensorBinding::tensor_parameter_name)
        .def_rw("accessor_name", &m2::TensorBinding::accessor_name);

    nb::class_<m2::KernelSpec::RuntimeArgSchema>(mod, "RuntimeArgSchema")
        .def(
            "__init__",
            [](m2::KernelSpec::RuntimeArgSchema* self,
               std::vector<std::string> runtime_arg_names,
               std::vector<std::string> common_runtime_arg_names) {
                new (self) m2::KernelSpec::RuntimeArgSchema{
                    .runtime_arg_names = std::move(runtime_arg_names),
                    .common_runtime_arg_names = std::move(common_runtime_arg_names)};
            },
            nb::arg("runtime_arg_names") = std::vector<std::string>{},
            nb::arg("common_runtime_arg_names") = std::vector<std::string>{})
        .def_rw("runtime_arg_names", &m2::KernelSpec::RuntimeArgSchema::runtime_arg_names)
        .def_rw("common_runtime_arg_names", &m2::KernelSpec::RuntimeArgSchema::common_runtime_arg_names);

    nb::class_<m2::KernelAdvancedOptions>(mod, "KernelAdvancedOptions")
        .def(
            "__init__",
            [](m2::KernelAdvancedOptions* self, uint32_t num_runtime_varargs, uint32_t num_common_runtime_varargs) {
                new (self) m2::KernelAdvancedOptions{};
                self->num_runtime_varargs = num_runtime_varargs;
                self->num_common_runtime_varargs = num_common_runtime_varargs;
            },
            nb::arg("num_runtime_varargs") = 0,
            nb::arg("num_common_runtime_varargs") = 0)
        .def_rw("num_runtime_varargs", &m2::KernelAdvancedOptions::num_runtime_varargs)
        .def_rw("num_common_runtime_varargs", &m2::KernelAdvancedOptions::num_common_runtime_varargs);

    nb::class_<m2::KernelSpec>(mod, "KernelSpec")
        .def(
            "__init__",
            [](m2::KernelSpec* self,
               m2::KernelSpecName unique_id,
               std::variant<std::filesystem::path, m2::KernelSpec::SourceCode> source,
               std::variant<m2::DataMovementHardwareConfig, m2::ComputeHardwareConfig> hw_config,
               uint32_t num_threads,
               m2::KernelSpec::CompilerOptions compiler_options,
               std::vector<m2::DFBBinding> dfb_bindings,
               std::vector<m2::SemaphoreBinding> semaphore_bindings,
               std::vector<m2::ScratchpadBinding> scratchpad_bindings,
               std::vector<m2::TensorBinding> tensor_bindings,
               m2::KernelSpec::CompileTimeArgs compile_time_args,
               m2::KernelSpec::RuntimeArgSchema runtime_arg_schema,
               m2::KernelAdvancedOptions advanced_options) {
                new (self) m2::KernelSpec{
                    .unique_id = std::move(unique_id),
                    .source = std::move(source),
                    .num_threads = num_threads,
                    .compiler_options = std::move(compiler_options),
                    .dfb_bindings = std::move(dfb_bindings),
                    .semaphore_bindings = std::move(semaphore_bindings),
                    .scratchpad_bindings = std::move(scratchpad_bindings),
                    .tensor_bindings = std::move(tensor_bindings),
                    .compile_time_args = std::move(compile_time_args),
                    .runtime_arg_schema = std::move(runtime_arg_schema),
                    .hw_config = std::move(hw_config),
                    .advanced_options = std::move(advanced_options)};
            },
            nb::arg("unique_id"),
            nb::arg("source"),
            nb::arg("hw_config"),
            nb::arg("num_threads") = 1,
            nb::arg("compiler_options") = m2::KernelSpec::CompilerOptions{},
            nb::arg("dfb_bindings") = std::vector<m2::DFBBinding>{},
            nb::arg("semaphore_bindings") = std::vector<m2::SemaphoreBinding>{},
            nb::arg("scratchpad_bindings") = std::vector<m2::ScratchpadBinding>{},
            nb::arg("tensor_bindings") = std::vector<m2::TensorBinding>{},
            nb::arg("compile_time_args") = m2::KernelSpec::CompileTimeArgs{},
            nb::arg("runtime_arg_schema") = m2::KernelSpec::RuntimeArgSchema{},
            nb::arg("advanced_options") = m2::KernelAdvancedOptions{})
        .def_rw("unique_id", &m2::KernelSpec::unique_id)
        .def_rw("source", &m2::KernelSpec::source)
        .def_rw("num_threads", &m2::KernelSpec::num_threads)
        .def_rw("compiler_options", &m2::KernelSpec::compiler_options)
        .def_rw("dfb_bindings", &m2::KernelSpec::dfb_bindings)
        .def_rw("semaphore_bindings", &m2::KernelSpec::semaphore_bindings)
        .def_rw("scratchpad_bindings", &m2::KernelSpec::scratchpad_bindings)
        .def_rw("tensor_bindings", &m2::KernelSpec::tensor_bindings)
        .def_rw("compile_time_args", &m2::KernelSpec::compile_time_args)
        .def_rw("runtime_arg_schema", &m2::KernelSpec::runtime_arg_schema)
        .def_rw("hw_config", &m2::KernelSpec::hw_config)
        .def_rw("advanced_options", &m2::KernelSpec::advanced_options);

    // ---------------------------------------------------------------- program spec

    nb::class_<m2::WorkUnitSpec>(mod, "WorkUnitSpec")
        .def(
            "__init__",
            [](m2::WorkUnitSpec* self,
               std::string name,
               std::vector<m2::KernelSpecName> kernels,
               m2::Nodes target_nodes) {
                new (self) m2::WorkUnitSpec{
                    .name = std::move(name), .kernels = std::move(kernels), .target_nodes = std::move(target_nodes)};
            },
            nb::arg("name"),
            nb::arg("kernels"),
            nb::arg("target_nodes"))
        .def_rw("name", &m2::WorkUnitSpec::name)
        .def_rw("kernels", &m2::WorkUnitSpec::kernels)
        .def_rw("target_nodes", &m2::WorkUnitSpec::target_nodes);

    nb::class_<m2::ProgramSpec>(mod, "ProgramSpec")
        .def(
            "__init__",
            [](m2::ProgramSpec* self,
               std::string name,
               std::vector<m2::KernelSpec> kernels,
               std::vector<m2::DataflowBufferSpec> dataflow_buffers,
               std::vector<m2::SemaphoreSpec> semaphores,
               std::vector<m2::ScratchpadSpec> scratchpads,
               std::vector<m2::TensorParameter> tensor_parameters,
               std::vector<m2::WorkUnitSpec> work_units) {
                new (self) m2::ProgramSpec{
                    .name = std::move(name),
                    .kernels = std::move(kernels),
                    .dataflow_buffers = std::move(dataflow_buffers),
                    .cross_node_dataflow_buffers = {},
                    .semaphores = std::move(semaphores),
                    .scratchpads = std::move(scratchpads),
                    .tensor_parameters = std::move(tensor_parameters),
                    .work_units = std::move(work_units)};
            },
            nb::arg("name") = std::string{},
            nb::arg("kernels") = std::vector<m2::KernelSpec>{},
            nb::arg("dataflow_buffers") = std::vector<m2::DataflowBufferSpec>{},
            nb::arg("semaphores") = std::vector<m2::SemaphoreSpec>{},
            nb::arg("scratchpads") = std::vector<m2::ScratchpadSpec>{},
            nb::arg("tensor_parameters") = std::vector<m2::TensorParameter>{},
            nb::arg("work_units") = std::vector<m2::WorkUnitSpec>{})
        .def_rw("name", &m2::ProgramSpec::name)
        .def_rw("kernels", &m2::ProgramSpec::kernels)
        .def_rw("dataflow_buffers", &m2::ProgramSpec::dataflow_buffers)
        .def_rw("semaphores", &m2::ProgramSpec::semaphores)
        .def_rw("scratchpads", &m2::ProgramSpec::scratchpads)
        .def_rw("tensor_parameters", &m2::ProgramSpec::tensor_parameters)
        .def_rw("work_units", &m2::ProgramSpec::work_units);

    // ---------------------------------------------------------------- run args

    nb::class_<m2::AdvancedKernelRunArgs>(mod, "AdvancedKernelRunArgs")
        .def(
            "__init__",
            [](m2::AdvancedKernelRunArgs* self,
               m2::Table<m2::NodeCoord, m2::AdvancedKernelRunArgs::Varargs> runtime_varargs,
               m2::AdvancedKernelRunArgs::Varargs common_runtime_varargs) {
                new (self) m2::AdvancedKernelRunArgs{
                    .runtime_varargs = std::move(runtime_varargs),
                    .common_runtime_varargs = std::move(common_runtime_varargs)};
            },
            nb::arg("runtime_varargs") = m2::Table<m2::NodeCoord, m2::AdvancedKernelRunArgs::Varargs>{},
            nb::arg("common_runtime_varargs") = m2::AdvancedKernelRunArgs::Varargs{})
        .def_rw("runtime_varargs", &m2::AdvancedKernelRunArgs::runtime_varargs)
        .def_rw("common_runtime_varargs", &m2::AdvancedKernelRunArgs::common_runtime_varargs);

    nb::class_<m2::KernelRunArgs>(mod, "KernelRunArgs")
        .def(
            "__init__",
            [](m2::KernelRunArgs* self,
               m2::KernelSpecName kernel,
               m2::KernelRunArgs::RuntimeArgValues runtime_arg_values,
               m2::KernelRunArgs::CommonRuntimeArgValues common_runtime_arg_values,
               m2::AdvancedKernelRunArgs advanced_options) {
                new (self) m2::KernelRunArgs{
                    .kernel = std::move(kernel),
                    .runtime_arg_values = std::move(runtime_arg_values),
                    .common_runtime_arg_values = std::move(common_runtime_arg_values),
                    .advanced_options = std::move(advanced_options)};
            },
            nb::arg("kernel"),
            nb::arg("runtime_arg_values") = m2::KernelRunArgs::RuntimeArgValues{},
            nb::arg("common_runtime_arg_values") = m2::KernelRunArgs::CommonRuntimeArgValues{},
            nb::arg("advanced_options") = m2::AdvancedKernelRunArgs{})
        .def_rw("kernel", &m2::KernelRunArgs::kernel)
        .def_rw("runtime_arg_values", &m2::KernelRunArgs::runtime_arg_values)
        .def_rw("common_runtime_arg_values", &m2::KernelRunArgs::common_runtime_arg_values)
        .def_rw("advanced_options", &m2::KernelRunArgs::advanced_options);

    nb::class_<m2::DFBRunOverrides>(mod, "DFBRunOverrides")
        .def(
            "__init__",
            [](m2::DFBRunOverrides* self,
               m2::DFBSpecName dfb,
               std::optional<uint32_t> entry_size,
               std::optional<uint32_t> num_entries) {
                new (self)
                    m2::DFBRunOverrides{.dfb = std::move(dfb), .entry_size = entry_size, .num_entries = num_entries};
            },
            nb::arg("dfb"),
            nb::arg("entry_size") = nb::none(),
            nb::arg("num_entries") = nb::none())
        .def_rw("dfb", &m2::DFBRunOverrides::dfb)
        .def_rw("entry_size", &m2::DFBRunOverrides::entry_size)
        .def_rw("num_entries", &m2::DFBRunOverrides::num_entries);

    // tensor_args is deliberately absent: TensorArgument holds a non-owning MeshTensor
    // reference that must alias one of the op's io tensors by pointer identity. generic_op
    // takes a {tensor parameter name -> io tensor index} map instead and builds the table itself.
    nb::class_<m2::ProgramRunArgs>(mod, "ProgramRunArgs")
        .def(
            "__init__",
            [](m2::ProgramRunArgs* self,
               std::vector<m2::KernelRunArgs> kernel_run_args,
               std::vector<m2::DFBRunOverrides> dfb_run_overrides) {
                new (self) m2::ProgramRunArgs{
                    .kernel_run_args = std::move(kernel_run_args),
                    .tensor_args = {},
                    .dfb_run_overrides = std::move(dfb_run_overrides)};
            },
            nb::arg("kernel_run_args") = std::vector<m2::KernelRunArgs>{},
            nb::arg("dfb_run_overrides") = std::vector<m2::DFBRunOverrides>{})
        .def_rw("kernel_run_args", &m2::ProgramRunArgs::kernel_run_args)
        .def_rw("dfb_run_overrides", &m2::ProgramRunArgs::dfb_run_overrides);
}

}  // namespace ttnn::program_specs
