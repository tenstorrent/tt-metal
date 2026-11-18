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

#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::program_specs {

namespace m2 = tt::tt_metal::experimental;

void py_module_types(nb::module_& mod) {
    export_enum<tt::tt_metal::UnpackMode>(mod, "UnpackMode");
    export_enum<tt::tt_metal::Precision>(mod, "Precision");
    export_enum<tt::tt_metal::DFBEndpointType>(mod, "DFBEndpointType");
    export_enum<tt::tt_metal::DFBAccessPattern>(mod, "DFBAccessPattern");

    // ---------------------------------------------------------------- tensor parameters

    nb::class_<tt::tt_metal::TensorSpecRelaxations>(mod, "TensorSpecRelaxations")
        .def(
            "__init__",
            [](tt::tt_metal::TensorSpecRelaxations* self, bool match_padded_shape_only, bool dynamic_tensor_shape) {
                new (self) tt::tt_metal::TensorSpecRelaxations{
                    .match_padded_shape_only = match_padded_shape_only, .dynamic_tensor_shape = dynamic_tensor_shape};
            },
            nb::arg("match_padded_shape_only") = false,
            nb::arg("dynamic_tensor_shape") = false)
        .def_rw("match_padded_shape_only", &tt::tt_metal::TensorSpecRelaxations::match_padded_shape_only)
        .def_rw("dynamic_tensor_shape", &tt::tt_metal::TensorSpecRelaxations::dynamic_tensor_shape);

    nb::class_<tt::tt_metal::TensorParameter>(mod, "TensorParameter")
        .def(
            "__init__",
            [](tt::tt_metal::TensorParameter* self,
               tt::tt_metal::TensorParamName unique_id,
               const tt::tt_metal::TensorSpec& spec,
               tt::tt_metal::TensorSpecRelaxations relaxations) {
                new (self)
                    tt::tt_metal::TensorParameter{.unique_id = std::move(unique_id), .spec = spec, .relaxations = relaxations};
            },
            nb::arg("unique_id"),
            nb::arg("spec"),
            nb::arg("relaxations") = tt::tt_metal::TensorSpecRelaxations{})
        .def_rw("unique_id", &tt::tt_metal::TensorParameter::unique_id)
        .def_rw("spec", &tt::tt_metal::TensorParameter::spec)
        .def_rw("relaxations", &tt::tt_metal::TensorParameter::relaxations);

    // ---------------------------------------------------------------- dataflow buffers

    nb::class_<m2::DFBAdvancedOptions>(mod, "DFBAdvancedOptions")
        .def(
            "__init__",
            [](m2::DFBAdvancedOptions* self,
               std::vector<tt::tt_metal::DFBSpecName> alias_with,
               bool allow_instance_multi_binding) {
                new (self) m2::DFBAdvancedOptions{
                    .alias_with = std::move(alias_with), .allow_instance_multi_binding = allow_instance_multi_binding};
            },
            nb::arg("alias_with") = std::vector<tt::tt_metal::DFBSpecName>{},
            nb::arg("allow_instance_multi_binding") = false)
        .def_rw("alias_with", &m2::DFBAdvancedOptions::alias_with)
        .def_rw("allow_instance_multi_binding", &m2::DFBAdvancedOptions::allow_instance_multi_binding);

    nb::class_<tt::tt_metal::DataflowBufferSpec>(mod, "DataflowBufferSpec")
        .def(
            // data_format is taken as a ttnn.DataType (as CBFormatDescriptor does) and converted.
            "__init__",
            [](tt::tt_metal::DataflowBufferSpec* self,
               tt::tt_metal::DFBSpecName unique_id,
               uint32_t entry_size,
               uint32_t num_entries,
               std::optional<ttnn::DataType> data_format,
               std::optional<tt::tt_metal::Tile> tile_format,
               std::optional<tt::tt_metal::TensorParamName> borrowed_from,
               m2::DFBAdvancedOptions advanced_options) {
                std::optional<tt::DataFormat> df;
                if (data_format.has_value()) {
                    df = tt::tt_metal::datatype_to_dataformat_converter(*data_format);
                }
                new (self) tt::tt_metal::DataflowBufferSpec{
                    .unique_id = std::move(unique_id),
                    .entry_size = entry_size,
                    .num_entries = num_entries,
                    .data_format_metadata = df,
                    .tile_format_metadata = tile_format,
                    .borrowed_from = std::move(borrowed_from),
                    .advanced_options = std::move(advanced_options)};
            },
            nb::arg("unique_id"),
            nb::arg("entry_size"),
            nb::arg("num_entries"),
            nb::arg("data_format") = nb::none(),
            nb::arg("tile_format") = nb::none(),
            nb::arg("borrowed_from") = nb::none(),
            nb::arg("advanced_options") = m2::DFBAdvancedOptions{})
        .def_rw("unique_id", &tt::tt_metal::DataflowBufferSpec::unique_id)
        .def_rw("entry_size", &tt::tt_metal::DataflowBufferSpec::entry_size)
        .def_rw("num_entries", &tt::tt_metal::DataflowBufferSpec::num_entries)
        // tt::DataFormat has no Python binding; surface it as its integer value.
        .def_prop_ro(
            "data_format_metadata",
            [](const tt::tt_metal::DataflowBufferSpec& self) -> std::optional<uint32_t> {
                if (!self.data_format_metadata.has_value()) {
                    return std::nullopt;
                }
                return static_cast<uint32_t>(*self.data_format_metadata);
            })
        .def_rw("tile_format_metadata", &tt::tt_metal::DataflowBufferSpec::tile_format_metadata)
        .def_rw("borrowed_from", &tt::tt_metal::DataflowBufferSpec::borrowed_from)
        .def_rw("advanced_options", &tt::tt_metal::DataflowBufferSpec::advanced_options);

    // Deriving constructor: build a DataflowBufferSpec from a sharded tensor instead of hand-typing
    // its entry-format fields. The Metal 2.0 counterpart of ttnn.cb_descriptor_from_sharded_tensor.
    mod.def(
        "dfb_spec_from_sharded_tensor",
        &ttnn::dfb_spec_from_sharded_tensor,
        nb::arg("unique_id"),
        nb::arg("tensor"),
        nb::arg("num_entries") = 0,
        nb::arg("borrowed_from") = nb::none(),
        nb::arg("page_as_tile") = false,
        nb::arg("tile_format") = nb::none(),
        nb::arg("advanced_options") = m2::DFBAdvancedOptions{},
        R"pbdoc(
            Create a DataflowBufferSpec from a sharded tensor.

            Derives every entry-format field from the tensor rather than making you retype what the
            tensor already knows: entry_size (the shard's page stride in L1), num_entries (pages per
            shard), data_format_metadata (from the dtype) and tile_format_metadata (the tensor's tile).
            The Metal 2.0 counterpart of ttnn.cb_descriptor_from_sharded_tensor.

            Args:
                unique_id: The DFB's name within the ProgramSpec.
                tensor: An allocated, sharded tensor to derive the entry format from.
                num_entries: DFB depth in entries. 0 (default) means one entry per page of the
                    shard, i.e. the DFB spans the whole resident shard. Pass a value for a shallower
                    DFB (e.g. a double-buffered staging DFB that only reuses the tensor's format).
                borrowed_from: Name of the TensorParameter whose L1 memory backs this DFB, or None
                    (default) for a normally allocated DFB. A tensor does not know the
                    TensorParameter name you gave it, so this cannot be defaulted for you.
                page_as_tile: Treat entries as tiles even for a ROW_MAJOR tensor - one entry becomes
                    one tile (or the whole shard, when the shard is smaller than a tile) instead of
                    one stick, and tile_format_metadata is set. No-op for a TILE tensor.
                tile_format: Replaces the derived tile_format_metadata. The tile carries the face
                    layout, so pass one when an entry holds fewer or shorter faces than the tensor's
                    tile - which is exactly the sub-tile shard case page_as_tile falls back to. Never
                    derived for you.
                advanced_options: Passed through to advanced_options.

            Returns:
                DataflowBufferSpec

            Raises:
                RuntimeError: if the tensor is not sharded, is not allocated, is asked to back a DFB
                    from non-L1 memory, or the resulting DFB does not fit in the borrowed shard.

            Example:
                >>> # Zero-copy: the DFB *is* the resident shard.
                >>> dfb = ttnn.dfb_spec_from_sharded_tensor("in", sharded_input, borrowed_from="a")
                >>>
                >>> # Row-major shard read as tiles by a compute kernel.
                >>> dfb = ttnn.dfb_spec_from_sharded_tensor(
                ...     "in", rm_sharded_input, borrowed_from="a", page_as_tile=True
                ... )
                >>>
                >>> # Separately allocated double-buffered staging DFB, tensor's format only.
                >>> dfb = ttnn.dfb_spec_from_sharded_tensor("stage", sharded_input, num_entries=2)
        )pbdoc");

    // ---------------------------------------------------------------- semaphores, scratchpads

    nb::class_<tt::tt_metal::SemaphoreSpec>(mod, "SemaphoreSpec")
        .def(
            "__init__",
            [](tt::tt_metal::SemaphoreSpec* self, tt::tt_metal::SemaphoreSpecName unique_id, tt::tt_metal::Nodes target_nodes) {
                new (self)
                    tt::tt_metal::SemaphoreSpec{.unique_id = std::move(unique_id), .target_nodes = std::move(target_nodes)};
            },
            nb::arg("unique_id"),
            nb::arg("target_nodes"))
        .def_rw("unique_id", &tt::tt_metal::SemaphoreSpec::unique_id)
        .def_rw("target_nodes", &tt::tt_metal::SemaphoreSpec::target_nodes);

    nb::class_<tt::tt_metal::ScratchpadSpec>(mod, "ScratchpadSpec")
        .def(
            "__init__",
            [](tt::tt_metal::ScratchpadSpec* self, tt::tt_metal::ScratchpadSpecName unique_id, uint32_t size_per_node) {
                new (self) tt::tt_metal::ScratchpadSpec{.unique_id = std::move(unique_id), .size_per_node = size_per_node};
            },
            nb::arg("unique_id"),
            nb::arg("size_per_node"))
        .def_rw("unique_id", &tt::tt_metal::ScratchpadSpec::unique_id)
        .def_rw("size_per_node", &tt::tt_metal::ScratchpadSpec::size_per_node);

    // ---------------------------------------------------------------- hardware configs

    using DM1XX = tt::tt_metal::DataMovementHardwareConfig::DataMovement1XXConfig;
    using DM2XX = tt::tt_metal::DataMovementHardwareConfig::DataMovement2XXConfig;
    using Compute1XX = tt::tt_metal::ComputeHardwareConfig::Compute1XXConfig;
    using Compute2XX = tt::tt_metal::ComputeHardwareConfig::Compute2XXConfig;
    using ComputeUnpackModes = tt::tt_metal::ComputeHardwareConfig::ComputeUnpackModes;

    nb::class_<DM1XX>(mod, "DataMovement1XXConfig")
        .def(
            "__init__",
            [](DM1XX* self,
               tt::tt_metal::DataMovementProcessor processor,
               tt::tt_metal::NOC noc,
               tt::tt_metal::NOC_MODE noc_mode) {
                new (self) DM1XX{.processor = processor, .noc = noc, .noc_mode = noc_mode};
            },
            nb::arg("processor"),
            nb::arg("noc"),
            nb::arg("noc_mode") = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC)
        .def_rw("processor", &DM1XX::processor)
        .def_rw("noc", &DM1XX::noc)
        .def_rw("noc_mode", &DM1XX::noc_mode);

    nb::class_<DM2XX>(mod, "DataMovement2XXConfig")
        .def(
            "__init__",
            [](DM2XX* self,
               std::vector<tt::tt_metal::DFBSpecName> disable_dfb_implicit_sync_for,
               bool disable_dfb_implicit_sync_for_all) {
                new (self) DM2XX{
                    .disable_dfb_implicit_sync_for = std::move(disable_dfb_implicit_sync_for),
                    .disable_dfb_implicit_sync_for_all = disable_dfb_implicit_sync_for_all};
            },
            nb::arg("disable_dfb_implicit_sync_for") = std::vector<tt::tt_metal::DFBSpecName>{},
            nb::arg("disable_dfb_implicit_sync_for_all") = false)
        .def_rw("disable_dfb_implicit_sync_for", &DM2XX::disable_dfb_implicit_sync_for)
        .def_rw("disable_dfb_implicit_sync_for_all", &DM2XX::disable_dfb_implicit_sync_for_all);

    nb::class_<tt::tt_metal::DataMovementHardwareConfig>(mod, "DataMovementHardwareConfig")
        .def(
            "__init__",
            [](tt::tt_metal::DataMovementHardwareConfig* self, std::optional<DM1XX> config_1xx, std::optional<DM2XX> config_2xx) {
                new (self)
                    tt::tt_metal::DataMovementHardwareConfig{.config_1xx = config_1xx, .config_2xx = std::move(config_2xx)};
            },
            nb::arg("config_1xx") = nb::none(),
            nb::arg("config_2xx") = nb::none())
        .def_rw("config_1xx", &tt::tt_metal::DataMovementHardwareConfig::config_1xx)
        .def_rw("config_2xx", &tt::tt_metal::DataMovementHardwareConfig::config_2xx);

    nb::class_<Compute1XX>(mod, "Compute1XXConfig")
        .def(
            "__init__",
            [](Compute1XX* self, tt::tt_metal::Precision bfp_pack_precision_mode) {
                new (self) Compute1XX{.bfp_pack_precision_mode = bfp_pack_precision_mode};
            },
            nb::arg("bfp_pack_precision_mode") = tt::tt_metal::Precision::Approximate)
        .def_rw("bfp_pack_precision_mode", &Compute1XX::bfp_pack_precision_mode);

    nb::class_<Compute2XX>(mod, "Compute2XXConfig").def(nb::init<>());

    nb::class_<tt::tt_metal::ComputeHardwareConfig>(mod, "ComputeHardwareConfig")
        .def(
            "__init__",
            [](tt::tt_metal::ComputeHardwareConfig* self,
               tt::tt_metal::MathFidelity fpu_math_fidelity,
               tt::tt_metal::Precision sfpu_precision_mode,
               bool enable_32_bit_dest,
               bool double_buffer_dest,
               ComputeUnpackModes unpack_modes,
               std::optional<Compute1XX> config_1xx,
               std::optional<Compute2XX> config_2xx) {
                new (self) tt::tt_metal::ComputeHardwareConfig{
                    .fpu_math_fidelity = fpu_math_fidelity,
                    .sfpu_precision_mode = sfpu_precision_mode,
                    .enable_32_bit_dest = enable_32_bit_dest,
                    .double_buffer_dest = double_buffer_dest,
                    .unpack_modes = std::move(unpack_modes),
                    .config_1xx = config_1xx,
                    .config_2xx = config_2xx};
            },
            nb::arg("fpu_math_fidelity") = tt::tt_metal::MathFidelity::HiFi4,
            nb::arg("sfpu_precision_mode") = tt::tt_metal::Precision::Precise,
            nb::arg("enable_32_bit_dest") = false,
            nb::arg("double_buffer_dest") = true,
            nb::arg("unpack_modes") = ComputeUnpackModes{},
            nb::arg("config_1xx") = nb::none(),
            nb::arg("config_2xx") = nb::none())
        .def_rw("fpu_math_fidelity", &tt::tt_metal::ComputeHardwareConfig::fpu_math_fidelity)
        .def_rw("sfpu_precision_mode", &tt::tt_metal::ComputeHardwareConfig::sfpu_precision_mode)
        .def_rw("enable_32_bit_dest", &tt::tt_metal::ComputeHardwareConfig::enable_32_bit_dest)
        .def_rw("double_buffer_dest", &tt::tt_metal::ComputeHardwareConfig::double_buffer_dest)
        .def_rw("unpack_modes", &tt::tt_metal::ComputeHardwareConfig::unpack_modes)
        .def_rw("config_1xx", &tt::tt_metal::ComputeHardwareConfig::config_1xx)
        .def_rw("config_2xx", &tt::tt_metal::ComputeHardwareConfig::config_2xx);

    mod.def("create_reader_dm_config", &m2::CreateReaderDataMovementConfig);
    mod.def("create_writer_dm_config", &m2::CreateWriterDataMovementConfig);

    // ---------------------------------------------------------------- kernels

    nb::class_<tt::tt_metal::KernelSpec::SourceCode>(mod, "SourceCode")
        .def(
            "__init__",
            [](tt::tt_metal::KernelSpec::SourceCode* self, std::string code) {
                new (self) tt::tt_metal::KernelSpec::SourceCode{.code = std::move(code)};
            },
            nb::arg("code"))
        .def_rw("code", &tt::tt_metal::KernelSpec::SourceCode::code);

    nb::class_<tt::tt_metal::KernelSpec::CompilerOptions>(mod, "CompilerOptions")
        .def(
            "__init__",
            [](tt::tt_metal::KernelSpec::CompilerOptions* self,
               std::vector<std::filesystem::path> include_paths,
               tt::tt_metal::KernelSpec::CompilerOptions::Defines defines,
               std::optional<tt::tt_metal::KernelBuildOptLevel> opt_level) {
                new (self) tt::tt_metal::KernelSpec::CompilerOptions{
                    .include_paths = std::move(include_paths),
                    .defines = std::move(defines),
                    .opt_level = opt_level.value_or(tt::tt_metal::KernelBuildOptLevel::O2)};
            },
            nb::arg("include_paths") = std::vector<std::filesystem::path>{},
            nb::arg("defines") = tt::tt_metal::KernelSpec::CompilerOptions::Defines{},
            // KernelBuildOptLevel is registered by ttnn.experimental.fabric_mux, which binds after this module.
            nb::arg("opt_level") = nb::none())
        .def_rw("include_paths", &tt::tt_metal::KernelSpec::CompilerOptions::include_paths)
        .def_rw("defines", &tt::tt_metal::KernelSpec::CompilerOptions::defines)
        .def_rw("opt_level", &tt::tt_metal::KernelSpec::CompilerOptions::opt_level);

    nb::class_<tt::tt_metal::DFBBinding>(mod, "DFBBinding")
        .def(
            "__init__",
            [](tt::tt_metal::DFBBinding* self,
               tt::tt_metal::DFBSpecName dfb_spec_name,
               std::string accessor_name,
               tt::tt_metal::DFBEndpointType endpoint_type,
               tt::tt_metal::DFBAccessPattern access_pattern) {
                new (self) tt::tt_metal::DFBBinding{
                    .dfb_spec_name = std::move(dfb_spec_name),
                    .accessor_name = std::move(accessor_name),
                    .endpoint_type = endpoint_type,
                    .access_pattern = access_pattern};
            },
            nb::arg("dfb_spec_name"),
            nb::arg("accessor_name"),
            nb::arg("endpoint_type"),
            nb::arg("access_pattern") = tt::tt_metal::DFBAccessPattern::STRIDED)
        .def_rw("dfb_spec_name", &tt::tt_metal::DFBBinding::dfb_spec_name)
        .def_rw("accessor_name", &tt::tt_metal::DFBBinding::accessor_name)
        .def_rw("endpoint_type", &tt::tt_metal::DFBBinding::endpoint_type)
        .def_rw("access_pattern", &tt::tt_metal::DFBBinding::access_pattern);

    mod.def("producer_of", &m2::ProducerOf, nb::arg("dfb_spec_name"), nb::arg("accessor_name"));
    mod.def("consumer_of", &m2::ConsumerOf, nb::arg("dfb_spec_name"), nb::arg("accessor_name"));
    mod.def("strided_consumer_of", &m2::StridedConsumerOf, nb::arg("dfb_spec_name"), nb::arg("accessor_name"));
    mod.def("all_consumer_of", &m2::AllConsumerOf, nb::arg("dfb_spec_name"), nb::arg("accessor_name"));

    nb::class_<tt::tt_metal::SemaphoreBinding>(mod, "SemaphoreBinding")
        .def(
            "__init__",
            [](tt::tt_metal::SemaphoreBinding* self, tt::tt_metal::SemaphoreSpecName semaphore_spec_name, std::string accessor_name) {
                new (self) tt::tt_metal::SemaphoreBinding{
                    .semaphore_spec_name = std::move(semaphore_spec_name), .accessor_name = std::move(accessor_name)};
            },
            nb::arg("semaphore_spec_name"),
            nb::arg("accessor_name"))
        .def_rw("semaphore_spec_name", &tt::tt_metal::SemaphoreBinding::semaphore_spec_name)
        .def_rw("accessor_name", &tt::tt_metal::SemaphoreBinding::accessor_name);

    nb::class_<tt::tt_metal::ScratchpadBinding>(mod, "ScratchpadBinding")
        .def(
            "__init__",
            [](tt::tt_metal::ScratchpadBinding* self, tt::tt_metal::ScratchpadSpecName scratchpad_spec_name, std::string accessor_name) {
                new (self) tt::tt_metal::ScratchpadBinding{
                    .scratchpad_spec_name = std::move(scratchpad_spec_name), .accessor_name = std::move(accessor_name)};
            },
            nb::arg("scratchpad_spec_name"),
            nb::arg("accessor_name"))
        .def_rw("scratchpad_spec_name", &tt::tt_metal::ScratchpadBinding::scratchpad_spec_name)
        .def_rw("accessor_name", &tt::tt_metal::ScratchpadBinding::accessor_name);

    nb::class_<tt::tt_metal::TensorBinding>(mod, "TensorBinding")
        .def(
            "__init__",
            [](tt::tt_metal::TensorBinding* self, tt::tt_metal::TensorParamName tensor_parameter_name, std::string accessor_name) {
                new (self) tt::tt_metal::TensorBinding{
                    .tensor_parameter_name = std::move(tensor_parameter_name),
                    .accessor_name = std::move(accessor_name)};
            },
            nb::arg("tensor_parameter_name"),
            nb::arg("accessor_name"))
        .def_rw("tensor_parameter_name", &tt::tt_metal::TensorBinding::tensor_parameter_name)
        .def_rw("accessor_name", &tt::tt_metal::TensorBinding::accessor_name);

    nb::class_<tt::tt_metal::KernelSpec::RuntimeArgSchema>(mod, "RuntimeArgSchema")
        .def(
            "__init__",
            [](tt::tt_metal::KernelSpec::RuntimeArgSchema* self,
               std::vector<std::string> runtime_arg_names,
               std::vector<std::string> common_runtime_arg_names) {
                new (self) tt::tt_metal::KernelSpec::RuntimeArgSchema{
                    .runtime_arg_names = std::move(runtime_arg_names),
                    .common_runtime_arg_names = std::move(common_runtime_arg_names)};
            },
            nb::arg("runtime_arg_names") = std::vector<std::string>{},
            nb::arg("common_runtime_arg_names") = std::vector<std::string>{})
        .def_rw("runtime_arg_names", &tt::tt_metal::KernelSpec::RuntimeArgSchema::runtime_arg_names)
        .def_rw("common_runtime_arg_names", &tt::tt_metal::KernelSpec::RuntimeArgSchema::common_runtime_arg_names);

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

    nb::class_<tt::tt_metal::KernelSpec>(mod, "KernelSpec")
        .def(
            "__init__",
            [](tt::tt_metal::KernelSpec* self,
               tt::tt_metal::KernelSpecName unique_id,
               std::variant<std::filesystem::path, tt::tt_metal::KernelSpec::SourceCode> source,
               std::variant<tt::tt_metal::DataMovementHardwareConfig, tt::tt_metal::ComputeHardwareConfig> hw_config,
               uint32_t num_threads,
               tt::tt_metal::KernelSpec::CompilerOptions compiler_options,
               std::vector<tt::tt_metal::DFBBinding> dfb_bindings,
               std::vector<tt::tt_metal::SemaphoreBinding> semaphore_bindings,
               std::vector<tt::tt_metal::ScratchpadBinding> scratchpad_bindings,
               std::vector<tt::tt_metal::TensorBinding> tensor_bindings,
               tt::tt_metal::KernelSpec::CompileTimeArgs compile_time_args,
               tt::tt_metal::KernelSpec::RuntimeArgSchema runtime_arg_schema,
               m2::KernelAdvancedOptions advanced_options) {
                new (self) tt::tt_metal::KernelSpec{
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
            nb::arg("compiler_options") = tt::tt_metal::KernelSpec::CompilerOptions{},
            nb::arg("dfb_bindings") = std::vector<tt::tt_metal::DFBBinding>{},
            nb::arg("semaphore_bindings") = std::vector<tt::tt_metal::SemaphoreBinding>{},
            nb::arg("scratchpad_bindings") = std::vector<tt::tt_metal::ScratchpadBinding>{},
            nb::arg("tensor_bindings") = std::vector<tt::tt_metal::TensorBinding>{},
            nb::arg("compile_time_args") = tt::tt_metal::KernelSpec::CompileTimeArgs{},
            nb::arg("runtime_arg_schema") = tt::tt_metal::KernelSpec::RuntimeArgSchema{},
            nb::arg("advanced_options") = m2::KernelAdvancedOptions{})
        .def_rw("unique_id", &tt::tt_metal::KernelSpec::unique_id)
        .def_rw("source", &tt::tt_metal::KernelSpec::source)
        .def_rw("num_threads", &tt::tt_metal::KernelSpec::num_threads)
        .def_rw("compiler_options", &tt::tt_metal::KernelSpec::compiler_options)
        .def_rw("dfb_bindings", &tt::tt_metal::KernelSpec::dfb_bindings)
        .def_rw("semaphore_bindings", &tt::tt_metal::KernelSpec::semaphore_bindings)
        .def_rw("scratchpad_bindings", &tt::tt_metal::KernelSpec::scratchpad_bindings)
        .def_rw("tensor_bindings", &tt::tt_metal::KernelSpec::tensor_bindings)
        .def_rw("compile_time_args", &tt::tt_metal::KernelSpec::compile_time_args)
        .def_rw("runtime_arg_schema", &tt::tt_metal::KernelSpec::runtime_arg_schema)
        .def_rw("hw_config", &tt::tt_metal::KernelSpec::hw_config)
        .def_rw("advanced_options", &tt::tt_metal::KernelSpec::advanced_options);

    // ---------------------------------------------------------------- program spec

    nb::class_<tt::tt_metal::WorkUnitSpec>(mod, "WorkUnitSpec")
        .def(
            "__init__",
            [](tt::tt_metal::WorkUnitSpec* self,
               std::string name,
               std::vector<tt::tt_metal::KernelSpecName> kernels,
               tt::tt_metal::Nodes target_nodes) {
                new (self) tt::tt_metal::WorkUnitSpec{
                    .name = std::move(name), .kernels = std::move(kernels), .target_nodes = std::move(target_nodes)};
            },
            nb::arg("name"),
            nb::arg("kernels"),
            nb::arg("target_nodes"))
        .def_rw("name", &tt::tt_metal::WorkUnitSpec::name)
        .def_rw("kernels", &tt::tt_metal::WorkUnitSpec::kernels)
        .def_rw("target_nodes", &tt::tt_metal::WorkUnitSpec::target_nodes);

    nb::class_<tt::tt_metal::ProgramSpec>(mod, "ProgramSpec")
        .def(
            "__init__",
            [](tt::tt_metal::ProgramSpec* self,
               std::string name,
               std::vector<tt::tt_metal::KernelSpec> kernels,
               std::vector<tt::tt_metal::DataflowBufferSpec> dataflow_buffers,
               std::vector<tt::tt_metal::SemaphoreSpec> semaphores,
               std::vector<tt::tt_metal::ScratchpadSpec> scratchpads,
               std::vector<tt::tt_metal::TensorParameter> tensor_parameters,
               std::vector<tt::tt_metal::WorkUnitSpec> work_units) {
                new (self) tt::tt_metal::ProgramSpec{
                    .name = std::move(name),
                    .kernels = std::move(kernels),
                    .dataflow_buffers = std::move(dataflow_buffers),
                    .semaphores = std::move(semaphores),
                    .scratchpads = std::move(scratchpads),
                    .tensor_parameters = std::move(tensor_parameters),
                    .work_units = std::move(work_units)};
            },
            nb::arg("name") = std::string{},
            nb::arg("kernels") = std::vector<tt::tt_metal::KernelSpec>{},
            nb::arg("dataflow_buffers") = std::vector<tt::tt_metal::DataflowBufferSpec>{},
            nb::arg("semaphores") = std::vector<tt::tt_metal::SemaphoreSpec>{},
            nb::arg("scratchpads") = std::vector<tt::tt_metal::ScratchpadSpec>{},
            nb::arg("tensor_parameters") = std::vector<tt::tt_metal::TensorParameter>{},
            nb::arg("work_units") = std::vector<tt::tt_metal::WorkUnitSpec>{})
        .def_rw("name", &tt::tt_metal::ProgramSpec::name)
        .def_rw("kernels", &tt::tt_metal::ProgramSpec::kernels)
        .def_rw("dataflow_buffers", &tt::tt_metal::ProgramSpec::dataflow_buffers)
        .def_rw("semaphores", &tt::tt_metal::ProgramSpec::semaphores)
        .def_rw("scratchpads", &tt::tt_metal::ProgramSpec::scratchpads)
        .def_rw("tensor_parameters", &tt::tt_metal::ProgramSpec::tensor_parameters)
        .def_rw("work_units", &tt::tt_metal::ProgramSpec::work_units);

    // ---------------------------------------------------------------- run args

    nb::class_<m2::AdvancedKernelRunArgs>(mod, "AdvancedKernelRunArgs")
        .def(
            "__init__",
            [](m2::AdvancedKernelRunArgs* self,
               tt::tt_metal::Table<tt::tt_metal::NodeCoord, m2::AdvancedKernelRunArgs::Varargs> runtime_varargs,
               m2::AdvancedKernelRunArgs::Varargs common_runtime_varargs) {
                new (self) m2::AdvancedKernelRunArgs{
                    .runtime_varargs = std::move(runtime_varargs),
                    .common_runtime_varargs = std::move(common_runtime_varargs)};
            },
            nb::arg("runtime_varargs") = tt::tt_metal::Table<tt::tt_metal::NodeCoord, m2::AdvancedKernelRunArgs::Varargs>{},
            nb::arg("common_runtime_varargs") = m2::AdvancedKernelRunArgs::Varargs{})
        .def_rw("runtime_varargs", &m2::AdvancedKernelRunArgs::runtime_varargs)
        .def_rw("common_runtime_varargs", &m2::AdvancedKernelRunArgs::common_runtime_varargs);

    nb::class_<tt::tt_metal::KernelRunArgs>(mod, "KernelRunArgs")
        .def(
            "__init__",
            [](tt::tt_metal::KernelRunArgs* self,
               tt::tt_metal::KernelSpecName kernel,
               tt::tt_metal::KernelRunArgs::RuntimeArgValues runtime_arg_values,
               tt::tt_metal::KernelRunArgs::CommonRuntimeArgValues common_runtime_arg_values,
               m2::AdvancedKernelRunArgs advanced_options) {
                new (self) tt::tt_metal::KernelRunArgs{
                    .kernel = std::move(kernel),
                    .runtime_arg_values = std::move(runtime_arg_values),
                    .common_runtime_arg_values = std::move(common_runtime_arg_values),
                    .advanced_options = std::move(advanced_options)};
            },
            nb::arg("kernel"),
            nb::arg("runtime_arg_values") = tt::tt_metal::KernelRunArgs::RuntimeArgValues{},
            nb::arg("common_runtime_arg_values") = tt::tt_metal::KernelRunArgs::CommonRuntimeArgValues{},
            nb::arg("advanced_options") = m2::AdvancedKernelRunArgs{})
        .def_rw("kernel", &tt::tt_metal::KernelRunArgs::kernel)
        .def_rw("runtime_arg_values", &tt::tt_metal::KernelRunArgs::runtime_arg_values)
        .def_rw("common_runtime_arg_values", &tt::tt_metal::KernelRunArgs::common_runtime_arg_values)
        .def_rw("advanced_options", &tt::tt_metal::KernelRunArgs::advanced_options);

    nb::class_<tt::tt_metal::DFBRunOverrides>(mod, "DFBRunOverrides")
        .def(
            "__init__",
            [](tt::tt_metal::DFBRunOverrides* self,
               tt::tt_metal::DFBSpecName dfb,
               std::optional<uint32_t> entry_size,
               std::optional<uint32_t> num_entries) {
                new (self)
                    tt::tt_metal::DFBRunOverrides{.dfb = std::move(dfb), .entry_size = entry_size, .num_entries = num_entries};
            },
            nb::arg("dfb"),
            nb::arg("entry_size") = nb::none(),
            nb::arg("num_entries") = nb::none())
        .def_rw("dfb", &tt::tt_metal::DFBRunOverrides::dfb)
        .def_rw("entry_size", &tt::tt_metal::DFBRunOverrides::entry_size)
        .def_rw("num_entries", &tt::tt_metal::DFBRunOverrides::num_entries);

    // tensor_args is deliberately absent: TensorArgument holds a non-owning MeshTensor
    // reference that must alias one of the op's io tensors by pointer identity. generic_op
    // takes a {tensor parameter name -> io tensor index} map instead and builds the table itself.
    nb::class_<tt::tt_metal::ProgramRunArgs>(mod, "ProgramRunArgs")
        .def(
            "__init__",
            [](tt::tt_metal::ProgramRunArgs* self,
               std::vector<tt::tt_metal::KernelRunArgs> kernel_run_args,
               std::vector<tt::tt_metal::DFBRunOverrides> dfb_run_overrides) {
                new (self) tt::tt_metal::ProgramRunArgs{
                    .kernel_run_args = std::move(kernel_run_args),
                    .tensor_args = {},
                    .dfb_run_overrides = std::move(dfb_run_overrides)};
            },
            nb::arg("kernel_run_args") = std::vector<tt::tt_metal::KernelRunArgs>{},
            nb::arg("dfb_run_overrides") = std::vector<tt::tt_metal::DFBRunOverrides>{})
        .def_rw("kernel_run_args", &tt::tt_metal::ProgramRunArgs::kernel_run_args)
        .def_rw("dfb_run_overrides", &tt::tt_metal::ProgramRunArgs::dfb_run_overrides);
}

}  // namespace ttnn::program_specs
