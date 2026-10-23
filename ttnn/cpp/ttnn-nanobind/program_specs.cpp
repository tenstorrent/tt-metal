// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "program_specs.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
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

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

// Python binding of the Metal 2.0 host API (tt::tt_metal::experimental::ProgramSpec and
// everything it is assembled from). Exposed as ttnn._ttnn.program_spec and re-exported as
// ttnn.experimental.metal2.
//
// Binding style, chosen because the host API is experimental and its structs change shape
// often: every aggregate is bound with a default constructor, one read-write property per
// field, and a keyword constructor that default-constructs the struct and then assigns the
// fields it was given. Adding a field to a struct is one def_rw line plus one keyword; a
// renamed or removed field fails to compile here instead of silently disappearing from
// Python. The static_asserts below pin each struct's field count so that a field added in
// tt_metal is noticed in this file.

namespace ttnn::program_specs {

namespace m2 = tt::tt_metal::experimental;

namespace {

// Compile-time field count of an aggregate via C++20 parenthesised aggregate initialisation:
// T is constructible from N stand-in arguments exactly when it has at least N fields. The
// stand-in converts to anything except T itself, so a copy constructor never counts as a field.
template <typename T>
struct AnyFieldOf {
    template <typename U>
        requires(!std::is_same_v<std::remove_cvref_t<U>, T>)
    operator U() const;
};

template <typename T, std::size_t>
using any_field_t = AnyFieldOf<T>;

template <typename T, std::size_t N>
constexpr bool constructible_from_n_fields_v = []<std::size_t... I>(std::index_sequence<I...>) {
    return std::is_constructible_v<T, any_field_t<T, I>...>;
}(std::make_index_sequence<N>{});

template <typename T, std::size_t N>
constexpr bool has_exactly_n_fields_v = constructible_from_n_fields_v<T, N> && !constructible_from_n_fields_v<T, N + 1>;

#define TTNN_M2_ASSERT_FIELD_COUNT(TYPE, N)                                                       \
    static_assert(                                                                                \
        has_exactly_n_fields_v<TYPE, N>,                                                          \
        #TYPE                                                                                     \
        " changed shape in tt_metal: update its binding below, the docstring, the field list in " \
        "tests/ttnn/unit_tests/operations/debug/test_metal2_bindings.py, and this count")

using DM1XX = m2::DataMovementHardwareConfig::DataMovement1XXConfig;
using DM2XX = m2::DataMovementHardwareConfig::DataMovement2XXConfig;
using Compute1XX = m2::ComputeHardwareConfig::Compute1XXConfig;
using Compute2XX = m2::ComputeHardwareConfig::Compute2XXConfig;
using ComputeUnpackModes = m2::ComputeHardwareConfig::ComputeUnpackModes;
using SourceCode = m2::KernelSpec::SourceCode;
using CompilerOptions = m2::KernelSpec::CompilerOptions;
using RuntimeArgSchema = m2::KernelSpec::RuntimeArgSchema;
using TensorBindingSequence = m2::KernelAdvancedOptions::TensorBindingSequence;
using PrefetcherPipeBinding = m2::KernelAdvancedOptions::PrefetcherPipeBinding;

// Deprecated members still count as fields for the sentinel; they are left unbound.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
TTNN_M2_ASSERT_FIELD_COUNT(m2::TensorSpecRelaxations, 4);
TTNN_M2_ASSERT_FIELD_COUNT(m2::TensorParameter, 3);
TTNN_M2_ASSERT_FIELD_COUNT(m2::DFBAdvancedOptions, 3);
TTNN_M2_ASSERT_FIELD_COUNT(m2::DataflowBufferSpec, 7);
TTNN_M2_ASSERT_FIELD_COUNT(m2::SemaphoreAdvancedOptions, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::SemaphoreSpec, 3);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ScratchpadSpec, 4);
TTNN_M2_ASSERT_FIELD_COUNT(DM1XX, 4);
TTNN_M2_ASSERT_FIELD_COUNT(DM2XX, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::DataMovementHardwareConfig, 2);
TTNN_M2_ASSERT_FIELD_COUNT(Compute1XX, 1);
TTNN_M2_ASSERT_FIELD_COUNT(Compute2XX, 0);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ComputeHardwareConfig, 7);
TTNN_M2_ASSERT_FIELD_COUNT(SourceCode, 1);
TTNN_M2_ASSERT_FIELD_COUNT(CompilerOptions, 3);
TTNN_M2_ASSERT_FIELD_COUNT(m2::DFBBinding, 5);
TTNN_M2_ASSERT_FIELD_COUNT(m2::SemaphoreBinding, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ScratchpadBinding, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::TensorBinding, 2);
TTNN_M2_ASSERT_FIELD_COUNT(RuntimeArgSchema, 2);
TTNN_M2_ASSERT_FIELD_COUNT(TensorBindingSequence, 2);
TTNN_M2_ASSERT_FIELD_COUNT(PrefetcherPipeBinding, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelAdvancedOptions, 6);
TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelSpec, 12);
TTNN_M2_ASSERT_FIELD_COUNT(m2::WorkUnitSpec, 3);
TTNN_M2_ASSERT_FIELD_COUNT(m2::PrefetcherPipeParameter, 4);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ProgramAdvancedOptions, 1);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ProgramSpec, 8);
TTNN_M2_ASSERT_FIELD_COUNT(m2::AdvancedKernelRunArgs, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelRunArgs, 4);
TTNN_M2_ASSERT_FIELD_COUNT(m2::DFBRunOverrides, 3);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ProgramRunArgs, 4);
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#undef TTNN_M2_ASSERT_FIELD_COUNT

// tt::DataFormat has no Python binding. The DFB and scratchpad format metadata are exposed as
// ttnn.DataType through these two converters, the one place a retyping of the C++ field lands.
std::optional<tt::DataFormat> to_data_format(const std::optional<tt::tt_metal::DataType>& data_type) {
    if (!data_type.has_value()) {
        return std::nullopt;
    }
    return tt::tt_metal::datatype_to_dataformat_converter(*data_type);
}

std::optional<tt::tt_metal::DataType> to_data_type(const std::optional<tt::DataFormat>& data_format) {
    if (!data_format.has_value()) {
        return std::nullopt;
    }
    return tt::tt_metal::dataformat_to_datatype_converter(*data_format);
}

}  // namespace

void py_module_types(nb::module_& mod) {
    mod.doc() =
        "Metal 2.0 program spec types (experimental). Python view of "
        "tt::tt_metal::experimental::ProgramSpec and ProgramRunArgs; re-exported as ttnn.experimental.metal2.";

    export_enum<tt::tt_metal::UnpackMode>(mod, "UnpackMode");
    export_enum<tt::tt_metal::Precision>(mod, "Precision");
    export_enum<m2::DFBEndpointType>(mod, "DFBEndpointType");
    export_enum<m2::DFBAccessPattern>(mod, "DFBAccessPattern");
    export_enum<m2::DedicatedNocSharing>(mod, "DedicatedNocSharing");

    // ---------------------------------------------------------------- tensor parameters

    nb::class_<m2::TensorSpecRelaxations>(
        mod, "TensorSpecRelaxations", "How loosely a TensorParameter's spec matches the tensor passed at run time.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::TensorSpecRelaxations* self,
               bool match_padded_shape_only,
               bool dynamic_tensor_shape,
               bool relax_logical_rank,
               bool match_page_size) {
                new (self) m2::TensorSpecRelaxations{};
                self->match_padded_shape_only = match_padded_shape_only;
                self->dynamic_tensor_shape = dynamic_tensor_shape;
                self->relax_logical_rank = relax_logical_rank;
                self->match_page_size = match_page_size;
            },
            nb::arg("match_padded_shape_only") = false,
            nb::arg("dynamic_tensor_shape") = false,
            nb::arg("relax_logical_rank") = false,
            nb::arg("match_page_size") = false)
        .def_rw("match_padded_shape_only", &m2::TensorSpecRelaxations::match_padded_shape_only)
        .def_rw("dynamic_tensor_shape", &m2::TensorSpecRelaxations::dynamic_tensor_shape)
        .def_rw("relax_logical_rank", &m2::TensorSpecRelaxations::relax_logical_rank)
        .def_rw("match_page_size", &m2::TensorSpecRelaxations::match_page_size);

    // TensorSpec has no default constructor, so this is the one struct built by designated
    // initialisation (a reordered or renamed field still fails to compile).
    nb::class_<m2::TensorParameter>(
        mod,
        "TensorParameter",
        "A tensor the program reads or writes, named so kernels can bind to it as tensor::<name>.")
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

    nb::class_<m2::DFBAdvancedOptions>(mod, "DFBAdvancedOptions", "Rarely needed DataflowBufferSpec settings.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::DFBAdvancedOptions* self,
               std::vector<m2::DFBSpecName> alias_with,
               bool allow_instance_multi_binding,
               std::vector<m2::PrefetcherPipeParamName> prefetcher_pipe_relays) {
                new (self) m2::DFBAdvancedOptions{};
                self->alias_with = std::move(alias_with);
                self->allow_instance_multi_binding = allow_instance_multi_binding;
                self->prefetcher_pipe_relays = std::move(prefetcher_pipe_relays);
            },
            nb::arg("alias_with") = std::vector<m2::DFBSpecName>{},
            nb::arg("allow_instance_multi_binding") = false,
            nb::arg("prefetcher_pipe_relays") = std::vector<m2::PrefetcherPipeParamName>{})
        .def_rw("alias_with", &m2::DFBAdvancedOptions::alias_with)
        .def_rw("allow_instance_multi_binding", &m2::DFBAdvancedOptions::allow_instance_multi_binding)
        .def_rw("prefetcher_pipe_relays", &m2::DFBAdvancedOptions::prefetcher_pipe_relays);

    nb::class_<m2::DataflowBufferSpec>(
        mod,
        "DataflowBufferSpec",
        "A ring of num_entries entries of entry_size bytes, allocated on every node of the work units that bind it. "
        "data_format and tile_format are the metadata compute kernels need; borrowed_from makes the ring alias a "
        "TensorParameter's resident L1 shard instead of allocating.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::DataflowBufferSpec* self,
               m2::DFBSpecName unique_id,
               uint32_t entry_size,
               uint32_t num_entries,
               std::optional<tt::tt_metal::DataType> data_format,
               std::optional<tt::tt_metal::Tile> tile_format,
               std::optional<m2::TensorParamName> borrowed_from,
               m2::DFBAdvancedOptions advanced_options) {
                new (self) m2::DataflowBufferSpec{};
                self->unique_id = std::move(unique_id);
                self->entry_size = entry_size;
                self->num_entries = num_entries;
                self->data_format_metadata = to_data_format(data_format);
                self->tile_format_metadata = tile_format;
                self->borrowed_from = std::move(borrowed_from);
                self->advanced_options = std::move(advanced_options);
            },
            nb::arg("unique_id"),
            nb::arg("entry_size"),
            nb::arg("num_entries"),
            nb::arg("data_format") = nb::none(),
            nb::arg("tile_format") = nb::none(),
            nb::arg("borrowed_from") = nb::none(),
            nb::arg("advanced_options") = m2::DFBAdvancedOptions{})
        .def_rw("unique_id", &m2::DataflowBufferSpec::unique_id)
        .def_rw("entry_size", &m2::DataflowBufferSpec::entry_size)
        .def_rw("num_entries", &m2::DataflowBufferSpec::num_entries)
        .def_prop_rw(
            "data_format",
            [](const m2::DataflowBufferSpec& self) { return to_data_type(self.data_format_metadata); },
            [](m2::DataflowBufferSpec& self, const std::optional<tt::tt_metal::DataType>& data_format) {
                self.data_format_metadata = to_data_format(data_format);
            },
            "The entries' data format as a ttnn.DataType, or None when the ring carries no format metadata.")
        .def_rw("tile_format", &m2::DataflowBufferSpec::tile_format_metadata)
        .def_rw("borrowed_from", &m2::DataflowBufferSpec::borrowed_from)
        .def_rw("advanced_options", &m2::DataflowBufferSpec::advanced_options);

    // ---------------------------------------------------------------- semaphores, scratchpads

    nb::class_<m2::SemaphoreAdvancedOptions>(mod, "SemaphoreAdvancedOptions", "Rarely needed SemaphoreSpec settings.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::SemaphoreAdvancedOptions* self, uint32_t max_value) {
                new (self) m2::SemaphoreAdvancedOptions{};
                self->max_value = max_value;
            },
            nb::arg("max_value") = 0)
        .def_rw("max_value", &m2::SemaphoreAdvancedOptions::max_value);

    nb::class_<m2::SemaphoreSpec>(
        mod, "SemaphoreSpec", "A semaphore allocated on target_nodes, bound by kernels as sem::<name>.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::SemaphoreSpec* self,
               m2::SemaphoreSpecName unique_id,
               m2::Nodes target_nodes,
               m2::SemaphoreAdvancedOptions advanced_options) {
                new (self) m2::SemaphoreSpec{};
                self->unique_id = std::move(unique_id);
                self->target_nodes = std::move(target_nodes);
                self->advanced_options = advanced_options;
            },
            nb::arg("unique_id"),
            nb::arg("target_nodes"),
            nb::arg("advanced_options") = m2::SemaphoreAdvancedOptions{})
        .def_rw("unique_id", &m2::SemaphoreSpec::unique_id)
        .def_rw("target_nodes", &m2::SemaphoreSpec::target_nodes)
        .def_rw("advanced_options", &m2::SemaphoreSpec::advanced_options);

    nb::class_<m2::ScratchpadSpec>(
        mod,
        "ScratchpadSpec",
        "Private L1 scratch of size_per_node bytes for the kernel that binds it as scratch::<name>.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::ScratchpadSpec* self,
               m2::ScratchpadSpecName unique_id,
               uint32_t size_per_node,
               std::optional<tt::tt_metal::DataType> data_format,
               std::optional<tt::tt_metal::Tile> tile_format) {
                new (self) m2::ScratchpadSpec{};
                self->unique_id = std::move(unique_id);
                self->size_per_node = size_per_node;
                self->data_format_metadata = to_data_format(data_format);
                self->tile_format_metadata = tile_format;
            },
            nb::arg("unique_id"),
            nb::arg("size_per_node"),
            nb::arg("data_format") = nb::none(),
            nb::arg("tile_format") = nb::none())
        .def_rw("unique_id", &m2::ScratchpadSpec::unique_id)
        .def_rw("size_per_node", &m2::ScratchpadSpec::size_per_node)
        .def_prop_rw(
            "data_format",
            [](const m2::ScratchpadSpec& self) { return to_data_type(self.data_format_metadata); },
            [](m2::ScratchpadSpec& self, const std::optional<tt::tt_metal::DataType>& data_format) {
                self.data_format_metadata = to_data_format(data_format);
            })
        .def_rw("tile_format", &m2::ScratchpadSpec::tile_format_metadata);

    // ---------------------------------------------------------------- hardware configs

    nb::class_<DM1XX>(
        mod,
        "DataMovement1XXConfig",
        "Wormhole/Blackhole placement of a data-movement kernel: which RISC-V, which NoC, and how the NoC is shared.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](DM1XX* self,
               tt::tt_metal::DataMovementProcessor processor,
               tt::tt_metal::NOC noc,
               tt::tt_metal::NOC_MODE noc_mode,
               m2::DedicatedNocSharing dedicated_noc_sharing) {
                new (self) DM1XX{};
                self->processor = processor;
                self->noc = noc;
                self->noc_mode = noc_mode;
                self->dedicated_noc_sharing = dedicated_noc_sharing;
            },
            nb::arg("processor"),
            nb::arg("noc"),
            nb::arg("noc_mode") = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            nb::arg("dedicated_noc_sharing") = m2::DedicatedNocSharing::EXCLUSIVE)
        .def_rw("processor", &DM1XX::processor)
        .def_rw("noc", &DM1XX::noc)
        .def_rw("noc_mode", &DM1XX::noc_mode)
        .def_rw("dedicated_noc_sharing", &DM1XX::dedicated_noc_sharing);

    nb::class_<DM2XX>(mod, "DataMovement2XXConfig", "Quasar settings of a data-movement kernel; ignored on 1xx.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](DM2XX* self,
               std::vector<m2::DFBSpecName> disable_dfb_implicit_sync_for,
               bool disable_dfb_implicit_sync_for_all) {
                new (self) DM2XX{};
                self->disable_dfb_implicit_sync_for = std::move(disable_dfb_implicit_sync_for);
                self->disable_dfb_implicit_sync_for_all = disable_dfb_implicit_sync_for_all;
            },
            nb::arg("disable_dfb_implicit_sync_for") = std::vector<m2::DFBSpecName>{},
            nb::arg("disable_dfb_implicit_sync_for_all") = false)
        .def_rw("disable_dfb_implicit_sync_for", &DM2XX::disable_dfb_implicit_sync_for)
        .def_rw("disable_dfb_implicit_sync_for_all", &DM2XX::disable_dfb_implicit_sync_for_all);

    nb::class_<m2::DataMovementHardwareConfig>(
        mod,
        "DataMovementHardwareConfig",
        "Marks a KernelSpec as a data-movement kernel. config_1xx is required on Wormhole/Blackhole; "
        "create_reader_dm_config() and create_writer_dm_config() fill it in.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::DataMovementHardwareConfig* self, std::optional<DM1XX> config_1xx, std::optional<DM2XX> config_2xx) {
                new (self) m2::DataMovementHardwareConfig{};
                self->config_1xx = config_1xx;
                self->config_2xx = std::move(config_2xx);
            },
            nb::arg("config_1xx") = nb::none(),
            nb::arg("config_2xx") = nb::none())
        .def_rw("config_1xx", &m2::DataMovementHardwareConfig::config_1xx)
        .def_rw("config_2xx", &m2::DataMovementHardwareConfig::config_2xx);

    nb::class_<Compute1XX>(mod, "Compute1XXConfig", "Wormhole/Blackhole settings of a compute kernel.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Compute1XX* self, tt::tt_metal::Precision bfp_pack_precision_mode) {
                new (self) Compute1XX{};
                self->bfp_pack_precision_mode = bfp_pack_precision_mode;
            },
            nb::arg("bfp_pack_precision_mode") = tt::tt_metal::Precision::Approximate)
        .def_rw("bfp_pack_precision_mode", &Compute1XX::bfp_pack_precision_mode);

    nb::class_<Compute2XX>(mod, "Compute2XXConfig", "Quasar settings of a compute kernel; none yet.").def(nb::init<>());

    nb::class_<m2::ComputeHardwareConfig>(
        mod,
        "ComputeHardwareConfig",
        "Marks a KernelSpec as a compute kernel and sets its math fidelity, SFPU precision, DST mode and "
        "per-DFB unpack modes (unpack_modes maps a DFB name to an UnpackMode).")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::ComputeHardwareConfig* self,
               tt::tt_metal::MathFidelity fpu_math_fidelity,
               tt::tt_metal::Precision sfpu_precision_mode,
               bool enable_32_bit_dest,
               bool double_buffer_dest,
               ComputeUnpackModes unpack_modes,
               std::optional<Compute1XX> config_1xx,
               std::optional<Compute2XX> config_2xx) {
                new (self) m2::ComputeHardwareConfig{};
                self->fpu_math_fidelity = fpu_math_fidelity;
                self->sfpu_precision_mode = sfpu_precision_mode;
                self->enable_32_bit_dest = enable_32_bit_dest;
                self->double_buffer_dest = double_buffer_dest;
                self->unpack_modes = std::move(unpack_modes);
                self->config_1xx = config_1xx;
                self->config_2xx = config_2xx;
            },
            nb::arg("fpu_math_fidelity") = tt::tt_metal::MathFidelity::HiFi4,
            nb::arg("sfpu_precision_mode") = tt::tt_metal::Precision::Precise,
            nb::arg("enable_32_bit_dest") = false,
            nb::arg("double_buffer_dest") = true,
            nb::arg("unpack_modes") = ComputeUnpackModes{},
            nb::arg("config_1xx") = nb::none(),
            nb::arg("config_2xx") = nb::none())
        .def_rw("fpu_math_fidelity", &m2::ComputeHardwareConfig::fpu_math_fidelity)
        .def_rw("sfpu_precision_mode", &m2::ComputeHardwareConfig::sfpu_precision_mode)
        .def_rw("enable_32_bit_dest", &m2::ComputeHardwareConfig::enable_32_bit_dest)
        .def_rw("double_buffer_dest", &m2::ComputeHardwareConfig::double_buffer_dest)
        .def_rw("unpack_modes", &m2::ComputeHardwareConfig::unpack_modes)
        .def_rw("config_1xx", &m2::ComputeHardwareConfig::config_1xx)
        .def_rw("config_2xx", &m2::ComputeHardwareConfig::config_2xx);

    mod.def(
        "create_reader_dm_config",
        &m2::CreateReaderDataMovementConfig,
        "The data-movement config of a reader kernel: RISCV_1 on NOC_0 on Wormhole/Blackhole.");
    mod.def(
        "create_writer_dm_config",
        &m2::CreateWriterDataMovementConfig,
        "The data-movement config of a writer kernel: RISCV_0 on NOC_1 on Wormhole/Blackhole.");

    // ---------------------------------------------------------------- kernels

    nb::class_<SourceCode>(mod, "SourceCode", "Inline kernel source, as an alternative to a path to a .cpp file.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](SourceCode* self, std::string code) {
                new (self) SourceCode{};
                self->code = std::move(code);
            },
            nb::arg("code"))
        .def_rw("code", &SourceCode::code);

    nb::class_<CompilerOptions>(mod, "CompilerOptions", "Extra include paths, -D defines and the optimisation level.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](CompilerOptions* self,
               std::vector<std::filesystem::path> include_paths,
               CompilerOptions::Defines defines,
               std::optional<tt::tt_metal::KernelBuildOptLevel> opt_level) {
                new (self) CompilerOptions{};
                self->include_paths = std::move(include_paths);
                self->defines = std::move(defines);
                self->opt_level = opt_level.value_or(CompilerOptions::OptLevel::O2);
            },
            nb::arg("include_paths") = std::vector<std::filesystem::path>{},
            nb::arg("defines") = CompilerOptions::Defines{},
            // KernelBuildOptLevel is registered by ttnn.experimental.fabric_mux, which binds after this
            // module, so the default is spelled as None here and resolved to O2 above.
            nb::arg("opt_level") = nb::none())
        .def_rw("include_paths", &CompilerOptions::include_paths)
        .def_rw("defines", &CompilerOptions::defines)
        .def_rw("opt_level", &CompilerOptions::opt_level);

    nb::class_<m2::DFBBinding>(
        mod,
        "DFBBinding",
        "Binds a DataflowBufferSpec to a kernel as dfb::<accessor_name>, as its producer or consumer. "
        "Prefer producer_of(), consumer_of(), strided_consumer_of(), all_consumer_of(), "
        "blocked_producer_of() and blocked_consumer_of().")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::DFBBinding* self,
               m2::DFBSpecName dfb_spec_name,
               std::string accessor_name,
               m2::DFBEndpointType endpoint_type,
               m2::DFBAccessPattern access_pattern,
               uint32_t block_size) {
                new (self) m2::DFBBinding{};
                self->dfb_spec_name = std::move(dfb_spec_name);
                self->accessor_name = std::move(accessor_name);
                self->endpoint_type = endpoint_type;
                self->access_pattern = access_pattern;
                self->block_size = block_size;
            },
            nb::arg("dfb_spec_name"),
            nb::arg("accessor_name"),
            nb::arg("endpoint_type"),
            nb::arg("access_pattern") = m2::DFBAccessPattern::STRIDED,
            nb::arg("block_size") = 0)
        .def_rw("dfb_spec_name", &m2::DFBBinding::dfb_spec_name)
        .def_rw("accessor_name", &m2::DFBBinding::accessor_name)
        .def_rw("endpoint_type", &m2::DFBBinding::endpoint_type)
        .def_rw("access_pattern", &m2::DFBBinding::access_pattern)
        .def_rw("block_size", &m2::DFBBinding::block_size);

    mod.def(
        "producer_of",
        &m2::ProducerOf,
        nb::arg("dfb_spec_name"),
        nb::arg("accessor_name"),
        "A STRIDED producer binding: thread t of the kernel fills entries t, t+N, ... of the ring.");
    mod.def(
        "consumer_of",
        &m2::ConsumerOf,
        nb::arg("dfb_spec_name"),
        nb::arg("accessor_name"),
        "A consumer binding with the default (STRIDED) access pattern.");
    mod.def(
        "strided_consumer_of",
        &m2::StridedConsumerOf,
        nb::arg("dfb_spec_name"),
        nb::arg("accessor_name"),
        "A STRIDED consumer binding: thread t of the kernel drains entries t, t+N, ... of the ring.");
    mod.def(
        "all_consumer_of",
        &m2::AllConsumerOf,
        nb::arg("dfb_spec_name"),
        nb::arg("accessor_name"),
        "An ALL consumer binding: every thread of the kernel sees every entry.");
    mod.def(
        "blocked_producer_of",
        &m2::BlockedProducerOf,
        nb::arg("dfb_spec_name"),
        nb::arg("accessor_name"),
        nb::arg("block_size"),
        "A BLOCKED producer binding: each thread owns contiguous blocks of block_size entries.");
    mod.def(
        "blocked_consumer_of",
        &m2::BlockedConsumerOf,
        nb::arg("dfb_spec_name"),
        nb::arg("accessor_name"),
        nb::arg("block_size"),
        "A BLOCKED consumer binding: each thread owns contiguous blocks of block_size entries.");

    nb::class_<m2::SemaphoreBinding>(
        mod, "SemaphoreBinding", "Binds a SemaphoreSpec to a kernel as sem::<accessor_name>.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::SemaphoreBinding* self, m2::SemaphoreSpecName semaphore_spec_name, std::string accessor_name) {
                new (self) m2::SemaphoreBinding{};
                self->semaphore_spec_name = std::move(semaphore_spec_name);
                self->accessor_name = std::move(accessor_name);
            },
            nb::arg("semaphore_spec_name"),
            nb::arg("accessor_name"))
        .def_rw("semaphore_spec_name", &m2::SemaphoreBinding::semaphore_spec_name)
        .def_rw("accessor_name", &m2::SemaphoreBinding::accessor_name);

    nb::class_<m2::ScratchpadBinding>(
        mod, "ScratchpadBinding", "Binds a ScratchpadSpec to a kernel as scratch::<accessor_name>.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::ScratchpadBinding* self, m2::ScratchpadSpecName scratchpad_spec_name, std::string accessor_name) {
                new (self) m2::ScratchpadBinding{};
                self->scratchpad_spec_name = std::move(scratchpad_spec_name);
                self->accessor_name = std::move(accessor_name);
            },
            nb::arg("scratchpad_spec_name"),
            nb::arg("accessor_name"))
        .def_rw("scratchpad_spec_name", &m2::ScratchpadBinding::scratchpad_spec_name)
        .def_rw("accessor_name", &m2::ScratchpadBinding::accessor_name);

    nb::class_<m2::TensorBinding>(
        mod, "TensorBinding", "Binds a TensorParameter to a kernel as tensor::<accessor_name>.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::TensorBinding* self, m2::TensorParamName tensor_parameter_name, std::string accessor_name) {
                new (self) m2::TensorBinding{};
                self->tensor_parameter_name = std::move(tensor_parameter_name);
                self->accessor_name = std::move(accessor_name);
            },
            nb::arg("tensor_parameter_name"),
            nb::arg("accessor_name"))
        .def_rw("tensor_parameter_name", &m2::TensorBinding::tensor_parameter_name)
        .def_rw("accessor_name", &m2::TensorBinding::accessor_name);

    nb::class_<RuntimeArgSchema>(
        mod,
        "RuntimeArgSchema",
        "The names of a kernel's per-node and common runtime args; kernels read them as get_arg(args::<name>).")
        .def(nb::init<>())
        .def(
            "__init__",
            [](RuntimeArgSchema* self,
               std::vector<std::string> runtime_arg_names,
               std::vector<std::string> common_runtime_arg_names) {
                new (self) RuntimeArgSchema{};
                self->runtime_arg_names = std::move(runtime_arg_names);
                self->common_runtime_arg_names = std::move(common_runtime_arg_names);
            },
            nb::arg("runtime_arg_names") = std::vector<std::string>{},
            nb::arg("common_runtime_arg_names") = std::vector<std::string>{})
        .def_rw("runtime_arg_names", &RuntimeArgSchema::runtime_arg_names)
        .def_rw("common_runtime_arg_names", &RuntimeArgSchema::common_runtime_arg_names);

    nb::class_<TensorBindingSequence>(
        mod,
        "TensorBindingSequence",
        "Groups tensor bindings into a device-side tuple tensor::<sequence_name>; members lists accessor names.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](TensorBindingSequence* self, std::string sequence_name, std::vector<std::string> members) {
                new (self) TensorBindingSequence{};
                self->sequence_name = std::move(sequence_name);
                self->members = std::move(members);
            },
            nb::arg("sequence_name"),
            nb::arg("members") = std::vector<std::string>{})
        .def_rw("sequence_name", &TensorBindingSequence::sequence_name)
        .def_rw("members", &TensorBindingSequence::members);

    nb::class_<PrefetcherPipeBinding>(
        mod, "PrefetcherPipeBinding", "Binds prefetcher pipe parameters to a kernel under one accessor name.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](PrefetcherPipeBinding* self,
               std::vector<m2::PrefetcherPipeParamName> pipe_parameter_names,
               std::string accessor_name) {
                new (self) PrefetcherPipeBinding{};
                self->pipe_parameter_names = std::move(pipe_parameter_names);
                self->accessor_name = std::move(accessor_name);
            },
            nb::arg("pipe_parameter_names"),
            nb::arg("accessor_name"))
        .def_rw("pipe_parameter_names", &PrefetcherPipeBinding::pipe_parameter_names)
        .def_rw("accessor_name", &PrefetcherPipeBinding::accessor_name);

    // num_runtime_varargs_per_node is deprecated in tt_metal and deliberately not bound.
    nb::class_<m2::KernelAdvancedOptions>(mod, "KernelAdvancedOptions", "Rarely needed KernelSpec settings.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::KernelAdvancedOptions* self,
               uint32_t num_runtime_varargs,
               uint32_t num_common_runtime_varargs,
               std::vector<uint32_t> compile_time_varargs,
               std::vector<TensorBindingSequence> tensor_binding_sequences,
               std::vector<PrefetcherPipeBinding> prefetcher_pipe_bindings) {
                new (self) m2::KernelAdvancedOptions{};
                self->num_runtime_varargs = num_runtime_varargs;
                self->num_common_runtime_varargs = num_common_runtime_varargs;
                self->compile_time_varargs = std::move(compile_time_varargs);
                self->tensor_binding_sequences = std::move(tensor_binding_sequences);
                self->prefetcher_pipe_bindings = std::move(prefetcher_pipe_bindings);
            },
            nb::arg("num_runtime_varargs") = 0,
            nb::arg("num_common_runtime_varargs") = 0,
            nb::arg("compile_time_varargs") = std::vector<uint32_t>{},
            nb::arg("tensor_binding_sequences") = std::vector<TensorBindingSequence>{},
            nb::arg("prefetcher_pipe_bindings") = std::vector<PrefetcherPipeBinding>{})
        .def_rw("num_runtime_varargs", &m2::KernelAdvancedOptions::num_runtime_varargs)
        .def_rw("num_common_runtime_varargs", &m2::KernelAdvancedOptions::num_common_runtime_varargs)
        .def_rw("compile_time_varargs", &m2::KernelAdvancedOptions::compile_time_varargs)
        .def_rw("tensor_binding_sequences", &m2::KernelAdvancedOptions::tensor_binding_sequences)
        .def_rw("prefetcher_pipe_bindings", &m2::KernelAdvancedOptions::prefetcher_pipe_bindings);

    nb::class_<m2::KernelSpec>(
        mod,
        "KernelSpec",
        "One kernel: its source (a path or SourceCode), its hardware config (DataMovementHardwareConfig or "
        "ComputeHardwareConfig), the specs it binds, its compile-time args (name -> value) and the schema of "
        "its runtime args. num_threads > 1 is supported on Quasar only.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::KernelSpec* self,
               m2::KernelSpecName unique_id,
               std::variant<std::filesystem::path, SourceCode> source,
               std::variant<m2::DataMovementHardwareConfig, m2::ComputeHardwareConfig> hw_config,
               uint32_t num_threads,
               CompilerOptions compiler_options,
               std::vector<m2::DFBBinding> dfb_bindings,
               std::vector<m2::SemaphoreBinding> semaphore_bindings,
               std::vector<m2::ScratchpadBinding> scratchpad_bindings,
               std::vector<m2::TensorBinding> tensor_bindings,
               m2::KernelSpec::CompileTimeArgs compile_time_args,
               RuntimeArgSchema runtime_arg_schema,
               m2::KernelAdvancedOptions advanced_options) {
                new (self) m2::KernelSpec{};
                self->unique_id = std::move(unique_id);
                self->source = std::move(source);
                self->hw_config = std::move(hw_config);
                self->num_threads = num_threads;
                self->compiler_options = std::move(compiler_options);
                self->dfb_bindings = std::move(dfb_bindings);
                self->semaphore_bindings = std::move(semaphore_bindings);
                self->scratchpad_bindings = std::move(scratchpad_bindings);
                self->tensor_bindings = std::move(tensor_bindings);
                self->compile_time_args = std::move(compile_time_args);
                self->runtime_arg_schema = std::move(runtime_arg_schema);
                self->advanced_options = std::move(advanced_options);
            },
            nb::arg("unique_id"),
            nb::arg("source"),
            nb::arg("hw_config"),
            nb::arg("num_threads") = 1,
            nb::arg("compiler_options") = CompilerOptions{},
            nb::arg("dfb_bindings") = std::vector<m2::DFBBinding>{},
            nb::arg("semaphore_bindings") = std::vector<m2::SemaphoreBinding>{},
            nb::arg("scratchpad_bindings") = std::vector<m2::ScratchpadBinding>{},
            nb::arg("tensor_bindings") = std::vector<m2::TensorBinding>{},
            nb::arg("compile_time_args") = m2::KernelSpec::CompileTimeArgs{},
            nb::arg("runtime_arg_schema") = RuntimeArgSchema{},
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
        .def_rw("advanced_options", &m2::KernelSpec::advanced_options)
        .def("is_data_movement_kernel", &m2::KernelSpec::is_data_movement_kernel)
        .def("is_compute_kernel", &m2::KernelSpec::is_compute_kernel);

    // ---------------------------------------------------------------- program spec

    nb::class_<m2::WorkUnitSpec>(
        mod,
        "WorkUnitSpec",
        "A named set of kernels that run together on target_nodes (a CoreCoord, CoreRange or CoreRangeSet).")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::WorkUnitSpec* self,
               std::string name,
               std::vector<m2::KernelSpecName> kernels,
               m2::Nodes target_nodes) {
                new (self) m2::WorkUnitSpec{};
                self->name = std::move(name);
                self->kernels = std::move(kernels);
                self->target_nodes = std::move(target_nodes);
            },
            nb::arg("name"),
            nb::arg("kernels"),
            nb::arg("target_nodes"))
        .def_rw("name", &m2::WorkUnitSpec::name)
        .def_rw("kernels", &m2::WorkUnitSpec::kernels)
        .def_rw("target_nodes", &m2::WorkUnitSpec::target_nodes);

    nb::class_<m2::PrefetcherPipeParameter>(
        mod, "PrefetcherPipeParameter", "A prefetcher pipe the program expects to be handed at run time.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::PrefetcherPipeParameter* self,
               m2::PrefetcherPipeParamName unique_id,
               m2::Nodes receivers,
               uint32_t ring_size,
               uint32_t entry_size) {
                new (self) m2::PrefetcherPipeParameter{};
                self->unique_id = std::move(unique_id);
                self->receivers = std::move(receivers);
                self->ring_size = ring_size;
                self->entry_size = entry_size;
            },
            nb::arg("unique_id"),
            nb::arg("receivers"),
            nb::arg("ring_size") = 0,
            nb::arg("entry_size") = 0)
        .def_rw("unique_id", &m2::PrefetcherPipeParameter::unique_id)
        .def_rw("receivers", &m2::PrefetcherPipeParameter::receivers)
        .def_rw("ring_size", &m2::PrefetcherPipeParameter::ring_size)
        .def_rw("entry_size", &m2::PrefetcherPipeParameter::entry_size);

    nb::class_<m2::ProgramAdvancedOptions>(mod, "ProgramAdvancedOptions", "Rarely needed ProgramSpec settings.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::ProgramAdvancedOptions* self, std::vector<m2::PrefetcherPipeParameter> prefetcher_pipe_parameters) {
                new (self) m2::ProgramAdvancedOptions{};
                self->prefetcher_pipe_parameters = std::move(prefetcher_pipe_parameters);
            },
            nb::arg("prefetcher_pipe_parameters") = std::vector<m2::PrefetcherPipeParameter>{})
        .def_rw("prefetcher_pipe_parameters", &m2::ProgramAdvancedOptions::prefetcher_pipe_parameters);

    nb::class_<m2::ProgramSpec>(
        mod,
        "ProgramSpec",
        "A whole program: kernels, the dataflow buffers, semaphores and scratchpads they bind, the tensors they "
        "touch, and the work units that place them on nodes. Container properties return copies: assign a new "
        "list rather than appending to the returned one.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::ProgramSpec* self,
               std::string name,
               std::vector<m2::KernelSpec> kernels,
               std::vector<m2::DataflowBufferSpec> dataflow_buffers,
               std::vector<m2::SemaphoreSpec> semaphores,
               std::vector<m2::ScratchpadSpec> scratchpads,
               std::vector<m2::TensorParameter> tensor_parameters,
               std::vector<m2::WorkUnitSpec> work_units,
               m2::ProgramAdvancedOptions advanced_options) {
                new (self) m2::ProgramSpec{};
                self->name = std::move(name);
                self->kernels = std::move(kernels);
                self->dataflow_buffers = std::move(dataflow_buffers);
                self->semaphores = std::move(semaphores);
                self->scratchpads = std::move(scratchpads);
                self->tensor_parameters = std::move(tensor_parameters);
                self->work_units = std::move(work_units);
                self->advanced_options = std::move(advanced_options);
            },
            nb::arg("name") = std::string{},
            nb::arg("kernels") = std::vector<m2::KernelSpec>{},
            nb::arg("dataflow_buffers") = std::vector<m2::DataflowBufferSpec>{},
            nb::arg("semaphores") = std::vector<m2::SemaphoreSpec>{},
            nb::arg("scratchpads") = std::vector<m2::ScratchpadSpec>{},
            nb::arg("tensor_parameters") = std::vector<m2::TensorParameter>{},
            nb::arg("work_units") = std::vector<m2::WorkUnitSpec>{},
            nb::arg("advanced_options") = m2::ProgramAdvancedOptions{})
        .def_rw("name", &m2::ProgramSpec::name)
        .def_rw("kernels", &m2::ProgramSpec::kernels)
        .def_rw("dataflow_buffers", &m2::ProgramSpec::dataflow_buffers)
        .def_rw("semaphores", &m2::ProgramSpec::semaphores)
        .def_rw("scratchpads", &m2::ProgramSpec::scratchpads)
        .def_rw("tensor_parameters", &m2::ProgramSpec::tensor_parameters)
        .def_rw("work_units", &m2::ProgramSpec::work_units)
        .def_rw("advanced_options", &m2::ProgramSpec::advanced_options);

    // ---------------------------------------------------------------- run args

    nb::class_<m2::AdvancedKernelRunArgs>(
        mod, "AdvancedKernelRunArgs", "Variadic runtime args: runtime_varargs maps a CoreCoord to a list of values.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::AdvancedKernelRunArgs* self,
               m2::Table<m2::NodeCoord, m2::AdvancedKernelRunArgs::Varargs> runtime_varargs,
               m2::AdvancedKernelRunArgs::Varargs common_runtime_varargs) {
                new (self) m2::AdvancedKernelRunArgs{};
                self->runtime_varargs = std::move(runtime_varargs);
                self->common_runtime_varargs = std::move(common_runtime_varargs);
            },
            nb::arg("runtime_varargs") = m2::Table<m2::NodeCoord, m2::AdvancedKernelRunArgs::Varargs>{},
            nb::arg("common_runtime_varargs") = m2::AdvancedKernelRunArgs::Varargs{})
        .def_rw("runtime_varargs", &m2::AdvancedKernelRunArgs::runtime_varargs)
        .def_rw("common_runtime_varargs", &m2::AdvancedKernelRunArgs::common_runtime_varargs);

    nb::class_<m2::KernelRunArgs>(
        mod,
        "KernelRunArgs",
        "The runtime arg values of one kernel. runtime_arg_values maps an arg name to {CoreCoord: value}; "
        "common_runtime_arg_values maps an arg name to one value for every node.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::KernelRunArgs* self,
               m2::KernelSpecName kernel,
               m2::KernelRunArgs::RuntimeArgValues runtime_arg_values,
               m2::KernelRunArgs::CommonRuntimeArgValues common_runtime_arg_values,
               m2::AdvancedKernelRunArgs advanced_options) {
                new (self) m2::KernelRunArgs{};
                self->kernel = std::move(kernel);
                self->runtime_arg_values = std::move(runtime_arg_values);
                self->common_runtime_arg_values = std::move(common_runtime_arg_values);
                self->advanced_options = std::move(advanced_options);
            },
            nb::arg("kernel"),
            nb::arg("runtime_arg_values") = m2::KernelRunArgs::RuntimeArgValues{},
            nb::arg("common_runtime_arg_values") = m2::KernelRunArgs::CommonRuntimeArgValues{},
            nb::arg("advanced_options") = m2::AdvancedKernelRunArgs{})
        .def_rw("kernel", &m2::KernelRunArgs::kernel)
        .def_rw("runtime_arg_values", &m2::KernelRunArgs::runtime_arg_values)
        .def_rw("common_runtime_arg_values", &m2::KernelRunArgs::common_runtime_arg_values)
        .def_rw("advanced_options", &m2::KernelRunArgs::advanced_options);

    nb::class_<m2::DFBRunOverrides>(
        mod, "DFBRunOverrides", "Per-run override of a DataflowBufferSpec's entry_size and/or num_entries.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::DFBRunOverrides* self,
               m2::DFBSpecName dfb,
               std::optional<uint32_t> entry_size,
               std::optional<uint32_t> num_entries) {
                new (self) m2::DFBRunOverrides{};
                self->dfb = std::move(dfb);
                self->entry_size = entry_size;
                self->num_entries = num_entries;
            },
            nb::arg("dfb"),
            nb::arg("entry_size") = nb::none(),
            nb::arg("num_entries") = nb::none())
        .def_rw("dfb", &m2::DFBRunOverrides::dfb)
        .def_rw("entry_size", &m2::DFBRunOverrides::entry_size)
        .def_rw("num_entries", &m2::DFBRunOverrides::num_entries);

    // tensor_args and advanced_options (prefetcher pipe arguments) hold non-owning references that
    // must alias objects the caller keeps alive, so they are not settable from Python. The generic_op
    // spec overload takes a {tensor parameter name: io tensor index} map and fills tensor_args itself.
    nb::class_<m2::ProgramRunArgs>(
        mod,
        "ProgramRunArgs",
        "The per-run values of a ProgramSpec: one KernelRunArgs per kernel with a runtime-arg schema, and "
        "optional DFB size overrides.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](m2::ProgramRunArgs* self,
               std::vector<m2::KernelRunArgs> kernel_run_args,
               std::vector<m2::DFBRunOverrides> dfb_run_overrides) {
                new (self) m2::ProgramRunArgs{};
                self->kernel_run_args = std::move(kernel_run_args);
                self->dfb_run_overrides = std::move(dfb_run_overrides);
            },
            nb::arg("kernel_run_args") = std::vector<m2::KernelRunArgs>{},
            nb::arg("dfb_run_overrides") = std::vector<m2::DFBRunOverrides>{})
        .def_rw("kernel_run_args", &m2::ProgramRunArgs::kernel_run_args)
        .def_rw("dfb_run_overrides", &m2::ProgramRunArgs::dfb_run_overrides);
}

}  // namespace ttnn::program_specs
