// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "program_spec.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/detail/nb_dict.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/strong_type.hpp>

#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/types.hpp"

// This file is built outside the unity build (see ttnn/CMakeLists.txt), so it does not see the
// NB_MAKE_OPAQUE(std::vector<uint32_t>) in program_descriptors.cpp: vararg values are Python lists here.

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

// The Metal 2.0 name types (KernelSpecName, DFBSpecName, ...) are string StrongTypes; Python passes and reads
// them as str.
template <typename Tag>
struct type_caster<ttsl::StrongType<std::string, Tag>> {
    using NameType = ttsl::StrongType<std::string, Tag>;
    NB_TYPE_CASTER(NameType, const_name("str"))

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        make_caster<std::string> caster;
        if (!caster.from_python(src, flags, cleanup)) {
            return false;
        }
        value = NameType(std::move(caster.value));
        return true;
    }

    static handle from_cpp(const NameType& name, rv_policy policy, cleanup_list* cleanup) noexcept {
        return make_caster<std::string>::from_cpp(*name, policy, cleanup);
    }
};

// Kernel argument values are uint32_t on the device; Python passes and reads them as int.
template <>
struct type_caster<tt::tt_metal::experimental::ArgValue> {
    NB_TYPE_CASTER(tt::tt_metal::experimental::ArgValue, const_name("int"))

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        make_caster<uint32_t> caster;
        if (!caster.from_python(src, flags, cleanup)) {
            return false;
        }
        value = caster.value;
        return true;
    }

    static handle from_cpp(
        const tt::tt_metal::experimental::ArgValue& arg, rv_policy policy, cleanup_list* cleanup) noexcept {
        return make_caster<uint32_t>::from_cpp(arg.to_uint32(), policy, cleanup);
    }
};

// Table<K, V> is a map with unique keys; Python passes and reads it as a dict.
template <typename K, typename V>
struct type_caster<tt::tt_metal::experimental::Table<K, V>>
    : dict_caster<tt::tt_metal::experimental::Table<K, V>, K, V> {
    using Base = dict_caster<tt::tt_metal::experimental::Table<K, V>, K, V>;

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        if (!Base::from_python(src, flags, cleanup)) {
            return false;
        }
        // Table keeps the first of two Python keys that convert to one C++ key, such as (0, 0) and
        // ttnn.CoreCoord(0, 0), and drops the other value. Reject such a dict instead.
        const Py_ssize_t size = PyObject_Length(src.ptr());
        if (size < 0) {
            PyErr_Clear();
            return false;
        }
        return static_cast<Py_ssize_t>(this->value.size()) == size;
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

namespace ttnn::program_spec {

namespace m2 = tt::tt_metal::experimental;

namespace {

using DMConfig = m2::DataMovementHardwareConfig;
using ComputeConfig = m2::ComputeHardwareConfig;

// Returned lists, dicts, and optional and variant fields are copies, so nothing read from them refers into a spec. A
// nested object that is always there is returned by reference instead (def_rw's default), so writes through it reach
// its parent, which it keeps alive.
constexpr auto COPY = nb::rv_policy::copy;

// Property setters refuse None unless told otherwise; these setters clear an std::optional field on None.
auto setter_accepts_none() { return nb::for_setter(nb::arg("value").none()); }

// Tripwires: adding a field to one of these Metal 2.0 types must be a deliberate choice. Bind it below,
// or note why Python leaves it out, then update the count. KernelAdvancedOptions and DFBAdvancedOptions
// have none: Python binds only the options Blaze uses, and the rest stay C++ only until a caller needs them.
static_assert(reflect::size<m2::TensorSpecRelaxations>() == 4);
static_assert(reflect::size<m2::TensorParameter>() == 3);
static_assert(reflect::size<m2::DataflowBufferSpec>() == 7);
static_assert(reflect::size<m2::SemaphoreSpec>() == 3, "advanced_options is C++ only");
static_assert(reflect::size<m2::ScratchpadSpec>() == 5, "advanced_options is C++ only");
static_assert(reflect::size<DMConfig::DataMovement1XXConfig>() == 4);
static_assert(reflect::size<DMConfig::DataMovement2XXConfig>() == 2);
static_assert(reflect::size<DMConfig>() == 2);
static_assert(reflect::size<ComputeConfig::Compute1XXConfig>() == 1);
static_assert(reflect::size<ComputeConfig::Compute12XConfig>() == 1);
static_assert(reflect::size<ComputeConfig::Compute2XXConfig>() == 1);
static_assert(reflect::size<ComputeConfig>() == 8);
static_assert(reflect::size<m2::KernelSpec::SourceCode>() == 1);
static_assert(reflect::size<m2::KernelSpec::CompilerOptions>() == 3);
static_assert(reflect::size<m2::KernelSpec::RuntimeArgSchema>() == 2);
static_assert(reflect::size<m2::DFBBinding>() == 5);
static_assert(reflect::size<m2::SemaphoreBinding>() == 2);
static_assert(reflect::size<m2::ScratchpadBinding>() == 2);
static_assert(reflect::size<m2::TensorBinding>() == 2);
static_assert(reflect::size<m2::KernelSpec>() == 12);
static_assert(reflect::size<m2::WorkUnitSpec>() == 3);
static_assert(reflect::size<m2::ProgramSpec>() == 8, "advanced_options is C++ only");
static_assert(reflect::size<m2::AdvancedKernelRunArgs>() == 2);
static_assert(reflect::size<m2::KernelRunArgs>() == 4);
static_assert(reflect::size<m2::DFBRunOverrides>() == 3);
static_assert(reflect::size<m2::ProgramRunArgs>() == 4, "advanced_options is C++ only");

// Data formats are ttnn data types in Python, so a format without one, such as Float16, cannot be set. A buffer
// takes the format that ttnn's own ops declare for a tensor of that data type: int8 becomes UInt8, because the
// unpacker and packer read Int8 as sign-magnitude, and so reads back as uint8.
std::optional<tt::DataFormat> to_data_format(const std::optional<ttnn::DataType>& data_type) {
    if (!data_type.has_value()) {
        return std::nullopt;
    }
    return tt::tt_metal::cb_dataformat_for(*data_type);
}

std::optional<ttnn::DataType> to_data_type(const std::optional<tt::DataFormat>& data_format) {
    if (!data_format.has_value()) {
        return std::nullopt;
    }
    return tt::tt_metal::dataformat_to_datatype_converter(*data_format);
}

// Points each tensor argument at the MeshTensor of a ttnn.Tensor. Copies of a ttnn.Tensor share one
// MeshTensor, so generic_op finds the same MeshTensor in its io_tensors.
m2::Table<m2::TensorParamName, m2::TensorArgument> make_tensor_args(const nb::dict& tensors) {
    m2::Table<m2::TensorParamName, m2::TensorArgument> tensor_args;
    for (const auto& [key, value] : tensors) {
        if (!nb::isinstance<nb::str>(key)) {
            throw nb::type_error("tensor_args keys must be tensor parameter names (str)");
        }
        auto name = nb::cast<std::string>(key);
        if (!nb::isinstance<ttnn::Tensor>(value)) {
            throw nb::type_error(fmt::format("tensor_args['{}'] must be a ttnn.Tensor", name).c_str());
        }
        const auto& tensor = nb::cast<const ttnn::Tensor&>(value);
        TT_FATAL(
            ttnn::is_device_tensor(tensor) && tensor.is_allocated(),
            "tensor_args['{}'] must be a tensor allocated on a device",
            name);
        tensor_args.emplace(m2::TensorParamName(std::move(name)), m2::TensorArgument{std::cref(tensor.mesh_tensor())});
    }
    return tensor_args;
}

void bind_enums(nb::module_& mod) {
    export_enum<tt::tt_metal::Precision>(mod, "Precision");
    export_enum<tt::tt_metal::UnpackMode>(mod, "UnpackMode");
    export_enum<m2::DedicatedNocSharing>(mod, "DedicatedNocSharing");
}

void bind_tensor_parameter(nb::module_& mod) {
    const m2::TensorSpecRelaxations relaxation_defaults{};
    nb::class_<m2::TensorSpecRelaxations>(
        mod,
        "TensorSpecRelaxations",
        "Ways a tensor argument's TensorSpec may differ from its TensorParameter's. The default requires an "
        "exact match. See tensor_spec_relaxations.hpp.")
        .def(
            "__init__",
            [](m2::TensorSpecRelaxations* self,
               bool match_padded_shape_only,
               bool dynamic_tensor_shape,
               bool relax_logical_rank,
               bool match_page_size) {
                new (self) m2::TensorSpecRelaxations{
                    .match_padded_shape_only = match_padded_shape_only,
                    .dynamic_tensor_shape = dynamic_tensor_shape,
                    .relax_logical_rank = relax_logical_rank,
                    .match_page_size = match_page_size};
            },
            nb::kw_only(),
            nb::arg("match_padded_shape_only") = relaxation_defaults.match_padded_shape_only,
            nb::arg("dynamic_tensor_shape") = relaxation_defaults.dynamic_tensor_shape,
            nb::arg("relax_logical_rank") = relaxation_defaults.relax_logical_rank,
            nb::arg("match_page_size") = relaxation_defaults.match_page_size)
        .def_rw("match_padded_shape_only", &m2::TensorSpecRelaxations::match_padded_shape_only)
        .def_rw("dynamic_tensor_shape", &m2::TensorSpecRelaxations::dynamic_tensor_shape)
        .def_rw("relax_logical_rank", &m2::TensorSpecRelaxations::relax_logical_rank)
        .def_rw("match_page_size", &m2::TensorSpecRelaxations::match_page_size);

    nb::class_<m2::TensorParameter>(
        mod,
        "TensorParameter",
        "A tensor the program's kernels use: its name and the single-device TensorSpec its tensor argument "
        "must have. relaxations refers into this object, so writes through it change the parameter. See "
        "tensor_parameter.hpp.")
        .def(
            "__init__",
            [](m2::TensorParameter* self,
               m2::TensorParamName unique_id,
               tt::tt_metal::TensorSpec spec,
               m2::TensorSpecRelaxations relaxations) {
                new (self) m2::TensorParameter{
                    .unique_id = std::move(unique_id), .spec = std::move(spec), .relaxations = relaxations};
            },
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("spec"),
            nb::arg("relaxations") = relaxation_defaults)
        .def_rw("unique_id", &m2::TensorParameter::unique_id)
        .def_rw("spec", &m2::TensorParameter::spec, COPY)
        .def_rw("relaxations", &m2::TensorParameter::relaxations);
}

void bind_resources(nb::module_& mod) {
    const m2::DFBAdvancedOptions dfb_advanced_defaults{};
    nb::class_<m2::DFBAdvancedOptions>(
        mod,
        "DFBAdvancedOptions",
        "Rarely used DataflowBufferSpec options. Python has alias_with only; the others are C++ only. See "
        "advanced_options.hpp.")
        .def(
            "__init__",
            [](m2::DFBAdvancedOptions* self, std::vector<m2::DFBSpecName> alias_with) {
                new (self) m2::DFBAdvancedOptions{};
                self->alias_with = std::move(alias_with);
            },
            nb::kw_only(),
            nb::arg("alias_with") = dfb_advanced_defaults.alias_with)
        .def_rw("alias_with", &m2::DFBAdvancedOptions::alias_with, COPY);

    const m2::DataflowBufferSpec dfb_defaults{};
    nb::class_<m2::DataflowBufferSpec>(
        mod,
        "DataflowBufferSpec",
        "A dataflow buffer (DFB): a FIFO between a producer kernel and a consumer kernel on each node. "
        "data_format_metadata is a ttnn data type; ttnn.int8 is stored as the UInt8 format, as in ttnn's own ops, and "
        "reads back as ttnn.uint8. advanced_options refers into this object. See dataflow_buffer_spec.hpp.")
        .def(
            "__init__",
            [](m2::DataflowBufferSpec* self,
               m2::DFBSpecName unique_id,
               uint32_t entry_size,
               uint32_t num_entries,
               std::optional<ttnn::DataType> data_format_metadata,
               std::optional<tt::tt_metal::Tile> tile_format_metadata,
               std::optional<m2::TensorParamName> borrowed_from,
               m2::DFBAdvancedOptions advanced_options) {
                new (self) m2::DataflowBufferSpec{
                    .unique_id = std::move(unique_id),
                    .entry_size = entry_size,
                    .num_entries = num_entries,
                    .data_format_metadata = to_data_format(data_format_metadata),
                    .tile_format_metadata = std::move(tile_format_metadata),
                    .borrowed_from = std::move(borrowed_from),
                    .advanced_options = std::move(advanced_options)};
            },
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("entry_size") = dfb_defaults.entry_size,
            nb::arg("num_entries") = dfb_defaults.num_entries,
            nb::arg("data_format_metadata") = to_data_type(dfb_defaults.data_format_metadata),
            nb::arg("tile_format_metadata") = dfb_defaults.tile_format_metadata,
            nb::arg("borrowed_from") = dfb_defaults.borrowed_from,
            nb::arg("advanced_options") = dfb_defaults.advanced_options)
        .def_rw("unique_id", &m2::DataflowBufferSpec::unique_id)
        .def_rw("entry_size", &m2::DataflowBufferSpec::entry_size)
        .def_rw("num_entries", &m2::DataflowBufferSpec::num_entries)
        .def_prop_rw(
            "data_format_metadata",
            [](const m2::DataflowBufferSpec& self) { return to_data_type(self.data_format_metadata); },
            [](m2::DataflowBufferSpec& self, std::optional<ttnn::DataType> data_format) {
                self.data_format_metadata = to_data_format(data_format);
            },
            setter_accepts_none())
        .def_rw("tile_format_metadata", &m2::DataflowBufferSpec::tile_format_metadata, COPY, setter_accepts_none())
        .def_rw("borrowed_from", &m2::DataflowBufferSpec::borrowed_from, setter_accepts_none())
        .def_rw("advanced_options", &m2::DataflowBufferSpec::advanced_options);

    nb::class_<m2::SemaphoreSpec>(
        mod, "SemaphoreSpec", "A semaphore on each of target_nodes, starting at 0. See semaphore_spec.hpp.")
        .def(
            "__init__",
            [](m2::SemaphoreSpec* self, m2::SemaphoreSpecName unique_id, m2::Nodes target_nodes) {
                new (self)
                    m2::SemaphoreSpec{.unique_id = std::move(unique_id), .target_nodes = std::move(target_nodes)};
            },
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("target_nodes"))
        .def_rw("unique_id", &m2::SemaphoreSpec::unique_id)
        .def_rw("target_nodes", &m2::SemaphoreSpec::target_nodes, COPY);

    const m2::ScratchpadSpec scratchpad_defaults{};
    nb::class_<m2::ScratchpadSpec>(
        mod,
        "ScratchpadSpec",
        "Private, uninitialized L1 memory for the kernel that binds it, on each of its nodes. "
        "data_format_metadata is a ttnn data type; ttnn.int8 is stored as the UInt8 format and reads back as "
        "ttnn.uint8. See scratchpad_spec.hpp.")
        .def(
            "__init__",
            [](m2::ScratchpadSpec* self,
               m2::ScratchpadSpecName unique_id,
               uint32_t size_per_node,
               std::optional<ttnn::DataType> data_format_metadata,
               std::optional<tt::tt_metal::Tile> tile_format_metadata) {
                new (self) m2::ScratchpadSpec{
                    .unique_id = std::move(unique_id),
                    .size_per_node = size_per_node,
                    .data_format_metadata = to_data_format(data_format_metadata),
                    .tile_format_metadata = std::move(tile_format_metadata)};
            },
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("size_per_node") = scratchpad_defaults.size_per_node,
            nb::arg("data_format_metadata") = to_data_type(scratchpad_defaults.data_format_metadata),
            nb::arg("tile_format_metadata") = scratchpad_defaults.tile_format_metadata)
        .def_rw("unique_id", &m2::ScratchpadSpec::unique_id)
        .def_rw("size_per_node", &m2::ScratchpadSpec::size_per_node)
        .def_prop_rw(
            "data_format_metadata",
            [](const m2::ScratchpadSpec& self) { return to_data_type(self.data_format_metadata); },
            [](m2::ScratchpadSpec& self, std::optional<ttnn::DataType> data_format) {
                self.data_format_metadata = to_data_format(data_format);
            },
            setter_accepts_none())
        .def_rw("tile_format_metadata", &m2::ScratchpadSpec::tile_format_metadata, COPY, setter_accepts_none());
}

void bind_hardware_configs(nb::module_& mod) {
    auto dm_config = nb::class_<DMConfig>(
        mod,
        "DataMovementHardwareConfig",
        "The hardware a data-movement kernel uses. config_1xx applies on Wormhole and Blackhole, config_2xx on "
        "Quasar. They are optional, so reading one returns a copy: assign a whole config to change it. See "
        "data_movement_hardware_config.hpp.");

    const DMConfig::DataMovement1XXConfig dm_1xx_defaults{};
    nb::class_<DMConfig::DataMovement1XXConfig>(dm_config, "DataMovement1XXConfig")
        .def(
            "__init__",
            [](DMConfig::DataMovement1XXConfig* self,
               tt::tt_metal::DataMovementProcessor processor,
               tt::tt_metal::NOC noc,
               tt::tt_metal::NOC_MODE noc_mode,
               m2::DedicatedNocSharing dedicated_noc_sharing) {
                new (self) DMConfig::DataMovement1XXConfig{
                    .processor = processor,
                    .noc = noc,
                    .noc_mode = noc_mode,
                    .dedicated_noc_sharing = dedicated_noc_sharing};
            },
            nb::kw_only(),
            nb::arg("processor"),
            nb::arg("noc"),
            nb::arg("noc_mode") = dm_1xx_defaults.noc_mode,
            nb::arg("dedicated_noc_sharing") = dm_1xx_defaults.dedicated_noc_sharing)
        .def_rw("processor", &DMConfig::DataMovement1XXConfig::processor)
        .def_rw("noc", &DMConfig::DataMovement1XXConfig::noc)
        .def_rw("noc_mode", &DMConfig::DataMovement1XXConfig::noc_mode)
        .def_rw("dedicated_noc_sharing", &DMConfig::DataMovement1XXConfig::dedicated_noc_sharing);

    const DMConfig::DataMovement2XXConfig dm_2xx_defaults{};
    nb::class_<DMConfig::DataMovement2XXConfig>(dm_config, "DataMovement2XXConfig")
        .def(
            "__init__",
            [](DMConfig::DataMovement2XXConfig* self,
               std::vector<m2::DFBSpecName> disable_dfb_implicit_sync_for,
               bool disable_dfb_implicit_sync_for_all) {
                new (self) DMConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for = std::move(disable_dfb_implicit_sync_for),
                    .disable_dfb_implicit_sync_for_all = disable_dfb_implicit_sync_for_all};
            },
            nb::kw_only(),
            nb::arg("disable_dfb_implicit_sync_for") = dm_2xx_defaults.disable_dfb_implicit_sync_for,
            nb::arg("disable_dfb_implicit_sync_for_all") = dm_2xx_defaults.disable_dfb_implicit_sync_for_all)
        .def_rw("disable_dfb_implicit_sync_for", &DMConfig::DataMovement2XXConfig::disable_dfb_implicit_sync_for, COPY)
        .def_rw(
            "disable_dfb_implicit_sync_for_all", &DMConfig::DataMovement2XXConfig::disable_dfb_implicit_sync_for_all);

    const DMConfig dm_defaults{};
    dm_config
        .def(
            "__init__",
            [](DMConfig* self,
               std::optional<DMConfig::DataMovement1XXConfig> config_1xx,
               std::optional<DMConfig::DataMovement2XXConfig> config_2xx) {
                new (self) DMConfig{.config_1xx = std::move(config_1xx), .config_2xx = std::move(config_2xx)};
            },
            nb::kw_only(),
            nb::arg("config_1xx") = dm_defaults.config_1xx,
            nb::arg("config_2xx") = dm_defaults.config_2xx)
        .def_rw("config_1xx", &DMConfig::config_1xx, COPY, setter_accepts_none())
        .def_rw("config_2xx", &DMConfig::config_2xx, COPY, setter_accepts_none());

    // ttnn's helpers, which also set config_2xx, stand in for Metal's CreateReaderDataMovementConfig and
    // CreateWriterDataMovementConfig.
    mod.def(
        "create_reader_datamovement_config",
        &ttnn::create_reader_datamovement_config,
        nb::kw_only(),
        nb::arg("disable_dfb_implicit_sync_for_all") = false,
        "The conventional DataMovementHardwareConfig of a kernel that reads from DRAM.");
    mod.def(
        "create_writer_datamovement_config",
        &ttnn::create_writer_datamovement_config,
        nb::kw_only(),
        nb::arg("disable_dfb_implicit_sync_for_all") = false,
        "The conventional DataMovementHardwareConfig of a kernel that writes to DRAM.");

    auto compute_config = nb::class_<ComputeConfig>(
        mod,
        "ComputeHardwareConfig",
        "The Tensix settings a compute kernel uses. config_1xx applies on Wormhole and Blackhole, config_12x "
        "on Blackhole and config_2xx on Quasar. They are optional, so reading one returns a copy: assign a whole "
        "config to change it. See compute_hardware_config.hpp.");

    const ComputeConfig::Compute1XXConfig compute_1xx_defaults{};
    nb::class_<ComputeConfig::Compute1XXConfig>(compute_config, "Compute1XXConfig")
        .def(
            "__init__",
            [](ComputeConfig::Compute1XXConfig* self, tt::tt_metal::Precision bfp_pack_precision_mode) {
                new (self) ComputeConfig::Compute1XXConfig{.bfp_pack_precision_mode = bfp_pack_precision_mode};
            },
            nb::kw_only(),
            nb::arg("bfp_pack_precision_mode") = compute_1xx_defaults.bfp_pack_precision_mode)
        .def_rw("bfp_pack_precision_mode", &ComputeConfig::Compute1XXConfig::bfp_pack_precision_mode);

    const ComputeConfig::Compute12XConfig compute_12x_defaults{};
    nb::class_<ComputeConfig::Compute12XConfig>(compute_config, "Compute12XConfig")
        .def(
            "__init__",
            [](ComputeConfig::Compute12XConfig* self, bool enable_pack_rvv) {
                new (self) ComputeConfig::Compute12XConfig{.enable_pack_rvv = enable_pack_rvv};
            },
            nb::kw_only(),
            nb::arg("enable_pack_rvv") = compute_12x_defaults.enable_pack_rvv)
        .def_rw("enable_pack_rvv", &ComputeConfig::Compute12XConfig::enable_pack_rvv);

    const ComputeConfig::Compute2XXConfig compute_2xx_defaults{};
    nb::class_<ComputeConfig::Compute2XXConfig>(compute_config, "Compute2XXConfig")
        .def(
            "__init__",
            [](ComputeConfig::Compute2XXConfig* self, bool enable_unpack_rvv) {
                new (self) ComputeConfig::Compute2XXConfig{.enable_unpack_rvv = enable_unpack_rvv};
            },
            nb::kw_only(),
            nb::arg("enable_unpack_rvv") = compute_2xx_defaults.enable_unpack_rvv)
        .def_rw("enable_unpack_rvv", &ComputeConfig::Compute2XXConfig::enable_unpack_rvv);

    const ComputeConfig compute_defaults{};
    compute_config
        .def(
            "__init__",
            [](ComputeConfig* self,
               tt::tt_metal::MathFidelity fpu_math_fidelity,
               tt::tt_metal::Precision sfpu_precision_mode,
               bool enable_32_bit_dest,
               bool double_buffer_dest,
               ComputeConfig::ComputeUnpackModes unpack_modes,
               std::optional<ComputeConfig::Compute1XXConfig> config_1xx,
               std::optional<ComputeConfig::Compute12XConfig> config_12x,
               std::optional<ComputeConfig::Compute2XXConfig> config_2xx) {
                new (self) ComputeConfig{
                    .fpu_math_fidelity = fpu_math_fidelity,
                    .sfpu_precision_mode = sfpu_precision_mode,
                    .enable_32_bit_dest = enable_32_bit_dest,
                    .double_buffer_dest = double_buffer_dest,
                    .unpack_modes = std::move(unpack_modes),
                    .config_1xx = config_1xx,
                    .config_12x = config_12x,
                    .config_2xx = config_2xx};
            },
            nb::kw_only(),
            nb::arg("fpu_math_fidelity") = compute_defaults.fpu_math_fidelity,
            nb::arg("sfpu_precision_mode") = compute_defaults.sfpu_precision_mode,
            nb::arg("enable_32_bit_dest") = compute_defaults.enable_32_bit_dest,
            nb::arg("double_buffer_dest") = compute_defaults.double_buffer_dest,
            nb::arg("unpack_modes") = compute_defaults.unpack_modes,
            nb::arg("config_1xx") = compute_defaults.config_1xx,
            nb::arg("config_12x") = compute_defaults.config_12x,
            nb::arg("config_2xx") = compute_defaults.config_2xx)
        .def_rw("fpu_math_fidelity", &ComputeConfig::fpu_math_fidelity)
        .def_rw("sfpu_precision_mode", &ComputeConfig::sfpu_precision_mode)
        .def_rw("enable_32_bit_dest", &ComputeConfig::enable_32_bit_dest)
        .def_rw("double_buffer_dest", &ComputeConfig::double_buffer_dest)
        .def_rw("unpack_modes", &ComputeConfig::unpack_modes, COPY)
        .def_rw("config_1xx", &ComputeConfig::config_1xx, COPY, setter_accepts_none())
        .def_rw("config_12x", &ComputeConfig::config_12x, COPY, setter_accepts_none())
        .def_rw("config_2xx", &ComputeConfig::config_2xx, COPY, setter_accepts_none());
}

void bind_kernel_spec(nb::module_& mod) {
    const m2::KernelAdvancedOptions kernel_advanced_defaults{};
    nb::class_<m2::KernelAdvancedOptions>(
        mod,
        "KernelAdvancedOptions",
        "Rarely used KernelSpec options. Python has the runtime vararg counts only; the others are C++ only. "
        "See advanced_options.hpp.")
        .def(
            "__init__",
            [](m2::KernelAdvancedOptions* self, uint32_t num_runtime_varargs, uint32_t num_common_runtime_varargs) {
                new (self) m2::KernelAdvancedOptions{};
                self->num_runtime_varargs = num_runtime_varargs;
                self->num_common_runtime_varargs = num_common_runtime_varargs;
            },
            nb::kw_only(),
            nb::arg("num_runtime_varargs") = kernel_advanced_defaults.num_runtime_varargs,
            nb::arg("num_common_runtime_varargs") = kernel_advanced_defaults.num_common_runtime_varargs)
        .def_rw("num_runtime_varargs", &m2::KernelAdvancedOptions::num_runtime_varargs)
        .def_rw("num_common_runtime_varargs", &m2::KernelAdvancedOptions::num_common_runtime_varargs);

    auto kernel_spec = nb::class_<m2::KernelSpec>(
        mod,
        "KernelSpec",
        "One compiled kernel: its source, compiler options, the buffers, semaphores, scratchpads and tensors it "
        "binds by name, its compile-time arguments and the names of its runtime arguments. compiler_options, "
        "runtime_arg_schema and advanced_options refer into this object, so writes through them change the kernel; "
        "source and hw_config are copies. See kernel_spec.hpp.");

    nb::class_<m2::KernelSpec::SourceCode>(kernel_spec, "SourceCode", "Kernel source code, given inline.")
        .def(
            "__init__",
            [](m2::KernelSpec::SourceCode* self, std::string code) {
                new (self) m2::KernelSpec::SourceCode{.code = std::move(code)};
            },
            nb::kw_only(),
            nb::arg("code"))
        .def_rw("code", &m2::KernelSpec::SourceCode::code);

    const m2::KernelSpec::CompilerOptions compiler_defaults{};
    nb::class_<m2::KernelSpec::CompilerOptions>(kernel_spec, "CompilerOptions")
        .def(
            "__init__",
            [](m2::KernelSpec::CompilerOptions* self,
               m2::KernelSpec::CompilerOptions::IncludePaths include_paths,
               m2::KernelSpec::CompilerOptions::Defines defines,
               tt::tt_metal::KernelBuildOptLevel opt_level) {
                new (self) m2::KernelSpec::CompilerOptions{
                    .include_paths = std::move(include_paths), .defines = std::move(defines), .opt_level = opt_level};
            },
            nb::kw_only(),
            nb::arg("include_paths") = compiler_defaults.include_paths,
            nb::arg("defines") = compiler_defaults.defines,
            nb::arg("opt_level") = compiler_defaults.opt_level)
        .def_rw("include_paths", &m2::KernelSpec::CompilerOptions::include_paths, COPY)
        .def_rw("defines", &m2::KernelSpec::CompilerOptions::defines, COPY)
        .def_rw("opt_level", &m2::KernelSpec::CompilerOptions::opt_level);

    const m2::KernelSpec::RuntimeArgSchema schema_defaults{};
    nb::class_<m2::KernelSpec::RuntimeArgSchema>(kernel_spec, "RuntimeArgSchema")
        .def(
            "__init__",
            [](m2::KernelSpec::RuntimeArgSchema* self,
               std::vector<std::string> runtime_arg_names,
               std::vector<std::string> common_runtime_arg_names) {
                new (self) m2::KernelSpec::RuntimeArgSchema{
                    .runtime_arg_names = std::move(runtime_arg_names),
                    .common_runtime_arg_names = std::move(common_runtime_arg_names)};
            },
            nb::kw_only(),
            nb::arg("runtime_arg_names") = schema_defaults.runtime_arg_names,
            nb::arg("common_runtime_arg_names") = schema_defaults.common_runtime_arg_names)
        .def_rw("runtime_arg_names", &m2::KernelSpec::RuntimeArgSchema::runtime_arg_names, COPY)
        .def_rw("common_runtime_arg_names", &m2::KernelSpec::RuntimeArgSchema::common_runtime_arg_names, COPY);

    auto dfb_binding = nb::class_<m2::DFBBinding>(
        kernel_spec,
        "DFBBinding",
        "Binds a kernel to a dataflow buffer, as its producer or its consumer. Kernel code reaches it as "
        "dfb::<accessor_name>.");
    export_enum<m2::DFBEndpointType>(dfb_binding, "EndpointType");
    export_enum<m2::DFBAccessPattern>(dfb_binding, "AccessPattern");
    const m2::DFBBinding dfb_binding_defaults{};
    dfb_binding
        .def(
            "__init__",
            [](m2::DFBBinding* self,
               m2::DFBSpecName dfb_spec_name,
               std::string accessor_name,
               m2::DFBEndpointType endpoint_type,
               m2::DFBAccessPattern access_pattern,
               uint32_t block_size) {
                new (self) m2::DFBBinding{
                    .dfb_spec_name = std::move(dfb_spec_name),
                    .accessor_name = std::move(accessor_name),
                    .endpoint_type = endpoint_type,
                    .access_pattern = access_pattern,
                    .block_size = block_size};
            },
            nb::kw_only(),
            nb::arg("dfb_spec_name"),
            nb::arg("accessor_name"),
            nb::arg("endpoint_type"),
            nb::arg("access_pattern") = dfb_binding_defaults.access_pattern,
            nb::arg("block_size") = dfb_binding_defaults.block_size)
        .def_rw("dfb_spec_name", &m2::DFBBinding::dfb_spec_name)
        .def_rw("accessor_name", &m2::DFBBinding::accessor_name)
        .def_rw("endpoint_type", &m2::DFBBinding::endpoint_type)
        .def_rw("access_pattern", &m2::DFBBinding::access_pattern)
        .def_rw("block_size", &m2::DFBBinding::block_size);

    nb::class_<m2::SemaphoreBinding>(
        kernel_spec,
        "SemaphoreBinding",
        "Binds a kernel to a semaphore. Kernel code reaches it as sem::<accessor_name>.")
        .def(
            "__init__",
            [](m2::SemaphoreBinding* self, m2::SemaphoreSpecName semaphore_spec_name, std::string accessor_name) {
                new (self) m2::SemaphoreBinding{
                    .semaphore_spec_name = std::move(semaphore_spec_name), .accessor_name = std::move(accessor_name)};
            },
            nb::kw_only(),
            nb::arg("semaphore_spec_name"),
            nb::arg("accessor_name"))
        .def_rw("semaphore_spec_name", &m2::SemaphoreBinding::semaphore_spec_name)
        .def_rw("accessor_name", &m2::SemaphoreBinding::accessor_name);

    nb::class_<m2::ScratchpadBinding>(
        kernel_spec,
        "ScratchpadBinding",
        "Binds a kernel to a scratchpad. Kernel code reaches it as scratch::<accessor_name>.")
        .def(
            "__init__",
            [](m2::ScratchpadBinding* self, m2::ScratchpadSpecName scratchpad_spec_name, std::string accessor_name) {
                new (self) m2::ScratchpadBinding{
                    .scratchpad_spec_name = std::move(scratchpad_spec_name), .accessor_name = std::move(accessor_name)};
            },
            nb::kw_only(),
            nb::arg("scratchpad_spec_name"),
            nb::arg("accessor_name"))
        .def_rw("scratchpad_spec_name", &m2::ScratchpadBinding::scratchpad_spec_name)
        .def_rw("accessor_name", &m2::ScratchpadBinding::accessor_name);

    nb::class_<m2::TensorBinding>(
        kernel_spec,
        "TensorBinding",
        "Binds a kernel to a tensor parameter. Kernel code reaches it as tensor::<accessor_name>.")
        .def(
            "__init__",
            [](m2::TensorBinding* self, m2::TensorParamName tensor_parameter_name, std::string accessor_name) {
                new (self) m2::TensorBinding{
                    .tensor_parameter_name = std::move(tensor_parameter_name),
                    .accessor_name = std::move(accessor_name)};
            },
            nb::kw_only(),
            nb::arg("tensor_parameter_name"),
            nb::arg("accessor_name"))
        .def_rw("tensor_parameter_name", &m2::TensorBinding::tensor_parameter_name)
        .def_rw("accessor_name", &m2::TensorBinding::accessor_name);

    const m2::KernelSpec kernel_defaults{};
    kernel_spec
        .def(
            "__init__",
            [](m2::KernelSpec* self,
               m2::KernelSpecName unique_id,
               std::variant<std::filesystem::path, m2::KernelSpec::SourceCode> source,
               std::variant<DMConfig, ComputeConfig> hw_config,
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
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("source"),
            nb::arg("hw_config"),
            nb::arg("num_threads") = kernel_defaults.num_threads,
            nb::arg("compiler_options") = kernel_defaults.compiler_options,
            nb::arg("dfb_bindings") = kernel_defaults.dfb_bindings,
            nb::arg("semaphore_bindings") = kernel_defaults.semaphore_bindings,
            nb::arg("scratchpad_bindings") = kernel_defaults.scratchpad_bindings,
            nb::arg("tensor_bindings") = kernel_defaults.tensor_bindings,
            nb::arg("compile_time_args") = kernel_defaults.compile_time_args,
            nb::arg("runtime_arg_schema") = kernel_defaults.runtime_arg_schema,
            nb::arg("advanced_options") = kernel_advanced_defaults)
        .def_rw("unique_id", &m2::KernelSpec::unique_id)
        .def_rw("source", &m2::KernelSpec::source, COPY)
        .def_rw("num_threads", &m2::KernelSpec::num_threads)
        .def_rw("compiler_options", &m2::KernelSpec::compiler_options)
        .def_rw("dfb_bindings", &m2::KernelSpec::dfb_bindings, COPY)
        .def_rw("semaphore_bindings", &m2::KernelSpec::semaphore_bindings, COPY)
        .def_rw("scratchpad_bindings", &m2::KernelSpec::scratchpad_bindings, COPY)
        .def_rw("tensor_bindings", &m2::KernelSpec::tensor_bindings, COPY)
        .def_rw("compile_time_args", &m2::KernelSpec::compile_time_args, COPY)
        .def_rw("runtime_arg_schema", &m2::KernelSpec::runtime_arg_schema)
        .def_rw("hw_config", &m2::KernelSpec::hw_config, COPY)
        .def_rw("advanced_options", &m2::KernelSpec::advanced_options)
        .def("is_data_movement_kernel", &m2::KernelSpec::is_data_movement_kernel)
        .def("is_compute_kernel", &m2::KernelSpec::is_compute_kernel);

    // Metal's DFBBinding factories, under snake_case names.
    mod.def(
        "producer_of",
        &m2::ProducerOf,
        nb::arg("dfb_spec_name"),
        nb::arg("accessor_name"),
        "A producer binding with the STRIDED access pattern.");
    mod.def(
        "blocked_producer_of",
        &m2::BlockedProducerOf,
        nb::arg("dfb_spec_name"),
        nb::arg("accessor_name"),
        nb::arg("block_size"),
        "A producer binding with the BLOCKED access pattern.");
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
        "A consumer binding with the STRIDED access pattern.");
    mod.def(
        "all_consumer_of",
        &m2::AllConsumerOf,
        nb::arg("dfb_spec_name"),
        nb::arg("accessor_name"),
        "A consumer binding with the ALL access pattern.");
    mod.def(
        "blocked_consumer_of",
        &m2::BlockedConsumerOf,
        nb::arg("dfb_spec_name"),
        nb::arg("accessor_name"),
        nb::arg("block_size"),
        "A consumer binding with the BLOCKED access pattern.");
}

void bind_program_spec(nb::module_& mod) {
    const m2::WorkUnitSpec work_unit_defaults{};
    nb::class_<m2::WorkUnitSpec>(
        mod,
        "WorkUnitSpec",
        "A set of kernels that run together on each of target_nodes. name is for messages only. See "
        "program_spec.hpp.")
        .def(
            "__init__",
            [](m2::WorkUnitSpec* self,
               m2::Nodes target_nodes,
               std::string name,
               std::vector<m2::KernelSpecName> kernels) {
                new (self) m2::WorkUnitSpec{
                    .name = std::move(name), .kernels = std::move(kernels), .target_nodes = std::move(target_nodes)};
            },
            nb::kw_only(),
            nb::arg("target_nodes"),
            nb::arg("name") = work_unit_defaults.name,
            nb::arg("kernels") = work_unit_defaults.kernels)
        .def_rw("name", &m2::WorkUnitSpec::name)
        .def_rw("kernels", &m2::WorkUnitSpec::kernels, COPY)
        .def_rw("target_nodes", &m2::WorkUnitSpec::target_nodes, COPY);

    const m2::ProgramSpec program_defaults{};
    nb::class_<m2::ProgramSpec>(
        mod,
        "ProgramSpec",
        "Everything fixed about a program: its kernels, buffers, semaphores, scratchpads, tensor parameters and "
        "the nodes they run on. name is for messages only. Lists are copied in and out, so assign a whole list "
        "to change one. See program_spec.hpp.")
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
                    .semaphores = std::move(semaphores),
                    .scratchpads = std::move(scratchpads),
                    .tensor_parameters = std::move(tensor_parameters),
                    .work_units = std::move(work_units)};
            },
            nb::kw_only(),
            nb::arg("name") = program_defaults.name,
            nb::arg("kernels") = program_defaults.kernels,
            nb::arg("dataflow_buffers") = program_defaults.dataflow_buffers,
            nb::arg("semaphores") = program_defaults.semaphores,
            nb::arg("scratchpads") = program_defaults.scratchpads,
            nb::arg("tensor_parameters") = program_defaults.tensor_parameters,
            nb::arg("work_units") = program_defaults.work_units)
        .def_rw("name", &m2::ProgramSpec::name)
        .def_rw("kernels", &m2::ProgramSpec::kernels, COPY)
        .def_rw("dataflow_buffers", &m2::ProgramSpec::dataflow_buffers, COPY)
        .def_rw("semaphores", &m2::ProgramSpec::semaphores, COPY)
        .def_rw("scratchpads", &m2::ProgramSpec::scratchpads, COPY)
        .def_rw("tensor_parameters", &m2::ProgramSpec::tensor_parameters, COPY)
        .def_rw("work_units", &m2::ProgramSpec::work_units, COPY);
}

void bind_program_run_args(nb::module_& mod) {
    const m2::AdvancedKernelRunArgs advanced_run_args_defaults{};
    nb::class_<m2::AdvancedKernelRunArgs>(
        mod,
        "AdvancedKernelRunArgs",
        "Vararg values: runtime_varargs maps each node to its list, and common_runtime_varargs is one list for "
        "every node. See advanced_options.hpp.")
        .def(
            "__init__",
            [](m2::AdvancedKernelRunArgs* self,
               m2::Table<m2::NodeCoord, m2::AdvancedKernelRunArgs::Varargs> runtime_varargs,
               m2::AdvancedKernelRunArgs::Varargs common_runtime_varargs) {
                new (self) m2::AdvancedKernelRunArgs{
                    .runtime_varargs = std::move(runtime_varargs),
                    .common_runtime_varargs = std::move(common_runtime_varargs)};
            },
            nb::kw_only(),
            nb::arg("runtime_varargs") = advanced_run_args_defaults.runtime_varargs,
            nb::arg("common_runtime_varargs") = advanced_run_args_defaults.common_runtime_varargs)
        .def_rw("runtime_varargs", &m2::AdvancedKernelRunArgs::runtime_varargs, COPY)
        .def_rw("common_runtime_varargs", &m2::AdvancedKernelRunArgs::common_runtime_varargs, COPY);

    // dynamic_attr: the Python object keeps its tensor arguments alive (see make_tensor_args). __setattr__ below
    // still rejects names that are not attributes of the class, as for the other classes; Python subclasses may add
    // attributes of their own.
    auto run_args = nb::class_<m2::ProgramRunArgs>(
        mod,
        "ProgramRunArgs",
        nb::dynamic_attr(),
        "The values of one program run: each kernel's runtime arguments, the tensor for each tensor "
        "parameter, and DFB size overrides. tensor_args maps each tensor parameter name to a ttnn.Tensor, which "
        "these run args keep alive. Lists and dicts are copied in and out. See program_run_args.hpp.");

    const m2::KernelRunArgs kernel_run_args_defaults{};
    nb::class_<m2::KernelRunArgs>(
        run_args,
        "KernelRunArgs",
        "One kernel's runtime argument values: runtime_arg_values maps each name to {node: value}, and "
        "common_runtime_arg_values maps each name to one value for every node. advanced_options refers into this "
        "object.")
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
            nb::kw_only(),
            nb::arg("kernel"),
            nb::arg("runtime_arg_values") = kernel_run_args_defaults.runtime_arg_values,
            nb::arg("common_runtime_arg_values") = kernel_run_args_defaults.common_runtime_arg_values,
            nb::arg("advanced_options") = kernel_run_args_defaults.advanced_options)
        .def_rw("kernel", &m2::KernelRunArgs::kernel)
        .def_rw("runtime_arg_values", &m2::KernelRunArgs::runtime_arg_values, COPY)
        .def_rw("common_runtime_arg_values", &m2::KernelRunArgs::common_runtime_arg_values, COPY)
        .def_rw("advanced_options", &m2::KernelRunArgs::advanced_options);

    const m2::DFBRunOverrides override_defaults{};
    nb::class_<m2::DFBRunOverrides>(
        run_args,
        "DFBRunOverrides",
        "New entry_size or num_entries for a dataflow buffer. An override stays in effect on later runs until "
        "it is changed.")
        .def(
            "__init__",
            [](m2::DFBRunOverrides* self,
               m2::DFBSpecName dfb,
               std::optional<uint32_t> entry_size,
               std::optional<uint32_t> num_entries) {
                new (self)
                    m2::DFBRunOverrides{.dfb = std::move(dfb), .entry_size = entry_size, .num_entries = num_entries};
            },
            nb::kw_only(),
            nb::arg("dfb"),
            nb::arg("entry_size") = override_defaults.entry_size,
            nb::arg("num_entries") = override_defaults.num_entries)
        .def_rw("dfb", &m2::DFBRunOverrides::dfb)
        .def_rw("entry_size", &m2::DFBRunOverrides::entry_size, setter_accepts_none())
        .def_rw("num_entries", &m2::DFBRunOverrides::num_entries, setter_accepts_none());

    const m2::ProgramRunArgs run_args_defaults{};
    run_args
        .def(
            "__init__",
            [](nb::pointer_and_handle<m2::ProgramRunArgs> self,
               std::vector<m2::KernelRunArgs> kernel_run_args,
               const nb::dict& tensor_args,
               std::vector<m2::DFBRunOverrides> dfb_run_overrides) {
                auto tensor_arg_table = make_tensor_args(tensor_args);
                new (self.p) m2::ProgramRunArgs{
                    .kernel_run_args = std::move(kernel_run_args),
                    .tensor_args = std::move(tensor_arg_table),
                    .dfb_run_overrides = std::move(dfb_run_overrides)};
                nb::setattr(self.h, TENSOR_ARGS_ATTRIBUTE, tensor_args.attr("copy")());
            },
            nb::kw_only(),
            nb::arg("kernel_run_args") = run_args_defaults.kernel_run_args,
            nb::arg("tensor_args") = nb::dict(),
            nb::arg("dfb_run_overrides") = run_args_defaults.dfb_run_overrides)
        .def(
            "__setattr__",
            [](nb::handle self, const nb::str& name, nb::handle value) {
                set_known_attribute(self, name, value, nb::type<m2::ProgramRunArgs>());
            },
            nb::arg("name"),
            nb::arg("value").none())
        .def_rw("kernel_run_args", &m2::ProgramRunArgs::kernel_run_args, COPY)
        .def_prop_rw(
            "tensor_args",
            [](nb::pointer_and_handle<m2::ProgramRunArgs> self) {
                return nb::getattr(self.h, TENSOR_ARGS_ATTRIBUTE, nb::dict()).attr("copy")();
            },
            [](nb::pointer_and_handle<m2::ProgramRunArgs> self, const nb::dict& tensor_args) {
                self.p->tensor_args = make_tensor_args(tensor_args);
                nb::setattr(self.h, TENSOR_ARGS_ATTRIBUTE, tensor_args.attr("copy")());
            })
        .def_rw("dfb_run_overrides", &m2::ProgramRunArgs::dfb_run_overrides, COPY);
}

}  // namespace

void set_known_attribute(nb::handle self, const nb::str& name, nb::handle value, nb::handle bound_type) {
    if (self.type().is(bound_type) && !nb::hasattr(bound_type, name) && !name.equal(nb::str(TENSOR_ARGS_ATTRIBUTE))) {
        throw nb::attribute_error(fmt::format(
                                      "'{}' object has no attribute '{}'",
                                      nb::cast<std::string>(self.type().attr("__name__")),
                                      nb::cast<std::string>(name))
                                      .c_str());
    }
    if (PyObject_GenericSetAttr(self.ptr(), name.ptr(), value.ptr()) != 0) {
        throw nb::python_error();
    }
}

void py_module_types(nb::module_& mod) {
    bind_enums(mod);
    bind_tensor_parameter(mod);
    bind_resources(mod);
    bind_hardware_configs(mod);
    bind_kernel_spec(mod);
    bind_program_spec(mod);
    bind_program_run_args(mod);

    // Namespace-level names that Metal also has.
    nb::object kernel_spec = mod.attr("KernelSpec");
    mod.attr("DFBBinding") = kernel_spec.attr("DFBBinding");
    mod.attr("SemaphoreBinding") = kernel_spec.attr("SemaphoreBinding");
    mod.attr("ScratchpadBinding") = kernel_spec.attr("ScratchpadBinding");
    mod.attr("TensorBinding") = kernel_spec.attr("TensorBinding");
    mod.attr("DFBEndpointType") = kernel_spec.attr("DFBBinding").attr("EndpointType");
    mod.attr("DFBAccessPattern") = kernel_spec.attr("DFBBinding").attr("AccessPattern");
    nb::object program_run_args = mod.attr("ProgramRunArgs");
    mod.attr("KernelRunArgs") = program_run_args.attr("KernelRunArgs");
    mod.attr("DFBRunOverrides") = program_run_args.attr("DFBRunOverrides");
    // The fabric mux operation binds KernelBuildOptLevel, the type of KernelSpec.CompilerOptions.opt_level.
    mod.attr("KernelBuildOptLevel") = nb::type<tt::tt_metal::KernelBuildOptLevel>();
}

}  // namespace ttnn::program_spec
