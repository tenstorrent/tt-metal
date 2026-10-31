// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_spec_op_nanobind.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

#include "generic_spec_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn-nanobind/export_enum.hpp"

namespace ttnn::operations::generic {

namespace {

namespace exp = tt::tt_metal::experimental;

using StringMap = std::map<std::string, std::string>;
using ArgMap = std::map<std::string, uint32_t>;
using NodeArgMap = std::map<std::string, std::map<tt::tt_metal::CoreCoord, uint32_t>>;
using HwConfig = std::variant<exp::DataMovementHardwareConfig, exp::ComputeHardwareConfig>;
using Nodes = std::variant<tt::tt_metal::CoreCoord, tt::tt_metal::CoreRange, tt::tt_metal::CoreRangeSet>;

template <typename Name>
std::vector<Name> to_names(const std::vector<std::string>& names) {
    std::vector<Name> result;
    result.reserve(names.size());
    for (const auto& name : names) {
        result.emplace_back(name);
    }
    return result;
}

template <typename Name>
std::vector<std::string> from_names(const std::vector<Name>& names) {
    std::vector<std::string> result;
    result.reserve(names.size());
    for (const auto& name : names) {
        result.push_back(*name);
    }
    return result;
}

exp::DataMovementHardwareConfig::DataMovement2XXConfig make_dm_2xx_config(
    bool disable_dfb_implicit_sync_for_all, const std::vector<std::string>& disable_dfb_implicit_sync_for) {
    return {
        .disable_dfb_implicit_sync_for = to_names<exp::DFBSpecName>(disable_dfb_implicit_sync_for),
        .disable_dfb_implicit_sync_for_all = disable_dfb_implicit_sync_for_all,
    };
}

// The old (pre-#63251) constructor keywords still work, with a DeprecationWarning.
void warn_deprecated(const char* message) {
    if (PyErr_WarnEx(PyExc_DeprecationWarning, message, 1) != 0) {
        throw nb::python_error();
    }
}

std::optional<tt::DataFormat> to_data_format(const std::optional<tt::tt_metal::DataType>& data_type) {
    return data_type.has_value() ? std::optional(tt::tt_metal::cb_dataformat_for(*data_type)) : std::nullopt;
}

std::optional<tt::tt_metal::DataType> to_data_type(const std::optional<tt::DataFormat>& data_format) {
    return data_format.has_value() ? std::optional(tt::tt_metal::dataformat_to_datatype_converter(*data_format))
                                   : std::nullopt;
}

// The RVV options postdate older tt-metal versions (whose Compute2XXConfig is empty and which lack
// config_12x); there they are rejected if requested, so this file builds against both.
template <typename Config>
void set_rvv_options(Config& config, bool enable_unpack_rvv, bool enable_pack_rvv) {
    config.config_2xx = typename decltype(config.config_2xx)::value_type{};
    if constexpr (requires { config.config_2xx->enable_unpack_rvv; }) {
        config.config_2xx->enable_unpack_rvv = enable_unpack_rvv;
    } else {
        TT_FATAL(!enable_unpack_rvv, "ComputeHardwareConfig: enable_unpack_rvv needs a newer tt-metal");
    }
    if constexpr (requires { config.config_12x; }) {
        config.config_12x = typename decltype(config.config_12x)::value_type{.enable_pack_rvv = enable_pack_rvv};
    } else {
        TT_FATAL(!enable_pack_rvv, "ComputeHardwareConfig: enable_pack_rvv needs a newer tt-metal");
    }
}

void bind_data_movement_config(nb::module_& mod) {
    using DM = exp::DataMovementHardwareConfig;
    export_enum<exp::DedicatedNocSharing>(mod, "DedicatedNocSharing");
    auto dm_config = nb::class_<DM>(mod, "DataMovementHardwareConfig", R"doc(
        Hardware config of a Metal 2.0 data movement kernel.

        config_1xx (DataMovement1XXConfig) is the Wormhole / Blackhole part: the RISC-V processor and NoC. Metal 2.0
        places data movement kernels itself on Quasar. ``ttnn.experimental.create_reader_datamovement_config()`` /
        ``create_writer_datamovement_config()`` give the conventional reader / writer placement.

        config_2xx (DataMovement2XXConfig) is the Quasar part: by default a dataflow buffer's credits are posted
        implicitly when the kernel's NoC transfers into or out of it complete. ``disable_dfb_implicit_sync_for``
        (unique_ids of DFBs the kernel binds) or ``disable_dfb_implicit_sync_for_all`` opts the kernel out, so its
        reserve_back / push_back (or wait_front / pop_front) alone move the credits. All producer (or all consumer)
        kernels of a DFB must agree.

        Reading config_1xx or config_2xx returns a copy: assign a whole config to change it.

        Deprecated: the keywords processor, noc, disable_dfb_implicit_sync_for_all and disable_dfb_implicit_sync_for,
        and reader() / writer().
    )doc");

    const DM::DataMovement1XXConfig dm_1xx_defaults{};
    nb::class_<DM::DataMovement1XXConfig>(dm_config, "DataMovement1XXConfig")
        .def(
            "__init__",
            [](DM::DataMovement1XXConfig* self,
               tt::tt_metal::DataMovementProcessor processor,
               tt::tt_metal::NOC noc,
               tt::tt_metal::NOC_MODE noc_mode,
               exp::DedicatedNocSharing dedicated_noc_sharing) {
                new (self) DM::DataMovement1XXConfig{
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
        .def_rw("processor", &DM::DataMovement1XXConfig::processor)
        .def_rw("noc", &DM::DataMovement1XXConfig::noc)
        .def_rw("noc_mode", &DM::DataMovement1XXConfig::noc_mode)
        .def_rw("dedicated_noc_sharing", &DM::DataMovement1XXConfig::dedicated_noc_sharing);

    nb::class_<DM::DataMovement2XXConfig>(dm_config, "DataMovement2XXConfig")
        .def(
            "__init__",
            [](DM::DataMovement2XXConfig* self,
               const std::vector<std::string>& disable_dfb_implicit_sync_for,
               bool disable_dfb_implicit_sync_for_all) {
                new (self) DM::DataMovement2XXConfig{
                    make_dm_2xx_config(disable_dfb_implicit_sync_for_all, disable_dfb_implicit_sync_for)};
            },
            nb::kw_only(),
            nb::arg("disable_dfb_implicit_sync_for") = std::vector<std::string>{},
            nb::arg("disable_dfb_implicit_sync_for_all") = false)
        .def_prop_rw(
            "disable_dfb_implicit_sync_for",
            [](const DM::DataMovement2XXConfig& self) { return from_names(self.disable_dfb_implicit_sync_for); },
            [](DM::DataMovement2XXConfig& self, const std::vector<std::string>& names) {
                self.disable_dfb_implicit_sync_for = to_names<exp::DFBSpecName>(names);
            })
        .def_rw("disable_dfb_implicit_sync_for_all", &DM::DataMovement2XXConfig::disable_dfb_implicit_sync_for_all);

    dm_config
        .def(
            "__init__",
            [](DM* self,
               std::optional<DM::DataMovement1XXConfig> config_1xx,
               std::optional<DM::DataMovement2XXConfig> config_2xx) {
                new (self) DM{.config_1xx = std::move(config_1xx), .config_2xx = std::move(config_2xx)};
            },
            nb::kw_only(),
            nb::arg("config_1xx") = nb::none(),
            nb::arg("config_2xx") = nb::none())
        .def(
            "__init__",
            [](DM* self,
               std::optional<tt::tt_metal::DataMovementProcessor> processor,
               std::optional<tt::tt_metal::NOC> noc,
               bool disable_dfb_implicit_sync_for_all,
               const std::vector<std::string>& disable_dfb_implicit_sync_for) {
                warn_deprecated(
                    "DataMovementHardwareConfig(processor=, noc=, disable_dfb_implicit_sync_for...=) is deprecated: "
                    "use config_1xx=DataMovement1XXConfig(...) and config_2xx=DataMovement2XXConfig(...)");
                TT_FATAL(
                    processor.has_value() == noc.has_value(),
                    "DataMovementHardwareConfig: give both processor and noc, or neither");
                new (self) DM{};
                if (processor.has_value()) {
                    self->config_1xx = DM::DataMovement1XXConfig{.processor = *processor, .noc = *noc};
                }
                self->config_2xx = make_dm_2xx_config(disable_dfb_implicit_sync_for_all, disable_dfb_implicit_sync_for);
            },
            nb::kw_only(),
            nb::arg("processor") = nb::none(),
            nb::arg("noc") = nb::none(),
            nb::arg("disable_dfb_implicit_sync_for_all") = false,
            nb::arg("disable_dfb_implicit_sync_for") = std::vector<std::string>{})
        .def_rw("config_1xx", &DM::config_1xx, nb::rv_policy::copy, nb::for_setter(nb::arg("value").none()))
        .def_rw("config_2xx", &DM::config_2xx, nb::rv_policy::copy, nb::for_setter(nb::arg("value").none()))
        .def_static(
            "reader",
            [](bool disable_dfb_implicit_sync_for_all, const std::vector<std::string>& disable_dfb_implicit_sync_for) {
                warn_deprecated(
                    "DataMovementHardwareConfig.reader() is deprecated: use "
                    "ttnn.experimental.create_reader_datamovement_config()");
                auto hw = exp::CreateReaderDataMovementConfig();
                hw.config_2xx = make_dm_2xx_config(disable_dfb_implicit_sync_for_all, disable_dfb_implicit_sync_for);
                return hw;
            },
            nb::kw_only(),
            nb::arg("disable_dfb_implicit_sync_for_all") = false,
            nb::arg("disable_dfb_implicit_sync_for") = std::vector<std::string>{})
        .def_static(
            "writer",
            [](bool disable_dfb_implicit_sync_for_all, const std::vector<std::string>& disable_dfb_implicit_sync_for) {
                warn_deprecated(
                    "DataMovementHardwareConfig.writer() is deprecated: use "
                    "ttnn.experimental.create_writer_datamovement_config()");
                auto hw = exp::CreateWriterDataMovementConfig();
                hw.config_2xx = make_dm_2xx_config(disable_dfb_implicit_sync_for_all, disable_dfb_implicit_sync_for);
                return hw;
            },
            nb::kw_only(),
            nb::arg("disable_dfb_implicit_sync_for_all") = false,
            nb::arg("disable_dfb_implicit_sync_for") = std::vector<std::string>{});

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
}

// A template so that the Compute12XConfig and RVV branches, which older tt-metal lacks, are discarded there.
template <typename Compute = exp::ComputeHardwareConfig>
void bind_compute_config(nb::module_& mod) {
    using Compute1XX = typename Compute::Compute1XXConfig;
    using Compute2XX = typename Compute::Compute2XXConfig;
    auto compute_config = nb::class_<Compute>(mod, "ComputeHardwareConfig", R"doc(
        Hardware config of a Metal 2.0 compute kernel.

        Args:
            fpu_math_fidelity (ttnn.MathFidelity): FPU math fidelity.
            sfpu_precision_mode (ttnn.Precision): SFPU precision (the legacy math_approx_mode).
            enable_32_bit_dest (bool): 32-bit dest register (the legacy fp32_dest_acc_en).
            double_buffer_dest (bool): Double-buffered dest register (the inverse of the legacy dst_full_sync_en).
            unpack_modes (Dict[str, ttnn.UnpackMode]): Unpack mode per consumed DataflowBufferSpec unique_id. On
                Wormhole / Blackhole, a Float32 dataflow buffer read with a 32-bit dest needs an entry.
            config_1xx (Compute1XXConfig, optional): Wormhole / Blackhole part (bfp_pack_precision_mode).
            config_12x (Compute12XConfig, optional): Blackhole part (enable_pack_rvv).
            config_2xx (Compute2XXConfig, optional): Quasar part (enable_unpack_rvv).

        Reading a config_* returns a copy: assign a whole config to change it. ``from_compute_kernel_config``
        converts a ttnn compute kernel config (e.g. ttnn.WormholeComputeKernelConfig).

        Deprecated: the keywords math_fidelity, bfp_pack_precision_mode, enable_unpack_rvv and enable_pack_rvv, and
        the math_fidelity attribute.
    )doc");

    const Compute1XX compute_1xx_defaults{};
    nb::class_<Compute1XX>(compute_config, "Compute1XXConfig")
        .def(
            "__init__",
            [](Compute1XX* self, tt::tt_metal::Precision bfp_pack_precision_mode) {
                new (self) Compute1XX{.bfp_pack_precision_mode = bfp_pack_precision_mode};
            },
            nb::kw_only(),
            nb::arg("bfp_pack_precision_mode") = compute_1xx_defaults.bfp_pack_precision_mode)
        .def_rw("bfp_pack_precision_mode", &Compute1XX::bfp_pack_precision_mode);

    if constexpr (requires { typename Compute::Compute12XConfig; }) {
        using Compute12X = typename Compute::Compute12XConfig;
        nb::class_<Compute12X>(compute_config, "Compute12XConfig")
            .def(
                "__init__",
                [](Compute12X* self, bool enable_pack_rvv) {
                    new (self) Compute12X{.enable_pack_rvv = enable_pack_rvv};
                },
                nb::kw_only(),
                nb::arg("enable_pack_rvv") = false)
            .def_rw("enable_pack_rvv", &Compute12X::enable_pack_rvv);
    }

    auto compute_2xx = nb::class_<Compute2XX>(compute_config, "Compute2XXConfig");
    if constexpr (requires(Compute2XX config) { config.enable_unpack_rvv; }) {
        compute_2xx
            .def(
                "__init__",
                [](Compute2XX* self, bool enable_unpack_rvv) {
                    new (self) Compute2XX{.enable_unpack_rvv = enable_unpack_rvv};
                },
                nb::kw_only(),
                nb::arg("enable_unpack_rvv") = false)
            .def_rw("enable_unpack_rvv", &Compute2XX::enable_unpack_rvv);
    } else {
        compute_2xx.def(nb::init<>());
    }

    const Compute compute_defaults{};
    compute_config
        .def(
            "__init__",
            [](Compute* self,
               tt::tt_metal::MathFidelity fpu_math_fidelity,
               tt::tt_metal::Precision sfpu_precision_mode,
               bool enable_32_bit_dest,
               bool double_buffer_dest,
               const std::map<std::string, tt::tt_metal::UnpackMode>& unpack_modes,
               std::optional<Compute1XX> config_1xx,
               nb::handle config_12x,
               std::optional<Compute2XX> config_2xx) {
                new (self) Compute{
                    .fpu_math_fidelity = fpu_math_fidelity,
                    .sfpu_precision_mode = sfpu_precision_mode,
                    .enable_32_bit_dest = enable_32_bit_dest,
                    .double_buffer_dest = double_buffer_dest,
                    .config_1xx = std::move(config_1xx),
                    .config_2xx = std::move(config_2xx),
                };
                for (const auto& [dfb_spec_name, mode] : unpack_modes) {
                    self->unpack_modes.emplace(exp::DFBSpecName{dfb_spec_name}, mode);
                }
                if (!config_12x.is_none()) {
                    if constexpr (requires { typename Compute::Compute12XConfig; }) {
                        self->config_12x = nb::cast<typename Compute::Compute12XConfig>(config_12x);
                    } else {
                        TT_THROW("ComputeHardwareConfig: config_12x needs a newer tt-metal");
                    }
                }
            },
            nb::kw_only(),
            nb::arg("fpu_math_fidelity") = compute_defaults.fpu_math_fidelity,
            nb::arg("sfpu_precision_mode") = compute_defaults.sfpu_precision_mode,
            nb::arg("enable_32_bit_dest") = compute_defaults.enable_32_bit_dest,
            nb::arg("double_buffer_dest") = compute_defaults.double_buffer_dest,
            nb::arg("unpack_modes") = std::map<std::string, tt::tt_metal::UnpackMode>{},
            nb::arg("config_1xx") = nb::none(),
            nb::arg("config_12x").none() = nb::none(),
            nb::arg("config_2xx") = nb::none())
        .def(
            "__init__",
            [](Compute* self,
               tt::tt_metal::MathFidelity math_fidelity,
               tt::tt_metal::Precision sfpu_precision_mode,
               bool enable_32_bit_dest,
               bool double_buffer_dest,
               const std::map<std::string, tt::tt_metal::UnpackMode>& unpack_modes,
               tt::tt_metal::Precision bfp_pack_precision_mode,
               bool enable_unpack_rvv,
               bool enable_pack_rvv) {
                warn_deprecated(
                    "ComputeHardwareConfig(math_fidelity=, bfp_pack_precision_mode=, enable_*_rvv=) is deprecated: use "
                    "fpu_math_fidelity= and config_1xx= / config_12x= / config_2xx=");
                new (self) Compute{
                    .fpu_math_fidelity = math_fidelity,
                    .sfpu_precision_mode = sfpu_precision_mode,
                    .enable_32_bit_dest = enable_32_bit_dest,
                    .double_buffer_dest = double_buffer_dest,
                    .config_1xx = Compute1XX{.bfp_pack_precision_mode = bfp_pack_precision_mode},
                };
                for (const auto& [dfb_spec_name, mode] : unpack_modes) {
                    self->unpack_modes.emplace(exp::DFBSpecName{dfb_spec_name}, mode);
                }
                set_rvv_options(*self, enable_unpack_rvv, enable_pack_rvv);
            },
            nb::kw_only(),
            nb::arg("math_fidelity") = tt::tt_metal::MathFidelity::HiFi4,
            nb::arg("sfpu_precision_mode") = tt::tt_metal::Precision::Precise,
            nb::arg("enable_32_bit_dest") = false,
            nb::arg("double_buffer_dest") = true,
            nb::arg("unpack_modes") = std::map<std::string, tt::tt_metal::UnpackMode>{},
            nb::arg("bfp_pack_precision_mode") = tt::tt_metal::Precision::Approximate,
            nb::arg("enable_unpack_rvv") = false,
            nb::arg("enable_pack_rvv") = false)
        .def_static(
            "from_compute_kernel_config",
            [](const ttnn::DeviceComputeKernelConfig& config) {
                auto hw = ttnn::to_compute_hardware_config(config);
                set_rvv_options(hw, /*enable_unpack_rvv=*/false, /*enable_pack_rvv=*/false);
                return hw;
            },
            nb::arg("compute_kernel_config"))
        .def_rw("fpu_math_fidelity", &Compute::fpu_math_fidelity)
        .def_rw("math_fidelity", &Compute::fpu_math_fidelity)
        .def_rw("sfpu_precision_mode", &Compute::sfpu_precision_mode)
        .def_rw("enable_32_bit_dest", &Compute::enable_32_bit_dest)
        .def_rw("double_buffer_dest", &Compute::double_buffer_dest)
        .def_prop_rw(
            "unpack_modes",
            [](const Compute& self) {
                std::map<std::string, tt::tt_metal::UnpackMode> modes;
                for (const auto& [name, mode] : self.unpack_modes) {
                    modes.emplace(*name, mode);
                }
                return modes;
            },
            [](Compute& self, const std::map<std::string, tt::tt_metal::UnpackMode>& modes) {
                self.unpack_modes = {};
                for (const auto& [name, mode] : modes) {
                    self.unpack_modes.emplace(exp::DFBSpecName{name}, mode);
                }
            })
        .def_rw("config_1xx", &Compute::config_1xx, nb::rv_policy::copy, nb::for_setter(nb::arg("value").none()))
        .def_rw("config_2xx", &Compute::config_2xx, nb::rv_policy::copy, nb::for_setter(nb::arg("value").none()))
        .def(
            "set_unpack_mode",
            [](Compute& self, const std::string& dfb_spec_name, tt::tt_metal::UnpackMode mode) {
                self.unpack_modes[exp::DFBSpecName{dfb_spec_name}] = mode;
            },
            nb::arg("dfb_spec_name"),
            nb::arg("mode"),
            "Set the unpack mode of one consumed DataflowBufferSpec.");
    if constexpr (requires { typename Compute::Compute12XConfig; }) {
        compute_config.def_rw(
            "config_12x", &Compute::config_12x, nb::rv_policy::copy, nb::for_setter(nb::arg("value").none()));
    }
}

void bind_hardware_configs(nb::module_& mod) {
    bind_data_movement_config(mod);
    export_enum<tt::tt_metal::Precision>(mod, "Precision");
    export_enum<tt::tt_metal::UnpackMode>(mod, "UnpackMode");
    bind_compute_config(mod);
}

void bind_resources(nb::module_& mod) {
    nb::class_<exp::DFBAdvancedOptions>(mod, "DFBAdvancedOptions", R"doc(
        Rarely used DataflowBufferSpec options. Python has alias_with and borrowed_offset only; the others are C++
        only.

        Args:
            alias_with (List[str]): unique_ids of DFBs that share this DFB's memory. Every member of an alias group
                lists every other member, and all have the same total size and nodes.
            borrowed_offset (int): Byte offset of a borrowed_from buffer within its tensor's shard on each core, so
                several buffers can share one tensor. A multiple of the L1 alignment, and the buffer must fit in the
                shard after it. Nothing checks that buffers in one tensor do not overlap. Defaults to 0.
    )doc")
        .def(
            "__init__",
            [](exp::DFBAdvancedOptions* self, const std::vector<std::string>& alias_with, uint32_t borrowed_offset) {
                new (self) exp::DFBAdvancedOptions{};
                self->alias_with = to_names<exp::DFBSpecName>(alias_with);
                self->borrowed_offset = borrowed_offset;
            },
            nb::kw_only(),
            nb::arg("alias_with") = std::vector<std::string>{},
            nb::arg("borrowed_offset") = 0)
        .def_prop_rw(
            "alias_with",
            [](const exp::DFBAdvancedOptions& self) { return from_names(self.alias_with); },
            [](exp::DFBAdvancedOptions& self, const std::vector<std::string>& names) {
                self.alias_with = to_names<exp::DFBSpecName>(names);
            })
        .def_rw("borrowed_offset", &exp::DFBAdvancedOptions::borrowed_offset);

    const exp::DataflowBufferSpec dfb_defaults{};
    nb::class_<exp::DataflowBufferSpec>(mod, "DataflowBufferSpec", R"doc(
        A Metal 2.0 dataflow buffer (the successor of a circular buffer).

        Args:
            unique_id (str): Name the kernels' DFBBindings refer to.
            entry_size (int): Bytes per entry (e.g. one tile).
            num_entries (int): Number of entries.
            data_format_metadata (ttnn.DataType, optional): Format of the entries, needed when a compute kernel reads
                or writes the buffer. ttnn.int8 is stored as UInt8, as in ttnn's own ops, and reads back as uint8.
            tile_format_metadata (ttnn.Tile, optional): Tile shape of the entries.
            borrowed_from (str, optional): unique_id of an L1-resident TensorParameter. The buffer then lives in that
                tensor's shard instead of its own L1 allocation, so its data is already in place: the producer only
                pushes entries, without copying them.
            advanced_options (DFBAdvancedOptions, optional): e.g. borrowed_offset, to place a borrowed_from buffer
                within its tensor. Refers into this object.

        Deprecated: the keywords data_format and tile.
    )doc")
        .def(
            "__init__",
            [](exp::DataflowBufferSpec* self,
               const std::string& unique_id,
               uint32_t entry_size,
               uint32_t num_entries,
               std::optional<tt::tt_metal::DataType> data_format_metadata,
               std::optional<tt::tt_metal::Tile> tile_format_metadata,
               std::optional<std::string> borrowed_from,
               const exp::DFBAdvancedOptions& advanced_options) {
                new (self) exp::DataflowBufferSpec{
                    .unique_id = exp::DFBSpecName{unique_id},
                    .entry_size = entry_size,
                    .num_entries = num_entries,
                    .data_format_metadata = to_data_format(data_format_metadata),
                    .tile_format_metadata = tile_format_metadata,
                    .borrowed_from =
                        borrowed_from.has_value() ? std::optional(exp::TensorParamName{*borrowed_from}) : std::nullopt,
                    .advanced_options = advanced_options,
                };
            },
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("entry_size") = dfb_defaults.entry_size,
            nb::arg("num_entries") = dfb_defaults.num_entries,
            nb::arg("data_format_metadata") = nb::none(),
            nb::arg("tile_format_metadata") = nb::none(),
            nb::arg("borrowed_from") = nb::none(),
            nb::arg("advanced_options") = exp::DFBAdvancedOptions{})
        .def(
            "__init__",
            [](exp::DataflowBufferSpec* self,
               const std::string& unique_id,
               uint32_t entry_size,
               uint32_t num_entries,
               std::optional<tt::tt_metal::DataType> data_format,
               std::optional<tt::tt_metal::Tile> tile,
               std::optional<std::string> borrowed_from,
               const exp::DFBAdvancedOptions& advanced_options) {
                warn_deprecated(
                    "DataflowBufferSpec(data_format=, tile=) is deprecated: use data_format_metadata= and "
                    "tile_format_metadata=");
                new (self) exp::DataflowBufferSpec{
                    .unique_id = exp::DFBSpecName{unique_id},
                    .entry_size = entry_size,
                    .num_entries = num_entries,
                    .data_format_metadata =
                        data_format.has_value()
                            ? std::optional(tt::tt_metal::datatype_to_dataformat_converter(*data_format))
                            : std::nullopt,
                    .tile_format_metadata = tile,
                    .borrowed_from =
                        borrowed_from.has_value() ? std::optional(exp::TensorParamName{*borrowed_from}) : std::nullopt,
                    .advanced_options = advanced_options,
                };
            },
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("entry_size"),
            nb::arg("num_entries"),
            nb::arg("data_format") = nb::none(),
            nb::arg("tile") = nb::none(),
            nb::arg("borrowed_from") = nb::none(),
            nb::arg("advanced_options") = exp::DFBAdvancedOptions{})
        .def_prop_ro("unique_id", [](const exp::DataflowBufferSpec& self) { return *self.unique_id; })
        .def_rw("entry_size", &exp::DataflowBufferSpec::entry_size)
        .def_rw("num_entries", &exp::DataflowBufferSpec::num_entries)
        .def_prop_rw(
            "data_format_metadata",
            [](const exp::DataflowBufferSpec& self) { return to_data_type(self.data_format_metadata); },
            [](exp::DataflowBufferSpec& self, std::optional<tt::tt_metal::DataType> data_format) {
                self.data_format_metadata = to_data_format(data_format);
            },
            nb::for_setter(nb::arg("value").none()))
        .def_rw(
            "tile_format_metadata",
            &exp::DataflowBufferSpec::tile_format_metadata,
            nb::rv_policy::copy,
            nb::for_setter(nb::arg("value").none()))
        .def_prop_rw(
            "borrowed_from",
            [](const exp::DataflowBufferSpec& self) -> std::optional<std::string> {
                return self.borrowed_from.has_value() ? std::optional(**self.borrowed_from) : std::nullopt;
            },
            [](exp::DataflowBufferSpec& self, std::optional<std::string> borrowed_from) {
                self.borrowed_from =
                    borrowed_from.has_value() ? std::optional(exp::TensorParamName{*borrowed_from}) : std::nullopt;
            },
            nb::for_setter(nb::arg("value").none()))
        .def_rw("advanced_options", &exp::DataflowBufferSpec::advanced_options);

    nb::class_<exp::TensorParameter>(mod, "TensorParameter", R"doc(
        A tensor the program reads or writes, named so kernels can bind it with a TensorBinding.

        Args:
            unique_id (str): Name the kernels' TensorBindings refer to.
            spec (ttnn.TensorSpec): Spec of the tensors bound to this parameter, e.g. ``tensor.spec``.
    )doc")
        .def(
            "__init__",
            [](exp::TensorParameter* self, const std::string& unique_id, const tt::tt_metal::TensorSpec& spec) {
                new (self) exp::TensorParameter{.unique_id = exp::TensorParamName{unique_id}, .spec = spec};
            },
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("spec"))
        .def_prop_ro("unique_id", [](const exp::TensorParameter& self) { return *self.unique_id; })
        .def_ro("spec", &exp::TensorParameter::spec);

    const exp::ScratchpadSpec scratchpad_defaults{};
    nb::class_<exp::ScratchpadSpec>(mod, "ScratchpadSpec", R"doc(
        Per-node L1 memory a kernel reads and writes freely for the life of the program.

        Args:
            unique_id (str): Name the kernels' ScratchpadBindings refer to.
            size_per_node (int): Bytes reserved on each node of the binding kernels.
            data_format_metadata (ttnn.DataType, optional): Format of the contents, needed when a compute kernel uses
                it.
            tile_format_metadata (ttnn.Tile, optional): Tile shape of the contents.

        Each node's scratchpad may be bound by only one kernel (multi-binding is not exposed).

        Deprecated: the keywords data_format and tile.
    )doc")
        .def(
            "__init__",
            [](exp::ScratchpadSpec* self,
               const std::string& unique_id,
               uint32_t size_per_node,
               std::optional<tt::tt_metal::DataType> data_format_metadata,
               std::optional<tt::tt_metal::Tile> tile_format_metadata) {
                new (self) exp::ScratchpadSpec{
                    .unique_id = exp::ScratchpadSpecName{unique_id},
                    .size_per_node = size_per_node,
                    .data_format_metadata = to_data_format(data_format_metadata),
                    .tile_format_metadata = tile_format_metadata,
                };
            },
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("size_per_node") = scratchpad_defaults.size_per_node,
            nb::arg("data_format_metadata") = nb::none(),
            nb::arg("tile_format_metadata") = nb::none())
        .def(
            "__init__",
            [](exp::ScratchpadSpec* self,
               const std::string& unique_id,
               uint32_t size_per_node,
               std::optional<tt::tt_metal::DataType> data_format,
               std::optional<tt::tt_metal::Tile> tile) {
                warn_deprecated(
                    "ScratchpadSpec(data_format=, tile=) is deprecated: use data_format_metadata= and "
                    "tile_format_metadata=");
                new (self) exp::ScratchpadSpec{
                    .unique_id = exp::ScratchpadSpecName{unique_id},
                    .size_per_node = size_per_node,
                    .data_format_metadata =
                        data_format.has_value()
                            ? std::optional(tt::tt_metal::datatype_to_dataformat_converter(*data_format))
                            : std::nullopt,
                    .tile_format_metadata = tile,
                };
            },
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("size_per_node"),
            nb::arg("data_format") = nb::none(),
            nb::arg("tile") = nb::none())
        .def_prop_ro("unique_id", [](const exp::ScratchpadSpec& self) { return *self.unique_id; })
        .def_rw("size_per_node", &exp::ScratchpadSpec::size_per_node)
        .def_prop_rw(
            "data_format_metadata",
            [](const exp::ScratchpadSpec& self) { return to_data_type(self.data_format_metadata); },
            [](exp::ScratchpadSpec& self, std::optional<tt::tt_metal::DataType> data_format) {
                self.data_format_metadata = to_data_format(data_format);
            },
            nb::for_setter(nb::arg("value").none()))
        .def_rw(
            "tile_format_metadata",
            &exp::ScratchpadSpec::tile_format_metadata,
            nb::rv_policy::copy,
            nb::for_setter(nb::arg("value").none()));

    nb::class_<exp::SemaphoreSpec>(mod, "SemaphoreSpec", R"doc(
        A semaphore on each of a set of nodes, starting at 0.

        Kernels on those nodes reach it as ``Semaphore s(sem::<accessor_name>)`` through a SemaphoreBinding, and
        can increment another node's instance over the NoC. A non-zero initial value is not offered: it is
        deprecated, and Quasar rejects it.

        Args:
            unique_id (str): Name the kernels' SemaphoreBindings refer to.
            target_nodes (ttnn.CoreCoord, ttnn.CoreRange or ttnn.CoreRangeSet): Nodes that get an instance.
    )doc")
        .def(
            "__init__",
            [](exp::SemaphoreSpec* self, const std::string& unique_id, const Nodes& target_nodes) {
                new (self) exp::SemaphoreSpec{
                    .unique_id = exp::SemaphoreSpecName{unique_id},
                    .target_nodes = target_nodes,
                };
            },
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("target_nodes"))
        .def_prop_ro("unique_id", [](const exp::SemaphoreSpec& self) { return *self.unique_id; })
        .def_rw("target_nodes", &exp::SemaphoreSpec::target_nodes, nb::rv_policy::copy);
}

void bind_kernel_spec(nb::module_& mod) {
    export_enum<exp::DFBEndpointType>(mod, "DFBEndpointType");
    export_enum<exp::DFBAccessPattern>(mod, "DFBAccessPattern");

    nb::class_<exp::DFBBinding>(mod, "DFBBinding", R"doc(
        Binds a DataflowBufferSpec to a kernel. The kernel reaches it as ``dfb::<accessor_name>``.
    )doc")
        .def(
            "__init__",
            [](exp::DFBBinding* self,
               const std::string& dfb_spec_name,
               const std::string& accessor_name,
               exp::DFBEndpointType endpoint_type,
               exp::DFBAccessPattern access_pattern,
               uint32_t block_size) {
                new (self) exp::DFBBinding{
                    .dfb_spec_name = exp::DFBSpecName{dfb_spec_name},
                    .accessor_name = accessor_name,
                    .endpoint_type = endpoint_type,
                    .access_pattern = access_pattern,
                    .block_size = block_size,
                };
            },
            nb::kw_only(),
            nb::arg("dfb_spec_name"),
            nb::arg("accessor_name"),
            nb::arg("endpoint_type"),
            nb::arg("access_pattern") = exp::DFBAccessPattern::STRIDED,
            nb::arg("block_size") = 0);

    nb::class_<exp::TensorBinding>(mod, "TensorBinding", R"doc(
        Binds a TensorParameter to a kernel. The kernel reaches it as ``tensor::<accessor_name>``.
    )doc")
        .def(
            "__init__",
            [](exp::TensorBinding* self, const std::string& tensor_parameter_name, const std::string& accessor_name) {
                new (self) exp::TensorBinding{
                    .tensor_parameter_name = exp::TensorParamName{tensor_parameter_name},
                    .accessor_name = accessor_name,
                };
            },
            nb::kw_only(),
            nb::arg("tensor_parameter_name"),
            nb::arg("accessor_name"));

    nb::class_<exp::SemaphoreBinding>(mod, "SemaphoreBinding", R"doc(
        Binds a SemaphoreSpec to a kernel. The kernel reaches it as ``sem::<accessor_name>``.
    )doc")
        .def(
            "__init__",
            [](exp::SemaphoreBinding* self, const std::string& semaphore_spec_name, const std::string& accessor_name) {
                new (self) exp::SemaphoreBinding{
                    .semaphore_spec_name = exp::SemaphoreSpecName{semaphore_spec_name},
                    .accessor_name = accessor_name,
                };
            },
            nb::kw_only(),
            nb::arg("semaphore_spec_name"),
            nb::arg("accessor_name"));

    nb::class_<exp::ScratchpadBinding>(mod, "ScratchpadBinding", R"doc(
        Binds a ScratchpadSpec to a kernel. The kernel reaches it as ``scratch::<accessor_name>``.
    )doc")
        .def(
            "__init__",
            [](exp::ScratchpadBinding* self,
               const std::string& scratchpad_spec_name,
               const std::string& accessor_name) {
                new (self) exp::ScratchpadBinding{
                    .scratchpad_spec_name = exp::ScratchpadSpecName{scratchpad_spec_name},
                    .accessor_name = accessor_name,
                };
            },
            nb::kw_only(),
            nb::arg("scratchpad_spec_name"),
            nb::arg("accessor_name"));

    auto kernel_spec = nb::class_<exp::KernelSpec>(mod, "KernelSpec", R"doc(
        A Metal 2.0 kernel.

        Args:
            unique_id (str): Name the WorkUnitSpecs and KernelRunArgs refer to.
            source (str or KernelSpec.SourceCode): Path of the kernel source file, resolved like any tt-metal kernel
                path, or the source itself.
            hw_config (DataMovementHardwareConfig or ComputeHardwareConfig): Picks the kernel type.
            num_threads (int): Threads per node (Quasar runs several data movement or compute threads).
            compiler_options (KernelSpec.CompilerOptions): include_paths, defines and opt_level.
            dfb_bindings (List[DFBBinding]): Dataflow buffers the kernel uses (``dfb::<name>``).
            semaphore_bindings (List[SemaphoreBinding]): Semaphores the kernel uses (``sem::<name>``).
            scratchpad_bindings (List[ScratchpadBinding]): Scratchpads the kernel uses (``scratch::<name>``).
            tensor_bindings (List[TensorBinding]): Tensors the kernel uses (``tensor::<name>``).
            compile_time_args (Dict[str, int]): Named compile-time args.
            runtime_arg_schema (KernelSpec.RuntimeArgSchema): Names of the per-node and common runtime args
                (``args::<name>``).

        Deprecated: the keywords source_code, runtime_arg_names, common_runtime_arg_names, include_paths, defines and
        opt_level.
    )doc");

    nb::class_<exp::KernelSpec::SourceCode>(kernel_spec, "SourceCode", "Kernel source code, given inline.")
        .def(
            "__init__",
            [](exp::KernelSpec::SourceCode* self, std::string code) {
                new (self) exp::KernelSpec::SourceCode{.code = std::move(code)};
            },
            nb::kw_only(),
            nb::arg("code"))
        .def_rw("code", &exp::KernelSpec::SourceCode::code);

    nb::class_<exp::KernelSpec::CompilerOptions>(kernel_spec, "CompilerOptions")
        .def(
            "__init__",
            [](exp::KernelSpec::CompilerOptions* self,
               const std::vector<std::string>& include_paths,
               const StringMap& defines,
               tt::tt_metal::KernelBuildOptLevel opt_level) {
                new (self) exp::KernelSpec::CompilerOptions{};
                for (const auto& include_path : include_paths) {
                    self->include_paths.emplace_back(include_path);
                }
                for (const auto& [name, value] : defines) {
                    self->defines.emplace(name, value);
                }
                self->opt_level = opt_level;
            },
            nb::kw_only(),
            nb::arg("include_paths") = std::vector<std::string>{},
            nb::arg("defines") = StringMap{},
            nb::arg("opt_level") = tt::tt_metal::KernelBuildOptLevel::O2)
        .def_prop_ro(
            "include_paths",
            [](const exp::KernelSpec::CompilerOptions& self) {
                std::vector<std::string> paths;
                for (const auto& path : self.include_paths) {
                    paths.push_back(path.string());
                }
                return paths;
            })
        .def_prop_ro(
            "defines",
            [](const exp::KernelSpec::CompilerOptions& self) {
                return StringMap(self.defines.begin(), self.defines.end());
            })
        .def_rw("opt_level", &exp::KernelSpec::CompilerOptions::opt_level);

    nb::class_<exp::KernelSpec::RuntimeArgSchema>(kernel_spec, "RuntimeArgSchema")
        .def(
            "__init__",
            [](exp::KernelSpec::RuntimeArgSchema* self,
               std::vector<std::string> runtime_arg_names,
               std::vector<std::string> common_runtime_arg_names) {
                new (self) exp::KernelSpec::RuntimeArgSchema{
                    .runtime_arg_names = std::move(runtime_arg_names),
                    .common_runtime_arg_names = std::move(common_runtime_arg_names)};
            },
            nb::kw_only(),
            nb::arg("runtime_arg_names") = std::vector<std::string>{},
            nb::arg("common_runtime_arg_names") = std::vector<std::string>{})
        .def_rw("runtime_arg_names", &exp::KernelSpec::RuntimeArgSchema::runtime_arg_names)
        .def_rw("common_runtime_arg_names", &exp::KernelSpec::RuntimeArgSchema::common_runtime_arg_names);

    kernel_spec
        .def(
            "__init__",
            [](exp::KernelSpec* self,
               const std::string& unique_id,
               const std::variant<std::filesystem::path, exp::KernelSpec::SourceCode>& source,
               const HwConfig& hw_config,
               uint32_t num_threads,
               const exp::KernelSpec::CompilerOptions& compiler_options,
               const std::vector<exp::DFBBinding>& dfb_bindings,
               const std::vector<exp::SemaphoreBinding>& semaphore_bindings,
               const std::vector<exp::ScratchpadBinding>& scratchpad_bindings,
               const std::vector<exp::TensorBinding>& tensor_bindings,
               const ArgMap& compile_time_args,
               const exp::KernelSpec::RuntimeArgSchema& runtime_arg_schema) {
                new (self) exp::KernelSpec{
                    .unique_id = exp::KernelSpecName{unique_id},
                    .source = source,
                    .num_threads = num_threads,
                    .compiler_options = compiler_options,
                    .dfb_bindings = dfb_bindings,
                    .semaphore_bindings = semaphore_bindings,
                    .scratchpad_bindings = scratchpad_bindings,
                    .tensor_bindings = tensor_bindings,
                    .runtime_arg_schema = runtime_arg_schema,
                    .hw_config = hw_config,
                };
                for (const auto& [name, value] : compile_time_args) {
                    self->compile_time_args.emplace(name, value);
                }
            },
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("source"),
            nb::arg("hw_config"),
            nb::arg("num_threads") = 1,
            nb::arg("compiler_options") = exp::KernelSpec::CompilerOptions{},
            nb::arg("dfb_bindings") = std::vector<exp::DFBBinding>{},
            nb::arg("semaphore_bindings") = std::vector<exp::SemaphoreBinding>{},
            nb::arg("scratchpad_bindings") = std::vector<exp::ScratchpadBinding>{},
            nb::arg("tensor_bindings") = std::vector<exp::TensorBinding>{},
            nb::arg("compile_time_args") = ArgMap{},
            nb::arg("runtime_arg_schema") = exp::KernelSpec::RuntimeArgSchema{})
        .def(
            "__init__",
            [](exp::KernelSpec* self,
               const std::string& unique_id,
               const std::optional<std::string>& source,
               const std::optional<std::string>& source_code,
               const HwConfig& hw_config,
               uint32_t num_threads,
               const std::vector<exp::DFBBinding>& dfb_bindings,
               const std::vector<exp::SemaphoreBinding>& semaphore_bindings,
               const std::vector<exp::ScratchpadBinding>& scratchpad_bindings,
               const std::vector<exp::TensorBinding>& tensor_bindings,
               const ArgMap& compile_time_args,
               const std::vector<std::string>& runtime_arg_names,
               const std::vector<std::string>& common_runtime_arg_names,
               const std::vector<std::string>& include_paths,
               const StringMap& defines,
               tt::tt_metal::KernelBuildOptLevel opt_level) {
                warn_deprecated(
                    "KernelSpec(source_code=, runtime_arg_names=, common_runtime_arg_names=, include_paths=, defines=, "
                    "opt_level=) is deprecated: use source=KernelSpec.SourceCode(...), "
                    "runtime_arg_schema=KernelSpec.RuntimeArgSchema(...) and "
                    "compiler_options=KernelSpec.CompilerOptions(...)");
                TT_FATAL(
                    source.has_value() != source_code.has_value(),
                    "KernelSpec '{}': give exactly one of source (a file path) and source_code",
                    unique_id);
                new (self) exp::KernelSpec{
                    .unique_id = exp::KernelSpecName{unique_id},
                    .num_threads = num_threads,
                    .dfb_bindings = dfb_bindings,
                    .semaphore_bindings = semaphore_bindings,
                    .scratchpad_bindings = scratchpad_bindings,
                    .tensor_bindings = tensor_bindings,
                    .runtime_arg_schema =
                        {.runtime_arg_names = runtime_arg_names, .common_runtime_arg_names = common_runtime_arg_names},
                    .hw_config = hw_config,
                };
                if (source.has_value()) {
                    self->source = std::filesystem::path(*source);
                } else {
                    self->source = exp::KernelSpec::SourceCode{.code = *source_code};
                }
                for (const auto& include_path : include_paths) {
                    self->compiler_options.include_paths.emplace_back(include_path);
                }
                for (const auto& [name, value] : defines) {
                    self->compiler_options.defines.emplace(name, value);
                }
                self->compiler_options.opt_level = opt_level;
                for (const auto& [name, value] : compile_time_args) {
                    self->compile_time_args.emplace(name, value);
                }
            },
            nb::kw_only(),
            nb::arg("unique_id"),
            nb::arg("source") = nb::none(),
            nb::arg("source_code") = nb::none(),
            nb::arg("hw_config"),
            nb::arg("num_threads") = 1,
            nb::arg("dfb_bindings") = std::vector<exp::DFBBinding>{},
            nb::arg("semaphore_bindings") = std::vector<exp::SemaphoreBinding>{},
            nb::arg("scratchpad_bindings") = std::vector<exp::ScratchpadBinding>{},
            nb::arg("tensor_bindings") = std::vector<exp::TensorBinding>{},
            nb::arg("compile_time_args") = ArgMap{},
            nb::arg("runtime_arg_names") = std::vector<std::string>{},
            nb::arg("common_runtime_arg_names") = std::vector<std::string>{},
            nb::arg("include_paths") = std::vector<std::string>{},
            nb::arg("defines") = StringMap{},
            nb::arg("opt_level") = tt::tt_metal::KernelBuildOptLevel::O2)
        .def_prop_ro("unique_id", [](const exp::KernelSpec& self) { return *self.unique_id; })
        .def_rw("num_threads", &exp::KernelSpec::num_threads)
        .def_rw("compiler_options", &exp::KernelSpec::compiler_options)
        .def_rw("runtime_arg_schema", &exp::KernelSpec::runtime_arg_schema)
        .def_rw("dfb_bindings", &exp::KernelSpec::dfb_bindings, nb::rv_policy::copy)
        .def_rw("semaphore_bindings", &exp::KernelSpec::semaphore_bindings, nb::rv_policy::copy)
        .def_rw("scratchpad_bindings", &exp::KernelSpec::scratchpad_bindings, nb::rv_policy::copy)
        .def_rw("tensor_bindings", &exp::KernelSpec::tensor_bindings, nb::rv_policy::copy)
        .def_rw("hw_config", &exp::KernelSpec::hw_config, nb::rv_policy::copy)
        .def_prop_ro("is_data_movement_kernel", &exp::KernelSpec::is_data_movement_kernel)
        .def_prop_ro("is_compute_kernel", &exp::KernelSpec::is_compute_kernel);

    // #63251 nests the bindings and their enums in KernelSpec / DFBBinding; the module-level names stay too.
    nb::object dfb_binding = mod.attr("DFBBinding");
    dfb_binding.attr("EndpointType") = mod.attr("DFBEndpointType");
    dfb_binding.attr("AccessPattern") = mod.attr("DFBAccessPattern");
    for (const char* name : {"DFBBinding", "SemaphoreBinding", "ScratchpadBinding", "TensorBinding"}) {
        kernel_spec.attr(name) = mod.attr(name);
    }

    auto def_binding_helper = [&mod](const char* name, auto make, const char* doc) {
        mod.def(
            name,
            [make](const std::string& dfb_spec_name, const std::string& accessor_name) {
                return make(exp::DFBSpecName{dfb_spec_name}, accessor_name);
            },
            nb::arg("dfb_spec_name"),
            nb::arg("accessor_name"),
            doc);
    };
    auto def_blocked_binding_helper = [&mod](const char* name, auto make, const char* doc) {
        mod.def(
            name,
            [make](const std::string& dfb_spec_name, const std::string& accessor_name, uint32_t block_size) {
                return make(exp::DFBSpecName{dfb_spec_name}, accessor_name, block_size);
            },
            nb::arg("dfb_spec_name"),
            nb::arg("accessor_name"),
            nb::arg("block_size"),
            doc);
    };
    def_binding_helper("producer_of", &exp::ProducerOf, "A producer binding with the STRIDED access pattern.");
    def_blocked_binding_helper(
        "blocked_producer_of", &exp::BlockedProducerOf, "A producer binding with the BLOCKED access pattern.");
    def_binding_helper(
        "consumer_of", &exp::ConsumerOf, "A consumer binding with the default (STRIDED) access pattern.");
    def_binding_helper(
        "strided_consumer_of", &exp::StridedConsumerOf, "A consumer binding with the STRIDED access pattern.");
    def_binding_helper("all_consumer_of", &exp::AllConsumerOf, "A consumer binding with the ALL access pattern.");
    def_blocked_binding_helper(
        "blocked_consumer_of", &exp::BlockedConsumerOf, "A consumer binding with the BLOCKED access pattern.");
}

void bind_program_spec(nb::module_& mod) {
    nb::class_<exp::WorkUnitSpec>(mod, "WorkUnitSpec", R"doc(
        Places a set of kernels on a set of nodes (cores).

        Args:
            name (str): Name of the work unit.
            kernels (List[str]): unique_ids of the KernelSpecs that run together on each node.
            target_nodes (ttnn.CoreCoord, ttnn.CoreRange or ttnn.CoreRangeSet): Nodes to run them on.
    )doc")
        .def(
            "__init__",
            [](exp::WorkUnitSpec* self,
               const std::string& name,
               const std::vector<std::string>& kernels,
               const Nodes& target_nodes) {
                new (self) exp::WorkUnitSpec{
                    .name = name, .kernels = to_names<exp::KernelSpecName>(kernels), .target_nodes = target_nodes};
            },
            nb::kw_only(),
            nb::arg("name") = std::string{},
            nb::arg("kernels") = std::vector<std::string>{},
            nb::arg("target_nodes"))
        .def_rw("name", &exp::WorkUnitSpec::name)
        .def_prop_rw(
            "kernels",
            [](const exp::WorkUnitSpec& self) { return from_names(self.kernels); },
            [](exp::WorkUnitSpec& self, const std::vector<std::string>& kernels) {
                self.kernels = to_names<exp::KernelSpecName>(kernels);
            })
        .def_rw("target_nodes", &exp::WorkUnitSpec::target_nodes, nb::rv_policy::copy);

    nb::class_<exp::ProgramSpec>(mod, "ProgramSpec", R"doc(
        A Metal 2.0 program, run with ``ttnn.experimental.generic_op(io_tensors, spec, run_args)``.

        Args:
            name (str): Name of the program.
            kernels (List[KernelSpec]): The kernels.
            dataflow_buffers (List[DataflowBufferSpec]): The dataflow buffers.
            semaphores (List[SemaphoreSpec]): The semaphores.
            scratchpads (List[ScratchpadSpec]): The scratchpads.
            tensor_parameters (List[TensorParameter]): The tensors the kernels bind.
            work_units (List[WorkUnitSpec]): Which kernels run on which nodes.
    )doc")
        .def(
            "__init__",
            [](exp::ProgramSpec* self,
               const std::string& name,
               const std::vector<exp::KernelSpec>& kernels,
               const std::vector<exp::DataflowBufferSpec>& dataflow_buffers,
               const std::vector<exp::SemaphoreSpec>& semaphores,
               const std::vector<exp::ScratchpadSpec>& scratchpads,
               const std::vector<exp::TensorParameter>& tensor_parameters,
               const std::vector<exp::WorkUnitSpec>& work_units) {
                new (self) exp::ProgramSpec{
                    .name = name,
                    .kernels = kernels,
                    .dataflow_buffers = dataflow_buffers,
                    .semaphores = semaphores,
                    .scratchpads = scratchpads,
                    .tensor_parameters = tensor_parameters,
                    .work_units = work_units,
                };
            },
            nb::kw_only(),
            nb::arg("name") = std::string{},
            nb::arg("kernels") = std::vector<exp::KernelSpec>{},
            nb::arg("dataflow_buffers") = std::vector<exp::DataflowBufferSpec>{},
            nb::arg("semaphores") = std::vector<exp::SemaphoreSpec>{},
            nb::arg("scratchpads") = std::vector<exp::ScratchpadSpec>{},
            nb::arg("tensor_parameters") = std::vector<exp::TensorParameter>{},
            nb::arg("work_units") = std::vector<exp::WorkUnitSpec>{})
        .def_rw("name", &exp::ProgramSpec::name)
        .def_rw("kernels", &exp::ProgramSpec::kernels, nb::rv_policy::copy)
        .def_rw("dataflow_buffers", &exp::ProgramSpec::dataflow_buffers, nb::rv_policy::copy)
        .def_rw("semaphores", &exp::ProgramSpec::semaphores, nb::rv_policy::copy)
        .def_rw("scratchpads", &exp::ProgramSpec::scratchpads, nb::rv_policy::copy)
        .def_rw("tensor_parameters", &exp::ProgramSpec::tensor_parameters, nb::rv_policy::copy)
        .def_rw("work_units", &exp::ProgramSpec::work_units, nb::rv_policy::copy);
}

void bind_run_args(nb::module_& mod) {
    nb::class_<exp::KernelRunArgs>(mod, "KernelRunArgs", R"doc(
        Runtime arg values of one kernel.

        Args:
            kernel (str): unique_id of the KernelSpec.
            runtime_arg_values (Dict[str, Dict[ttnn.CoreCoord, int]]): Per-node values of each name in the
                kernel's runtime_arg_names; every node of the kernel's work units needs a value.
            common_runtime_arg_values (Dict[str, int]): Values of the kernel's common_runtime_arg_names.
    )doc")
        .def(
            "__init__",
            [](exp::KernelRunArgs* self,
               const std::string& kernel,
               const NodeArgMap& runtime_arg_values,
               const ArgMap& common_runtime_arg_values) {
                new (self) exp::KernelRunArgs{.kernel = exp::KernelSpecName{kernel}};
                for (const auto& [name, values_per_node] : runtime_arg_values) {
                    for (const auto& [node, value] : values_per_node) {
                        self->runtime_arg_values[name][node] = value;
                    }
                }
                for (const auto& [name, value] : common_runtime_arg_values) {
                    self->common_runtime_arg_values.emplace(name, value);
                }
            },
            nb::kw_only(),
            nb::arg("kernel"),
            nb::arg("runtime_arg_values") = NodeArgMap{},
            nb::arg("common_runtime_arg_values") = ArgMap{})
        .def_prop_ro("kernel", [](const exp::KernelRunArgs& self) { return *self.kernel; })
        .def(
            "add_runtime_args_for_node",
            [](exp::KernelRunArgs& self, const tt::tt_metal::CoreCoord& node, const ArgMap& named_values) {
                for (const auto& [name, value] : named_values) {
                    self.runtime_arg_values[name][node] = value;
                }
            },
            nb::arg("node"),
            nb::arg("named_values"),
            "Set the runtime arg values {name: value} of one node.");

    nb::class_<ttnn::ProgramSpecRunArgs>(mod, "ProgramSpecRunArgs", R"doc(
        Run-time arguments of a ProgramSpec passed to ``ttnn.generic_op``.

        Args:
            kernel_run_args (List[KernelRunArgs]): Runtime arg values, one entry per kernel that has runtime args.
            tensor_args (Dict[str, ttnn.Tensor]): The tensor bound to each TensorParameter. Every tensor must also
                be in io_tensors.
            custom_program_hash (int, optional): Program-cache key to use instead of the ProgramSpec's hash.
    )doc")
        .def(
            "__init__",
            [](ttnn::ProgramSpecRunArgs* self,
               const std::vector<exp::KernelRunArgs>& kernel_run_args,
               const std::map<std::string, Tensor>& tensor_args,
               std::optional<ttsl::hash::hash_t> custom_program_hash) {
                new (self) ttnn::ProgramSpecRunArgs{
                    .kernel_run_args = kernel_run_args,
                    .tensor_args = {tensor_args.begin(), tensor_args.end()},
                    .custom_program_hash = custom_program_hash,
                };
            },
            nb::kw_only(),
            nb::arg("kernel_run_args") = std::vector<exp::KernelRunArgs>{},
            nb::arg("tensor_args") = std::map<std::string, Tensor>{},
            nb::arg("custom_program_hash") = nb::none())
        .def_rw("kernel_run_args", &ttnn::ProgramSpecRunArgs::kernel_run_args)
        .def_rw("custom_program_hash", &ttnn::ProgramSpecRunArgs::custom_program_hash);
}

}  // namespace

void bind_program_spec_types(nb::module_& mod) {
    bind_hardware_configs(mod);
    bind_resources(mod);
    bind_kernel_spec(mod);
    bind_program_spec(mod);
    bind_run_args(mod);
}

}  // namespace ttnn::operations::generic
