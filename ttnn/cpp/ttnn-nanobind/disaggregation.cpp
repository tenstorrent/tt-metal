// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "disaggregation.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <map>
#include <span>

#include <tt-metalium/internal/disaggregation/kv_chunk_address_table.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

#include "ttnn/experimental/disaggregation/tensor_helpers.hpp"

namespace tt::tt_metal::internal::disaggregation {
// Protobuf serializer free-functions. Declared in impl/.../kv_chunk_address_table_protobuf.hpp,
// which is not on ttnn's include path; the definitions link from libtt_metal (the .cpp is
// compiled into the `impl` target). Forward-declared here to bind without the impl header.
std::string export_to_protobuf(const KvChunkAddressTable& table);
void export_to_protobuf_file(const KvChunkAddressTable& table, const std::string& path);
}  // namespace tt::tt_metal::internal::disaggregation

namespace ttnn::disaggregation {

void bind_disaggregation_api(nb::module_& mod) {
    using namespace tt::tt_metal::internal::disaggregation;

    // DeviceGroupIndex - StrongType wrapper around uint32_t
    nb::class_<DeviceGroupIndex>(mod, "DeviceGroupIndex", R"(
        Strongly-typed index into the device group table.
        Used to reference device groups in KvCacheLocation.
    )")
        .def(nb::init<uint32_t>(), nb::arg("value"), "Create a DeviceGroupIndex from an integer value.")
        .def(
            "__int__", [](const DeviceGroupIndex& idx) { return *idx; }, "Convert DeviceGroupIndex to integer.")
        .def(
            "__eq__",
            [](const DeviceGroupIndex& lhs, const DeviceGroupIndex& rhs) { return lhs == rhs; },
            nb::arg("other"))
        .def("__repr__", [](const DeviceGroupIndex& idx) { return fmt::format("DeviceGroupIndex({})", *idx); });

    // DeviceGroup - Group of fabric nodes with replicas
    nb::class_<DeviceGroup>(mod, "DeviceGroup", R"(
        A unique group of fabric nodes that hold replicas of a KV cache chunk.
        FabricNodeIds are stored sorted for deduplication.
    )")
        .def(nb::init<>(), "Create an empty DeviceGroup.")
        .def_rw("fabric_node_ids", &DeviceGroup::fabric_node_ids, "List of FabricNodeIds in this group (sorted).")
        .def("__eq__", [](const DeviceGroup& lhs, const DeviceGroup& rhs) { return lhs == rhs; }, nb::arg("other"));

    // KvCacheLocation - Physical location of a KV cache chunk
    nb::class_<KvCacheLocation>(mod, "KvCacheLocation", R"(
        Describes the physical location of a single KV cache chunk in device memory.
        Contains NOC address, size, and reference to the device group holding replicas.
    )")
        .def(nb::init<>(), "Create a KvCacheLocation with default values.")
        .def_rw("noc_addr", &KvCacheLocation::noc_addr, "NOC address of the KV cache chunk.")
        .def_rw("size_bytes", &KvCacheLocation::size_bytes, "Size of the KV cache chunk in bytes.")
        .def_rw("device_group_index", &KvCacheLocation::device_group_index, "Index into the device group table.");

    // KvChunkAddressTableConfig - Configuration struct
    nb::class_<KvChunkAddressTableConfig>(mod, "KvChunkAddressTableConfig", R"(
        Configuration for constructing a KvChunkAddressTable.
        Defines the dimensions and chunking parameters.
    )")
        .def(nb::init<>(), "Create a config with default values.")
        .def_rw("num_layers", &KvChunkAddressTableConfig::num_layers, "Number of transformer layers.")
        .def_rw(
            "max_sequence_length",
            &KvChunkAddressTableConfig::max_sequence_length,
            "Maximum sequence length in tokens.")
        .def_rw("num_slots", &KvChunkAddressTableConfig::num_slots, "Number of KV cache slots.")
        .def_rw(
            "chunk_n_tokens",
            &KvChunkAddressTableConfig::chunk_n_tokens,
            "Tokens per chunk (KV atomic block granularity). Default: 32")
        .def_rw(
            "chunk_size_bytes",
            &KvChunkAddressTableConfig::chunk_size_bytes,
            "Physical size of one chunk in bytes. Default: 19584 (18 x 1088 bfp8 tiles)");

    // KvChunkAddressTable - Main lookup table class
    nb::class_<KvChunkAddressTable>(mod, "KvChunkAddressTable", R"(
        Lookup table mapping (layer, position, slot) -> KvCacheLocation.

        Describes how a KV cache is allocated across a multi-host, multi-chip,
        multi-memory system. Used by the migration layer to locate KV cache chunks
        for transfer.
    )")
        .def(
            nb::init<const KvChunkAddressTableConfig&>(),
            nb::arg("config"),
            "Construct a single-config KvChunkAddressTable (config id 0, name \"0\").")
        .def(
            "__init__",
            [](KvChunkAddressTable* self, const std::vector<KvChunkAddressTableConfig>& configs) {
                new (self) KvChunkAddressTable(std::span<const KvChunkAddressTableConfig>(configs));
            },
            nb::arg("configs"),
            R"(
            Construct a multi-config KvChunkAddressTable from a list of configs.
            Config i is named "i", so string accessors resolve "0".."N-1" to ids 0..N-1.
            )")
        .def(
            nb::init<const std::map<std::string, KvChunkAddressTableConfig>&>(),
            nb::arg("configs"),
            R"(
            Construct a multi-config KvChunkAddressTable from a name->config map.
            Config ids are assigned in sorted key order; each config's name is its key.
            )")

        // Device group management
        .def(
            "add_device_group",
            &KvChunkAddressTable::add_device_group,
            nb::arg("fabric_node_ids"),
            R"(
            Register a device group (set of replica FabricNodeIds).
            The FabricNodeIds are sorted internally for deduplication.
            Returns the index for this group. If an identical sorted group
            already exists, returns the existing index.
            )")
        .def(
            "get_device_group",
            &KvChunkAddressTable::get_device_group,
            nb::arg("index"),
            nb::rv_policy::reference_internal,
            "Lookup a device group by index. Returns a reference to the DeviceGroup.")
        .def("num_device_groups", &KvChunkAddressTable::num_device_groups, "Number of unique device groups registered.")

        // Mutators
        .def(
            "set",
            static_cast<void (KvChunkAddressTable::*)(uint32_t, uint32_t, uint32_t, KvCacheLocation, uint32_t)>(
                &KvChunkAddressTable::set),
            nb::arg("layer"),
            nb::arg("position"),
            nb::arg("slot"),
            nb::arg("location"),
            nb::arg("config_id") = 0,
            R"(
            Set the location for a specific (layer, position, slot, config_id).
            Position is in tokens and must be chunk-aligned (multiple of the config's chunk_n_tokens).
            config_id defaults to 0 (the single-config case).
            )")
        .def(
            "set",
            static_cast<void (KvChunkAddressTable::*)(uint32_t, uint32_t, uint32_t, KvCacheLocation, const std::string&)>(
                &KvChunkAddressTable::set),
            nb::arg("layer"),
            nb::arg("position"),
            nb::arg("slot"),
            nb::arg("location"),
            nb::arg("config"),
            "Set the location for a specific (layer, position, slot, config-name).")
        .def(
            "set_fabric_node_host",
            &KvChunkAddressTable::set_fabric_node_host,
            nb::arg("node_id"),
            nb::arg("host_name"),
            "Register a mapping from FabricNodeId to its host name.")

        // Accessors
        .def(
            "lookup",
            static_cast<const KvCacheLocation& (KvChunkAddressTable::*)(uint32_t, uint32_t, uint32_t, uint32_t) const>(
                &KvChunkAddressTable::lookup),
            nb::arg("layer"),
            nb::arg("position"),
            nb::arg("slot"),
            nb::arg("config_id") = 0,
            nb::rv_policy::reference_internal,
            R"(
            Lookup a single entry by (layer, position, slot, config_id). Position is in tokens (chunk-aligned).
            config_id defaults to 0. Returns a reference to the KvCacheLocation.
            )")
        .def(
            "lookup",
            static_cast<
                const KvCacheLocation& (KvChunkAddressTable::*)(uint32_t, uint32_t, uint32_t, const std::string&) const>(
                &KvChunkAddressTable::lookup),
            nb::arg("layer"),
            nb::arg("position"),
            nb::arg("slot"),
            nb::arg("config"),
            nb::rv_policy::reference_internal,
            "Lookup a single entry by (layer, position, slot, config-name).")
        .def(
            "lookup_range",
            [](const KvChunkAddressTable& table,
               uint32_t layer,
               uint32_t start_pos,
               uint32_t end_pos,
               uint32_t slot,
               uint32_t config_id) {
                auto span = table.lookup_range(layer, start_pos, end_pos, slot, config_id);
                // Convert span to vector for Python
                return std::vector<KvCacheLocation>(span.begin(), span.end());
            },
            nb::arg("layer"),
            nb::arg("start_pos"),
            nb::arg("end_pos"),
            nb::arg("slot"),
            nb::arg("config_id") = 0,
            R"(
            Lookup a contiguous range of position chunks for a given (layer, slot, config_id).
            Returns a list of KvCacheLocation entries. config_id defaults to 0.
            start_pos must be chunk-aligned. end_pos need not be aligned.
            Returns entries for chunks covering positions [start_pos, end_pos).
            )")
        .def(
            "lookup_range",
            [](const KvChunkAddressTable& table,
               uint32_t layer,
               uint32_t start_pos,
               uint32_t end_pos,
               uint32_t slot,
               const std::string& config) {
                auto span = table.lookup_range(layer, start_pos, end_pos, slot, config);
                return std::vector<KvCacheLocation>(span.begin(), span.end());
            },
            nb::arg("layer"),
            nb::arg("start_pos"),
            nb::arg("end_pos"),
            nb::arg("slot"),
            nb::arg("config"),
            "Lookup a contiguous range of position chunks for a given (layer, slot, config-name).")
        .def(
            "get_host",
            &KvChunkAddressTable::get_host,
            nb::arg("node_id"),
            nb::rv_policy::reference,
            "Resolve a FabricNodeId to its host name. Throws if not found.")
        .def(
            "has_host",
            &KvChunkAddressTable::has_host,
            nb::arg("node_id"),
            "Check if a FabricNodeId has a registered host mapping.")

        // Properties
        .def(
            "config",
            &KvChunkAddressTable::config,
            nb::arg("config_id") = 0,
            nb::rv_policy::reference_internal,
            "Get a config by id (default 0, the lone config of a single-config table).")
        .def("num_configs", &KvChunkAddressTable::num_configs, "Number of configs (\"groups\") held by this table.")
        .def(
            "config_name",
            &KvChunkAddressTable::config_name,
            nb::arg("config_id"),
            nb::rv_policy::reference_internal,
            "Name of a config by id.")
        .def(
            "config_id_of",
            &KvChunkAddressTable::config_id_of,
            nb::arg("name"),
            "Resolve a config name to its id (throws if unknown).")
        .def(
            "num_position_chunks",
            &KvChunkAddressTable::num_position_chunks,
            nb::arg("config_id") = 0,
            "Number of position chunks for a config (default 0).")
        .def("total_entries", &KvChunkAddressTable::total_entries, "Total number of entries summed across all configs.")

        // Device reads
        .def(
            "read_device_chunk",
            [](const KvChunkAddressTable& table, uint32_t layer, uint32_t position, uint32_t slot, uint32_t config_id) {
                auto buf = table.read_device_chunk(layer, position, slot, config_id);
                return nb::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
            },
            nb::arg("layer"),
            nb::arg("position"),
            nb::arg("slot"),
            nb::arg("config_id") = 0,
            R"(
            Read the raw bytes of a single chunk from the primary replica device.
            Resolves the device internally via the global ControlPlane.
            Position is in tokens (chunk-aligned). config_id defaults to 0.
            )")
        .def(
            "read_device_chunk",
            [](const KvChunkAddressTable& table,
               uint32_t layer,
               uint32_t position,
               uint32_t slot,
               const std::string& config) {
                auto buf = table.read_device_chunk(layer, position, slot, config);
                return nb::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
            },
            nb::arg("layer"),
            nb::arg("position"),
            nb::arg("slot"),
            nb::arg("config"),
            "Read the raw bytes of a single chunk (config addressed by name).");

    mod.def(
        "tensor_from_bfp8_bytes",
        [](const nb::bytes& raw_bytes, const std::vector<uint32_t>& shape) {
            return ttnn::experimental_disaggregation::tensor_from_bfp8_bytes(
                std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(raw_bytes.c_str()), raw_bytes.size()), shape);
        },
        nb::arg("raw_bytes"),
        nb::arg("shape"),
        R"(
        Wrap raw bfp8-packed bytes (uint32-aligned, TILE layout) as a host-side ttnn.Tensor
        with the given shape — no quantization round-trip.
        Used to compare KV-table reads against the live KV cache byte-for-byte.
        )");

    // Protobuf serialization — the runner publishes the table to the
    // migration_worker (SET_TABLE consumes a serialized protobuf file path).
    mod.def(
        "export_to_protobuf_file",
        &export_to_protobuf_file,
        nb::arg("table"),
        nb::arg("path"),
        "Serialize a KvChunkAddressTable to a protobuf file at `path`.");
}

}  // namespace ttnn::disaggregation
