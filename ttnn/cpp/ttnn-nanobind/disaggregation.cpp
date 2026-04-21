// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "disaggregation.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/disaggregation/kv_chunk_address_table.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace ttnn::disaggregation {

void bind_disaggregation_api(nb::module_& mod) {
    using namespace tt::tt_metal::experimental::disaggregation;

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
            "Construct a KvChunkAddressTable from configuration.")

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
            &KvChunkAddressTable::set,
            nb::arg("layer"),
            nb::arg("position"),
            nb::arg("slot"),
            nb::arg("location"),
            R"(
            Set the location for a specific (layer, position, slot).
            Position is in tokens and must be chunk-aligned (multiple of chunk_n_tokens).
            )")
        .def(
            "set_fabric_node_host",
            &KvChunkAddressTable::set_fabric_node_host,
            nb::arg("node_id"),
            nb::arg("host_name"),
            "Register a mapping from FabricNodeId to its host name.")

        // Accessors
        .def(
            "lookup",
            &KvChunkAddressTable::lookup,
            nb::arg("layer"),
            nb::arg("position"),
            nb::arg("slot"),
            nb::rv_policy::reference_internal,
            R"(
            Lookup a single entry. Position is in tokens (chunk-aligned).
            Returns a reference to the KvCacheLocation.
            )")
        .def(
            "lookup_range",
            [](const KvChunkAddressTable& table, uint32_t layer, uint32_t start_pos, uint32_t end_pos, uint32_t slot) {
                auto span = table.lookup_range(layer, start_pos, end_pos, slot);
                // Convert span to vector for Python
                return std::vector<KvCacheLocation>(span.begin(), span.end());
            },
            nb::arg("layer"),
            nb::arg("start_pos"),
            nb::arg("end_pos"),
            nb::arg("slot"),
            R"(
            Lookup a contiguous range of position chunks for a given (layer, slot).
            Returns a list of KvCacheLocation entries.
            start_pos must be chunk-aligned. end_pos need not be aligned.
            Returns entries for chunks covering positions [start_pos, end_pos).
            )")
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
            nb::rv_policy::reference_internal,
            "Get the configuration used to construct this table.")
        .def(
            "num_position_chunks",
            &KvChunkAddressTable::num_position_chunks,
            "Number of position chunks (computed from config).")
        .def("total_entries", &KvChunkAddressTable::total_entries, "Total number of entries in the table.");
}

}  // namespace ttnn::disaggregation
