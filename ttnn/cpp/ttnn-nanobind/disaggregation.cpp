// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "disaggregation.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/internal/disaggregation/kv_chunk_address_table.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

#include "ttnn/experimental/disaggregation/tensor_helpers.hpp"

namespace tt::tt_metal::internal::disaggregation {
// Protobuf serializer free-functions. Declared in impl/.../kv_chunk_address_table_protobuf.hpp,
// which is not on ttnn's include path; the definitions link from libtt_metal (the .cpp is
// compiled into the `impl` target). Forward-declared here to bind without the impl header.
std::string export_to_protobuf(const KvChunkAddressTable& table);
void export_to_protobuf_file(const KvChunkAddressTable& table, const std::string& path);
// Deserializers — same TU/target as the exporters above, so they link the same way.
KvChunkAddressTable import_from_protobuf(const std::string& data);
KvChunkAddressTable import_from_protobuf_file(const std::string& path);
// UMD-backed device-less DRAM read (defined in umd_dram_reader.cpp, links from libtt_metal):
// reads a chunk over a bare tt::umd::Cluster (no start_device / no CHIP_IN_USE lock); chip selected
// by ASIC unique_id, noc_addr = (dram_view << 32) | local_addr.
std::vector<uint8_t> read_dram_umd(uint64_t unique_id, uint64_t noc_addr, uint32_t size_bytes);
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
        .def("total_entries", &KvChunkAddressTable::total_entries, "Total number of entries in the table.")

        // Device reads
        .def(
            "read_device_chunk",
            [](const KvChunkAddressTable& table, uint32_t layer, uint32_t position, uint32_t slot) {
                auto buf = table.read_device_chunk(layer, position, slot);
                return nb::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
            },
            nb::arg("layer"),
            nb::arg("position"),
            nb::arg("slot"),
            R"(
            Read the raw bytes of a single chunk from the primary replica device.
            Resolves the device internally via the global ControlPlane.
            Position is in tokens (chunk-aligned).
            )");

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

    // Deserialization — an external consumer (e.g. the prefill_h2d producer) reconstructs the
    // table the runner exported. Pure data: builds the KvChunkAddressTable from the protobuf with
    // no device / ControlPlane access, so it is safe to call from a device-less process. (Only a
    // subsequent read_device_chunk() needs the device to be open in the caller's process.)
    mod.def(
        "import_from_protobuf_file",
        &import_from_protobuf_file,
        nb::arg("path"),
        "Deserialize a KvChunkAddressTable from a protobuf file at `path`.");
    mod.def(
        "import_from_protobuf",
        &import_from_protobuf,
        nb::arg("data"),
        "Deserialize a KvChunkAddressTable from a serialized protobuf byte string.");

    // UMD-backed device-less read: reads a KV chunk's DRAM bytes over a bare tt::umd::Cluster that
    // never calls start_device() (no CHIP_IN_USE flock), so a producer can read a live server's KV
    // CONCURRENTLY with the runner — the mechanism the migration worker uses
    // (disaggregation/migration/src/worker/device_io.cpp). The chip is selected by ASIC unique_id
    // (the caller resolves fabric_node -> unique_id from the runner's device-map sidecar); noc_addr
    // is (dram_view << 32) | local_addr, size_bytes is the chunk size (19584 for a bfp8 KV chunk).
    mod.def(
        "read_dram_umd",
        [](uint64_t unique_id, uint64_t noc_addr, uint32_t size_bytes) {
            auto buf = tt::tt_metal::experimental::disaggregation::read_dram_umd(unique_id, noc_addr, size_bytes);
            return nb::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
        },
        nb::arg("unique_id"),
        nb::arg("noc_addr"),
        nb::arg("size_bytes"),
        R"(
        Read a KV chunk's raw bytes over UMD from a device-less process, CONCURRENT with the running
        server. Uses a bare tt::umd::Cluster (no start_device / no CHIP_IN_USE lock), selecting the
        chip by ASIC unique_id and the DRAM view/offset from noc_addr = (view << 32) | local_addr.
        Mirrors the migration worker's device_io read path.
        )");
}

}  // namespace ttnn::disaggregation
