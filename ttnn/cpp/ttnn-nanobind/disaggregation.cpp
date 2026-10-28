// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "disaggregation.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <map>
#include <span>

#include <internal/disaggregation/cache_tensor_layout_spec.hpp>
#include <internal/disaggregation/kv_chunk_address_table.hpp>
#include <internal/disaggregation/kv_chunk_table_cache.hpp>
#include <internal/disaggregation/to_chunk_map.hpp>
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
// CacheLayoutModel serializers (impl/.../cache_layout_model_protobuf.hpp), linked the same way.
void export_to_protobuf_file(const CacheLayoutModel& model, const std::string& path);
CacheLayoutModel import_cache_layout_model_from_protobuf_file(const std::string& path);
// UMD-backed DRAM read (defined in umd_dram_reader.cpp, links from libtt_metal):
// reads a chunk over a bare tt::umd::Cluster; chip selected
// by ASIC unique_id, noc_addr = (dram_view << 32) | local_addr.
std::vector<uint8_t> read_dram_umd(uint64_t unique_id, uint64_t noc_addr, uint32_t size_bytes);
}  // namespace tt::tt_metal::internal::disaggregation

namespace ttnn::disaggregation {

namespace {

// The model-facing cache layout: a model declares each cache as a CacheConfig and derives its
// KvChunkAddressTable with to_chunk_map, or exports the CacheLayoutModel for a consumer to derive it.
void bind_cache_layout_api(nb::module_& mod) {
    using namespace tt::tt_metal::internal::disaggregation;

    nb::class_<temporal::Dense>(mod, "TemporalDense", "Retains [0, prefix_len): full KV, MLA latent.")
        .def(nb::init<>());
    nb::class_<temporal::Window>(mod, "TemporalWindow", "Retains [prefix_len - width, prefix_len): SWA.")
        .def(
            "__init__",
            [](temporal::Window* t, uint32_t width_tokens) { new (t) temporal::Window{WindowTokens{width_tokens}}; },
            nb::arg("width_tokens"))
        .def_prop_ro("width_tokens", [](const temporal::Window& t) { return t.width.get(); });
    nb::class_<temporal::BlockLocal>(mod, "TemporalBlockLocal", "Retains the current attention chunk only.")
        .def(
            "__init__",
            [](temporal::BlockLocal* t, uint32_t span_tokens) {
                new (t) temporal::BlockLocal{BlockLocalSpan{span_tokens}};
            },
            nb::arg("span_tokens"))
        .def_prop_ro("span_tokens", [](const temporal::BlockLocal& t) { return t.span.get(); });
    nb::class_<temporal::Rolling>(mod, "TemporalRolling", "A conv / token-shift ring; no sequence axis.")
        .def(nb::init<>());
    nb::class_<temporal::Static>(mod, "TemporalStatic", "Computed once at prefill and frozen.").def(nb::init<>());
    nb::class_<temporal::None>(mod, "TemporalNone", "A fixed-size recurrent summary; no sequence axis.")
        .def(nb::init<>());

    nb::class_<KvLayoutSpec>(mod, "KvLayoutSpec", R"(
        One cached KV/state tensor: its distributed TensorSpec, how its sequence coordinate is
        interpreted, and the target arch.
    )")
        .def(
            "__init__",
            [](KvLayoutSpec* t, const tt::tt_metal::TensorSpec& tensor, const TemporalPolicy& temporal, tt::ARCH arch) {
                new (t) KvLayoutSpec{.tensor = tensor, .temporal = temporal, .arch = arch};
            },
            nb::arg("tensor"),
            nb::arg("temporal"),
            nb::arg("arch"))
        .def_rw("tensor", &KvLayoutSpec::tensor)
        .def_rw("temporal", &KvLayoutSpec::temporal)
        .def_rw("arch", &KvLayoutSpec::arch)
        .def("has_sequence", &KvLayoutSpec::has_sequence)
        .def("sequence_axis", &KvLayoutSpec::sequence_axis);

    nb::enum_<BankScheme>(mod, "BankScheme", "How a chunk index maps to (bank, per-bank offset).")
        .value("Natural", BankScheme::Natural)
        .value("MlaShard", BankScheme::MlaShard)
        .value("Block", BankScheme::Block)
        .value("Cyclic", BankScheme::Cyclic)
        .value("BlockCyclic", BankScheme::BlockCyclic);
    nb::enum_<BankOrder>(mod, "BankOrder", "The DRAM bank permutation the cache's writer used.")
        .value("Identity", BankOrder::Identity)
        .value("Optimal", BankOrder::Optimal);

    nb::class_<GenerationPolicy>(mod, "GenerationPolicy", R"(
        What the cache's writer decides that its tensor does not carry: the bank scheme and order, the
        k-chunk size, and the sequence-shard stride (device_chunk_size; None => k_chunk_size * banks).
    )")
        .def(
            "__init__",
            [](GenerationPolicy* t,
               BankScheme bank_scheme,
               BankOrder bank_order,
               uint32_t k_chunk_size,
               std::optional<uint32_t> device_chunk_size,
               uint32_t banks_per_head,
               uint32_t num_blocks) {
                new (t) GenerationPolicy{
                    .bank_scheme = bank_scheme,
                    .bank_order = bank_order,
                    .k_chunk_size = KChunkSize{k_chunk_size},
                    .device_chunk_size = device_chunk_size.has_value()
                                             ? std::optional<DeviceChunkSize>(DeviceChunkSize{*device_chunk_size})
                                             : std::nullopt,
                    .banks_per_head = BanksPerHead{banks_per_head},
                    .num_blocks = num_blocks};
            },
            nb::arg("bank_scheme") = BankScheme::Natural,
            nb::arg("bank_order") = BankOrder::Optimal,
            nb::arg("k_chunk_size") = 128,
            nb::arg("device_chunk_size") = nb::none(),
            nb::arg("banks_per_head") = 0,
            nb::arg("num_blocks") = 0)
        .def_rw("bank_scheme", &GenerationPolicy::bank_scheme)
        .def_rw("bank_order", &GenerationPolicy::bank_order)
        .def_prop_ro("k_chunk_size", [](const GenerationPolicy& p) { return p.k_chunk_size.get(); })
        .def_prop_ro(
            "device_chunk_size",
            [](const GenerationPolicy& p) -> std::optional<uint32_t> {
                return p.device_chunk_size.has_value() ? std::optional<uint32_t>(p.device_chunk_size->get())
                                                       : std::nullopt;
            })
        .def_prop_ro("banks_per_head", [](const GenerationPolicy& p) { return p.banks_per_head.get(); })
        .def_rw("num_blocks", &GenerationPolicy::num_blocks);

    nb::class_<CacheConfig>(mod, "CacheConfig", R"(
        One co-resident cache to address: its KvLayoutSpec, the mesh distribution its chunks really have
        (TensorTopology), its GenerationPolicy and buffer address, its table config name (empty => its
        index; all empty or all set), the layers it publishes (ascending; empty => every layer it holds),
        the model layers [first_layer, first_layer + num_layers) it holds (num_layers 0 => through the
        last), its mesh (None => the model's), and for a head-sharded cache the one head it holds (None =>
        every head, fanned into slots). Caches sharing a name are one table config, one per pipeline stage.
    )")
        .def(
            "__init__",
            [](CacheConfig* t,
               const KvLayoutSpec& spec,
               const tt::tt_metal::TensorTopology& topology,
               const GenerationPolicy& policy,
               uint64_t base_addr,
               const std::string& name,
               const std::vector<uint32_t>& layers,
               uint32_t first_layer,
               uint32_t num_layers,
               std::optional<tt::tt_fabric::MeshId> mesh_id,
               std::optional<uint32_t> head) {
                new (t) CacheConfig{
                    .spec = spec,
                    .topology = topology,
                    .policy = policy,
                    .base_addr = base_addr,
                    .name = name,
                    .layers = layers,
                    .first_layer = first_layer,
                    .num_layers = num_layers,
                    .mesh_id = mesh_id,
                    .head = head};
            },
            nb::arg("spec"),
            nb::arg("topology"),
            nb::arg("policy"),
            nb::arg("base_addr"),
            nb::arg("name") = "",
            nb::arg("layers") = std::vector<uint32_t>{},
            nb::arg("first_layer") = 0,
            nb::arg("num_layers") = 0,
            nb::arg("mesh_id") = nb::none(),
            nb::arg("head") = nb::none())
        .def_rw("spec", &CacheConfig::spec)
        .def_rw("topology", &CacheConfig::topology)
        .def_rw("policy", &CacheConfig::policy)
        .def_rw("base_addr", &CacheConfig::base_addr)
        .def_rw("name", &CacheConfig::name)
        .def_rw("layers", &CacheConfig::layers)
        .def_rw("first_layer", &CacheConfig::first_layer)
        .def_rw("num_layers", &CacheConfig::num_layers)
        .def_rw("mesh_id", &CacheConfig::mesh_id)
        .def_rw("head", &CacheConfig::head);

    nb::class_<MapGeometry>(mod, "MapGeometry", "The (layer, slot, position) grid a table is enumerated over.")
        .def(
            "__init__",
            [](MapGeometry* t, uint32_t num_layers, uint32_t num_slots, uint32_t max_seq_len, uint32_t position_step) {
                new (t) MapGeometry{
                    .num_layers = num_layers,
                    .num_slots = num_slots,
                    .max_seq_len = max_seq_len,
                    .position_step = position_step};
            },
            nb::arg("num_layers"),
            nb::arg("num_slots"),
            nb::arg("max_seq_len"),
            nb::arg("position_step") = tt::constants::TILE_WIDTH)
        .def_rw("num_layers", &MapGeometry::num_layers)
        .def_rw("num_slots", &MapGeometry::num_slots)
        .def_rw("max_seq_len", &MapGeometry::max_seq_len)
        .def_rw("position_step", &MapGeometry::position_step);

    nb::class_<CacheLayoutModel>(mod, "CacheLayoutModel", R"(
        A model's caches and the mesh and geometry they are enumerated over: everything to_chunk_map reads.
    )")
        .def(
            "__init__",
            [](CacheLayoutModel* t,
               const std::vector<CacheConfig>& caches,
               tt::tt_fabric::MeshId mesh_id,
               const tt::tt_metal::distributed::MeshShape& mesh_shape,
               const MapGeometry& geometry) {
                new (t) CacheLayoutModel{
                    .caches = caches, .mesh_id = mesh_id, .mesh_shape = mesh_shape, .geometry = geometry};
            },
            nb::arg("caches"),
            nb::arg("mesh_id"),
            nb::arg("mesh_shape"),
            nb::arg("geometry"))
        .def_rw("caches", &CacheLayoutModel::caches)
        .def_rw("mesh_id", &CacheLayoutModel::mesh_id)
        .def_rw("mesh_shape", &CacheLayoutModel::mesh_shape)
        .def_rw("geometry", &CacheLayoutModel::geometry);

    mod.def(
        "to_chunk_map",
        nb::overload_cast<const CacheLayoutModel&>(&to_chunk_map),
        nb::arg("model"),
        "Derive the KvChunkAddressTable of a CacheLayoutModel.");
    mod.def(
        "first_table_difference",
        &first_table_difference,
        nb::arg("expected"),
        nb::arg("actual"),
        R"(
        The first way `actual` differs from `expected` as a description, or None when they agree: the same
        configs and, at every (config, layer, slot, position), the same address, size, device-group members
        and members' hosts. Device-group indices are not compared.
        )");
    mod.def(
        "export_cache_layout_model_to_protobuf_file",
        nb::overload_cast<const CacheLayoutModel&, const std::string&>(&export_to_protobuf_file),
        nb::arg("model"),
        nb::arg("path"),
        "Serialize a CacheLayoutModel to a protobuf file at `path`.");
    mod.def(
        "import_cache_layout_model_from_protobuf_file",
        &import_cache_layout_model_from_protobuf_file,
        nb::arg("path"),
        "Deserialize a CacheLayoutModel from a protobuf file at `path`.");
}

}  // namespace

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

    // KvChunkTensorType - Cached tensor kind (maps 1:1 onto a TemporalPolicy alternative)
    nb::enum_<KvChunkTensorType>(mod, "KvChunkTensorType", R"(
        What kind of cached tensor a config's chunks belong to.
        Each value corresponds to one TemporalPolicy alternative.
    )")
        .value("Kv", KvChunkTensorType::Kv, "Dense: full KV / MLA latent over [0, prefix_len).")
        .value(
            "Snapshot",
            KvChunkTensorType::Snapshot,
            "None: fixed-size state keyed by the prefix_len it was snapshotted at (SSM / KDA state).");

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
            "Physical size of one chunk in bytes. Default: 19584 (18 x 1088 bfp8 tiles)")
        .def_rw(
            "tensor_type",
            &KvChunkAddressTableConfig::tensor_type,
            "Cached tensor kind for this config's chunks. Default: KvChunkTensorType.Kv");

    // ChunkCompression - per-config in-memory / on-wire representation
    nb::enum_<ChunkCompression>(mod, "ChunkCompression", R"(
        How a config's chunk map is represented: UNROLLED (one entry per chunk) or
        STRIDED_ROWS (per (slot, layer) row, a periodic base + stride per residue class).
    )")
        .value("UNROLLED", ChunkCompression::kUnrolled)
        .value("STRIDED_ROWS", ChunkCompression::kStridedRows);

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
        .def(
            "__init__",
            [](KvChunkAddressTable* self,
               const std::map<std::string, KvChunkAddressTableConfig>& configs,
               const std::map<std::string, ChunkCompression>& compressions) {
                std::map<std::string, KvChunkAddressTable::NamedConfigInit> inits;
                for (const auto& [name, cfg] : configs) {
                    inits[name] = {.config = cfg, .compression = ChunkCompression::kUnrolled};
                }
                for (const auto& [name, compression] : compressions) {
                    auto it = inits.find(name);
                    TT_FATAL(it != inits.end(), "compression given for unknown config '{}'", name);
                    it->second.compression = compression;
                }
                new (self) KvChunkAddressTable(inits);
            },
            nb::arg("configs"),
            nb::arg("compressions"),
            R"(
            Construct a multi-config KvChunkAddressTable from a name->config map, choosing each
            config's representation from `compressions` (name -> ChunkCompression; configs not
            listed are UNROLLED). A STRIDED_ROWS config allocates no per-chunk grid and is
            authored with set_strided_row().
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
            static_cast<void (KvChunkAddressTable::*)(
                uint32_t, uint32_t, uint32_t, KvCacheLocation, const std::string&)>(&KvChunkAddressTable::set),
            nb::arg("layer"),
            nb::arg("position"),
            nb::arg("slot"),
            nb::arg("location"),
            nb::arg("config"),
            "Set the location for a specific (layer, position, slot, config-name).")
        .def(
            "set_strided_row",
            [](KvChunkAddressTable& table,
               uint32_t layer,
               uint32_t slot,
               uint32_t size_bytes,
               std::vector<uint64_t> bases,
               std::vector<int64_t> strides,
               const std::vector<uint32_t>& device_group_indices,
               uint32_t config_id) {
                std::vector<DeviceGroupIndex> groups;
                groups.reserve(device_group_indices.size());
                for (uint32_t g : device_group_indices) {
                    groups.emplace_back(g);
                }
                table.set_strided_row(
                    layer, slot, size_bytes, std::move(bases), std::move(strides), std::move(groups), config_id);
            },
            nb::arg("layer"),
            nb::arg("slot"),
            nb::arg("size_bytes"),
            nb::arg("bases"),
            nb::arg("strides"),
            nb::arg("device_group_indices"),
            nb::arg("config_id") = 0,
            R"(
            Author one (slot, layer) row of a STRIDED_ROWS config: chunk c resolves to
            bases[c % step] + (c // step) * strides[c % step] on device group
            device_group_indices[c % step], where step = len(bases). device_group_indices
            takes ints or the DeviceGroupIndex objects add_device_group() returns.
            )")
        .def(
            "set_strided_row",
            [](KvChunkAddressTable& table,
               uint32_t layer,
               uint32_t slot,
               uint32_t size_bytes,
               std::vector<uint64_t> bases,
               std::vector<int64_t> strides,
               std::vector<DeviceGroupIndex> device_group_indices,
               uint32_t config_id) {
                table.set_strided_row(
                    layer,
                    slot,
                    size_bytes,
                    std::move(bases),
                    std::move(strides),
                    std::move(device_group_indices),
                    config_id);
            },
            nb::arg("layer"),
            nb::arg("slot"),
            nb::arg("size_bytes"),
            nb::arg("bases"),
            nb::arg("strides"),
            nb::arg("device_group_indices"),
            nb::arg("config_id") = 0,
            "Overload of set_strided_row taking the DeviceGroupIndex objects add_device_group() returns.")
        .def(
            "set_fabric_node_host",
            &KvChunkAddressTable::set_fabric_node_host,
            nb::arg("node_id"),
            nb::arg("host_name"),
            "Register a mapping from FabricNodeId to its host name.")

        // Accessors
        .def(
            "lookup",
            static_cast<KvCacheLocation (KvChunkAddressTable::*)(uint32_t, uint32_t, uint32_t, uint32_t) const>(
                &KvChunkAddressTable::lookup),
            nb::arg("layer"),
            nb::arg("position"),
            nb::arg("slot"),
            nb::arg("config_id") = 0,
            R"(
            Lookup a single entry by (layer, position, slot, config_id). Position is in tokens (chunk-aligned).
            config_id defaults to 0. Returns the KvCacheLocation by value.
            )")
        .def(
            "lookup",
            static_cast<KvCacheLocation (KvChunkAddressTable::*)(uint32_t, uint32_t, uint32_t, const std::string&)
                            const>(&KvChunkAddressTable::lookup),
            nb::arg("layer"),
            nb::arg("position"),
            nb::arg("slot"),
            nb::arg("config"),
            "Lookup a single entry by (layer, position, slot, config-name).")
        .def(
            "lookup_range",
            [](const KvChunkAddressTable& table,
               uint32_t layer,
               uint32_t start_pos,
               uint32_t end_pos,
               uint32_t slot,
               uint32_t config_id) {
                std::vector<KvCacheLocation> out;
                table.visit_range(layer, start_pos, end_pos, slot, config_id, [&out](const auto& range) {
                    out.reserve(range.size());
                    out.insert(out.end(), range.begin(), range.end());
                });
                return out;
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
                std::vector<KvCacheLocation> out;
                table.visit_range(layer, start_pos, end_pos, slot, table.config_id_of(config), [&out](const auto& range) {
                    out.reserve(range.size());
                    out.insert(out.end(), range.begin(), range.end());
                });
                return out;
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
        .def(
            "compression",
            &KvChunkAddressTable::compression,
            nb::arg("config_id") = 0,
            "Representation of a config's chunk map (ChunkCompression).")
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

    mod.def(
        "tensor_from_bf16_bytes",
        [](const nb::bytes& raw_bytes, const std::vector<uint32_t>& shape) {
            return ttnn::experimental_disaggregation::tensor_from_bf16_bytes(
                std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(raw_bytes.c_str()), raw_bytes.size()), shape);
        },
        nb::arg("raw_bytes"),
        nb::arg("shape"),
        R"(
        Wrap raw bfloat16 bytes (2 bytes/element, ROW_MAJOR layout) as a host-side ttnn.Tensor
        with the given shape — no quantization round-trip. The bf16/ROW_MAJOR analogue of
        tensor_from_bfp8_bytes, for uncompressed (bf16 ROW_MAJOR) KV caches.
        )");

    // Protobuf serialization — the runner publishes the table to the
    // migration_worker (SET_TABLE consumes a serialized protobuf file path).
    mod.def(
        "export_to_protobuf_file",
        nb::overload_cast<const KvChunkAddressTable&, const std::string&>(&export_to_protobuf_file),
        nb::arg("table"),
        nb::arg("path"),
        "Serialize a KvChunkAddressTable to a protobuf file at `path`.");

    // Deserialization — an external consumer (e.g. the prefill_producer) reconstructs the
    // table the runner exported.
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

    mod.def(
        "get_or_build_kv_chunk_table",
        [](const std::string& seed, const nb::handle& key, const nb::callable& build, const std::string& out_path) {
            nb::object dumps = nb::module_::import_("json").attr("dumps");
            return get_or_build_kv_chunk_table(
                default_kv_chunk_table_cache_dir(),
                seed,
                nb::cast<std::string>(dumps(key, nb::arg("sort_keys") = true)),
                [&build](const std::string& path) {
                    export_to_protobuf_file(nb::cast<const KvChunkAddressTable&>(build()), path);
                },
                out_path);
        },
        nb::arg("seed"),
        nb::arg("key").none(),
        nb::arg("build"),
        nb::arg("out_path"),
        R"(
        Write the KV chunk table for (seed, key) to `out_path`, reusing the tt-metal cache on a hit.
        `build()` returns the KvChunkAddressTable and runs only on a miss. `seed` is the commit of the repo
        that owns the table layout ("" skips the cache); `key` is JSON-serializable and must cover every
        table input, including DRAM bases and fabric node ids. Returns True on a hit.
        )");

    // UMD-backed read: reads a KV chunk's DRAM bytes over a bare tt::umd::Cluster. The mechanism
    // the migration worker uses (disaggregation/migration/src/worker/device_io.cpp). The chip is
    // selected by ASIC unique_id (the caller resolves fabric_node -> unique_id from the runner's
    // device-map sidecar); noc_addr is (dram_view << 32) | local_addr, size_bytes is the chunk size
    // (19584 for a bfp8 KV chunk).
    mod.def(
        "read_dram_umd",
        [](uint64_t unique_id, uint64_t noc_addr, uint32_t size_bytes) {
            auto buf = tt::tt_metal::internal::disaggregation::read_dram_umd(unique_id, noc_addr, size_bytes);
            return nb::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
        },
        nb::arg("unique_id"),
        nb::arg("noc_addr"),
        nb::arg("size_bytes"),
        R"(
        Read a KV chunk's raw bytes over UMD from a device-less process, CONCURRENT with the running
        server. Uses a bare tt::umd::Cluster, selecting the chip by ASIC unique_id and the DRAM
        view/offset from noc_addr = (dram_view << 32) | local_addr. Mirrors the migration worker's device_io
        read path.
        )");

    bind_cache_layout_api(mod);
}

}  // namespace ttnn::disaggregation
