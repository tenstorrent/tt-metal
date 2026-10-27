// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The flat routed expert with another op's program appended on every chip (one program per chip, so the two run
// together on disjoint cores): the other op comes as a MeshProgramDescriptor (e.g. a Python-built combine), its
// buffers inside the flat expert's arena on its own cells (the caller owns the arena: persistent, so the other op may
// keep state there across calls). Every flat y writer reports each local expert, once its rows have landed, by an
// increment of counts[e] on the collector core (se_cmb_done.hpp, slot order, no go handshake: counts are cumulative,
// the other op compares them with calls x writers).
#pragma once

#include <array>
#include <memory>
#include <tuple>
#include <vector>

#include <tt-metalium/experimental/mesh_program_descriptor.hpp>

#include "flat_routed_expert_program_factory.hpp"
#include "flat_routed_expert_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::flat_routed_expert {

struct FlatWithProgramParams {
    FlatRoutedExpertConfig flat;  // with cmb_rt set
    // (shared, never copied with the attributes: a non-owning view of the caller's descriptor for the call; null when
    // the caller passed an empty one)
    std::shared_ptr<const tt::tt_metal::experimental::MeshProgramDescriptor> extra;
    tt::tt_metal::CoreCoord collector{0, 0};  // logical
    uint32_t counts_addr = 0;
    uint32_t writers = 0;
    // patch_addresses = false (the op before): every buffer address the program bakes in (the extra program's are its
    // own args) is part of the cache key, plus the caller's key for the extra program, so a cache hit never needs a
    // patch, and a call whose buffers land elsewhere builds a new program (DeepSeek-V4.1 FIX.5).
    std::vector<uint32_t> addrs;
    std::vector<uint32_t> flat_key;
    uint64_t extra_key = 0;
    // y_l1: per appended program (same order), the y writers' L1 hand-off args (se_yrm.hpp SE_Y_L1; the
    // writer index is filled in here)
    std::vector<std::vector<uint32_t>> l1_args;
    // patch_addresses = true (DeepSeek-V4.1 FIX.5): no address in the key. A hit re-applies every buffer address:
    // the flat program's (its AddrPatch list and arena CBs, as flat_routed_expert's own hits), the writers' report
    // args (counts_addr) and L1 hand-off args, the appended programs' arena-backed CBs (moved to this call's arena at
    // their offsets), and, when extra_addrs differ from the ones the cached program last got, the appended programs'
    // runtime args copied from `extra` (which the call must then carry). extra_addrs: the caller's list of the
    // addresses its appended programs bake into their runtime args (compared, never hashed).
    bool patch_addresses = false;
    std::vector<uint32_t> extra_addrs;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "flat_key", "addrs", "extra_key", "writers", "counts_addr", "collector_x", "collector_y", "patch_addresses");
    auto attribute_values() const {
        return std::forward_as_tuple(
            flat_key,
            addrs,
            extra_key,
            writers,
            counts_addr,
            static_cast<uint32_t>(collector.x),
            static_cast<uint32_t>(collector.y),
            patch_addresses);
    }
};

// patch_addresses: one of the flat program's address args, its RuntimeArgsData cached (the kernel's per-core entry: a
// stable object, updated in place when the program is dispatched)
struct FlatWithProgramArgPatch {
    tt::tt_metal::RuntimeArgsData* args;
    uint32_t index;
    uint32_t offset;
};

struct FlatWithProgramSharedVariables {
    // (patch_addresses: what a hit re-applies on this range's program, and what it last applied: a hit writes only
    // what moved)
    FlatRoutedExpertSharedVariables flat;  // the flat program's arena CBs and y writers
    uint32_t prog_i = 0;                   // this range's appended program (the l1_args index)
    uint32_t extra_k0 = 0;                 // the first appended kernel's handle
    std::vector<std::pair<tt::tt_metal::CBHandle, uint32_t>> extra_cbs;  // appended arena-backed CBs (handle, offset)
    std::vector<uint32_t> extra_addrs;  // the extra_addrs the appended runtime args were last set for
    std::array<std::vector<FlatWithProgramArgPatch>, static_cast<size_t>(AddrSrc::Count)> by_src;  // flat AddrPatch
    std::array<uint32_t, static_cast<size_t>(AddrSrc::Count)> addr{};  // the buffer addresses last applied
    uint32_t counts_addr = 0;                                          // the writers' report args last applied
    std::vector<uint32_t> l1;                                          // the writers' L1 hand-off args last applied
};

struct FlatWithProgramFactory {
    using shared_variables_t = FlatWithProgramSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const FlatWithProgramParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const FlatRoutedExpertInputs& tensor_args,
        Tensor& tensor_return_value);

    // patch_addresses: re-applies every buffer address on a hit (see FlatWithProgramParams); otherwise nothing (the
    // key holds the addresses)
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const FlatWithProgramParams& operation_attributes,
        const FlatRoutedExpertInputs& tensor_args,
        Tensor& tensor_return_value);
};

struct FlatWithProgramDeviceOperation {
    using operation_attributes_t = FlatWithProgramParams;
    using tensor_args_t = FlatRoutedExpertInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<FlatWithProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    // the program is defined by the attributes (every baked buffer address, the plan key, the extra program's key) and
    // the tensors' specs: a custom hash over exactly those (the default canonical key also encodes the attribute
    // struct's other members and the tensors' mesh coordinates, so a caller could not predict a hit).
    // patch_addresses: no address at all (they are patched on a hit): the plan key, the extra program's key, the
    // writers / collector, the L1 hand-off arg counts, and every tensor's spec (optional ones with their presence).
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t& p, const tensor_args_t& t) {
        if (p.patch_addresses) {
            std::vector<uint32_t> l1_sizes;
            l1_sizes.reserve(p.l1_args.size());
            for (const auto& v : p.l1_args) {
                l1_sizes.push_back(static_cast<uint32_t>(v.size()));
            }
            auto h = ttsl::hash::hash_objects_with_default_seed(
                uint32_t{0xF5A7C0DEu},  // (another key space than the address-keyed mode)
                p.flat_key,
                p.extra_key,
                p.writers,
                static_cast<uint32_t>(p.collector.x),
                static_cast<uint32_t>(p.collector.y),
                l1_sizes,
                t.x.tensor_spec(),
                t.counts.tensor_spec(),
                t.regions.tensor_spec(),
                t.global_expert_ids.tensor_spec(),
                t.gate_up_weights.tensor_spec(),
                t.down_weights.tensor_spec(),
                t.done_words.tensor_spec(),
                t.arena.tensor_spec(),
                t.words.tensor_spec(),
                t.output.tensor_spec());
            for (const auto* o : {&t.reader_down_weights, &t.token_index}) {
                h = o->has_value() ? ttsl::hash::hash_objects(h, uint32_t{1}, (*o)->tensor_spec())
                                   : ttsl::hash::hash_objects(h, uint32_t{0});
            }
            return h;
        }
        return ttsl::hash::hash_objects_with_default_seed(
            p.flat_key,
            p.addrs,
            p.extra_key,
            p.writers,
            p.counts_addr,
            static_cast<uint32_t>(p.collector.x),
            static_cast<uint32_t>(p.collector.y),
            t.x.tensor_spec(),
            t.arena.tensor_spec(),
            t.output.tensor_spec());
    }
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& t) {
        return t.output.tensor_spec();
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t& t) {
        return t.output;
    }
};

}  // namespace ttnn::operations::bringup::flat_routed_expert

namespace ttnn::prim::bringup {
ttnn::Tensor flat_with_program(
    const ttnn::operations::bringup::flat_routed_expert::FlatWithProgramParams& params,
    const ttnn::operations::bringup::flat_routed_expert::FlatRoutedExpertInputs& inputs);
}  // namespace ttnn::prim::bringup
