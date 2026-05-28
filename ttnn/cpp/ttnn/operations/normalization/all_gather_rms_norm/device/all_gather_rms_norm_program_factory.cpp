// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_rms_norm_program_factory.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/normalization/all_gather_rms_norm/device/kernels/";

// Circular-buffer index assignments (see header / plan).  Kept in one place so the host arg lists and
// the kernel stubs stay in sync.
namespace cb {
constexpr uint32_t input = tt::CBIndex::c_0;           // streamed input tiles
constexpr uint32_t reduce_scalar = tt::CBIndex::c_1;   // 1/W reduce scalar (1 tile)
constexpr uint32_t x_squared = tt::CBIndex::c_2;       // x^2 intermediate
constexpr uint32_t local_stats = tt::CBIndex::c_3;     // per-device E[x^2] (1 tile)
constexpr uint32_t gathered_stats = tt::CBIndex::c_4;  // ring_size tiles after all-gather
constexpr uint32_t eps = tt::CBIndex::c_5;             // epsilon (1 tile)
constexpr uint32_t gamma = tt::CBIndex::c_6;           // weight (optional)
constexpr uint32_t beta = tt::CBIndex::c_7;            // bias (optional)
constexpr uint32_t var = tt::CBIndex::c_8;             // mean(E[x^2]) over devices
constexpr uint32_t recip_sqrt = tt::CBIndex::c_10;     // 1/sqrt(var + eps)
constexpr uint32_t x_normed = tt::CBIndex::c_12;       // x * 1/sqrt(var + eps)
constexpr uint32_t gamma_out = tt::CBIndex::c_13;      // x_normed * gamma (only when gamma AND beta)
constexpr uint32_t output = tt::CBIndex::c_14;         // final output
constexpr uint32_t packet_header = tt::CBIndex::c_24;  // reserved for fabric packet headers (clear of compute CBs)
}  // namespace cb

CBDescriptor make_cb(uint32_t index, uint32_t num_tiles, tt::DataFormat df, const CoreRangeSet& cores) {
    const uint32_t tile_bytes = tt::tile_size(df);
    return CBDescriptor{
        .total_size = num_tiles * tile_bytes,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = df,
            .page_size = tile_bytes,
        }}},
    };
}

// Build the per-coord ProgramDescriptor.  The fabric args depend on the sender coordinate (ring_index +
// forward/backward neighbor lookups), so this runs once per coord inside create_workload_descriptor().
//
// TODO(LLK): this lays out CBs, kernels (pointing at the stub kernel files), semaphores and runtime args
// for the fused pre-stats -> all-gather -> post-normalize pipeline.  The actual compute/dataflow math
// lives in the kernel stubs and is not yet implemented.
tt::tt_metal::ProgramDescriptor build_program_descriptor_at(
    const AllGatherRMSNormParams& args,
    const ttnn::MeshCoordinate& sender_device_coord,
    const AllGatherRMSNormInputs& tensor_args,
    Tensor& output,
    const tt::tt_metal::GlobalSemaphore& out_ready_semaphore,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    tt::tt_metal::ProgramDescriptor desc;

    const auto& a = tensor_args.input;
    const auto& residual = tensor_args.residual_input_tensor;
    const auto& gamma_tensor = tensor_args.weight;
    const auto& beta_tensor = tensor_args.bias;

    auto* mesh_device = a.device();
    const uint32_t ring_size = args.ring_size;
    const bool single_device = ring_size == 1;
    const uint32_t ring_index = single_device ? 0
                                              : ::ttnn::ccl::get_linearized_index_from_physical_coord(
                                                    a, sender_device_coord, args.cluster_axis);
    (void)ring_index;

    // Ring/line fabric neighbors for the all-gather of the stats. Only meaningful for ring_size > 1; a
    // single device performs the whole reduce locally (no fabric).
    std::optional<MeshCoordinate> forward_coord;
    std::optional<MeshCoordinate> backward_coord;
    if (!single_device) {
        forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            a, sender_device_coord, 1, args.topology, args.cluster_axis);
        backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            a, sender_device_coord, -1, args.topology, args.cluster_axis);
        TT_FATAL(
            forward_coord.has_value() || backward_coord.has_value(),
            "all_gather_rms_norm: device has no forward or backward fabric neighbor");
    }

    // Data formats.
    const tt::DataFormat in_df = datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    const tt::DataFormat gamma_df = gamma_tensor.has_value()
                                        ? datatype_to_dataformat_converter(gamma_tensor.value().dtype())
                                        : tt::DataFormat::Float16_b;
    const tt::DataFormat beta_df = beta_tensor.has_value()
                                       ? datatype_to_dataformat_converter(beta_tensor.value().dtype())
                                       : tt::DataFormat::Float16_b;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(mesh_device->arch(), args.compute_kernel_config);
    const tt::DataFormat interm_df = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

    // Shape / work split.  Reduction is over the (local, per-device) last dim; rows = product of leading
    // dims.  We split tile-rows across the available worker cores (1 worker per link for the stats fabric
    // transfer; the row work is spread over the worker grid).
    const auto& shape = a.padded_shape();
    const uint32_t Wt = shape[-1] / TILE_WIDTH;  // local width in tiles
    const uint32_t num_tile_rows = a.physical_volume() / shape[-1] / TILE_HEIGHT;
    const uint32_t block_size = std::min<uint32_t>(Wt, 4);  // streaming block; tuned later

    auto subdevice_id = args.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const uint32_t num_workers_per_link = 1;
    const auto [worker_core_range, worker_cores] = ::ttnn::ccl::choose_worker_cores(
        args.num_links, num_workers_per_link, mesh_device, subdevice_id, CoreCoord(0, 0), std::nullopt);

    // ----- Circular buffers -----
    const bool has_gamma = gamma_tensor.has_value();
    const bool has_beta = beta_tensor.has_value();
    const bool fuse_pre_add = residual.has_value();

    // The reduce keeps all Wt local-width tiles resident per row (cumulative wait + indexed pack), so
    // input/x_squared/gamma/beta are sized to Wt (input double-buffered); the per-row scalar CBs are 1 tile.
    desc.cbs.push_back(make_cb(cb::input, Wt * 2, in_df, worker_core_range));
    desc.cbs.push_back(make_cb(cb::reduce_scalar, 1, tt::DataFormat::Float16_b, worker_core_range));
    desc.cbs.push_back(make_cb(cb::x_squared, Wt, interm_df, worker_core_range));
    desc.cbs.push_back(make_cb(cb::local_stats, 1, interm_df, worker_core_range));
    desc.cbs.push_back(make_cb(cb::gathered_stats, ring_size, interm_df, worker_core_range));
    desc.cbs.push_back(make_cb(cb::eps, 1, tt::DataFormat::Float16_b, worker_core_range));
    if (has_gamma) {
        desc.cbs.push_back(make_cb(cb::gamma, Wt, gamma_df, worker_core_range));
    }
    if (has_beta) {
        desc.cbs.push_back(make_cb(cb::beta, Wt, beta_df, worker_core_range));
    }
    desc.cbs.push_back(make_cb(cb::var, 1, interm_df, worker_core_range));
    desc.cbs.push_back(make_cb(cb::recip_sqrt, 1, interm_df, worker_core_range));
    // x_normed (and the gamma intermediate) hold a full row of Wt tiles: the single compute kernel fills
    // all Wt in the normalize loop before the (sequential) gamma/beta loop consumes them, so sizing these
    // to a block would deadlock. The output CB is drained concurrently by the writer, so a block suffices.
    desc.cbs.push_back(make_cb(cb::x_normed, Wt, interm_df, worker_core_range));
    if (has_gamma && has_beta) {
        desc.cbs.push_back(make_cb(cb::gamma_out, Wt, interm_df, worker_core_range));
    }
    desc.cbs.push_back(make_cb(cb::output, block_size * 2, out_df, worker_core_range));

    // Reserved CB for fabric packet headers (atomic incs / unicast headers).
    static constexpr uint32_t num_packet_headers_storable = 8;
    const uint32_t packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_packet_headers_storable * packet_header_size_bytes * 2,
        .core_ranges = worker_core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb::packet_header),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes,
        }}},
    });

    // ----- Semaphores (local mcast / stats-ready coordination) -----
    // The all-gather out-ready (drain) semaphore and the cross-device init barrier are workload-scoped
    // GlobalSemaphores passed in by address.  The single local semaphore below coordinates "local stats
    // computed -> ready to all-gather" between the worker cores on this device.
    const uint32_t stats_ready_semaphore_id = 0;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = stats_ready_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = worker_core_range,
        .initial_value = 0,
    });

    // ----- Kernels -----
    union {
        float f;
        uint32_t u;
    } eps_bits{};
    eps_bits.f = args.eps;
    const uint32_t reduce_factor = (shape[-1] * ring_size);  // global reduction width

    std::map<std::string, std::string> defines;
    defines["FUSE_GAMMA"] = has_gamma ? "1" : "0";
    defines["FUSE_BETA"] = has_beta ? "1" : "0";
    defines["FUSE_PRE_ADD"] = fuse_pre_add ? "1" : "0";
    const KernelDescriptor::Defines defines_vec{defines.begin(), defines.end()};

    // A Tensix core runs exactly ONE compute kernel, so the fused op uses a single compute kernel that
    // does pre-reduce -> (all-gather of stats, ring_size > 1) -> post-normalize. The reader and writer are
    // data-movement kernels. The ring_size == 1 path performs the whole reduce locally (no fabric).
    (void)reduce_factor;

    // Reader.  reduce_factor (= local_W * ring_size = global reduction width) lets the reader generate the
    // AVG reduce scalar so the reduction yields the GLOBAL mean E[x^2], not a per-device mean.
    std::vector<uint32_t> reader_ct_args = {
        cb::input,
        cb::reduce_scalar,
        cb::eps,
        cb::gamma,
        cb::beta,
        Wt,
        block_size,
        static_cast<uint32_t>(has_gamma),
        static_cast<uint32_t>(has_beta),
        reduce_factor};
    TensorAccessorArgs(a.buffer()).append_to(reader_ct_args);
    if (has_gamma) {
        TensorAccessorArgs(gamma_tensor.value().buffer()).append_to(reader_ct_args);
    }
    if (has_beta) {
        TensorAccessorArgs(beta_tensor.value().buffer()).append_to(reader_ct_args);
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kKernelDir) + "dataflow/all_gather_rms_norm_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = worker_core_range;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.defines = defines_vec;
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer.  For ring_size == 1 it simply writes the output tiles back to DRAM; for ring_size > 1 it also
    // performs the fabric all-gather of the per-device stats (see TODO in the kernel).
    std::vector<uint32_t> writer_ct_args = {
        cb::output,
        block_size,
        ring_size,
        cb::local_stats,
        cb::gathered_stats,
        cb::packet_header,
        num_packet_headers_storable,
        args.num_links};
    TensorAccessorArgs(output.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = std::string(kKernelDir) + "dataflow/all_gather_rms_norm_writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = worker_core_range;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.defines = defines_vec;
    writer_desc.config = WriterConfigDescriptor{};

    // Single fused compute kernel.
    std::vector<uint32_t> compute_ct_args = {
        cb::input,
        cb::reduce_scalar,
        cb::x_squared,
        cb::eps,
        cb::gamma,
        cb::beta,
        cb::var,
        cb::recip_sqrt,
        cb::x_normed,
        cb::gamma_out,
        cb::output,
        cb::local_stats,
        cb::gathered_stats,
        Wt,
        block_size,
        ring_size,
        static_cast<uint32_t>(has_gamma),
        static_cast<uint32_t>(has_beta),
        static_cast<uint32_t>(fp32_dest_acc_en)};
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(kKernelDir) + "compute/all_gather_rms_norm_compute.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = worker_core_range;
    compute_desc.compile_time_args = std::move(compute_ct_args);
    compute_desc.defines = defines_vec;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    // Push kernels in a stable order; the index is the KernelHandle used by the fabric helper below.
    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    constexpr tt::tt_metal::KernelHandle reader_kernel_id = 0;
    constexpr tt::tt_metal::KernelHandle writer_kernel_id = 1;
    constexpr tt::tt_metal::KernelHandle compute_kernel_id = 2;

    // ----- Runtime args -----
    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
    std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
    if (forward_coord.has_value()) {
        dst_nodes.push_back(mesh_device->get_fabric_node_id(forward_coord.value()));
    }
    if (backward_coord.has_value()) {
        dst_nodes.push_back(mesh_device->get_fabric_node_id(backward_coord.value()));
    }
    const uint32_t num_connections = dst_nodes.size();

    CoreCoord drain_sync_core;
    for (uint32_t link = 0; link < args.num_links; ++link) {
        const CoreCoord core = worker_cores[link];
        if (link == 0) {
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        // Per-core row range for the reduction work.
        const uint32_t rows_per_worker = (num_tile_rows + args.num_links - 1) / args.num_links;
        const uint32_t row_start = std::min(link * rows_per_worker, num_tile_rows);
        const uint32_t row_end = std::min((link + 1) * rows_per_worker, num_tile_rows);
        const uint32_t num_rows_this_worker = row_end - row_start;
        const uint32_t tile_offset = row_start * Wt;

        // Reader RT args. Buffer base addresses are bound as Buffer* so the framework patches them on the
        // cache-hit fast path.
        KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(a.buffer());  // input address (Buffer* binding)
        reader_rt_args.push_back(num_rows_this_worker);
        reader_rt_args.push_back(tile_offset);
        reader_rt_args.push_back(eps_bits.u);
        if (has_gamma) {
            reader_rt_args.push_back(gamma_tensor.value().buffer());
        }
        if (has_beta) {
            reader_rt_args.push_back(beta_tensor.value().buffer());
        }
        desc.kernels[reader_kernel_id].emplace_runtime_args(core, reader_rt_args);

        // Compute RT args: number of tile-rows this worker processes.
        KernelDescriptor::RTArgList compute_rt_args;
        compute_rt_args.push_back(num_rows_this_worker);
        desc.kernels[compute_kernel_id].emplace_runtime_args(core, compute_rt_args);

        // Writer RT args. Index 0 is the output buffer base address (bound as Buffer*).
        std::vector<uint32_t> writer_rt_args = {
            0,                          // placeholder for output buffer address (replaced by Buffer* below)
            num_rows_this_worker * Wt,  // num output tiles
            tile_offset,
        };
        if (!single_device) {
            // ring_size > 1: append the stats-all-gather coordination + fabric connection args.
            writer_rt_args.push_back(out_ready_semaphore.address());
            writer_rt_args.push_back(barrier_semaphore.address());
            writer_rt_args.push_back(drain_sync_core.x);
            writer_rt_args.push_back(drain_sync_core.y);
            writer_rt_args.push_back(ring_size * args.num_links);  // out_ready_sem_wait_value
            writer_rt_args.push_back(num_connections);
            writer_rt_args.push_back(stats_ready_semaphore_id);
            tt::tt_metal::KernelHandle writer_kernel_id_mut = writer_kernel_id;
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                sender_fabric_node_id, dst_nodes, {link}, desc, writer_kernel_id_mut, core, writer_rt_args);
        }

        KernelDescriptor::RTArgList writer_rt_args_builder;
        writer_rt_args_builder.reserve(writer_rt_args.size());
        writer_rt_args_builder.push_back(output.buffer());  // index 0: output Buffer* binding
        for (size_t i = 1; i < writer_rt_args.size(); ++i) {
            writer_rt_args_builder.push_back(writer_rt_args[i]);
        }
        desc.kernels[writer_kernel_id].emplace_runtime_args(core, writer_rt_args_builder);
    }

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor AllGatherRMSNormProgramFactory::create_workload_descriptor(
    const AllGatherRMSNormParams& operation_attributes,
    const AllGatherRMSNormInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    auto* mesh_device = tensor_args.input.device();
    auto subdevice_id = operation_attributes.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    ttnn::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // Allocate the workload-scoped init-barrier GlobalSemaphore (parked so it outlives the cached
    // workload).  The user-supplied operation_attributes.semaphore is the all-gather out-ready / drain
    // semaphore and is referenced by address in the writer runtime args.
    workload_descriptor.semaphores.push_back(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, tt::tt_metal::BufferType::L1));
    const auto& barrier_semaphore = workload_descriptor.semaphores[0];

    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);

    // One ProgramDescriptor per coord (ring_index + fabric neighbors are coord-dependent).
    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_program_descriptor_at(
            operation_attributes,
            coord,
            tensor_args,
            tensor_return_value,
            operation_attributes.semaphore,
            barrier_semaphore);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return workload_descriptor;
}

}  // namespace ttnn::prim
