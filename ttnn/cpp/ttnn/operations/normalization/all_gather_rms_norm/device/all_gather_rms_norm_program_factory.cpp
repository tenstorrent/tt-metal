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
constexpr uint32_t reduce_one = tt::CBIndex::c_9;      // SUM scaler (1.0) for the gather-reduce (ring>1)
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
// Lays out the fused pre-stats -> all-gather -> post-normalize pipeline: each worker core computes its
// row-slice's per-device E[x^2] partial, the multi-device path all-gathers those partials across the ring
// over the fabric mux, and the compute kernel finishes the normalization (+ optional gamma/beta and the
// num_heads head-split).  A single device performs the whole reduce locally (no fabric).
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

    // Line-mcast extent for the stats all-gather (how many hops to send in each direction).
    uint32_t num_targets_forward = 0;
    uint32_t num_targets_backward = 0;
    if (!single_device) {
        std::tie(num_targets_forward, num_targets_backward) =
            ::ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, args.topology, false);
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
    // Use fp32 for all intermediate VALUES (x^2, the local/gathered stat partials, global E[x^2], 1/rms, and
    // the normalized/gamma intermediates) + fp32 dest accumulation, so the stats path carries no bf16
    // rounding. The reduce/eps SCALAR CBs stay Float16_b: making them fp32 corrupts the reduce on the
    // round-robin path (rsqrt blows up to inf), and they're constants, not intermediate values.
    fp32_dest_acc_en = true;
    const tt::DataFormat interm_df = tt::DataFormat::Float32;

    // Shape / work split.  Reduction is over the (local, per-device) last dim; rows = product of leading
    // dims.  We split tile-rows across the available worker cores (1 worker per link for the stats fabric
    // transfer; the row work is spread over the worker grid).
    const auto& shape = a.padded_shape();
    const uint32_t Wt = shape[-1] / TILE_WIDTH;  // local width in tiles
    const uint32_t num_tile_rows = a.physical_volume() / shape[-1] / TILE_HEIGHT;
    const uint32_t block_size = std::min<uint32_t>(Wt, 4);  // streaming block; tuned later

    // Per-row pipeline (Option A): gather_chunk == 1 -> the chunk loop degenerates to a per-row loop
    // (read row -> reduce -> send one partial -> one fabric barrier -> normalize -> write -> next row).
    // This removes the multi-row resident-chunk / batched-barrier machinery (one fabric barrier per row).
    const uint32_t gather_chunk = 1u;

    const auto grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t num_links = args.num_links;

    // Work split: fan the tile-rows across the compute grid (one core per tile-row) so reduce/normalize run
    // in parallel (chunk size 1). Designated-gather (multi-device): reserve the bottom grid row for ONE
    // dedicated direct-fabric gather core; every other core is a worker handling exactly one tile-row. The
    // workers relay their per-row stat partials to the gather core over NoC; the gather core owns the direct
    // fabric connection (no mux) and line-multicasts them. This decouples compute parallelism (full grid)
    // from fabric (the gather core), so the worker count is no longer capped by mux channels.
    CoreRangeSet worker_core_range;
    std::vector<CoreCoord> worker_cores;
    CoreCoord gather_core_logical;
    CoreCoord gather_core_virtual;
    if (single_device) {
        const uint32_t n = std::max<uint32_t>(1, std::min<uint32_t>(num_tile_rows, grid.x * grid.y));
        worker_core_range = num_cores_to_corerangeset(n, grid, /*row_wise=*/true);
        worker_cores = corerange_to_cores(worker_core_range, n, /*row_wise=*/true);
    } else {
        const CoreCoord worker_grid{grid.x, grid.y - 1};  // reserve bottom row for the gather core
        const uint32_t max_worker_cores = static_cast<uint32_t>(worker_grid.x * worker_grid.y);
        // Use the full worker grid; when NCHt > workers each worker handles ceil(NCHt/workers) contiguous
        // rows, processed one-per-round, and the gather core barriers per round (rows-per-worker = "super
        // chunks"). The contiguous split is monotonic in worker index, so each round's active set is a prefix.
        const uint32_t n = std::max<uint32_t>(1, std::min<uint32_t>(num_tile_rows, max_worker_cores));
        worker_core_range = num_cores_to_corerangeset(n, worker_grid, /*row_wise=*/true);
        worker_cores = corerange_to_cores(worker_core_range, n, /*row_wise=*/true);
        gather_core_logical = CoreCoord(0, grid.y - 1);
        gather_core_virtual = mesh_device->worker_core_from_logical_core(gather_core_logical);
    }
    const uint32_t num_cores = worker_cores.size();

    // ----- Circular buffers -----
    const bool has_gamma = gamma_tensor.has_value();
    const bool has_beta = beta_tensor.has_value();
    const bool fuse_pre_add = residual.has_value();

    // The reduce keeps all Wt local-width tiles resident per row (cumulative wait + indexed pack), so
    // input/x_squared/gamma/beta are sized to Wt; the per-row scalar CBs are 1 tile. input is single-buffered
    // (the fused kernel reuses the resident row for both x^2 and normalize) to keep the L1 footprint down.
    // Multi-device keeps a whole gather_chunk of rows resident across the gather (so pass 2 doesn't re-read
    // input); single-device streams one row at a time (gather_chunk == 1 -> Wt, unchanged).
    desc.cbs.push_back(make_cb(cb::input, gather_chunk * Wt, in_df, worker_core_range));
    desc.cbs.push_back(make_cb(cb::reduce_scalar, 1, tt::DataFormat::Float16_b, worker_core_range));
    if (ring_size > 1) {
        desc.cbs.push_back(make_cb(cb::reduce_one, 1, tt::DataFormat::Float16_b, worker_core_range));
    }
    desc.cbs.push_back(make_cb(cb::x_squared, Wt, interm_df, worker_core_range));
    // Stats CBs hold a chunk of per-row partials, double-buffered so a peer's fabric write of chunk k+1
    // cannot land on a region the compute kernel is still consuming for chunk k.
    desc.cbs.push_back(make_cb(cb::local_stats, gather_chunk * 2, interm_df, worker_core_range));
    desc.cbs.push_back(make_cb(cb::gathered_stats, ring_size * gather_chunk * 2, interm_df, worker_core_range));
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
    // x_normed / gamma_out are block-sized: the compute kernel fuses normalize -> gamma -> beta per block,
    // consuming each block immediately, so these never need to hold a full Wt row. This is what lets the
    // fused single kernel fit L1 at full DiT width (Wt=128). x_normed only feeds the gamma path; the plain
    // case writes the normalize result straight to output.
    if (has_gamma) {
        desc.cbs.push_back(make_cb(cb::x_normed, block_size * 2, interm_df, worker_core_range));
    }
    if (has_gamma && has_beta) {
        desc.cbs.push_back(make_cb(cb::gamma_out, block_size * 2, interm_df, worker_core_range));
    }
    desc.cbs.push_back(make_cb(cb::output, block_size * 2, out_df, worker_core_range));
    // No packet-header CB: only the dedicated gather core touches fabric, and PacketHeaderPool allocates
    // from a fixed reserved L1 region (MEM_PACKET_HEADER_POOL_BASE), not a user CB.

    // ----- Semaphores (local mcast / stats-ready coordination) -----
    // The all-gather out-ready (drain) semaphore and the cross-device init barrier are workload-scoped
    // GlobalSemaphores passed in by address.  The single local semaphore below coordinates "local stats
    // computed -> ready to all-gather" between the worker cores on this device.
    // Designated-gather semaphores. Ids are deterministic by index, so get_semaphore(id) gives the same L1
    // address on every core -> workers and the gather core can reference each other's sem by id:
    //   id 0 (relay_ready): lives on the gather core; each worker bumps it after staging its relay slot.
    //   id 1 (done): lives on the workers; the gather core bumps it as back-pressure (unused while
    //                NCHt <= workers, i.e. one row per worker, but declared for the general path).
    // Declared on the combined worker+gather range so both addresses are reserved/valid everywhere.
    static constexpr uint32_t relay_ready_semaphore_id = 0;
    static constexpr uint32_t done_semaphore_id = 1;
    if (!single_device) {
        std::vector<CoreRange> all_ranges(worker_core_range.ranges().begin(), worker_core_range.ranges().end());
        all_ranges.emplace_back(gather_core_logical);
        CoreRangeSet all_cores(all_ranges);
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = relay_ready_semaphore_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = done_semaphore_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
    }

    // ----- Direct-fabric route + relay buffer (designated-gather, multi-device) -----
    // The gather core connects directly to the fabric EDM toward its ring neighbor(s) and line-multicasts
    // each worker's partial. Workers stage their partials into the gather core's relay buffer -- a raw L1
    // region at the unreserved base (the gather core has no CBs, so the base is free, and the address is
    // host-known so workers can address it). dst_nodes = the immediate fwd/bwd neighbors; num_connections =
    // how many exist (the line-mcast hops then reach all ring peers in each direction).
    std::optional<tt::tt_fabric::FabricNodeId> sender_fabric_node_id;
    std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
    size_t relay_base = 0;
    const uint32_t stat_tile_bytes = tt::tile_size(interm_df);
    const uint32_t relay_slot_stride = stat_tile_bytes + 32;  // partial tile + 16B metadata, 32B aligned
    (void)barrier_semaphore;  // workload init-barrier sem; not referenced by the designated-gather path
    if (!single_device) {
        TT_FATAL(num_links == 1, "all_gather_rms_norm designated-gather supports num_links == 1 (got {})", num_links);
        sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
        if (forward_coord.has_value()) {
            dst_nodes.push_back(mesh_device->get_fabric_node_id(forward_coord.value()));
        }
        if (backward_coord.has_value()) {
            dst_nodes.push_back(mesh_device->get_fabric_node_id(backward_coord.value()));
        }
        relay_base = mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    }
    const uint32_t num_connections = static_cast<uint32_t>(dst_nodes.size());

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
    if (!single_device) {
        // Gates the fabric stats all-gather path (and its fabric includes) in the writer kernel.
        defines["RING_GT_1"] = "1";
    }
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
        reduce_factor,
        cb::reduce_one,
        ring_size,
        gather_chunk};
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
        Wt,
        ring_index,
        // Head-split: width-tile w of global tile-row gr -> head h = w/head_dim_tiles, within-head tile
        // e = w%head_dim_tiles, output page h*per_head_stride + gr*head_dim_tiles + e. num_heads == 1 ->
        // head_dim_tiles == Wt -> the scatter reduces to a contiguous write.
        Wt / args.num_heads,       // head_dim_tiles
        num_tile_rows,             // m_tiles (global)
        relay_ready_semaphore_id,  // gather-core sem the worker bumps after staging its relay
        relay_slot_stride,         // gather-core relay slot stride (bytes)
        done_semaphore_id};        // our local back-pressure sem the gather core bumps per round
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
        static_cast<uint32_t>(fp32_dest_acc_en),
        cb::reduce_one,
        gather_chunk};
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

    // Push kernels in a stable order; the index is the KernelHandle.
    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    constexpr tt::tt_metal::KernelHandle reader_kernel_id = 0;
    constexpr tt::tt_metal::KernelHandle writer_kernel_id = 1;
    constexpr tt::tt_metal::KernelHandle compute_kernel_id = 2;

    // Dedicated direct-fabric gather kernel (multi-device): one core on the reserved bottom row owns the
    // fabric connection and line-multicasts every worker's relayed partial. Pushed AFTER reader/writer/
    // compute so their handles stay 0/1/2.
    if (!single_device) {
        std::vector<uint32_t> gather_ct_args = {
            stat_tile_bytes,
            relay_slot_stride,
            relay_ready_semaphore_id,
            done_semaphore_id,
            forward_coord.has_value() ? 1u : 0u,                      // start_hops_forward
            forward_coord.has_value() ? num_targets_forward : 0u,     // range_hops_forward
            backward_coord.has_value() ? 1u : 0u,                     // start_hops_backward
            backward_coord.has_value() ? num_targets_backward : 0u};  // range_hops_backward
        KernelDescriptor gather_desc;
        gather_desc.kernel_source = std::string(kKernelDir) + "dataflow/all_gather_rms_norm_gather.cpp";
        gather_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        gather_desc.core_ranges = CoreRangeSet(CoreRange(gather_core_logical));
        gather_desc.compile_time_args = std::move(gather_ct_args);
        gather_desc.config = DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
        };
        desc.kernels.push_back(std::move(gather_desc));
        tt::tt_metal::KernelHandle gather_kernel_id = desc.kernels.size() - 1;

        // Rows-per-worker rounds ("super chunks") + the per-round active-worker count. The contiguous row
        // split gives worker c rows [min(c*rpc,N), min((c+1)*rpc,N)); num_rows(c) is monotonically
        // non-increasing in c, so the workers active in round sc are exactly the prefix 0..active_count[sc]-1.
        const uint32_t rpc = (num_tile_rows + num_cores - 1) / num_cores;
        std::vector<uint32_t> active_count(rpc, 0);
        for (uint32_t c = 0; c < num_cores; ++c) {
            const uint32_t rs = std::min(c * rpc, num_tile_rows);
            const uint32_t re = std::min((c + 1) * rpc, num_tile_rows);
            for (uint32_t sc = 0; sc < re - rs; ++sc) {
                active_count[sc]++;
            }
        }
        // Gather RT args: relay_base, num_workers, num_super_chunks, num_connections, [active_count per
        // round], [worker virt coords...], then the direct-fabric connection args from the routing-plane helper.
        std::vector<uint32_t> gather_rt_args = {
            static_cast<uint32_t>(relay_base),
            num_cores,
            rpc,  // num_super_chunks
            num_connections};
        for (uint32_t sc = 0; sc < rpc; ++sc) {
            gather_rt_args.push_back(active_count[sc]);
        }
        for (uint32_t c = 0; c < num_cores; ++c) {
            const CoreCoord wv = mesh_device->worker_core_from_logical_core(worker_cores[c]);
            gather_rt_args.push_back(wv.x);
            gather_rt_args.push_back(wv.y);
        }
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
            sender_fabric_node_id.value(),
            dst_nodes,
            {0u},
            desc,
            gather_kernel_id,
            gather_core_logical,
            gather_rt_args);

        KernelDescriptor::RTArgList gather_rt;
        gather_rt.reserve(gather_rt_args.size());
        for (uint32_t v : gather_rt_args) {
            gather_rt.push_back(v);
        }
        desc.kernels[gather_kernel_id].emplace_runtime_args(gather_core_logical, gather_rt);
    }

    // ----- Runtime args -----
    // Fan the tile-rows across the worker cores: core c handles rows [c*rows_per_core, (c+1)*rows_per_core).
    const uint32_t rows_per_core = (num_tile_rows + num_cores - 1) / num_cores;
    for (uint32_t c = 0; c < num_cores; ++c) {
        const CoreCoord core = worker_cores[c];

        const uint32_t row_start = std::min(c * rows_per_core, num_tile_rows);
        const uint32_t row_end = std::min((c + 1) * rows_per_core, num_tile_rows);
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
            // out-ready: peers' gather cores line-multicast their partials straight into THIS worker's
            // gathered-stats slots and atomic-inc this semaphore, so we wait for the (ring_size-1) peers.
            // The worker relays its own partial to its slot on the (single) gather core, then signals it.
            writer_rt_args.push_back(out_ready_semaphore.address());
            writer_rt_args.push_back(ring_size - 1);  // out_ready_sem_wait_value (peers only)
            writer_rt_args.push_back(gather_core_virtual.x);
            writer_rt_args.push_back(gather_core_virtual.y);
            writer_rt_args.push_back(static_cast<uint32_t>(relay_base));
            writer_rt_args.push_back(c);  // this worker's relay slot index on the gather core
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
