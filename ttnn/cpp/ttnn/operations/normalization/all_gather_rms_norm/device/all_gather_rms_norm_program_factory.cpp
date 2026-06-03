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
    const tt::DataFormat interm_df = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

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

    // Work split: fan the tile-rows across the compute grid (one core per row-slice) so reduce/normalize run
    // in parallel. Multi-device reserves the bottom grid row for the fabric mux cores and routes every
    // worker's stats all-gather through the mux; each worker uses one mux channel per direction, so the
    // worker count is capped by the mux channels-per-core budget.
    static constexpr uint32_t kMaxMuxChannels = 8;  // full-size channels per mux core
    CoreRangeSet worker_core_range;
    std::vector<CoreCoord> worker_cores;
    if (single_device) {
        const uint32_t n = std::max<uint32_t>(1, std::min<uint32_t>(num_tile_rows, grid.x * grid.y));
        worker_core_range = num_cores_to_corerangeset(n, grid, /*row_wise=*/true);
        worker_cores = corerange_to_cores(worker_core_range, n, /*row_wise=*/true);
    } else {
        const CoreCoord worker_grid{grid.x, grid.y - 1};  // reserve bottom row for mux cores
        const uint32_t max_worker_cores = static_cast<uint32_t>(worker_grid.x * worker_grid.y);
        const uint32_t mux_cap = num_links * kMaxMuxChannels;
        const uint32_t n = std::max<uint32_t>(1, std::min({num_tile_rows, max_worker_cores, mux_cap}));
        worker_core_range = num_cores_to_corerangeset(n, worker_grid, /*row_wise=*/true);
        worker_cores = corerange_to_cores(worker_core_range, n, /*row_wise=*/true);
    }
    const uint32_t num_cores = worker_cores.size();
    // One mux channel per worker per direction; with num_links links, workers are striped across links.
    const uint32_t num_workers_per_link = single_device ? 1u : ((num_cores + num_links - 1) / num_links);

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

    // Reserved CB for fabric packet headers (atomic incs / unicast headers). Only needed for the
    // multi-device fabric path; get_tt_fabric_packet_header_size_bytes() touches the (uninitialized on a
    // single device) fabric context, so skip the CB entirely when single_device.
    static constexpr uint32_t num_packet_headers_storable = 8;
    if (!single_device) {
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
    }

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

    // Per-worker fabric-mux handshake semaphores (one set of 5 per direction): termination_sync,
    // local_status, local_flow_control, local_teardown, local_buffer_index. Declared with the SAME ids on
    // every worker core (same L1 address) so the termination master's sync semaphore is reachable from peers.
    // Forward uses ids [1..5], backward [6..10].
    static constexpr uint32_t kMuxSemFwdBase = 1;  // ids 1..5
    static constexpr uint32_t kMuxSemBwdBase = 6;  // ids 6..10
    if (!single_device) {
        for (uint32_t id = kMuxSemFwdBase; id < kMuxSemBwdBase + 5; ++id) {
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = id,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = worker_core_range,
                .initial_value = 0,
            });
        }
    }

    // ----- Fabric mux setup (multi-device) -----
    // One mux core per direction opens the fabric EDM connection; every worker connects to it as a client
    // (one full-size channel each). Mux cores live on the reserved bottom grid row, disjoint from workers.
    static constexpr uint32_t kMuxNumBuffersPerChannel = 8;
    std::optional<tt::tt_fabric::FabricMuxConfig> mux_cfg;
    std::optional<tt::tt_fabric::FabricNodeId> sender_fabric_node_id;
    std::optional<tt::tt_fabric::FabricNodeId> forward_node_id;
    std::optional<tt::tt_fabric::FabricNodeId> backward_node_id;
    CoreCoord mux_fwd_logical, mux_bwd_logical;
    CoreCoord mux_fwd_virtual, mux_bwd_virtual;
    (void)barrier_semaphore;  // workload init-barrier sem; not referenced by the mux path
    if (!single_device) {
        // A single mux pair (forward/backward) is placed; multi-link striping is future work.
        TT_FATAL(num_links == 1, "all_gather_rms_norm mux path currently supports num_links == 1 (got {})", num_links);
        sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
        if (forward_coord.has_value()) {
            forward_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
        }
        if (backward_coord.has_value()) {
            backward_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
        }
        const size_t mux_base_l1 = mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
        const size_t mux_buf_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
        mux_cfg.emplace(
            static_cast<uint8_t>(num_workers_per_link),      // num_full_size_channels
            static_cast<uint8_t>(0),                         // num_header_only_channels
            static_cast<uint8_t>(kMuxNumBuffersPerChannel),  // num_buffers_full_size_channel
            static_cast<uint8_t>(0),                         // num_buffers_header_only_channel
            mux_buf_bytes,                                   // buffer_size_bytes_full_size_channel
            mux_base_l1);
        mux_fwd_logical = CoreCoord(0, grid.y - 1);
        mux_bwd_logical = CoreCoord(1, grid.y - 1);
        mux_fwd_virtual = mesh_device->worker_core_from_logical_core(mux_fwd_logical);
        mux_bwd_virtual = mesh_device->worker_core_from_logical_core(mux_bwd_logical);
    }

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
        cb::packet_header,
        num_packet_headers_storable,
        args.num_links,
        Wt,
        ring_index,
        num_targets_forward,
        num_targets_backward,
        forward_coord.has_value() ? 1u : 0u,                     // start_distance_in_hops_forward
        forward_coord.has_value() ? num_targets_forward : 0u,    // range_hops_forward
        backward_coord.has_value() ? 1u : 0u,                    // start_distance_in_hops_backward
        backward_coord.has_value() ? num_targets_backward : 0u,  // range_hops_backward
        gather_chunk,
        // Fabric-mux compile-time args (0 on single device; the writer reads them only under RING_GT_1).
        // The TensorAccessor offset below stays fixed regardless, so these are always present.
        mux_cfg
            ? static_cast<uint32_t>(mux_cfg->get_num_buffers(tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL))
            : 0u,
        mux_cfg ? static_cast<uint32_t>(
                      mux_cfg->get_buffer_size_bytes(tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL))
                : 0u,
        mux_cfg ? static_cast<uint32_t>(mux_cfg->get_status_address()) : 0u,
        mux_cfg ? static_cast<uint32_t>(mux_cfg->get_termination_signal_address()) : 0u,
        num_workers_per_link,  // num_mux_clients
        // Head-split: the writer scatters each row's Wt output tiles into the (1, num_heads, M, head_dim)
        // output. head_dim_tiles = Wt/num_heads (per-head width in tiles), m_tiles = global tile-row count.
        // num_heads == 1 -> head_dim_tiles == Wt -> the scatter reduces to a contiguous write.
        Wt / args.num_heads,  // head_dim_tiles
        num_tile_rows};       // m_tiles (global)
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

    // Fabric mux kernel (multi-device): one kernel instance covering the (≤2) mux cores, each opening the
    // fabric EDM connection toward its neighbor. Pushed AFTER reader/writer/compute so their handles stay 0/1/2.
    if (!single_device) {
        std::vector<CoreRange> mux_ranges;
        if (forward_coord.has_value()) {
            mux_ranges.emplace_back(mux_fwd_logical);
        }
        if (backward_coord.has_value()) {
            mux_ranges.emplace_back(mux_bwd_logical);
        }
        KernelDescriptor mux_desc;
        mux_desc.kernel_source = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
        mux_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        mux_desc.core_ranges = CoreRangeSet(mux_ranges);
        mux_desc.compile_time_args = mux_cfg->get_fabric_mux_compile_time_args();
        mux_desc.config = DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            // The fabric mux MUST run on NOC 0 to match its RISCV_0 processor (and the EDM's NoC
            // convention) -- the same config the proven imperative mux ops (all_gather_async,
            // reduce_scatter_minimal_async) use. Pairing RISCV_0 with RISCV_1_default (NOC 1) works on a
            // quiet fabric but races/corrupts against the model's fabric traffic in-context.
            .noc = tt::tt_metal::NOC::RISCV_0_default,
        };
        desc.kernels.push_back(std::move(mux_desc));
        const tt::tt_metal::KernelHandle mux_kernel_id = desc.kernels.size() - 1;

        auto set_mux_rt = [&](const CoreCoord& mux_logical, const tt::tt_fabric::FabricNodeId& dst) {
            auto rt = mux_cfg->get_fabric_mux_run_time_args<tt::tt_metal::ProgramDescriptor>(
                sender_fabric_node_id.value(), dst, /*link_idx=*/0, desc, mux_logical);
            KernelDescriptor::RTArgList mux_rt;
            mux_rt.reserve(rt.size());
            for (uint32_t v : rt) {
                mux_rt.push_back(v);
            }
            desc.kernels[mux_kernel_id].emplace_runtime_args(mux_logical, mux_rt);
        };
        if (forward_coord.has_value()) {
            set_mux_rt(mux_fwd_logical, forward_node_id.value());
        }
        if (backward_coord.has_value()) {
            set_mux_rt(mux_bwd_logical, backward_node_id.value());
        }
    }

    // ----- Runtime args -----
    // Fan the tile-rows across the worker cores: core c handles rows [c*rows_per_core, (c+1)*rows_per_core).
    const uint32_t rows_per_core = (num_tile_rows + num_cores - 1) / num_cores;
    // Termination master for the mux teardown handshake is worker core 0 (per direction).
    const CoreCoord term_master_virtual =
        single_device ? CoreCoord{} : mesh_device->worker_core_from_logical_core(worker_cores[0]);
    for (uint32_t c = 0; c < num_cores; ++c) {
        const CoreCoord core = worker_cores[c];
        const CoreCoord core_virtual = mesh_device->worker_core_from_logical_core(core);

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
            // Per-core out-ready handshake: peers' same-coord core multicasts an atomic-inc to THIS core's
            // out-ready semaphore; wait value is ring_size (one inc per peer, including the local inc).
            const uint32_t worker_id = c / num_links;  // mux channel index within the link
            writer_rt_args.push_back(out_ready_semaphore.address());
            writer_rt_args.push_back(core_virtual.x);
            writer_rt_args.push_back(core_virtual.y);
            writer_rt_args.push_back(ring_size);  // out_ready_sem_wait_value

            // Mux connection args, forward then backward (17 each, matching parse_mux_connection_args).
            using CT = tt::tt_fabric::FabricMuxChannelType;
            auto push_mux_args = [&](bool valid, const CoreCoord& mux_virtual, uint32_t sem_base) {
                writer_rt_args.push_back(valid ? 1u : 0u);   // mux_connection_valid
                writer_rt_args.push_back(c == 0 ? 1u : 0u);  // is_termination_master
                writer_rt_args.push_back(mux_virtual.x);     // fabric_mux_x
                writer_rt_args.push_back(mux_virtual.y);     // fabric_mux_y
                writer_rt_args.push_back(
                    static_cast<uint32_t>(mux_cfg->get_channel_base_address(CT::FULL_SIZE_CHANNEL, worker_id)));
                writer_rt_args.push_back(
                    static_cast<uint32_t>(mux_cfg->get_connection_info_address(CT::FULL_SIZE_CHANNEL, worker_id)));
                writer_rt_args.push_back(
                    static_cast<uint32_t>(mux_cfg->get_connection_handshake_address(CT::FULL_SIZE_CHANNEL, worker_id)));
                writer_rt_args.push_back(
                    static_cast<uint32_t>(mux_cfg->get_flow_control_address(CT::FULL_SIZE_CHANNEL, worker_id)));
                writer_rt_args.push_back(
                    static_cast<uint32_t>(mux_cfg->get_buffer_index_address(CT::FULL_SIZE_CHANNEL, worker_id)));
                writer_rt_args.push_back(
                    static_cast<uint32_t>(mux_cfg->get_channel_credits_stream_id(CT::FULL_SIZE_CHANNEL, worker_id)));
                writer_rt_args.push_back(sem_base + 0u);  // termination_sync id
                writer_rt_args.push_back(sem_base + 1u);  // local_fabric_mux_status id
                writer_rt_args.push_back(sem_base + 2u);  // local_flow_control id
                writer_rt_args.push_back(sem_base + 3u);  // local_teardown id
                writer_rt_args.push_back(sem_base + 4u);  // local_buffer_index id
                writer_rt_args.push_back(term_master_virtual.x);
                writer_rt_args.push_back(term_master_virtual.y);
            };
            push_mux_args(forward_coord.has_value(), mux_fwd_virtual, kMuxSemFwdBase);
            push_mux_args(backward_coord.has_value(), mux_bwd_virtual, kMuxSemBwdBase);
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
