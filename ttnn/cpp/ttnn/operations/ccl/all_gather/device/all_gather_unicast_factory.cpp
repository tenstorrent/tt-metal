// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_unicast_factory.hpp"

#include "chunk_plan.hpp"

#include <tt-metalium/kernel_types.hpp>  // for tt::tt_metal::NOC
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::ccl {

using namespace ::ttnn::ccl;

////////////////////////////////////////////////////////////////
// Store-and-forward AllGather (line/ring over a single mesh axis; for Fabric 1D and 2D)
//
// Every device relays stripes to its neighbor one hop at a time; a shard reaches far devices by being
// re-forwarded at each hop. Forward and backward directions run on separate cores. Per direction: the reader
// (CB producer, no fabric) reads hop 0 from local input and later hops from what upstream relayed
// into our output; the writer (CB consumer) unicasts each stripe one hop to the neighbor's output (same
// address on every device). Direction/topology are runtime args, so both kernels compile once and run on all
// cores. Two semaphores: barrier_sem (init handshake) and data_valid_sem (relay gate + completion).
////////////////////////////////////////////////////////////////

AllGatherUnicastFactory::cached_mesh_workload_t AllGatherUnicastFactory::create_mesh_workload(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    if (operation_attributes.sub_core_grid.has_value()) {
        available_cores = available_cores.intersection(operation_attributes.sub_core_grid.value());
    }
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // Kernel needs to wait to receive all remote data before exiting, and in some cases needs to wait
    // for all remote devices to be ready before beginning operation.
    // Since Fabric doesn't provide such capability within kernels, we need to manually sync using global semaphores.
    // Allocate the semaphore in L1_SMALL to avoid fragmenting the larger L1 memory pool.
    // Two semaphores:
    // - barrier_sem: one-shot init handshake ("I'm alive") to the neighbor.
    // - data_valid_sem: chunks upstream has relayed into our output (relay gate + completion).
    bool l1_small_size = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL);
    auto sem_buffer_type = l1_small_size > 0 ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    if (sem_buffer_type != tt::tt_metal::BufferType::L1_SMALL) {
        log_warning(
            tt::LogOp,
            "Allocating semaphores in L1, which may fragment L1 and reduce headroom for subsequent op "
            "allocations. Configure an L1_SMALL region to mitigate this.");
    }
    auto barrier_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    auto data_valid_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args,
            output_tensor,
            barrier_sem,
            data_valid_sem,
            available_cores.num_cores());
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

AllGatherUnicastFactory::cached_program_t AllGatherUnicastFactory::create_at(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const AllGatherInputs& tensor_args,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& barrier_sem,
    const tt::tt_metal::GlobalSemaphore& data_valid_sem,
    uint32_t num_available_cores) {
    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::Program program{};
    auto* mesh_device = input_tensor.device();

    ////////////////////////////////////////////////////////////////
    // Fabric setup
    //
    // Glossary:
    //   relay          -- re-forward a received stripe from upstream one hop to downstream.
    //   slice          -- portion of tensor stripes allocated to this (link, worker)
    //   sink direction -- a direction that forwards nothing (num_hops == 0), e.g. a line endpoint's dead side.
    //   sink stripe    -- a stripe consumed here, not relayed onward (a line endpoint's incoming, or a ring
    //                     antipode).
    //   antipode       -- on a ring, the device N/2 hops away.
    ////////////////////////////////////////////////////////////////

    TT_FATAL(!operation_attributes.is_true_2d(), "all_gather unicast algorithm does not support true 2D topologies");

    const uint32_t axis = operation_attributes.get_1d_axis();
    const auto topology = operation_attributes.axis_topology[axis];
    const bool is_ring = tt::tt_fabric::is_ring_or_torus(topology);

    const uint32_t num_devices = operation_attributes.num_devices;
    const uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    auto fwd_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, sender_device_coord, 1, topology, axis);
    auto bwd_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, sender_device_coord, -1, topology, axis);

    // Relay steps a direction takes from a device: ring -> N/2; line fwd -> d+1, bwd -> N-d; 0 at a dead
    // endpoint. Also queried for the downstream device, to choose fine-grained vs single data_valid signalling.
    auto relay_hops = [&](uint32_t idx, bool is_forward) -> uint32_t {
        if (is_ring) {
            return num_devices / 2;
        }
        return is_forward ? (idx + 1 < num_devices ? idx + 1 : 0) : (idx > 0 ? num_devices - idx : 0);
    };
    const uint32_t fwd_hops = relay_hops(device_idx, true);
    const uint32_t bwd_hops = relay_hops(device_idx, false);
    TT_FATAL(fwd_hops > 0 || bwd_hops > 0, "device participates in neither direction");

    // Even-sized ring: for load balancing, the antipode device receives the antipode stripe as halves from both
    // forward and backward directions.
    const bool ring_even_split = is_ring && (num_devices % 2 == 0);
    // We use an init barrier to wait for remote output tensors to be allocated.
    // But we cannot skip init_barrier when persistent output buffer is used since:
    // - The persistent output buffer is also an input source (store-and-forward in the relay hops).
    // - The persistent output buffer may be reused across multiple invocations of this CCL.
    const bool do_init_barrier = true;

    const uint32_t fabric_packet_size = operation_attributes.packet_size;
    const auto arch = input_tensor.device()->arch();

    ////////////////////////////////////////////////////////////////
    // Chunks
    //
    // Where the two machineries meet. See the layout map in all_gather_device_operation.cpp.
    //
    // Glossary, one language everywhere:
    //   chunk           -- the unit we move. min(input page, output page).
    //   chunks_per_page -- how many chunks share one page. One number per side, and one of them is 1.
    //   bank_step       -- page-id step to the next chunk sitting next to this one in memory.
    //   run             -- chunks next to each other in memory. One NOC command, one packet segment.
    //   run_max         -- most chunks one run may take.
    //   payload_chunks  -- chunks the fabric payload could hold.
    //   packet_chunks   -- chunks a packet really holds: min(payload_chunks, 4 runs).
    //   entry           -- one CB page, in chunks. Always a whole number of packets.
    //   stripe          -- our chunks per output row. The one number all_gather hands the chunk
    //                      machinery: a run stops at the row edge, past which sits another device.
    //   hop             -- one relay step. Hop 0 sends our own input, later hops relay.
    //   sink            -- a stripe consumed here, not relayed on.
    //
    // The host sizes, the kernels walk. Reader and writer read bank_step off the TensorAccessor
    // themselves, so no layout is special-cased here and the two cannot disagree.
    ////////////////////////////////////////////////////////////////

    const auto chunks = chunk_sizes(input_tensor, output_tensor);
    const uint32_t chunk_size = chunks.chunk_size;
    const uint32_t in_chunks_per_page = chunks.in_chunks_per_page;
    const uint32_t out_chunks_per_page = chunks.out_chunks_per_page;

    // A hop past the first reads our own output back into the CB, which is packed at chunk stride, so
    // the output's aligned page must be exactly tiled by chunks. select_program_factory routes these
    // away; reaching here means unicast was forced (2D bent axis or neighbour-exchange fabric).
    TT_FATAL(
        output_tensor.buffer()->aligned_page_size() == out_chunks_per_page * chunk_size,
        "all_gather unicast cannot relay a padded output page: {} B of content in a {} B page holding {} chunk(s) of "
        "{} B. Use a row size that is a multiple of the output's memory alignment, or the same buffer type on both "
        "sides.",
        output_tensor.buffer()->page_size(),
        output_tensor.buffer()->aligned_page_size(),
        out_chunks_per_page,
        chunk_size);

    const uint32_t num_input_pages = input_tensor.buffer()->num_pages();
    const uint32_t num_output_chunks = num_input_pages * in_chunks_per_page;
    TT_FATAL(
        num_output_chunks / in_chunks_per_page == num_input_pages,
        "all_gather output chunk count overflowed uint32: {} input pages x {} chunks per page",
        num_input_pages,
        in_chunks_per_page);

    // --- Stripe ---
    // input_pages_per_stripe = num input pages along [gather dim .. last dim] this
    // device contributes per stripe. For a last-dim RM gather this is the *page* count,
    // which handles sharded RM input (> 1 input page per row).
    const uint32_t input_content_page = input_tensor.buffer()->page_size();
    const auto& input_shape = input_tensor.padded_shape();
    const auto tile_spec =
        input_tensor.layout() == Layout::TILE ? input_tensor.tensor_spec().tile() : tt::tt_metal::Tile();
    uint32_t input_pages_per_stripe = 1;
    for (int32_t i = operation_attributes.dim_from_end; i < 0; i++) {
        uint32_t extent;
        if (i == -1) {
            if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
                extent = input_shape[i] / tile_spec.get_width();
            } else {
                // This is a page count, so divide by the unaligned page size, not aligned
                extent = (input_shape[i] * input_tensor.element_size()) / input_content_page;
            }
        } else if (input_tensor.layout() == ttnn::TILE_LAYOUT && i == -2) {
            extent = input_shape[i] / tile_spec.get_height();
        } else {
            extent = input_shape[i];
        }
        input_pages_per_stripe *= extent;
    }

    // Our chunks per output row. Measured in chunks (not output pages) so multi-shard concat works:
    // a stripe's chunks are laid across output pages via the inner byte offset and may straddle pages.
    const uint32_t stripe = input_pages_per_stripe * in_chunks_per_page;
    TT_FATAL(stripe > 0, "stripe must be > 0");

    // --- Packets and runs ---
    // Whole chunks only: a packet cannot use a tail shorter than one chunk, and leaving that tail out
    // of the figure the CB entry is sized from keeps entry and packet boundaries coincident.
    // bank_step is 1 on a sharded output -- its chunks are already neighbours, so runs are long and
    // nothing below caps them.
    const uint32_t payload_chunks = std::max(1u, fabric_packet_size / chunk_size);
    const uint32_t packet_size = payload_chunks * chunk_size;
    const uint32_t bank_step =
        output_tensor.memory_config().is_sharded()
            ? 1u
            : std::max(1u, mesh_device->allocator()->get_num_banks(output_tensor.buffer()->buffer_type()));
    const uint32_t packet_chunks = packet_chunks_of(payload_chunks, stripe, bank_step);
    report_payload(packet_size, chunk_size, stripe, bank_step, arch);

    ////////////////////////////////////////////////////////////////
    // Core selection
    //
    // Each link runs two directions: forward (dir 0) and backward (dir 1). With workers_per_dir == 1 each
    // direction is a single core connected directly to its neighbor's ERISC. With > 1 the workers of a direction
    // can't each open a direct connection (an ERISC exposes one worker sender channel per direction), so they
    // share a fabric mux: one mux core per direction per link owns the connection and multiplexes their traffic.
    //
    // One (mux?, workers) group per direction per link, ordered link-major:
    //   group index = (link * num_directions) + dir
    ////////////////////////////////////////////////////////////////

    uint32_t num_links = operation_attributes.axis_num_links[axis];

    // Packed bytes of the gathered output crossing one link. At the *requested* link count, so if
    // the core grid later forces links down (which warns) the choices below stay the ones for the
    // wider config.
    const uint64_t per_link_bytes =
        output_tensor.tensor_spec().compute_packed_buffer_size_bytes() / std::max(1u, num_links);

    // --- Per-arch tuning ---
    uint32_t workers_per_dir = 1;  // >1 needs a fabric mux core per direction per link
    uint32_t packets_per_cb_entry = 1;
    uint32_t run_cap_bytes = 0;
    uint8_t mux_slots_per_channel = 2;
    const uint32_t cb_depth = 2;  // two entries: one filling while the other drains.
    if (arch == tt::ARCH::WORMHOLE_B0) {
        // Calibrated on T3000 at 2, 4 and 8 devices.
        //
        // workers_per_dir: a second worker needs a fabric mux -- 6 cores per link instead of 2, plus a
        // hop per packet -- so take it only where it is not slower, i.e. while the op is latency-bound.
        // A long stripe outlasts a transfer many times over, so a worker's runs stay inside one row and
        // its writes land sequentially at the destination; short stripes straddle a row edge on every
        // transfer, which is where the extra worker starts paying.
        const bool long_stripe = stripe >= 64;
        // Keyed on this device's own share of the link, not the whole gathered output: that share is what
        // actually transfers, and the boundaries below sit at the same per-device figure at 2, 4 and 8
        // devices. Stated against the total they would only be right at 8.
        const uint64_t device_bytes_per_link = per_link_bytes / std::max(1u, num_devices);
        const bool small = device_bytes_per_link <= 64 * 1024;
        const bool long_stripe_wins = long_stripe && (is_ring || device_bytes_per_link <= 512 * 1024);
        workers_per_dir = (small || long_stripe_wins) ? 1 : 2;
        packets_per_cb_entry = 3;  // multicast wants 1 here; every cell was swept on its own
        // mux_slots_per_channel: two slots let a worker stage its next packet while the mux forwards the
        // last. On a ring at scale that only interleaves the two workers' packets more finely at the
        // receiver, scattering its DRAM writes, so one slot wins there; a line, whose per-hop relay is
        // already serialised, keeps two. Ring only, and a ring cannot be shortened on a T3000 (the
        // wrap-around link needs all 8), so this one stays in whole-output terms.
        mux_slots_per_channel = (is_ring && per_link_bytes >= 2 * 1024 * 1024) ? 1 : 2;
        // No run cap: the best run length lands at the hardware ceiling (7616 B), so any value settable
        // here is already above it and would only cost payload.
    } else if (arch == tt::ARCH::BLACKHOLE) {
        // Calibrated at 8 devices x 2 links, ring and line, from 16 KB to 200 MB of output, at 2 KB and
        // 1088 B chunks. Stripe length moves none of these boundaries any more -- the chunk machinery
        // handles it -- so only volume and topology appear below.
        //
        // workers_per_dir: two, or four on a ring once the output is big enough to keep four fed. Each
        // extra worker feeds a link harder but costs a core and, past one, a mux core plus a NOC hop per
        // packet. Two is enough almost everywhere now that a packet is filled: two beats four by 5-9%
        // from 4 to 10 MB, ties at 20 MB, and loses 1.6-2.7% from 50 MB up, so the crossover sits
        // between 20 and 50 MB. One worker is never right in this range (+27% at 4 MB, +75% at 20 MB),
        // and three never beats the better of two and four by more than the run-to-run spread. Below
        // ~2 MB the choice is inside noise (within 4% across 16 KB to 1.7 MB).
        //
        // A line stays at two at every size: its relay crosses N-1 hops, so the per-hop send is the
        // serialising step and more feeders cannot help -- four measured 1.5% slower at 50 MB and level
        // at 100 MB. A ring sends both directions at once and does saturate its links, which is why the
        // volume rule is ring-only.
        const uint64_t total_output_bytes = per_link_bytes * num_links;
        workers_per_dir = (is_ring && total_output_bytes >= 32u * 1024u * 1024u) ? 4u : 2u;
        // packets_per_cb_entry: one. The CB entry rule below sizes the entry from what a packet
        // actually carries, which is what the old stripe band was reaching for by hand.
        packets_per_cb_entry = 1;
        // run_cap_bytes: capping a run costs packet fill but stops the walk parking in one DRAM bank. On
        // a ring the cap follows *link count*: every link runs its own 2 x workers_per_dir readers
        // against one shared DRAM, so the more links, the shorter each run has to be to keep the banks
        // spread...
        // ...but only once there are enough links to create that pressure, and that is now the whole
        // gate: at 2 links capping is a straight loss on both topologies -- ring 14% at 10 MB of output,
        // 21% at 50 MB, 22% at 100 MB; line 25% at 10 MB -- and irrelevant below that. At 4 links it was
        // worth +22% on a ring. The earlier line rule keyed on whether the slice split was even; that no
        // longer separates anything, so it is gone.
        // The 4-link figure predates the output-alignment and init-barrier fixes and has not been
        // re-measured since, so treat it as the weaker half of this rule.
        const bool cap_pays = num_links >= 4 && total_output_bytes >= 8u * 1024u * 1024u;
        run_cap_bytes = cap_pays ? (16384u / std::max(1u, num_links)) : 0u;
        // mux_slots_per_channel: one slot wins only on an even split, and loses more when the split is
        // uneven than it wins there, so the even case is not worth special-casing.
        mux_slots_per_channel = 2;
    }

    // Two upper bounds on a run: the geometry rule (chunk_plan.hpp) and the per-arch cap above. Take
    // the tighter of the two. 0 means no cap, which is what the kernels expect.
    uint32_t run_max = run_max_of(packet_chunks, payload_chunks, chunk_size);
    if (run_cap_bytes != 0) {
        const uint32_t arch_run_max = std::max(1u, run_cap_bytes / chunk_size);
        run_max = run_max != 0 ? std::min(run_max, arch_run_max) : arch_run_max;
    }

    // Shrink core usage to fit available core grid. Shrink workers_per_link first, and then shrink num_links.
    auto cores_per_link = [](uint32_t workers) { return 2u * (workers + (workers > 1u ? 1u : 0u)); };
    const uint32_t wanted_workers = workers_per_dir;
    const uint32_t wanted_links = num_links;
    while (workers_per_dir > 1 && num_links * cores_per_link(workers_per_dir) > num_available_cores) {
        --workers_per_dir;
    }
    if (num_links * cores_per_link(workers_per_dir) > num_available_cores) {
        // Even one worker per direction per link does not fit; drop links (workers_per_dir is 1 by now).
        num_links = num_available_cores / cores_per_link(workers_per_dir);
        TT_FATAL(
            num_links > 0,
            "all_gather needs at least {} worker cores but only {} are available; provide a larger sub_core_grid.",
            cores_per_link(workers_per_dir),
            num_available_cores);
    }
    if (workers_per_dir != wanted_workers || num_links != wanted_links) {
        log_warning(
            tt::LogOp,
            "all_gather scaled down from {} links x {} workers/direction to {} links x {} workers/direction to fit "
            "the {} available worker cores. This may lead to performance loss.",
            wanted_links,
            wanted_workers,
            num_links,
            workers_per_dir,
            num_available_cores);
    }

    constexpr uint32_t num_directions = 2;  // 0 = forward, 1 = backward
    const bool use_mux = workers_per_dir > 1;

    const auto core_groups = ttnn::ccl::choose_mux_worker_cores(
        num_links * num_directions,
        workers_per_dir,
        mesh_device,
        operation_attributes.subdevice_id,
        operation_attributes.sub_core_grid);

    // Helpers to index into the core placement (dir: 0 = forward, 1 = backward).
    auto group_at = [&](uint32_t link, uint32_t dir) -> const ttnn::ccl::MuxWorkerGroup& {
        return core_groups[(link * num_directions) + dir];
    };
    auto mux_core = [&](uint32_t link, uint32_t dir) -> const CoreCoord& { return group_at(link, dir).mux.value(); };
    auto worker_core = [&](uint32_t link, uint32_t dir, uint32_t w) -> const CoreCoord& {
        return group_at(link, dir).workers[w];
    };
    auto dir_neighbor = [&](uint32_t dir) { return dir == 0 ? fwd_coord : bwd_coord; };
    auto dir_hops = [&](uint32_t dir) { return dir == 0 ? fwd_hops : bwd_hops; };

    // Reader/writer kernels + CB run on worker cores only; the mux kernels run on their own cores.
    std::vector<CoreCoord> worker_cores;
    worker_cores.reserve(num_links * num_directions * workers_per_dir);
    std::set<CoreRange> worker_core_set;
    for (uint32_t link = 0; link < num_links; ++link) {
        for (uint32_t dir = 0; dir < num_directions; ++dir) {
            for (uint32_t w = 0; w < workers_per_dir; ++w) {
                worker_cores.push_back(worker_core(link, dir, w));
                worker_core_set.emplace(worker_core(link, dir, w));
            }
        }
    }
    const CoreRangeSet worker_core_range(worker_core_set);

    ////////////////////////////////////////////////////////////////
    // Circular Buffer and Kernel creation
    ////////////////////////////////////////////////////////////////

    // --- CB entry ---
    // An entry holds a whole number of packets, and a whole number of pages on whichever side shares
    // one, so an entry boundary never cuts either; one of those two counts is always 1.
    //
    // The unit is what a packet actually carries -- not what the payload could carry, which is a
    // different number whenever the segment count ends the packet first. Sizing from the payload
    // leaves the entry a fraction of a packet and cuts one at every boundary: a 4-chunk stripe
    // measured 250-255 us at every such entry against 223-227 us at every multiple of what a packet
    // holds.
    const uint32_t chunks_per_group = std::max(in_chunks_per_page, out_chunks_per_page);
    uint32_t entry_unit = packet_chunks;
    entry_unit = std::max(chunks_per_group, (entry_unit / chunks_per_group) * chunks_per_group);
    // One packet, or enough packets to reach 14 KB. Below that the reader/writer handshake shows: at
    // a 8192 B fabric payload a one-packet entry measured 22% slower at every stripe. Above it, more
    // packets per entry only adds writer lag -- two measured 3-7% slower than one at stripes 16-64.
    // 14 KB is also where both per-arch tunings landed on their own: Wormhole's 7616 B packet at 3
    // packets/entry is 18 KB, Blackhole's 15232 B at 1 packet is 14 KB.
    //
    // Known gap: at a 1088 B chunk, stripes 10 and 12 want a ~30 KB entry -- five or six packets, and
    // not a whole number of them -- and lose 4-9% to this rule. Every setting that serves them costs
    // more elsewhere (the same entries are 5-9% slower at stripes 4 and 8 of that same chunk size),
    // so it is left as measured rather than special-cased.
    const uint32_t entry_floor_bytes = 14 * 1024;
    const uint32_t max_l1_space = ttnn::operations::data_movement::get_max_l1_space(input_tensor);
    const uint32_t entry_chunks =
        entry_chunks_of(entry_unit, chunk_size, packets_per_cb_entry, entry_floor_bytes, cb_depth, max_l1_space);
    const uint32_t cb_page_size = entry_chunks * chunk_size;

    // Input and relay CB
    uint32_t cb0_id = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_depth * cb_page_size, {{cb0_id, df}}).set_page_size(cb0_id, cb_page_size);
    CreateCircularBuffer(program, worker_core_range, cb_src0_config);

    // --- signal rate ---
    // signal_every: data_valid is signalled once per this many entries, so a downstream can start
    // relaying before the whole stripe arrives. Larger = fewer syncs, smaller = finer pipelining.
    // Sized for exactly 2 signals per stripe: the downstream relay starts at the halfway point without
    // paying for a signal per entry. Both neighbours are worse, and the two divides have to round up --
    // truncating them lands on 3-4 signals instead.
    const uint32_t total_slices = num_links * workers_per_dir;
    const uint32_t chunks_per_slice = std::max(1u, num_output_chunks / total_slices);
    const uint32_t entries_per_stripe = std::max(1u, tt::div_up(chunks_per_slice, entry_chunks));
    const uint32_t signals_per_stripe = 2;
    const uint32_t signal_every = std::max(1u, tt::div_up(entries_per_stripe, signals_per_stripe));

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        // --- moving chunks ---
        chunk_size,           // the unit we move
        in_chunks_per_page,   // chunks per input page
        out_chunks_per_page,  // chunks per output page
        packet_size,          // bytes a packet may carry
        run_max,              // most chunks one run may take; 0 = the whole payload
        entry_chunks,         // one CB entry, in chunks
        // --- all_gather ---
        stripe,       // our chunks per output row
        num_devices,  // how far apart our rows sit in the output
        // --- this kernel ---
        cb0_id,           // cb id
        do_init_barrier,  // wait for remote output allocation before relaying
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

    // Writer
    std::vector<uint32_t> writer_compile_args = {
        // --- moving chunks ---
        chunk_size,           // the unit we move
        out_chunks_per_page,  // chunks per output page
        packet_size,          // bytes a packet may carry
        run_max,              // most chunks one run may take; 0 = the whole payload
        entry_chunks,         // one CB entry, in chunks
        // --- all_gather ---
        stripe,       // our chunks per output row
        num_devices,  // how far apart our rows sit in the output
        // --- this kernel ---
        cb0_id,           // cb id
        do_init_barrier,  // send init handshake before relaying
        signal_every,     // signal data_valid once per this many entries
    };
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);

    std::map<std::string, std::string> writer_defines;
    if (use_mux) {
        writer_defines["USE_WORKER_MUX"] = "1";
    }

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_reader.cpp",
        worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_writer.cpp",
        worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args, writer_defines));

    // Fabric mux
    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
    // The fabric maximum is also the smallest safe value here: our payload is the max payload, and an
    // undersized slot silently overruns the next one.
    const size_t channel_buffer_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    tt::tt_fabric::FabricMuxV2Config mux_config(
        /*num_channels=*/static_cast<uint8_t>(workers_per_dir),
        /*num_buffers_per_channel=*/mux_slots_per_channel,
        /*channel_buffer_size_bytes=*/channel_buffer_size_bytes,
        /*base_l1_address=*/mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    if (use_mux) {
        // A mux owns exactly one downstream connection (dst node + link), so every dir and link needs its own.
        for (uint32_t link = 0; link < num_links; ++link) {
            for (uint32_t dir = 0; dir < num_directions; ++dir) {
                // MuxV2 only exits once every channel has been opened and closed, so a mux with a channel nobody
                // connects to hangs the op. Hence mux must be created under the same condition as worker connection.
                if (dir_hops(dir) == 0) {
                    continue;
                }
                const auto dst_node = mesh_device->get_fabric_node_id(*dir_neighbor(dir));
                // Forwarder on NOC 0: our writer pushes into the mux on NOC 1, and sharing one NOC makes
                // them contend.
                tt::tt_fabric::add_fabric_mux_v2_to_program(
                    program,
                    mux_config,
                    mux_core(link, dir),
                    sender_fabric_node_id,
                    dst_node,
                    link,
                    tt::tt_metal::NOC::NOC_0);
            }
        }
    }

    ////////////////////////////////////////////////////////////////
    // Runtime args
    //
    // The page split is per (link, worker) -- num_links * workers_per_dir slices -- and direction-independent,
    // so a (link, worker)'s forward and backward directions relay the same slice. Core assignment is
    // deterministic, so worker w has the same coords on every device: data_valid signals target the mirror
    // worker w on the neighbor.
    ////////////////////////////////////////////////////////////////

    const uint32_t input_addr = input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();

    for (uint32_t link = 0; link < num_links; ++link) {
        for (uint32_t w = 0; w < workers_per_dir; ++w) {
            const uint32_t slice_idx = (link * workers_per_dir) + w;
            const uint32_t input_pages_per_slice = num_input_pages / total_slices;
            const uint32_t remainder = num_input_pages % total_slices;
            const uint32_t input_tile_id_start = (slice_idx * input_pages_per_slice) + std::min(slice_idx, remainder);
            const uint32_t input_tile_id_end =
                ((slice_idx + 1) * input_pages_per_slice) + std::min(slice_idx + 1, remainder);
            // Map this slice of input pages to its slice of output chunks. num_output_chunks is
            // num_input_pages * in_chunks_per_page, so the map is just a scale by that count.
            const uint32_t local_output_start = input_tile_id_start * in_chunks_per_page;
            const uint32_t local_output_end = input_tile_id_end * in_chunks_per_page;
            const uint32_t num_worker_output_chunks = local_output_end - local_output_start;
            const uint32_t half = num_worker_output_chunks / 2;

            // Both directions (dir: 0 = forward, 1 = backward). mirror_core is this core's coords, reused as the
            // data_valid_sem target on the neighbor's mirror core; partner_core is the opposite-direction worker
            // (same index w), the barrier_sem target.
            for (uint32_t dir = 0; dir < num_directions; ++dir) {
                const bool is_forward = (dir == 0);
                const CoreCoord core = worker_core(link, dir, w);
                const CoreCoord partner = worker_core(link, 1 - dir, w);
                const CoreCoord mirror_core = mesh_device->worker_core_from_logical_core(core);
                const CoreCoord partner_core = mesh_device->worker_core_from_logical_core(partner);
                const auto neighbor = dir_neighbor(dir);

                const uint32_t stripe_step = is_forward ? num_devices - 1 : 1;
                const uint32_t num_hops = dir_hops(dir);
                const uint32_t num_recv =
                    is_ring ? num_devices / 2 : (is_forward ? device_idx : num_devices - 1 - device_idx);
                const bool do_local_write = is_forward ? (fwd_hops > 0) : (fwd_hops == 0);

                // data_valid sem is granularly incremented when downstream needs to relay the stripe.
                // data_valid sem is incremented just once when downstream is a sink (doesn't need to relay).
                uint32_t relayed_hops = 0;
                if (num_hops > 0) {
                    const uint32_t downstream_hops =
                        is_ring ? num_devices / 2
                                : relay_hops(is_forward ? device_idx + 1 : device_idx - 1, is_forward);
                    relayed_hops = downstream_hops > 0 ? downstream_hops - 1 : 0;
                }

                // Even ring: the antipode stripe is split between the two directions. Expressed as a range
                // of positions in the emission order, not chunk ids, so the downstream reader's data_valid
                // arithmetic holds once the walk is strided.
                uint32_t final_skip = 0;
                uint32_t final_take = num_worker_output_chunks;
                if (ring_even_split) {
                    final_skip = is_forward ? 0 : half;
                    final_take = is_forward ? half : (num_worker_output_chunks - half);
                }

                // Chunks the upstream delivers into our output (relayed full stripes + sink). The even-ring
                // antipode arrives as a half, so it contributes final_take instead of a full stripe.
                const uint32_t total_chunks = num_recv * num_worker_output_chunks -
                                              (ring_even_split ? (num_worker_output_chunks - final_take) : 0);

                std::vector<uint32_t> reader_rt_args = {
                    input_addr,                // input tensor address
                    output_addr,               // output tensor address
                    device_idx,                // this device's index (initial stripe)
                    stripe_step,               // stripe index step per hop
                    num_hops,                  // relay steps this direction runs
                    total_chunks,              // chunks upstream delivers (completion wait)
                    local_output_start,        // this worker's slice start (chunk id)
                    num_worker_output_chunks,  // this worker's slice length (chunks)
                    final_skip,                // last hop: where in the emission order to start (even-ring split)
                    final_take,                // last hop: how many to take (even-ring split)
                    barrier_sem.address(),     // barrier_sem L1 address
                    data_valid_sem.address(),  // data_valid_sem L1 address
                };
                tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt_args);

                // route for Fabric_2D
                const auto route_node =
                    neighbor.has_value() ? mesh_device->get_fabric_node_id(*neighbor) : sender_fabric_node_id;
                std::vector<uint32_t> writer_rt_args = {
                    output_addr,                   // output tensor address
                    device_idx,                    // this device's index (initial stripe)
                    stripe_step,                   // stripe index step per hop
                    num_hops,                      // relay steps this direction runs
                    local_output_start,            // this worker's slice start (chunk id)
                    num_worker_output_chunks,      // this worker's slice length (chunks)
                    final_skip,                    // last hop: where in the emission order to start (even-ring split)
                    final_take,                    // last hop: how many to take (even-ring split)
                    do_local_write ? 1u : 0u,      // write local data into local output on hop 0
                    barrier_sem.address(),         // barrier_sem L1 address
                    data_valid_sem.address(),      // data_valid_sem L1 address
                    (uint32_t)partner_core.x,      // barrier_sem target (neighbor partner core x)
                    (uint32_t)partner_core.y,      // barrier_sem target (neighbor partner core y)
                    (uint32_t)mirror_core.x,       // data_valid_sem target (neighbor mirror core x)
                    (uint32_t)mirror_core.y,       // data_valid_sem target (neighbor mirror core y)
                    relayed_hops,                  // leading sends the downstream relays
                    (uint32_t)route_node.chip_id,  // neighbor chip id (packet header 2D route)
                    (uint32_t)(*route_node.mesh_id),  // neighbor mesh id (packet header 2D route)
                };
                TT_FATAL(num_hops == 0 || neighbor.has_value(), "an active direction must have a neighbor");
                if (num_hops > 0) {
                    if (use_mux) {
                        const CoreCoord mux_vc = mesh_device->worker_core_from_logical_core(mux_core(link, dir));
                        const auto flow_control_sem_id = tt::tt_metal::CreateSemaphore(program, core, 0);
                        const auto teardown_sem_id = tt::tt_metal::CreateSemaphore(program, core, 0);
                        mux_config.append_client_connection_rt_args(
                            mux_vc,
                            /*logical_channel_id=*/static_cast<uint8_t>(w),
                            tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{
                                .flow_control_sem_id = flow_control_sem_id,
                                .teardown_sem_id = teardown_sem_id,
                            },
                            writer_rt_args);
                    } else {
                        std::vector<tt::tt_fabric::FabricNodeId> dst = {mesh_device->get_fabric_node_id(*neighbor)};
                        append_routing_plane_connection_manager_rt_args(
                            sender_fabric_node_id,
                            dst,
                            {link},
                            program,
                            writer_kernel_id,
                            {core},
                            writer_rt_args,
                            tt::tt_fabric::FabricApiType::Linear);
                    }
                }
                tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt_args);
            }
        }
    }

    shared_variables_t shared_variables{
        .worker_cores = worker_cores,
        .reader_kernel_id = reader_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .barrier_sem = barrier_sem,
        .data_valid_sem = data_valid_sem,
    };

    return {std::move(program), std::move(shared_variables)};
}

void AllGatherUnicastFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherParams& /*operation_attributes*/,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);
        const uint32_t barrier_addr = shared_vars.barrier_sem.address();
        const uint32_t data_valid_addr = shared_vars.data_valid_sem.address();

        auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernel_id);
        for (const auto& core : shared_vars.worker_cores) {
            // reader: [0]=input_addr, [1]=output_addr, [10]=barrier_sem, [11]=data_valid_sem
            auto& reader_args = reader_args_by_core[core.x][core.y];
            reader_args[0] = input_addr;
            reader_args[1] = output_addr;
            reader_args[10] = barrier_addr;
            reader_args[11] = data_valid_addr;
            // writer: [0]=output_addr, [9]=barrier_sem, [10]=data_valid_sem
            auto& writer_args = writer_args_by_core[core.x][core.y];
            writer_args[0] = output_addr;
            writer_args[9] = barrier_addr;
            writer_args[10] = data_valid_addr;
        }
    }
}

}  // namespace ttnn::operations::ccl
