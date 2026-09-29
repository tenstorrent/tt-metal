// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilizer kernel (reader RISC, NOC_0). Stages the tokens one ring direction's senders are about to want
// into its own L1, and hands them over a batch at a time.
//
// What a batch is and which ones exist is combine_fabric2d_group_walk.hpp; this core takes those where
// batch % num_peers == my_index. The rest of the group takes the others, and since every core of the group
// builds the same walk, that split needs no coordination.
//
// Two counters, the same monotonic single-writer idiom the reader and sender already use between themselves.
// `produced` lives on each consumer's core and is bumped here; `freed[c]` lives here, one per consumer,
// and is bumped there. Per consumer rather than summed: consumers run far apart -- a group's senders take
// alternating halves of each run -- and one sum would let the leading one's credit release a slot the
// trailing one is still reading.
//
// A TILED dispatched buffer is read a block of tiles at a time and untilized by the compute kernel on this
// core, a whole tile-row per batch because that is the least an untilize can produce. A ROW_MAJOR one needs
// no untilize and has no compute kernel, so the rows go straight into the batch and only the ones the walk
// asked for are read at all.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "combine_fabric2d_untilizer_ct_args.hpp"
#include "combine_fabric2d_untilizer_rt_args.hpp"
#include "combine_fabric2d_group_walk.hpp"

namespace cmbf2d_ns = cmbf2d;

#include "untilizer_combine_fabric2d_body.hpp"
