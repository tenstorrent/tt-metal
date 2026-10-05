// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel (reader RISC, NOC_0). Feeds the L1 ring the sender sends from and makes ALL routing
// decisions. The sender writes L1 -> eth over NOC_1, so the two do not contend.
//
// Every packet the sender sends travels exactly one hop, to the chip across its own cable, so a token
// bound further is staged into that neighbour's forwarding buffer and re-sent from there by the neighbour's
// reader on the same (plane, direction). Hence two kinds of work:
//
//   A. Its OWN assignments — its plane's share of this chip's movements. Each destination is either our
//      immediate neighbour (=> CMD_FINAL_WRITE, the neighbour writes it straight into its output region) or
//      further away (=> CMD_FORWARD into the neighbour's forwarding buffer).
//
//   B. The chunks that arrived in OUR region of the forwarding buffer, written by the upstream chip's
//      sender on the same (plane, direction). Each is pushed one more hop: final-write if its destination
//      is now our neighbour, re-forward otherwise.
//
// Both are done one LOCAL EXPERT at a time: the whole schedule runs for expert 0, then again for expert 1,
// and so on, so a chunk is one (origin chip -> destination chip, expert) term. Every chip walks its experts
// in the same order, so a relay at iteration e is passing on chunks its upstream produced at iteration e —
// the ordering the dense forwarding buffer needs is unchanged, there are just more, smaller chunks. This is
// what lets an upstream stage produce one expert's tokens at a time.
//
// The forwarding buffer is DENSE: a chunk occupies exactly as many pages as it has tokens, and starts where
// the previous one ended. Nothing is exchanged to make the writer and the reader of a region agree on those
// boundaries — both compute every chunk's length from the same replicated expert_offsets, using the chunk
// descriptors the host packed in the same order the upstream sender emits them.
//
// The stream ends with a CMD_END slot, because its length is not knowable up front: how much this core
// re-forwards depends on chunk sizes decided upstream.
//
// The forwarding-buffer page layout is deliberately [token][final_addr][dst_chip], which is EXACTLY the
// first token_size+16 bytes of a ring slot. Re-forwarding is therefore a single DRAM read of
// token_size+16 bytes straight into the slot base: payload and both metadata words land where the sender
// already expects them, with no unpacking.
//
// The ring is hand-rolled rather than a metal CB because the packet headers live at fixed offsets from the
// L1 allocator base — where a CB would be allocated — and a sender on another chip addresses them there.
//
// Two monotonic single-writer counters, each bumped by a NoC atomic to our OWN core. `filled` is ours,
// `freed` is the sender's, and each side keeps its own local count and works on the difference, so there is
// no read-modify-write to race. The atomic is what makes a bump visible to the other RISC; the slot metadata
// it announces rides on plain stores, ordered ahead of it in the same store stream, exactly as cb_push_back
// announces tile data.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "combine_fabric2d_reader_ct_args.hpp"
#include "combine_fabric2d_reader_rt_args.hpp"
#include "combine_fabric2d_group_walk.hpp"

#define CMBF2D_OVERLAPPED 1
// Overlapped, combine always takes the routed expert's bfloat8_b TILE output; there is no row-major path.
#define TILE 1
namespace cmbf2d_ns = hyb_cmbf2d;

#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/reader_combine_fabric2d_body.hpp"
