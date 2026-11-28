// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, semaphores and argument slots of the grouped_matmul kernels (gmm_reader.cpp,
// gmm_bank_reader.cpp, gmm_writer.cpp, gmm_compute.cpp). The host and the kernels index every slot through
// these names.
namespace grouped_matmul {

// The generic op's ids, unchanged (CB_OUT is 16). The kernels also receive them at their compile-time-arg
// positions below, as they always did.
namespace cb {
constexpr uint32_t IN0 = 0;
constexpr uint32_t IN1 = 1;
constexpr uint32_t PART = 2;
constexpr uint32_t OUT = 16;
}  // namespace cb

// Semaphore ids the readers read with get_semaphore(): READY (receivers credit the sender), VALID (the
// sender's multicast flag). The factory takes them from find_available_semaphore_id and TT_FATALs they match.
namespace sem {
constexpr uint32_t READY = 0;
constexpr uint32_t VALID = 1;
}  // namespace sem

// Compile-time args of gmm_reader.cpp (interleaved path); x's then w's TensorAccessorArgs follow from COUNT.
namespace reader_ct {
enum : uint32_t { MT, KT, NT, BW, PN, GX, CB_IN0, CB_IN1, IN0_TB, IN1_TB, PNX, COUNT };
}  // namespace reader_ct

// Compile-time args of gmm_bank_reader.cpp (bank path); x's TensorAccessorArgs follow from COUNT.
namespace bank_reader_ct {
enum : uint32_t { MT, KT, BW, PN, PNX, CB_IN0, CB_IN1, IN0_TB, IN1_TB, DEPTH, W_INFLIGHT, X_AHEAD, COUNT };
}  // namespace bank_reader_ct

// Compile-time args of gmm_writer.cpp; y's TensorAccessorArgs follow from COUNT.
namespace writer_ct {
enum : uint32_t { MT, NT, PN, CB_OUT, OUT_TB, COUNT };
}  // namespace writer_ct

// Compile-time args of gmm_compute.cpp.
namespace compute_ct {
enum : uint32_t { MT, BW, PN, NBLK, CB_IN0, CB_IN1, CB_PART, CB_OUT, RT, PNX, COUNT };
}  // namespace compute_ct

// Common runtime args of both readers.
namespace reader_common {
enum : uint32_t { X_ADDR, W_ADDR, COUNT };
}  // namespace reader_common

// Common runtime args of the writer.
namespace writer_common {
enum : uint32_t { Y_ADDR, COUNT };
}  // namespace writer_common

// Per-core runtime args of gmm_reader.cpp (MC_* and SENDER_* are physical NoC coordinates; MC_* are 0 when GX == 1).
namespace reader_rt {
enum : uint32_t { G, N0, SENDER, MC_X0, MC_X1, MC_Y, SENDER_X, SENDER_Y, COUNT };
}  // namespace reader_rt

// Per-core runtime args of gmm_bank_reader.cpp. PN is present only under GMM_PN_RT (one_group): COUNT without it,
// COUNT_PN_RT with it.
namespace bank_reader_rt {
enum : uint32_t { G, TILE0, BANK, SENDER, MC_X0, MC_Y0, MC_X1, MC_Y1, NDEST, SENDER_X, SENDER_Y, COUNT };
constexpr uint32_t PN = COUNT;
constexpr uint32_t COUNT_PN_RT = COUNT + 1;
}  // namespace bank_reader_rt

// Per-core runtime args of gmm_writer.cpp; PN only under GMM_PN_RT (bank path, one_group).
namespace writer_rt {
enum : uint32_t { G, N0, COUNT };
constexpr uint32_t PN = COUNT;
constexpr uint32_t COUNT_PN_RT = COUNT + 1;
}  // namespace writer_rt

// Per-core runtime args of gmm_compute.cpp: none, except PN under GMM_PN_RT.
namespace compute_rt {
constexpr uint32_t PN = 0;
constexpr uint32_t COUNT = 0;
constexpr uint32_t COUNT_PN_RT = 1;
}  // namespace compute_rt

// Kernels are pushed in groups of three (one group per distinct core width pn, in the Python's order of first
// appearance; one group when one_group): kernel index = group * KERNELS_PER_GROUP + role.
namespace kernel_role {
enum : uint32_t { READER, WRITER, COMPUTE, KERNELS_PER_GROUP };
}  // namespace kernel_role

}  // namespace grouped_matmul
