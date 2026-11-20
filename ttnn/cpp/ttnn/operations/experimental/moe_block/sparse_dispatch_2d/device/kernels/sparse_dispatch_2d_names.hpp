// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The sparse 2D dispatch's named compile-time args: one name for each meaning, shared by the host (program factory) and
// the port reader and sender. Plain constexpr only (no includes): the kernels include it as is.
//
// Both kernels (through sparse_dispatch_2d_common.hpp) read the mesh names; the rest are per kernel:
//
//   name                | meaning                                              | reader | sender
//   --------------------+------------------------------------------------------+--------+-------
//   mesh_rows           | mesh rows R                                          | x      | x
//   mesh_columns        | mesh columns C                                       | x      | x
//   rows_per_chip       | own rows a chip (S)                                  | x      | x
//   last_generation     | (C - 1) + (R - 1) - 1: the last END generation       | x      | x
//   route_words         | route words per own row (1: mask | order << 16 |     | x      | x
//                       |   link << 17; 2: mask, order | link << 1)            |        |
//   wrap_x, wrap_y      | 0 / 1: the axis is a ring                            | x      | x
//   manifest_cb         | the manifest CB                                      | x      | x
//   rows_cb, meta_cb    | the CBs of the rows and of their 16-byte entries     | x      | x
//   row_bytes           | bytes of a row (the page of xbuf)                    | x      | x
//   group               | rows a reader -> sender hand-off                     | x      | x
//   plan_cb, plan_bytes | the route plan's CB and bytes                        | x      |
//   stage_cb            | the CB the sender stages manifest entries in         |        | x
//   manifest_batch      | manifest entries a stream sends at once              |        | x
//   rows_per_packet     | rows a fabric packet (1 or 2)                        |        | x
//
// The two ring span tables stay defines (SPARSE_DISPATCH_XRING, SPARSE_DISPATCH_YRING): comma lists that initialise the
// kernels' constexpr arrays. The TensorAccessorArgs chains follow at compile-time arg offset 0 (the kernels have no
// other positional compile-time args).
#pragma once

namespace sparse_dispatch_2d_names {

inline constexpr const char* kMeshRows = "mesh_rows";
inline constexpr const char* kMeshColumns = "mesh_columns";
inline constexpr const char* kRowsPerChip = "rows_per_chip";
inline constexpr const char* kLastGeneration = "last_generation";
inline constexpr const char* kRouteWords = "route_words";
inline constexpr const char* kWrapX = "wrap_x";
inline constexpr const char* kWrapY = "wrap_y";
inline constexpr const char* kManifestCb = "manifest_cb";
inline constexpr const char* kRowsCb = "rows_cb";
inline constexpr const char* kMetaCb = "meta_cb";
inline constexpr const char* kRowBytes = "row_bytes";
inline constexpr const char* kGroup = "group";
inline constexpr const char* kPlanCb = "plan_cb";
inline constexpr const char* kPlanBytes = "plan_bytes";
inline constexpr const char* kStageCb = "stage_cb";
inline constexpr const char* kManifestBatch = "manifest_batch";
inline constexpr const char* kRowsPerPacket = "rows_per_packet";
inline constexpr const char* kRowPieces = "row_pieces";

}  // namespace sparse_dispatch_2d_names
