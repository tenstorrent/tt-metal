# SPDX-License-Identifier: Apache-2.0
#
# BH.0 heartbeat / coexistence-gate tool.
#
# IMPORTANT: this builds IN-TREE ONLY. Active-eth kernel creation uses the INTERNAL
# tt-metal API `EthernetConfig` (tt_metal/impl/kernels/kernel.hpp), which is NOT part
# of the public installed TT-Metalium (verified: the public CreateKernel only accepts
# variant<DataMovementConfig, ComputeConfig>). So it CANNOT be built out-of-tree via
# find_package(TT-Metalium). It must be built as part of the tt-metal-external-eth build,
# where the internal headers + `test_metal_common_libs` (which the deployment/eth tests
# link and which pulls EthernetConfig + MeshDevice) are available.
#
# Wire it into the internal build by adding, in tests/tt_metal/tt_metal/CMakeLists.txt
# (next to `add_subdirectory(.../deployment)`):
#     add_subdirectory(${PROJECT_SOURCE_DIR}/tt_metal/tt_rdma/bh0 tt_rdma_bh0)
# then build the repo normally (./build_metal.sh). Run with TT_METAL_HOME=<repo root>.
#
# Alternative (guaranteed to compile, less tool-like): drop bh0_heartbeat_host.cpp into the
# deployment/eth gtest suite (add to UNIT_TESTS_DEPLOYMENT_SRC) and drive it from a TEST()
# using the CommandQueueFixture — that path already uses EthernetConfig + MeshDevice.

add_executable(bh0_heartbeat bh0_heartbeat_host.cpp)

# Repo root so `#include "tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h"` resolves.
target_include_directories(bh0_heartbeat PRIVATE ${PROJECT_SOURCE_DIR})

# Internal test lib: provides EthernetConfig, distributed::MeshDevice, and the launch API
# used by the deployment/eth tests. (If a standalone main() cannot link against it because
# of a gtest main dependency, use the gtest alternative above instead.)
target_link_libraries(bh0_heartbeat PRIVATE test_metal_common_libs)

# BH.1 / M-1a — first TT-RDMA-v1 PROBE frame on the wire (loads bh_rdma_send_probe.cpp).
add_executable(bh1_send_probe bh1_send_probe_host.cpp)
target_include_directories(bh1_send_probe PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(bh1_send_probe PRIVATE test_metal_common_libs)

# BH.1 / M-1b — receive an 0x1af6 frame FROM the BF3 (loads bh_rdma_recv_probe.cpp).
add_executable(bh1_recv_probe bh1_recv_probe_host.cpp)
target_include_directories(bh1_recv_probe PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(bh1_recv_probe PRIVATE test_metal_common_libs)

# BH.2a / M-2 — RISC-off-datapath TX ring drainer (loads bh_rdma_tx_ring.cpp).
add_executable(bh1_tx_ring bh1_tx_ring_host.cpp)
target_include_directories(bh1_tx_ring PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(bh1_tx_ring PRIVATE test_metal_common_libs)

# BH.2-RX — TT-RDMA receive + opcode dispatch + MR WRITE (loads bh_rdma_rx_dispatch.cpp).
add_executable(bh1_rx_dispatch bh1_rx_dispatch_host.cpp)
target_include_directories(bh1_rx_dispatch PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(bh1_rx_dispatch PRIVATE test_metal_common_libs)

# ICRC probe — characterize the ETH-CTRL ROCE_ICRC HW CRC engine (loads bh_rdma_icrc_probe.cpp).
add_executable(bh1_icrc_probe bh1_icrc_probe_host.cpp)
target_include_directories(bh1_icrc_probe PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(bh1_icrc_probe PRIVATE test_metal_common_libs)
