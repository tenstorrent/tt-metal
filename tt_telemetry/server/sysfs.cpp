/*
 * sysfs.cpp
 *
 * Sysfs metrics discovery and printing functionality.
 * Equivalent to the Python sysfs scraping code for discovering Tenstorrent card metrics.
 *
 * TODO:
 * -----
 * - Joel Smith mentioned an issue that could occur:

Joel Smith
  7 minutes ago
If you're working with WH 6U and telemetry there are two things you should know regarding reset:
if you reset the chips, you'll lose telemetry: https://tenstorrent.atlassian.net/browse/SYS-1920
you cannot assume /dev/tenstorrent/x will still be the same ASIC pre- and post-reset
I have solutions/workarounds for both of these, but they are not deployed anywhere.  Just prototypes.





Joel Smith
  6 minutes ago
Quick workaround for the first one is to wait about a minute after reset, then unload/reload the driver.  Happy to chat if you run into any issues here.


Bart Trzynadlowski
  3 minutes ago
Ah I won’t be aware of a reset


Bart Trzynadlowski
  3 minutes ago
Should I just always read the corresponding serial as well and match at sample time?
New


Joel Smith
  Just now
There's an attribute named asic_id that will show up if your FW is new enough.  I think it's guaranteed to be unique.  I am not sure if you can/should rely on serial being unique, although maybe it's a reasonable assumption for 6U.
 */

 #include <iostream>
 #include <fstream>
 #include <string>
 #include <vector>
 #include <map>
 #include <regex>
 #include <filesystem>
 #include <optional>
 #include <unistd.h>
 #include <cstdlib>
 #include <array>
 #include <memory>
#include <thread>

#include "impl/context/metal_context.hpp"
#include <telemetry/ethernet/ethernet_endpoint.hpp>
#include <telemetry/arc/arc_telemetry_reader.hpp>
#include <tt-metalium/control_plane.hpp>


#include <third_party/umd/device/api/umd/device/tt_device/tt_device.h>

#include <telemetry/ethernet/chip_identifier.hpp>

 static auto make_ordered_ethernet_connections(const auto& unordered_connections) {
     std::map<
         tt::umd::chip_id_t,
         std::map<tt::umd::ethernet_channel_t, std::tuple<tt::umd::chip_id_t, tt::umd::ethernet_channel_t>>>
         ordered_connections;

     for (const auto& [chip_id, channel_map] : unordered_connections) {
         for (const auto& [channel, connection_tuple] : channel_map) {
             ordered_connections[chip_id][channel] = connection_tuple;
         }
     }

     return ordered_connections;
 }

 std::unordered_map<tt::umd::ethernet_channel_t, CoreCoord> map_ethernet_channel_to_core_coord(
     const tt_SocDescriptor& soc_desc, tt::umd::chip_id_t chip_id) {
     // logical_eth_core_to_chan_map should be a 1:1 mapping and therefore easily invertible
     std::unordered_map<tt::umd::ethernet_channel_t, CoreCoord> ethernet_channel_to_core_coord;
     for (auto channel = 0; channel < soc_desc.get_num_eth_channels(); channel++) {
         ethernet_channel_to_core_coord.insert({channel, soc_desc.get_eth_core_for_channel(channel)});
     }
     return ethernet_channel_to_core_coord;
 }

 static std::unordered_map<tt::umd::chip_id_t, std::unique_ptr<tt::umd::TTDevice>> get_pcie_devices(
     const std::unique_ptr<tt::umd::tt_ClusterDescriptor>& cluster_descriptor) {
     std::unordered_map<tt::umd::chip_id_t, std::unique_ptr<tt::umd::TTDevice>> pcie_device_by_chip_id;
     for (auto [chip_id, pcie_id] : cluster_descriptor->get_chips_with_mmio()) {
         std::unique_ptr<tt::umd::TTDevice> device = tt::umd::TTDevice::create(pcie_id);
         device->init_tt_device();
         pcie_device_by_chip_id.emplace(std::make_pair(chip_id, std::move(device)));
     }
     return pcie_device_by_chip_id;
 }

 static bool is_link_up(const std::unique_ptr<tt::umd::Cluster>& cluster, EthernetEndpoint ep) {
     uint32_t link_up_value = 0;
     tt::umd::CoreCoord ethernet_core = tt::umd::CoreCoord(
         ep.ethernet_core.x, ep.ethernet_core.y, tt::umd::CoreType::ETH, tt::umd::CoordSystem::LOGICAL);
     cluster->read_from_device(&link_up_value, ep.chip.id, ethernet_core, 0x1ed4, sizeof(uint32_t));

     if (cluster->get_tt_device(ep.chip.id)->get_arch() == tt::ARCH::WORMHOLE_B0) {
         return link_up_value == 6;  // see eth_fw_api.h
     } else if (cluster->get_tt_device(ep.chip.id)->get_arch() == tt::ARCH::BLACKHOLE) {
         return link_up_value == 1;
     }

     TT_ASSERT(false, "Unsupported architecture for chip {}", ep.chip);
     return false;
 }

 void test_umd() {
     std::cout << "Num PCIE devices: " << PCIDevice::enumerate_devices_info().size() << std::endl;
     std::unique_ptr<tt::umd::Cluster> cluster =
         std::make_unique<tt::umd::Cluster>();
    std::cout << "Got here" << std::endl;
     auto connections = make_ordered_ethernet_connections(cluster->get_cluster_description()->get_ethernet_connections());
     std::cout << "Connections: " << cluster->get_cluster_description()->get_ethernet_connections().size() << std::endl;
    //  std::unordered_map<tt::umd::chip_id_t, std::unique_ptr<tt::umd::TTDevice>> pcie_devices_by_chip_id =
    //      get_pcie_devices(cluster_descriptor);
     while (true) {
         auto endpoint_by_chip = get_ethernet_endpoints_by_chip(cluster);
         for (auto& [chip_id, endpoints] : endpoint_by_chip) {
             std::cout << chip_id << ":" << std::endl;
             for (auto& endpoint : endpoints) {
                 std::cout << "  " << endpoint << " = " << (is_link_up(cluster, endpoint) ? "UP" : "DOWN") << std::endl;
             }
         }
         std::cout << "--" << std::endl;
         std::this_thread::sleep_for(std::chrono::seconds(5));
     }
     std::cout << "Finished" << std::endl;
     return;
 }