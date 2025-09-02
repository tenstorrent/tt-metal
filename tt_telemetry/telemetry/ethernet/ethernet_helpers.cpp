#include <tt-metalium/assert.hpp>

#include <telemetry/ethernet/ethernet_helpers.hpp>

static auto make_ordered_ethernet_connections(const auto &unordered_connections) {
    std::map<
        tt::umd::chip_id_t,
        std::map<
            tt::umd::ethernet_channel_t,
            std::tuple<tt::umd::chip_id_t, tt::umd::ethernet_channel_t>
        >
    > ordered_connections;

    for (const auto& [chip_id, channel_map] : unordered_connections) {
        for (const auto& [channel, connection_tuple] : channel_map) {
            ordered_connections[chip_id][channel] = connection_tuple;
        }
    }

    return ordered_connections;
}

std::map<
    tt::umd::chip_id_t,
    std::map<tt::umd::ethernet_channel_t, std::tuple<tt::umd::chip_id_t, tt::umd::ethernet_channel_t>>>
get_ordered_ethernet_connections(const std::unique_ptr<tt::umd::Cluster>& cluster) {
    return make_ordered_ethernet_connections(cluster->get_cluster_description()->get_ethernet_connections());
}

// bool is_ethernet_endpoint_up(const tt::Cluster &cluster, const EthernetEndpoint &ep, bool force_refresh_link_status) {
//     tt_cxy_pair virtual_eth_core = tt_cxy_pair(ep.chip.id, cluster.get_virtual_coordinate_from_logical_coordinates(ep.chip.id, ep.ethernet_core, CoreType::ETH));
//     if (force_refresh_link_status && cluster.arch() == tt::ARCH::BLACKHOLE) {
//         // On Blackhole, we should use the mailbox to request a link status update.
//         // The link status should be auto-updated periodically by code running on RISC0 but this
//         // may not be guaranteed in some cases, so we retain the option to do it explicitly.
//         tt::llrt::internal_::send_msg_to_eth_mailbox(
//             ep.chip.id,         // device ID (chip_id_t)
//             virtual_eth_core,   // virtual core (CoreCoord == tt_cxy_pair)
//             tt::tt_metal::FWMailboxMsg::ETH_MSG_LINK_STATUS_CHECK,
//             {0xffffffff},       // args: arg0=copy_addr -- we don't want to copy this anywhere so as not to overwrite anything, just perform update of existing struct we will read LINK_UP from
//             50                  // timeout ms
//         );
//     }
//     uint32_t link_up_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::LINK_UP);
//     uint32_t link_up_value = 0;
//     cluster.read_core(&link_up_value, sizeof(uint32_t), virtual_eth_core, link_up_addr);
//     if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
//         return link_up_value == 6;  // see eth_fw_api.h
//     } else if (cluster.arch() == tt::ARCH::BLACKHOLE) {
//         return link_up_value == 1;
//     }
//     TT_ASSERT(false, "Unsupported architecture for chip {}", ep.chip);
//     return false;
// }

// TODO: hard-coded address is WH only. Need to add a UMD HAL layer.
bool is_ethernet_endpoint_up(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const EthernetEndpoint& endpoint,
    bool force_refresh_link_status) {
    uint32_t wormhole_link_up_addr = 0x1ed4;
    tt::umd::TTDevice* device = cluster->get_tt_device(endpoint.chip.id);
    TT_ASSERT(device->get_arch() == tt::ARCH::WORMHOLE_B0, "Unsupported architecture for chip {}", endpoint.chip);

    uint32_t link_up_value = 0;
    tt::umd::CoreCoord ethernet_core = tt::umd::CoreCoord(
        endpoint.ethernet_core.x, endpoint.ethernet_core.y, tt::umd::CoreType::ETH, tt::umd::CoordSystem::LOGICAL);
    cluster->read_from_device(&link_up_value, endpoint.chip.id, ethernet_core, wormhole_link_up_addr, sizeof(uint32_t));

    if (cluster->get_tt_device(endpoint.chip.id)->get_arch() == tt::ARCH::WORMHOLE_B0) {
        return link_up_value == 6;  // see eth_fw_api.h
    } else if (cluster->get_tt_device(endpoint.chip.id)->get_arch() == tt::ARCH::BLACKHOLE) {
        return link_up_value == 1;
    }

    TT_ASSERT(false, "Unsupported architecture for chip {}", ep.chip);
    return false;
}
