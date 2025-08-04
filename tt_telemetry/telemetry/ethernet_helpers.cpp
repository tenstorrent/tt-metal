#include <telemetry/ethernet_helpers.hpp>

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
    std::map<
        tt::umd::ethernet_channel_t, 
        std::tuple<tt::umd::chip_id_t, tt::umd::ethernet_channel_t>
    >
> get_ordered_ethernet_connections(const tt::Cluster &cluster) {
    return make_ordered_ethernet_connections(cluster.get_ethernet_connections());
}

std::unordered_map<tt::umd::ethernet_channel_t, CoreCoord> map_ethernet_channel_to_core_coord(const tt::Cluster &cluster, tt::umd::chip_id_t chip_id) {
    // logical_eth_core_to_chan_map should be a 1:1 mapping and therefore easily invertible
    std::unordered_map<tt::umd::ethernet_channel_t, CoreCoord> ethernet_channel_to_core_coord;
    for (const auto &[core, channel]: cluster.get_soc_desc(chip_id).logical_eth_core_to_chan_map) {
        TT_ASSERT(ethernet_channel_to_core_coord.count(channel) == 0, "Multiple Ethernet cores in logical_eth_core_to_chan_map map to Ethernet channel {}", channel);
        ethernet_channel_to_core_coord.insert({ channel, core });
    }
    return ethernet_channel_to_core_coord;
}