#include <map>
#include <unordered_map>
#include <vector>

#include <telemetry/ethernet/ethernet_link.hpp>
#include <telemetry/ethernet/ethernet_helpers.hpp>

// TODO: using get_ethernet_connections() in here. This will only map links on the same host. We will want a more generic scheme eventually for remote ones.
static std::unordered_map<EthernetEndpoint, EthernetEndpoint> map_endpoints_to_remote_endpoints(const tt::Cluster &cluster) {
    std::unordered_map<EthernetEndpoint, EthernetEndpoint> endpoint_to_remote;

    for (const auto &[chip_id, remote_chip_and_channel_by_channel]: cluster.get_ethernet_connections()) {
        ChipIdentifier from_chip = get_chip_identifier_from_umd_chip_id(cluster, chip_id);

        auto ethernet_channel_to_core_coord = map_ethernet_channel_to_core_coord(cluster, chip_id);

        for (const auto &[channel, remote_chip_and_channel]: remote_chip_and_channel_by_channel) {
            TT_ASSERT(ethernet_channel_to_core_coord.count(channel) != 0, "Channel {} missing in ethernet_channel_to_core_coord map for {}", channel, from_chip);

            CoreCoord from_ethernet_core = ethernet_channel_to_core_coord[channel];
            EthernetEndpoint from{ .chip = from_chip, .ethernet_core = from_ethernet_core, .channel = channel };

            tt::umd::chip_id_t remote_chip_id;
            tt::umd::ethernet_channel_t remote_channel;
            std::tie(remote_chip_id, remote_channel) = remote_chip_and_channel;

            ChipIdentifier to_chip = get_chip_identifier_from_umd_chip_id(cluster, remote_chip_id);

            auto remote_ethernet_channel_to_core_coord = map_ethernet_channel_to_core_coord(cluster, remote_chip_id);

            TT_ASSERT(remote_ethernet_channel_to_core_coord.count(remote_channel) != 0, "Remote channel {} missing in remote_ethernet_channel_to_core_coord map for {}", remote_channel, remote_chip_id);

            CoreCoord remote_ethernet_core = remote_ethernet_channel_to_core_coord[remote_channel];
            EthernetEndpoint to{ .chip = to_chip, .ethernet_core = remote_ethernet_core, .channel = remote_channel };

            endpoint_to_remote[from] = to;
        }
    }

    return endpoint_to_remote;
}

std::vector<EthernetLink> get_ethernet_links(const tt::Cluster &cluster) {
    std::vector<EthernetLink> links;

    auto endpoint_to_remote = map_endpoints_to_remote_endpoints(cluster);

    auto it = endpoint_to_remote.begin();
    while (it != endpoint_to_remote.end()) {
        const EthernetEndpoint &from = it->first;
        const EthernetEndpoint &to = it->second;

        // Ensure the reverse mapping exists
        auto reverse_it = endpoint_to_remote.find(to);
        if (reverse_it != endpoint_to_remote.end() && reverse_it->second == from) {
            // Yes. Bidirectional link confirmed. Add it to output, with "lesser" ordered endpoint
            // first.
            if (from < to) {
                links.push_back(std::make_pair(from, to));
            } else {
                links.push_back(std::make_pair(to, from));
            }

            // Carefully remove reverse mapping
            if (reverse_it == it) {
                ++it;
                endpoint_to_remote.erase(reverse_it);
            } else {
                endpoint_to_remote.erase(reverse_it);
                ++it;
            }
        } else {
            // No reverse mapping.
            log_fatal(tt::LogAlways, "Endpoint {} -> {} exists but the reverse does not", from, to);
            TT_THROW("Endpoint appears to be one-sided: from->to does not have a corresponding to->from");
            ++it;
        }
    }

    std::sort(links.begin(), links.end());
    return links;
}
