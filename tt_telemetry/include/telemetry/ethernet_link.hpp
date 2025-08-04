#pragma once

/*
 * ethernet_link.hpp
 *
 * Representation of a bi-directional link (two endpoints). The idea here is to reduce the number
 * of monitored entities.
 */

#include <vector>

#include <telemetry/ethernet_endpoint.hpp>

using EthernetLink = std::pair<EthernetEndpoint, EthernetEndpoint>;

std::vector<EthernetLink> get_ethernet_links(const tt::Cluster &cluster);


