# TT-Telemetry

# Overview

Standalone application that currently presents a web-based GUI. To build and run:

```
./build_metal.sh --build-telemetry
build/tt_telemetry/tt_telemetry_server
```

Currently, we run an HTTP server on port 5555 (the former Debuda port, which is exposed on IRD machines). To find the corresponding
external port:

```
echo $P_USER_DBD_PORT
```

This will often be e.g. 54168. Assuming you are on e.g. `aus-glx-02` you can then point your browser to: `http://aus-glx-02:54168/static/index.html`.

# TODO

- Update to BH style telemetry:  https://github.com/tenstorrent/tt-umd/commit/1b6fc8c8fd29f9a2b32e3b879a02ab26be496e0d
- HAL probably needs to be moved to UMD somehow, although we use the Metal one for now without creating a Metal context.
- When factory descriptor becomes available, use that to identify chips, connections, and produce telemetry paths.
- Multi-host: initially, simply communicate with peer tt_telemetry instances.
- Wait until each `TelemetrySubscriber` has finished processing a buffer before fetching a new one and continue to use existing buffer until ready for hand off. May not be necessary but we should at least watch for slow consumers causing the number of buffers
being allocated to increase (ideally, there should only be two at any given time -- one being written, one being consumed).
- On Blackhole, Ethernet link status update rate is not clear and we may want to investigate triggering a forced update, although this may have adverse performance effects. Requires further discussion. Sample code:

```
bool is_ethernet_endpoint_up(const tt::Cluster &cluster, const EthernetEndpoint &ep, bool force_refresh_link_status) {
    tt_cxy_pair virtual_eth_core = tt_cxy_pair(ep.chip.id, cluster.get_virtual_coordinate_from_logical_coordinates(ep.chip.id, ep.ethernet_core, CoreType::ETH));
    if (force_refresh_link_status && cluster.arch() == tt::ARCH::BLACKHOLE) {
        // On Blackhole, we should use the mailbox to request a link status update.
        // The link status should be auto-updated periodically by code running on RISC0 but this
        // may not be guaranteed in some cases, so we retain the option to do it explicitly.
        tt::llrt::internal_::send_msg_to_eth_mailbox(
            ep.chip.id,         // device ID (chip_id_t)
            virtual_eth_core,   // virtual core (CoreCoord == tt_cxy_pair)
            tt::tt_metal::FWMailboxMsg::ETH_MSG_LINK_STATUS_CHECK,
            {0xffffffff},       // args: arg0=copy_addr -- we don't want to copy this anywhere so as not to overwrite anything, just perform update of existing struct we will read LINK_UP from
            50                  // timeout ms
        );
    }
    uint32_t link_up_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::LINK_UP);
    uint32_t link_up_value = 0;
    cluster.read_core(&link_up_value, sizeof(uint32_t), virtual_eth_core, link_up_addr);
    if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
        return link_up_value == 6;  // see eth_fw_api.h
    } else if (cluster.arch() == tt::ARCH::BLACKHOLE) {
        return link_up_value == 1;
    }
    TT_ASSERT(false, "Unsupported architecture for chip {}", ep.chip);
    return false;
}
```
