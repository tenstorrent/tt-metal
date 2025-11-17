# TT-Telemetry

# Overview

Standalone application that currently presents a web-based GUI. To build and run:

```
./build_metal.sh --build-telemetry
build/tt_telemetry/tt_telemetry_server --fsd=/path/to/system/fsd/file
```

A factory system descriptor (FSD) is required to run telemetry now.

If running on an IRD machine, the Debuda port (usually 5555) is exposed and can be used as the web server port:

```
build/tt_telemetry/tt_telemetry_server --port=$P_USER_DBD_PORT
```

This will often be exposed externally as e.g. 54168. Assuming you are on e.g. `aus-glx-02` you can then point your browser to: `http://aus-glx-02:54168`.

## Collector Mode

By default, tt_telemetry runs in collector mode, collecting telemetry on the local host and providing a socket server on port 8081 for an aggregator instance to connect to. This port can be changed with `--collector-port`.

## Aggregator Mode

In aggregator mode, tt_telemetry connects to collector instances specified with the `--aggregate-from` argument. It will also collect telemetry on its own host unless told not to with `--disable-telemetry`. For example:

```
build/tt_telemetry/tt_telemetry_server --aggregate-from=ws://metal-wh-14:8081,ws://metal-wh-15:8081/
```

Assuming the above was run on host metal-wh-13, this instance of the service will provide metrics for metal-wh-13, metal-wh-14, and metal-wh-15.

## MPI

MPI is not strictly needed but is used by some of the Metal library components. To run with MPI:

1. Create a hosts.txt on e.g. metal-wh-13 containing:

```
metal-wh-13 slots=1
metal-wh-14 slots=1
metal-wh-15 slots=1
```

2. We will need a shell script that uses the MPI rank to determine which command line arguments to pass. We want to run one machine as aggregator and the rest as collectors. Below is an example script.

```
#!/bin/bash

#
# Run using:
#
# mpirun -x TT_METAL_HOME -hostfile hosts.txt ./run_telemetry.sh
#

# Get MPI rank
RANK=${OMPI_COMM_WORLD_RANK:-0}

# Common arguments
COMMON_ARGS="--fsd=/home/btrzynadlowski/factory_system_descriptor_16_n300_lb.textproto"

if [ $RANK -eq 0 ]; then
    # Master node (rank 0) - Aggregator mode
    echo "Starting as AGGREGATOR on rank $RANK"
    exec /home/btrzynadlowski/tt-metal/build/tt_telemetry/tt_telemetry_server \
        $COMMON_ARGS \
        --aggregate-from=ws://metal-wh-14:8081
else
    # Worker nodes (rank > 0) - Collector mode
    echo "Starting as COLLECTOR on rank $RANK"
    exec /home/btrzynadlowski/tt-metal/build/tt_telemetry/tt_telemetry_server \
        $COMMON_ARGS
fi
```

3. Use `mpirun` and make sure that TT_METAL_HOME is exported.

```
mpirun -x TT_METAL_HOME -hostfile hosts.txt /home/btrzynadlowski/tt-metal/run_telemetry.sh
```

# Docker

A Dockerfile is provided in `tt_telemetry/docker/Dockerfile` with two targets: `dev` and `release`. Currently, this is a work-in-progress and intended for internal Tenstorrent use.

To build the development container, which assumes the tt-metal repo is in the context directory:

```
tt_telemetry/docker/build.sh
```

The image will be `ghcr.io/btrzynadlowski-tt/tt-telemetry-dev:latest`. If you have the appropriate credentials, log in and push it using:

```
echo $GITHUB_TOKEN | docker login ghcr.io -u btrzynadlowski-tt --password-stdin
docker push ghcr.io/btrzynadlowski-tt/tt-telemetry-dev:latest
```

To pass in an FSD file, use `-v` to mount it and then pass it in as a commandline argument:

```
docker run -h `hostname` --device /dev/tenstorrent -p 8080:8080 -p 8081:8081 -v /path/to/fsd.textproto:/path/inside/container/fsd.textproto ghcr.io/btrzynadlowski-tt/tt-telemetry-dev:latest --fsd=/path/inside/container/fsd.textproto
```

The container host name is set with `-h` to match the actual machine host name. Ports 8080 and 8081 must be exposed for the web server and intra-process communication, respectively. UMD requires `/dev/tenstorrent` to be passed through. For systems that require it (systems that do not use IOMMU for DMA), `-v /dev/hugepages-1G:/dev/hugepages-1G` will need to be added.

# Generating Factory System Descriptors

For basic single-node machine types, one of the unit tests will produce FSD files, which must be hand-edited to populate host names.

```
./build_metal.sh --build-metal-tests
./build/test/tools/scaleout/test_factory_system_descriptor --gtest_filter="Cluster.TestFactorySystemDescriptorSingleNodeTypes"
```

The files will be placed in `fsd/`. FSD files should be placed in `/var/telemetry` to be usable with the Docker container.

# TODO

- If a machine is down, the aggregator won't receive its metrics and that host will not appear in the GUI. We need to instead somehow mark the machine as being down, perhaps by having the aggregator create metrics associated with hosts it expects to see, or by having a fake bool metric e.g. "/hostname".
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
            ep.chip.id,         // device ID (ChipId)
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
