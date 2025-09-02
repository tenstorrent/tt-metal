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

- Blackhole addresses for ARC and Ethernet metrics, test on Blackhole box
- Update to BH style telemetry:  https://github.com/tenstorrent/tt-umd/commit/1b6fc8c8fd29f9a2b32e3b879a02ab26be496e0d
- HAL needs to be moved to UMD somehow.
- When factory descriptor becomes available, use that to identify chips, connections, and produce telemetry paths.
- Multi-host: initially, simply communicate with peer tt_telemetry instances.
- Wait until each `TelemetrySubscriber` has finished processing a buffer before fetching a new one and continue to use existing buffer until ready for hand off. May not be necessary but we should at least watch for slow consumers causing the number of buffers
being allocated to increase (ideally, there should only be two at any given time -- one being written, one being consumed).
