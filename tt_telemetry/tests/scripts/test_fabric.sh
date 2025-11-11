#!/bin/bash

echo ""
echo "========================================"
echo "Running: Fabric2DFixture.TextUnicastRaw"
echo "========================================"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2DFixture.TestUnicastRaw"

echo ""
echo "========================================"
echo "Running: test_tt_fabric with FlowControlMeshDynamic"
echo "========================================"

# Generate the test configuration YAML
cat > tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_long_running_dont_clear_telemetry_buffers.yaml << 'EOF'
Tests:
  - name: "FlowControlMeshDynamic"
    enable_flow_control: true
    fabric_setup:
      topology: Mesh
      routing_type: Dynamic
    defaults:
      ftype: unicast
      ntype: unicast_write
      size: 4096
      num_packets: 10000000
    patterns:
      - type: all_to_all
EOF

# Run the test
./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_long_running_dont_clear_telemetry_buffers.yaml

# Clean up the generated YAML file
rm -f tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_long_running_dont_clear_telemetry_buffers.yaml

echo ""
echo "========================================"
echo "Test suite completed"
echo "========================================"
