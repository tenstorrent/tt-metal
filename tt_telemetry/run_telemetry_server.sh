#!/bin/bash

set -e  # Exit on error

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Step 1: Build metal with telemetry and tests
echo "========================================"
echo "Step 1: Building metal with telemetry and tests..."
echo "========================================"
./build_metal.sh --build-telemetry --build-metal-tests

# Step 2: Run the factory system descriptor test
echo "========================================"
echo "Step 2: Running factory system descriptor test..."
echo "========================================"
./build/test/tools/scaleout/test_factory_system_descriptor \
    --gtest_filter="Cluster.TestFactorySystemDescriptorSingleNodeTypes"

# Step 3: Copy the factory system descriptor file
echo "========================================"
echo "Step 3: Copying factory system descriptor..."
echo "========================================"
cp fsd/factory_system_descriptor_N300_LB.textproto ./fsd_n300.textproto

# Step 4: Replace "host" with actual hostname
echo "========================================"
echo "Step 4: Replacing 'host' with actual hostname..."
echo "========================================"
HOSTNAME=$(hostname)
echo "Hostname: $HOSTNAME"
sed -i "s/\"host\"/\"$HOSTNAME\"/g" ./fsd_n300.textproto

# Step 5: Run the telemetry server
echo "========================================"
echo "Step 5: Starting telemetry server..."
echo "========================================"
TT_METAL_HOME=$(pwd) build/tt_telemetry/tt_telemetry_server --fsd=fsd_n300.textproto
