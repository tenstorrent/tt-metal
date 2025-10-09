#!/bin/bash
# Quick script to verify coordinator selection works

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Coordinator Selection Verification Script                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if allocation monitor is running
if ! pgrep -x "allocation_monitor" > /dev/null; then
    echo "⚠️  WARNING: Allocation monitor not detected!"
    echo "   Start it in another terminal:"
    echo "   cd tt_metal/programming_examples/alexp_examples/memory_utilization_monitor"
    echo "   ./allocation_monitor_client -a"
    echo ""
fi

export TT_ALLOC_TRACKING_ENABLED=1

echo "Test 1: Default coordinator (device 0)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
unset TT_COORDINATOR_DEVICE_ID
pytest test_custom_coordinator.py::test_coordinator_memory_pattern -v -s --tb=short || true
echo ""
sleep 2

echo "Test 2: Custom coordinator (device 3)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
export TT_COORDINATOR_DEVICE_ID=3
pytest test_custom_coordinator.py::test_coordinator_memory_pattern -v -s --tb=short || true
echo ""
sleep 2

echo "Test 3: Custom coordinator (device 7)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
export TT_COORDINATOR_DEVICE_ID=7
pytest test_custom_coordinator.py::test_coordinator_memory_pattern -v -s --tb=short || true
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Verification Complete!                                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Check your allocation monitor output:"
echo "  - Test 1: Device 0 should have ~3MB L1"
echo "  - Test 2: Device 3 should have ~3MB L1"
echo "  - Test 3: Device 7 should have ~3MB L1"
echo ""
echo "The high L1 device is the coordinator!"
