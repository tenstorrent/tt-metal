#!/bin/bash
# Run cluster validation before tests

echo "Running cluster validation..."
tt-run \
   --tcp-interface enp194s0f0np0 \
   --rank-binding $TT_METAL_HOME/tests/tt_metal/distributed/config/dual_bh_lb_rank_bindings.yaml \
   --mpi-args "--host bh-lb-02,bh-lb-03" \
   $TT_METAL_HOME/build/tools/scaleout/run_cluster_validation --print-connectivity --send-traffic

echo "Cluster validation complete!"
