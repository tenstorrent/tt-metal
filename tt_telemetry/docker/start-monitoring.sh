#!/bin/bash
# Start Prometheus + Grafana monitoring stack for TT-Metal telemetry
#
# Usage:
#   ./start-monitoring.sh [HOST:PORT] [HOST:PORT] ...
#
# Examples:
#   # Scrape localhost TT-Metal telemetry (default)
#   ./start-monitoring.sh
#
#   # Scrape remote TT-Metal telemetry
#   ./start-monitoring.sh sjc-wh-05:53494
#
#   # Scrape multiple hosts
#   ./start-monitoring.sh sjc-wh-05:53494 sjc-wh-06:53494

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command-line arguments for target hosts
TARGETS=("$@")

# Show help if requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    grep "^#" "$0" | tail -n +2 | cut -c 3-
    exit 0
fi

# Default: scrape localhost telemetry if no targets specified
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=("host.docker.internal:8080")
fi

# Generate prometheus.yml with targets
generate_prometheus_config() {
    cat > prometheus.yml <<EOF
# Prometheus configuration file for scraping TT-Metal telemetry metrics

global:
  scrape_interval: 15s     # Scrape targets every 15 seconds
  evaluation_interval: 15s # Evaluate rules every 15 seconds

scrape_configs:
  - job_name: 'tt-telemetry'
    metrics_path: '/api/metrics'
    static_configs:
      - targets: [
EOF

    for i in "${!TARGETS[@]}"; do
        if [ $i -eq 0 ]; then
            echo -n "          '${TARGETS[$i]}'" >> prometheus.yml
        else
            echo "," >> prometheus.yml
            echo -n "          '${TARGETS[$i]}'" >> prometheus.yml
        fi
    done

    cat >> prometheus.yml <<EOF

        ]
EOF
}

echo "==================================================================="
echo "TT-Metal Telemetry Monitoring Stack"
echo "==================================================================="
echo ""
echo "TT-Metal Telemetry endpoints:"
for target in "${TARGETS[@]}"; do
    echo "  - http://$target/api/metrics"
done

echo ""
echo "Generating Prometheus configuration..."
generate_prometheus_config

# Detect docker compose vs docker-compose
if docker compose version &>/dev/null; then
    DOCKER_COMPOSE="docker compose"
elif docker-compose version &>/dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "Error: Neither 'docker compose' nor 'docker-compose' found"
    exit 1
fi

echo "Starting Docker containers..."
$DOCKER_COMPOSE down 2>/dev/null || true
$DOCKER_COMPOSE up -d

echo ""
echo "==================================================================="
echo "Monitoring stack is running!"
echo "==================================================================="
echo ""
echo "Access points:"
echo "  Prometheus: http://localhost:9092"
echo "  Grafana:    http://localhost:3002 (admin/admin)"
echo ""
echo "To stop the monitoring stack:"
echo "  $DOCKER_COMPOSE down"
echo ""
echo "To view logs:"
echo "  $DOCKER_COMPOSE logs -f"
echo "==================================================================="
