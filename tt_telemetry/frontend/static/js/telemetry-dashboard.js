// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

import { LitElement, html, css } from 'https://cdn.jsdelivr.net/gh/lit/dist@3/all/lit-all.min.js';
import { HierarchicalTelemetryStore } from './hierarchical_telemetry_store.js';
import { arrayOfPairsToObject } from './utils.js';
import './status-grid.js';
import './metric-sidebar.js';

// MAIN DASHBOARD COMPONENT - Manages navigation and data
export class TelemetryDashboard extends LitElement {
    static styles = css`
        :host {
            display: block;
        }

        .controls {
            margin-bottom: 20px;
            text-align: center;
        }

        .controls button {
            background: #444;
            color: white;
            border: 1px solid #666;
            padding: 10px 20px;
            margin: 0 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
        }

        .controls button:hover {
            background: #666;
        }

        .navigation {
            margin-bottom: 20px;
            text-align: center;
        }

        .breadcrumb {
            font-size: 18px;
            color: #ccc;
            margin-bottom: 10px;
        }

        .back-button {
            background: #444;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
        }

        .back-button:hover {
            background: #666;
        }

        .back-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
    `;

    static properties = {
        currentPath: { type: Array },
        selectedMetric: { type: Object },
        sidebarOpen: { type: Boolean }
    };

    constructor() {
        super();

        this.currentPath = [];
        this.selectedMetric = null;
        this.sidebarOpen = false;
        this._eventSource = null;
        this._telemetryStore = new HierarchicalTelemetryStore();

        // Setup Server-Sent Events
        this._setupSSE();
    }

    _setupSSE() {
        try {
            // Replace '/api/events' with your actual SSE endpoint
            this._eventSource = new EventSource('/api/stream');

            this._eventSource.onmessage = (event) => {
                let didUpdate = false;

                // Handle message
                try {
                    const data = JSON.parse(event.data);

                    const boolMetricIds = data["bool_metric_ids"];
                    const boolMetricNames = data["bool_metric_names"];    // optional, and indicates new telemetry metrics if present
                    const boolMetricValues = data["bool_metric_values"];
                    const boolMetricTimestamps = data["bool_metric_timestamps"] || [];

                    if (boolMetricIds.length != boolMetricValues.length) {
                        console.log(`SSE error: Received differing id and value counts (${boolMetricIds.length} vs. ${boolMetricValues.length})`);
                        return;
                    }

                    if (boolMetricTimestamps.length > 0 && boolMetricTimestamps.length != boolMetricIds.length) {
                        console.log(`SSE error: Received differing id and timestamp counts (${boolMetricIds.length} vs. ${boolMetricTimestamps.length})`);
                        return;
                    }

                    if (boolMetricNames.length > 0 && boolMetricNames.length != boolMetricIds.length) {
                        console.log(`SSE error: Received differing name and id counts (${boolMetricNames.length} vs. ${boolMetricIds.length})`);
                        return;
                    }

                    if (boolMetricNames.length > 0) {
                        // Adding new
                        for (let i = 0; i < boolMetricIds.length; i++) {
                            const path = boolMetricNames[i];
                            const id = boolMetricIds[i];
                            const value = boolMetricValues[i];
                            const timestamp = boolMetricTimestamps[i];
                            this._telemetryStore.addPath(path, id, value, true, timestamp);
                            didUpdate = true;
                        }
                    } else {
                        // Delta updates
                        for (let i = 0; i < boolMetricIds.length; i++) {
                            const id = boolMetricIds[i];
                            const value = boolMetricValues[i];
                            const timestamp = boolMetricTimestamps[i];
                            this._telemetryStore.updateBoolValue(id, value, timestamp);
                            didUpdate = true;
                        }
                    }

                    const uintMetricIds = data["uint_metric_ids"];
                    const uintMetricNames = data["uint_metric_names"];    // optional, and indicates new telemetry metrics if present
                    const uintMetricUnits = data["uint_metric_units"] || [];
                    const uintMetricValues = data["uint_metric_values"];
                    const uintMetricTimestamps = data["uint_metric_timestamps"] || [];
                    const unitDisplayLabels = arrayOfPairsToObject(data["metric_unit_display_label_by_code"]);
                    const unitFullLabels = arrayOfPairsToObject(data["metric_unit_full_label_by_code"]);

                    if (uintMetricIds.length != uintMetricValues.length) {
                        console.log(`SSE error: Received differing id and value counts (${uintMetricIds.length} vs. ${uintMetricValues.length})`);
                        return;
                    }

                    if (uintMetricTimestamps.length > 0 && uintMetricTimestamps.length != uintMetricIds.length) {
                        console.log(`SSE error: Received differing id and timestamp counts (${uintMetricIds.length} vs. ${uintMetricTimestamps.length})`);
                        return;
                    }

                    if (uintMetricNames.length > 0 && uintMetricNames.length != uintMetricIds.length) {
                        console.log(`SSE error: Received differing name and id counts (${uintMetricNames.length} vs. ${uintMetricIds.length})`);
                        return;
                    }

                    if (uintMetricUnits.length > 0 && uintMetricUnits.length != uintMetricIds.length) {
                        console.log(`SSE error: Received differing units and id counts (${uintMetricUnits.length} vs. ${uintMetricIds.length})`);
                        return;
                    }

                    // Update unit label maps if present
                    this._telemetryStore.updateUnitLabelMaps(unitDisplayLabels, unitFullLabels);

                    if (uintMetricNames.length > 0) {
                        // Adding new metrics with unit information
                        for (let i = 0; i < uintMetricIds.length; i++) {
                            const path = uintMetricNames[i];
                            const id = uintMetricIds[i];
                            const value = uintMetricValues[i];
                            const timestamp = uintMetricTimestamps.length > 0 ? uintMetricTimestamps[i] : null;
                            const unitCode = uintMetricUnits.length > 0 ? uintMetricUnits[i] : null;
                            this._telemetryStore.addPath(path, id, value, false, timestamp, unitCode);
                            didUpdate = true;
                        }
                    } else {
                        // Delta updates
                        for (let i = 0; i < uintMetricIds.length; i++) {
                            const id = uintMetricIds[i];
                            const value = uintMetricValues[i];
                            const timestamp = uintMetricTimestamps.length > 0 ? uintMetricTimestamps[i] : null;
                            this._telemetryStore.updateUIntValue(id, value, timestamp);
                            didUpdate = true;
                        }
                    }

                    const doubleMetricIds = data["double_metric_ids"];
                    const doubleMetricNames = data["double_metric_names"];    // optional, and indicates new telemetry metrics if present
                    const doubleMetricUnits = data["double_metric_units"] || [];
                    const doubleMetricValues = data["double_metric_values"];
                    const doubleMetricTimestamps = data["double_metric_timestamps"] || [];

                    if (doubleMetricIds.length != doubleMetricValues.length) {
                        console.log(`SSE error: Received differing id and value counts (${doubleMetricIds.length} vs. ${doubleMetricValues.length})`);
                        return;
                    }

                    if (doubleMetricTimestamps.length > 0 && doubleMetricTimestamps.length != doubleMetricIds.length) {
                        console.log(`SSE error: Received differing id and timestamp counts (${doubleMetricIds.length} vs. ${doubleMetricTimestamps.length})`);
                        return;
                    }

                    if (doubleMetricNames.length > 0 && doubleMetricNames.length != doubleMetricIds.length) {
                        console.log(`SSE error: Received differing name and id counts (${doubleMetricNames.length} vs. ${doubleMetricIds.length})`);
                        return;
                    }

                    if (doubleMetricUnits.length > 0 && doubleMetricUnits.length != doubleMetricIds.length) {
                        console.log(`SSE error: Received differing units and id counts (${doubleMetricUnits.length} vs. ${doubleMetricIds.length})`);
                        return;
                    }

                    if (doubleMetricNames.length > 0) {
                        // Adding new metrics with unit information
                        for (let i = 0; i < doubleMetricIds.length; i++) {
                            const path = doubleMetricNames[i];
                            const id = doubleMetricIds[i];
                            const value = doubleMetricValues[i];
                            const timestamp = doubleMetricTimestamps.length > 0 ? doubleMetricTimestamps[i] : null;
                            const unitCode = doubleMetricUnits.length > 0 ? doubleMetricUnits[i] : null;
                            this._telemetryStore.addPath(path, id, value, false, timestamp, unitCode);
                            didUpdate = true;
                        }
                    } else {
                        // Delta updates
                        for (let i = 0; i < doubleMetricIds.length; i++) {
                            const id = doubleMetricIds[i];
                            const value = doubleMetricValues[i];
                            const timestamp = doubleMetricTimestamps.length > 0 ? doubleMetricTimestamps[i] : null;
                            this._telemetryStore.updateDoubleValue(id, value, timestamp);
                            didUpdate = true;
                        }
                    }
                } catch (error) {
                    // If it's not JSON, append as-is
                    console.error(`SSE error: Error processing data: ${error}`);
                }

                if (didUpdate) {
                    this.requestUpdate();
                }
            };

            this._eventSource.onerror = (error) => {
                console.error('SSE error:', error);
            };

            this._eventSource.onopen = () => {
                console.log('SSE connection opened');
            };
        } catch (error) {
            console.error('Error setting up SSE:', error);
        }
    }

    _handleGridBoxClick(event) {
        // Append the thing we clicked on to the path
        const metricName = event.detail.name;
        const newPath = [ ...this.currentPath, metricName ].join("/");

        console.log(`Clicked ${metricName}: newPath=${newPath}`);

        // Get children, if any, to drill in deeper
        const childNames = this._telemetryStore.getChildNames(newPath);
        console.log(childNames);

        if (childNames.length > 0) {
            // This is an intermediate node - drill down
            this.currentPath = [...this.currentPath, metricName]; // descend one level deeper (don't use push, we need to mutate array to trigger state update)
        } else {
            // This is a leaf node - open sidebar with details
            const metricData = this._telemetryStore.getData(newPath);
            this.selectedMetric = {
                name: metricName,
                fullPath: newPath,
                value: metricData ? metricData.value : null,
                timestamp: metricData ? metricData.timestamp : null,
                unitDisplayLabel: metricData ? metricData.unitDisplayLabel : null,
                unitFullLabel: metricData ? metricData.unitFullLabel : null
            };
            this.sidebarOpen = true;
        }
    }

    _handleBackClick() {
        if (this.currentPath.length > 0) {
            this.currentPath = this.currentPath.slice(0, -1);   // remove last
        }
    }

    _handleSidebarClose() {
        this.sidebarOpen = false;
        this.selectedMetric = null;
    }

    _getBreadcrumb() {
        if (this.currentPath.length == 0) {
            return "Cluster Overview";
        } else {
            return this.currentPath.join(" > ");
        }
    }

    _getCurrentMetrics() {
        // Get children of currently displayed path
        const currentPath = this.currentPath.join("/");
        const metricNames = this._telemetryStore.getChildNames(currentPath);

        // Create metric boxes
        const metrics = [];
        for (const metricName of metricNames) {
            const metricPath = [...this.currentPath, metricName].join("/");
            const hasChildren = this._telemetryStore.getChildNames(metricPath).length > 0;
            const metricData = this._telemetryStore.getData(metricPath);
            metrics.push({
                name: metricName,
                value: metricData ? metricData.value : null,
                timestamp: metricData ? metricData.timestamp : null,
                unitDisplayLabel: metricData ? metricData.unitDisplayLabel : null,
                unitFullLabel: metricData ? metricData.unitFullLabel : null,
                isLeaf: !hasChildren  // True if this is a final node with no children
            });
        }

        return metrics;
    }

    render() {
        console.log("Render", this._getCurrentMetrics());
        return html`
            <div class="navigation">
                <div class="breadcrumb">${this._getBreadcrumb()}</div>
                <button class="back-button"
                        ?disabled="${this.currentPath.length === 0}"
                        @click="${this._handleBackClick}">
                    ← Back
                </button>
            </div>

            <status-grid
                .metrics="${this._getCurrentMetrics()}"
                @grid-box-click="${this._handleGridBoxClick}">
            </status-grid>

            <metric-sidebar
                ?open="${this.sidebarOpen}"
                .metricName="${this.selectedMetric?.name || ''}"
                .metricValue="${this.selectedMetric?.value}"
                .metricTimestamp="${this.selectedMetric?.timestamp}"
                .unitDisplayLabel="${this.selectedMetric?.unitDisplayLabel}"
                .unitFullLabel="${this.selectedMetric?.unitFullLabel}"
                @sidebar-close="${this._handleSidebarClose}">
            </metric-sidebar>
        `;
    }
}

// Register the custom element
customElements.define('telemetry-dashboard', TelemetryDashboard);
