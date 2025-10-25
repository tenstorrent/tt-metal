// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

import { LitElement, html, css } from 'https://cdn.jsdelivr.net/gh/lit/dist@3/all/lit-all.min.js';
import { HierarchicalTelemetryStore } from './hierarchical_telemetry_store.js';
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
                // Handle message
                try {
                    const snapshot = JSON.parse(event.data);
                    const didUpdate = this._telemetryStore.processSnapshot(snapshot);

                    if (didUpdate) {
                        this.requestUpdate();
                    }
                } catch (error) {
                    console.error(`SSE error: Error processing data: ${error}`);
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
