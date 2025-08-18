import { LitElement, html, css } from 'https://cdn.jsdelivr.net/gh/lit/dist@3/all/lit-all.min.js';
import { HierarchicalTelemetryStore } from './hierarchical_telemetry_store.js';
import './status-grid.js';

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
        currentPath: { type: Array }
    };

    constructor() {
        super();

        this.currentPath = [];
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
                let did_update = false;

                // Handle message
                try {
                    const data = JSON.parse(event.data);

                    const bool_metric_ids = data["bool_metric_ids"];
                    const bool_metric_names = data["bool_metric_names"];    // optional, and indicates new telemetry metrics if present
                    const bool_metric_values = data["bool_metric_values"];
                    const is_absolute = data["is_absolute"];                // absolute update

                    if (bool_metric_ids.length != bool_metric_values.length) {
                        console.log(`SSE error: Received differing id and value counts (${bool_metric_ids.length} vs. ${bool_metric_values.length})`);
                        return;
                    }

                    if (bool_metric_names.length > 0 && bool_metric_names.length != bool_metric_ids.length) {
                        console.log(`SSE error: Received differing name and id counts (${bool_metric_names.length} vs. ${bool_metric_ids.length})`);
                        return;
                    }

                    if (bool_metric_names.length > 0) {
                        // Adding new
                        for (let i = 0; i < bool_metric_ids.length; i++) {
                            const path = bool_metric_names[i];
                            const id = bool_metric_ids[i];
                            const value = bool_metric_values[i];
                            this._telemetryStore.addPath(path, id, value, true);
                            did_update = true;
                        }
                    } else {
                        // Delta updates
                        for (let i = 0; i < bool_metric_ids.length; i++) {
                            const id = bool_metric_ids[i];
                            const value = bool_metric_values[i];
                            this._telemetryStore.updateBoolValue(id, value);
                            did_update = true;
                        }
                    }

                    const uint_metric_ids = data["uint_metric_ids"];
                    const uint_metric_names = data["uint_metric_names"];    // optional, and indicates new telemetry metrics if present
                    const uint_metric_values = data["uint_metric_values"];

                    if (uint_metric_ids.length != uint_metric_values.length) {
                        console.log(`SSE error: Received differing id and value counts (${uint_metric_ids.length} vs. ${uint_metric_values.length})`);
                        return;
                    }

                    if (uint_metric_names.length > 0 && uint_metric_names.length != uint_metric_ids.length) {
                        console.log(`SSE error: Received differing name and id counts (${uint_metric_names.length} vs. ${uint_metric_ids.length})`);
                        return;
                    }

                    if (uint_metric_names.length > 0) {
                        // Adding new
                        for (let i = 0; i < uint_metric_ids.length; i++) {
                            const path = uint_metric_names[i];
                            const id = uint_metric_ids[i];
                            const value = uint_metric_values[i];
                            this._telemetryStore.addPath(path, id, value, false);
                            did_update = true;
                        }
                    } else {
                        // Delta updates
                        for (let i = 0; i < uint_metric_ids.length; i++) {
                            const id = uint_metric_ids[i];
                            const value = uint_metric_values[i];
                            this._telemetryStore.updateUIntValue(id, value);
                            did_update = true;
                        }
                    }
                } catch (error) {
                    // If it's not JSON, append as-is
                    console.error(`SSE error: Error processing data: ${error}`);
                }

                if (did_update) {
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

        console.log(`Clicked ${metricName}: new_path=${newPath}`);
        
        // Get children, if any, to drill in deeper
        const child_names = this._telemetryStore.getChildNames(newPath);
        console.log(child_names);
        if (child_names.length > 0) {
            this.currentPath = [...this.currentPath, metricName]; // descend one level deeper (don't use push, we need to mutate array to trigger state update)
        }
    }

    _handleBackClick() {
        if (this.currentPath.length > 0) {
            this.currentPath = this.currentPath.slice(0, -1);   // remove last
        }
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
        const current_path = this.currentPath.join("/");
        const metric_names = this._telemetryStore.getChildNames(current_path);

        // Create metric boxes
        const metrics = [];
        for (const metric_name of metric_names) {
            const metric_path = [...this.currentPath, metric_name].join("/");
            metrics.push({
                name: metric_name,
                value: this._telemetryStore.getValue(metric_path)
            });
        }

        // Sort by name
        metrics.sort((metric1, metric2) => metric1.name.localeCompare(metric2.name));

        return metrics;
    }

    render() {
        console.log("Render", this._getCurrentMetrics());
        return html`
            <!--
            <div class="controls">
                <button @click="${this.randomizeStatus}">Randomize Status</button>
                <button @click="${() => this.setCpuCount(2)}">2 CPUs</button>
                <button @click="${() => this.setCpuCount(4)}">4 CPUs</button>
                <button @click="${() => this.setCpuCount(8)}">8 CPUs</button>
                <button @click="${() => this.setConditionCount(3)}">3 Conditions</button>
                <button @click="${() => this.setConditionCount(6)}">6 Conditions</button>
            </div>
            -->

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
        `;
    }
}

// Register the custom element
customElements.define('telemetry-dashboard', TelemetryDashboard);
