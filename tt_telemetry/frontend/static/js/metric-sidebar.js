import { LitElement, html, css } from 'https://cdn.jsdelivr.net/gh/lit/dist@3/all/lit-all.min.js';
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// METRIC SIDEBAR COMPONENT - Shows detailed information about a selected metric
export class MetricSidebar extends LitElement {
    static styles = css`
        :host {
            display: block;
            position: fixed;
            top: 0;
            right: 0;
            width: 400px;
            height: 100vh;
            background: #2d3748;
            color: white;
            box-shadow: -5px 0 15px rgba(0, 0, 0, 0.3);
            transform: translateX(100%);
            transition: transform 0.3s ease-in-out;
            z-index: 1000;
            overflow-y: auto;
        }

        :host([open]) {
            transform: translateX(0);
        }

        .sidebar-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            border-bottom: 1px solid #4a5568;
            background: #1a202c;
        }

        .sidebar-title {
            font-size: 1.5em;
            font-weight: bold;
            margin: 0;
            flex-grow: 1;
            padding-right: 10px;
            word-wrap: break-word;
        }

        .close-button {
            background: none;
            border: none;
            color: white;
            font-size: 1.5em;
            cursor: pointer;
            padding: 5px;
            border-radius: 3px;
            transition: background-color 0.2s;
            flex-shrink: 0;
        }

        .close-button:hover {
            background: #4a5568;
        }

        .sidebar-content {
            padding: 20px;
        }

        .info-section {
            margin-bottom: 25px;
        }

        .info-section h3 {
            color: #e2e8f0;
            font-size: 1.1em;
            margin-bottom: 10px;
            border-bottom: 1px solid #4a5568;
            padding-bottom: 5px;
        }

        .info-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #374151;
        }

        .info-item:last-child {
            border-bottom: none;
        }

        .info-label {
            font-weight: 500;
            color: #cbd5e0;
        }

        .info-value {
            color: #f7fafc;
            font-family: monospace;
            background: #374151;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }

        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }

        .status-good {
            background: #48bb78;
            color: white;
        }

        .status-bad {
            background: #f56565;
            color: white;
        }

        .status-unknown {
            background: #718096;
            color: white;
        }

        .metric-chart {
            height: 200px;
            background: #374151;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #a0aec0;
            font-style: italic;
        }

        .recent-events {
            max-height: 300px;
            overflow-y: auto;
        }

        .event-item {
            padding: 10px;
            background: #374151;
            margin-bottom: 8px;
            border-radius: 6px;
            border-left: 4px solid #4299e1;
        }

        .event-time {
            font-size: 0.8em;
            color: #a0aec0;
            margin-bottom: 4px;
        }

        .event-message {
            color: #e2e8f0;
        }
    `;

    static properties = {
        open: { type: Boolean, reflect: true },
        metricName: { type: String },
        metricValue: { type: Object },
        metricTimestamp: { type: Object },
        unitDisplayLabel: { type: String },
        unitFullLabel: { type: String }
    };

    constructor() {
        super();
        this.open = false;
        this.metricName = '';
        this.metricValue = null;
        this.metricTimestamp = null;
        this.unitDisplayLabel = null;
        this.unitFullLabel = null;
    }

    _handleClose() {
        this.dispatchEvent(new CustomEvent('sidebar-close', {
            bubbles: true
        }));
    }

    _getMockData() {
        // Mock data - in the future this would come from props or API calls
        const lastUpdated = this.metricTimestamp ?
            this.metricTimestamp.toLocaleString() :
            'Never';

        return {
            lastUpdated,
            updateFrequency: '5 seconds',
            dataType: typeof this.metricValue === 'boolean' ? 'Boolean Health' : 'Numeric Value',
            source: 'Hardware Monitor',
            threshold: typeof this.metricValue === 'boolean' ? 'N/A' : '> 100',
            unit: typeof this.metricValue === 'boolean' ? 'Status' : 'Count',
            events: [
                // {
                //     time: '2 minutes ago',
                //     message: 'Metric value updated successfully'
                // },
                // {
                //     time: '5 minutes ago',
                //     message: 'Data collection cycle completed'
                // },
                // {
                //     time: '8 minutes ago',
                //     message: 'Connection established to hardware'
                // }
            ]
        };
    }

    _renderStatusBadge(status) {
        const statusClass = `status-badge status-${status ? 'good' : 'bad'}`;
        const statusText = status ? 'GOOD' : 'BAD';
        return html`<span class="${statusClass}">${statusText}</span>`;
    }

    _getTimestampAge() {
        if (!this.metricTimestamp) return 'Unknown';

        const now = new Date();
        const diffMs = now.getTime() - this.metricTimestamp.getTime();
        const diffSeconds = Math.floor(diffMs / 1000);

        if (diffSeconds < 60) {
            return `${diffSeconds} seconds ago`;
        } else if (diffSeconds < 3600) {
            const minutes = Math.floor(diffSeconds / 60);
            return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
        } else if (diffSeconds < 86400) {
            const hours = Math.floor(diffSeconds / 3600);
            return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
        } else {
            const days = Math.floor(diffSeconds / 86400);
            return `${days} day${days !== 1 ? 's' : ''} ago`;
        }
    }

    render() {
        if (!this.open) return html``;

        const mockData = this._getMockData();

        const isBoolean = typeof this.metricValue === 'boolean'
        let statusBadge = '';
        if (isBoolean) {
            statusBadge = html`
            <div class="info-item">
                <span class="info-label">Status:</span>
                ${this._renderStatusBadge(this.metricValue)}
            </div>
            `;
        }

        return html`
            <div class="sidebar-header">
                <h2 class="sidebar-title">${this.metricName}</h2>
                <button class="close-button" @click="${this._handleClose}" title="Close">
                    ×
                </button>
            </div>

            <div class="sidebar-content">
                <div class="info-section">
                    <h3>Current Status</h3>
                    ${statusBadge}
                     <div class="info-item">
                        <span class="info-label">Value:</span>
                        <span class="info-value">${this.metricValue}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Last Changed:</span>
                        <span class="info-value">${mockData.lastUpdated}</span>
                    </div>
                    ${this.metricTimestamp ? html`
                    <div class="info-item">
                        <span class="info-label"></span>
                        <span class="info-value">${this._getTimestampAge()}</span>
                    </div>
                    ` : ''}
                </div>

                <div class="info-section">
                    <h3>Configuration</h3>
                    <div class="info-item">
                        <span class="info-label">Data Type:</span>
                        <span class="info-value">${mockData.dataType}</span>
                    </div>
                    ${this.unitDisplayLabel || this.unitFullLabel ? html`
                    <div class="info-item">
                        <span class="info-label">Units:</span>
                        <span class="info-value">
                            ${this.unitDisplayLabel || ''}${this.unitFullLabel ? ` (${this.unitFullLabel})` : ''}
                        </span>
                    </div>
                    ` : ''}
                    <div class="info-item">
                        <span class="info-label">Update Frequency:</span>
                        <span class="info-value">${mockData.updateFrequency}</span>
                    </div>
                </div>

                <div class="info-section">
                    <h3>Historical Data</h3>
                    <div class="metric-chart">
                        Chart visualization coming soon...
                    </div>
                </div>

                <div class="info-section">
                    <h3>Recent Events</h3>
                    <div class="recent-events">
                        ${mockData.events.map(event => html`
                            <div class="event-item">
                                <div class="event-time">${event.time}</div>
                                <div class="event-message">${event.message}</div>
                            </div>
                        `)}
                    </div>
                </div>
            </div>
        `;
    }
}

// Register the custom element
customElements.define('metric-sidebar', MetricSidebar);
