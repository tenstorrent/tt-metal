import { LitElement, html, css } from 'https://cdn.jsdelivr.net/gh/lit/dist@3/all/lit-all.min.js';
import { isBool } from './hierarchical_telemetry_store.js';
import './status-box.js';

// STATUS GRID COMPONENT - Manages layout of multiple status boxes
export class StatusGrid extends LitElement {
    static styles = css`
        :host {
            display: block;
            padding: 20px;
        }

        .grid {
            display: grid;
            gap: 15px;
            max-width: 1200px;
            margin: 0 auto;
        }

        .grid.level-any {
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        }
    `;

    static properties = {
        metrics: { type: Array },
    };

    constructor() {
        super();
        this.metrics = [];
    }

    _handleBoxClick(event) {
        // Bubble up the box click event
        this.dispatchEvent(new CustomEvent('grid-box-click', {
            detail: event.detail,
            bubbles: true
        }));
    }

    render() {
        const gridClass = `grid level-any`;

        // Sort metrics: boolean metrics first, then alphabetically by name
        const sortedMetrics = [...this.metrics].sort((metric1, metric2) => {
            const isBool1 = isBool(metric1.value);
            const isBool2 = isBool(metric2.value);
            
            // If one is bool and the other isn't, prioritize the bool metric
            if (isBool1 && !isBool2) return -1;
            if (!isBool1 && isBool2) return 1;
            
            // If both are same type (both bool or both non-bool), sort alphabetically
            return metric1.name.localeCompare(metric2.name);
        });

        return html`
            <div class="${gridClass}" @box-click="${this._handleBoxClick}">
                ${sortedMetrics.map(metric => html`
                    <status-box 
                        name="${metric.name}"
                        .value="${metric.value}"
                        .clickable="true"
                        type="${isBool(metric.value) ? "health" : "valued"}">
                    </status-box>
                `)}
            </div>
        `;
    }
}

// Register the custom element
customElements.define('status-grid', StatusGrid);
