import { LitElement, html, css } from 'https://cdn.jsdelivr.net/gh/lit/dist@3/all/lit-all.min.js';

// STATUS BOX COMPONENT - Individual clickable status boxes
export class StatusBox extends LitElement {
    static styles = css`
        :host {
            display: block;
        }

        .status-box {
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s ease;
            border: 2px solid transparent;
            position: relative;
            overflow: hidden;
            min-height: 100px;
        }

        .status-box:hover {
            transform: scale(1.05);
            border-color: rgba(255, 255, 255, 0.3);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }

        .status-box.good {
            background: linear-gradient(135deg, #4ade80, #22c55e);
            color: white;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }

        .status-box.bad {
            background: linear-gradient(135deg, #f87171, #ef4444);
            color: white;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }

        .status-box.valued {
            background: linear-gradient(135deg, #e5e7eb, #d1d5db);
            color: #374151;
            text-shadow: none;
            flex-direction: column;
            gap: 8px;
        }

        .valued-name {
            font-weight: bold;
            font-size: 1em;
        }

        .valued-value {
            font-weight: normal;
            font-size: 0.9em;
            opacity: 0.8;
        }

        .status-box.level-1 {
            font-size: 24px;
            min-height: 200px;
        }

        .status-box.level-2 {
            font-size: 18px;
            min-height: 120px;
        }

        .status-box.level-3 {
            font-size: 14px;
            min-height: 100px;
        }

        .status-box.level-any {
            font-size: 18px;
            min-height: 120px;
        }

        .status-box.not-clickable {
            cursor: default;
        }

        .status-box.not-clickable:hover {
            transform: none;
            border-color: transparent;
            box-shadow: none;
        }

        .status-indicator {
            position: absolute;
            top: 10px;
            right: 10px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.8);
            border: 2px solid currentColor;
        }
    `;

    static properties = {
        name: { type: String },
        level: { type: Number },
        clickable: { type: Boolean },
        type: { type: String },
        value: { type: Object }
    };

    constructor() {
        super();
        this.name = '';
        this.clickable = true;
        this.type = 'health';
        this.value = null;
    }

    _handleClick() {
        if (this.clickable) {
            this.dispatchEvent(new CustomEvent('box-click', {
                detail: { name: this.name },
                bubbles: true
            }));
        }
    }

    render() {
        let statusClass;
        let content;
        
        if (this.type === 'valued') {
            statusClass = 'valued';
            content = html`
                <div class="valued-name">${this.name}</div>
                <div class="valued-value">${this.value !== null ? this.value : 'N/A'}</div>
            `;
        } else {
            // Default to health type - treat value as boolean for health status
            const isHealthy = typeof this.value === 'boolean' ? this.value : true;
            statusClass = isHealthy ? 'good' : 'bad';
            content = html`
                <div class="status-indicator"></div>
                ${this.name}
            `;
        }
        
        const levelClass = `level-any`;
        const clickableClass = this.clickable ? '' : 'not-clickable';

        return html`
            <div class="status-box ${statusClass} ${levelClass} ${clickableClass}" 
                 @click="${this._handleClick}">
                ${content}
            </div>
        `;
    }
}

// Register the custom element
customElements.define('status-box', StatusBox);
