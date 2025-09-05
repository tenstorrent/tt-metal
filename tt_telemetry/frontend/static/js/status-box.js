// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

import { LitElement, html, css } from 'https://cdn.jsdelivr.net/gh/lit/dist@3/all/lit-all.min.js';
import { formatFloat } from './utils.js';

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

        .valued-value {
            font-weight: normal;
            font-size: 0.9em;
            opacity: 0.8;
            text-align: center;
            word-wrap: break-word;
            overflow-wrap: break-word;
            max-width: 100%;
            padding: 0 8px;
            line-height: 1.2;
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

        .label-text {
            text-align: center;
            word-wrap: break-word;
            overflow-wrap: break-word;
            hyphens: auto;
            max-width: 100%;
            padding: 0 8px;
            line-height: 1.2;
            display: -webkit-box;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 2;
            overflow: hidden;
            font-size: inherit;
        }

        .label-text.scale-down {
            font-size: 0.85em;
        }

        .label-text.scale-down-more {
            font-size: 0.7em;
        }

        .valued-name {
            font-weight: bold;
            font-size: 1em;
            text-align: center;
            word-wrap: break-word;
            overflow-wrap: break-word;
            hyphens: auto;
            max-width: 100%;
            padding: 0 8px;
            line-height: 1.2;
            display: -webkit-box;
            -webkit-box-orient: vertical;
            -webkit-line-clamp: 2;
            overflow: hidden;
        }

        .valued-name.scale-down {
            font-size: 0.85em;
        }

        .valued-name.scale-down-more {
            font-size: 0.7em;
        }
    `;

    static properties = {
        name: { type: String },
        level: { type: Number },
        clickable: { type: Boolean },
        type: { type: String },
        value: { type: Object },
        isLeaf: { type: Boolean },
        unitDisplayLabel: { type: String },
        unitFullLabel: { type: String }
    };

    constructor() {
        super();
        this.name = '';
        this.clickable = true;
        this.type = 'health';
        this.value = null;
        this.isLeaf = false;
        this.unitDisplayLabel = null;
        this.unitFullLabel = null;
        this._cachedFittedText = null;
    }

    updated(changedProperties) {
        super.updated(changedProperties);

        // If the name changed or this is the first render, recalculate text fitting
        // after the DOM has been updated with actual dimensions
        if (changedProperties.has('name') && this.name) {
            // Use requestAnimationFrame to ensure DOM is fully rendered
            requestAnimationFrame(() => {
                const newFittedText = this._getFittedText(this.name, true);

                // If the fitted text changed, trigger a re-render
                if (JSON.stringify(newFittedText) !== JSON.stringify(this._cachedFittedText)) {
                    this._cachedFittedText = newFittedText;
                    this.requestUpdate();
                }
            });
        }
    }

    _handleClick() {
        if (this.clickable) {
            this.dispatchEvent(new CustomEvent('box-click', {
                detail: { name: this.name },
                bubbles: true
            }));
        }
    }

    /**
     * Fits text to available space by trying different approaches:
     * 1. Use original text if it fits in one line
     * 2. Scale down font size for medium-length text
     * 3. Allow wrapping up to 2 lines
     * 4. Abbreviate with "..." if it would require more than 2 lines
     */
    _getFittedText(text, forceRecalculate = false) {
        if (!text) return { text: '', scaleClass: '' };

        // Use cached result if text hasn't changed and we're not forcing recalculation
        if (!forceRecalculate && this._cachedFittedText && this._cachedFittedText.text === text) {
            return this._cachedFittedText;
        }

        // Get available dimensions with padding margin
        const availableWidth = this._getAvailableWidth();
        const availableHeight = this._getAvailableHeight();

        // Try different sizing approaches in order of preference
        const approaches = [
            { scaleClass: '', fontSize: 1.0 },           // Normal size
            { scaleClass: 'scale-down', fontSize: 0.85 }, // Slightly smaller
            { scaleClass: 'scale-down-more', fontSize: 0.7 } // Much smaller
        ];

        for (const approach of approaches) {
            // Check if text fits in one line at this size
            if (this._textFitsInOneLine(text, availableWidth, approach.fontSize)) {
                const result = { text, scaleClass: approach.scaleClass };
                return result;
            }

            // Check if text fits in two lines at this size
            if (this._textFitsInTwoLines(text, availableWidth, availableHeight, approach.fontSize)) {
                const result = { text, scaleClass: approach.scaleClass };
                return result;
            }
        }

        // If still doesn't fit, abbreviate text and use smallest size
        const abbreviatedText = this._abbreviateText(text, availableWidth, availableHeight, 0.7);
        const result = { text: abbreviatedText, scaleClass: 'scale-down-more' };
        return result;
    }

    /**
     * Gets the available width for text, accounting for padding and indicators
     */
    _getAvailableWidth() {
        // Get actual box dimensions from DOM
        const boxElement = this.shadowRoot?.querySelector('.status-box');
        let boxWidth = 200; // Fallback default
        let isActualMeasurement = false;

        if (boxElement) {
            const rect = boxElement.getBoundingClientRect();
            if (rect.width > 0) {
                boxWidth = rect.width;
                isActualMeasurement = true;
            }
        }

        const horizontalPadding = 16; // 8px on each side from CSS
        const indicatorSpace = this.type === 'health' ? 20 : 0; // Space for status indicator
        const margin = 10; // Safety margin

        const availableWidth = Math.max(50, boxWidth - horizontalPadding - indicatorSpace - margin);

        // Debug output for development
        // if (this.name && isActualMeasurement) {
        //     console.log(`[StatusBox] ${this.name}: actual width=${boxWidth}px, available=${availableWidth}px`);
        // }

        return availableWidth;
    }

    /**
     * Gets the available height for text
     */
    _getAvailableHeight() {
        // Get actual box dimensions from DOM
        const boxElement = this.shadowRoot?.querySelector('.status-box');
        let boxHeight = 120; // Fallback default
        let isActualMeasurement = false;

        if (boxElement) {
            const rect = boxElement.getBoundingClientRect();
            if (rect.height > 0) {
                boxHeight = rect.height;
                isActualMeasurement = true;
            }
        }

        const verticalPadding = 16; // From CSS
        const valueSpace = this.type === 'valued' ? 30 : 0; // Space for value display
        const margin = 10; // Safety margin

        const availableHeight = Math.max(20, boxHeight - verticalPadding - valueSpace - margin);

        // Debug output for development
        // if (this.name && isActualMeasurement) {
        //     console.log(`[StatusBox] ${this.name}: actual height=${boxHeight}px, available=${availableHeight}px`);
        // }

        return availableHeight;
    }

    /**
     * Estimates if text fits in one line at given font size
     */
    _textFitsInOneLine(text, availableWidth, fontSizeRatio) {
        // More accurate estimation considering font characteristics
        const baseFontSize = 18; // Base font size from CSS
        const actualFontSize = baseFontSize * fontSizeRatio;

        // For bold fonts (like status labels), characters are wider
        const isBold = this.type === 'valued' || this.type === 'health';
        const fontWidthMultiplier = isBold ? 0.65 : 0.6;

        const avgCharWidth = actualFontSize * fontWidthMultiplier;
        const estimatedWidth = text.length * avgCharWidth;

        return estimatedWidth <= availableWidth;
    }

    /**
     * Estimates if text fits in two lines at given font size
     */
    _textFitsInTwoLines(text, availableWidth, availableHeight, fontSizeRatio) {
        const baseFontSize = 18;
        const actualFontSize = baseFontSize * fontSizeRatio;
        const lineHeight = actualFontSize * 1.2; // line-height: 1.2 from CSS
        const twoLineHeight = lineHeight * 2;

        // Check if two lines fit vertically
        if (twoLineHeight > availableHeight) {
            return false;
        }

        // Estimate how much text fits in two lines
        const isBold = this.type === 'valued' || this.type === 'health';
        const fontWidthMultiplier = isBold ? 0.65 : 0.6;
        const avgCharWidth = actualFontSize * fontWidthMultiplier;
        const charsPerLine = Math.floor(availableWidth / avgCharWidth);
        const maxCharsInTwoLines = charsPerLine * 2;

        return text.length <= maxCharsInTwoLines;
    }

    /**
     * Abbreviates text to fit within the given constraints
     */
    _abbreviateText(text, availableWidth, availableHeight, fontSizeRatio) {
        const baseFontSize = 18;
        const actualFontSize = baseFontSize * fontSizeRatio;
        const lineHeight = actualFontSize * 1.2;
        const twoLineHeight = lineHeight * 2;

        // If even two lines don't fit vertically, use very short text
        if (twoLineHeight > availableHeight) {
            return text.substring(0, 8) + '...';
        }

        // Calculate how many characters fit in two lines, minus space for "..."
        const isBold = this.type === 'valued' || this.type === 'health';
        const fontWidthMultiplier = isBold ? 0.65 : 0.6;
        const avgCharWidth = actualFontSize * fontWidthMultiplier;
        const charsPerLine = Math.floor(availableWidth / avgCharWidth);
        const maxChars = (charsPerLine * 2) - 3; // Reserve 3 chars for "..."

        if (text.length <= maxChars) {
            return text;
        }

        return text.substring(0, Math.max(1, maxChars)) + '...';
    }

    render() {
        let statusClass;
        let content;

        const fittedText = this._getFittedText(this.name);
        this._cachedFittedText = fittedText;
        const { text: displayName, scaleClass } = fittedText;

        if (this.type === 'valued') {
            statusClass = 'valued';
            const valueDisplay = formatFloat(this.value);
            const unitDisplay = this.unitDisplayLabel ? ` ${this.unitDisplayLabel}` : '';
            content = html`
                <div class="valued-name ${scaleClass}">${displayName}</div>
                <div class="valued-value">${valueDisplay}${unitDisplay}</div>
            `;
        } else {
            // Default to health type - treat value as boolean for health status
            const isHealthy = typeof this.value === 'boolean' ? this.value : true;
            statusClass = isHealthy ? 'good' : 'bad';
            content = html`
                ${!this.isLeaf ? html`<div class="status-indicator"></div>` : ''}
                <div class="label-text ${scaleClass}">${displayName}</div>
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
