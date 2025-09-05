// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * utils.js
 *
 * Utility functions for telemetry frontend.
 */

/**
 * Converts an array of key-value pairs to a Map object.
 * This is needed because nlohmann::json serializes maps with non-string keys
 * as arrays of pairs, e.g.: [[0,"MHz"], [1,"mW"]]
 *
 * @param {Array} pairsArray - Array of [key, value] pairs
 * @returns {Map} - Map object with the key-value pairs
 */
export function arrayOfPairsToMap(pairsArray) {
    if (!pairsArray || !Array.isArray(pairsArray)) {
        return new Map();
    }

    const map = new Map();
    for (const pair of pairsArray) {
        if (Array.isArray(pair) && pair.length >= 2) {
            map.set(pair[0], pair[1]);
        }
    }

    return map;
}

/**
 * Converts an array of key-value pairs to a plain JavaScript object.
 * Alternative to arrayOfPairsToMap when you need a plain object instead of a Map.
 *
 * @param {Array} pairsArray - Array of [key, value] pairs
 * @returns {Object} - Plain object with the key-value pairs
 */
export function arrayOfPairsToObject(pairsArray) {
    if (!pairsArray || !Array.isArray(pairsArray)) {
        return {};
    }

    const obj = {};
    for (const pair of pairsArray) {
        if (Array.isArray(pair) && pair.length >= 2) {
            obj[pair[0]] = pair[1];
        }
    }

    return obj;
}

/**
 * Formats a numeric value to display with up to 2 decimal places.
 * Removes trailing zeros and unnecessary decimal points.
 * Examples: 4 -> "4", 4.1 -> "4.1", 4.12 -> "4.12", 4.123 -> "4.12"
 *
 * @param {any} value - The value to format
 * @returns {string} - Formatted string representation
 */
export function formatFloat(value) {
    if (value === null || value === undefined) {
        return 'N/A';
    }

    // If it's not a number, return as-is
    if (typeof value !== 'number') {
        return String(value);
    }

    // If it's an integer, return without decimal places
    if (Number.isInteger(value)) {
        return String(value);
    }

    // Format to 2 decimal places and remove trailing zeros
    return parseFloat(value.toFixed(2)).toString();
}
