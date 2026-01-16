// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * HierarchicalTelemetryStore.js
 *
 * Stores telemetry data, named by slash-delimited paths, in a hierarchical tree. Leaf nodes
 * contain actual reported telemetry data and intermediate nodes are aggregated from children.
 *
 * Updated to use string-based paths directly (no more integer IDs) for much simpler processing
 * of telemetry snapshots.
 */

import { arrayOfPairsToMap } from './utils.js';

export function isBool(value) {
    return typeof value === "boolean" || value instanceof Boolean;
}

function isInt(value) {
    return Number.isInteger(value);
}

export class HierarchicalTelemetryStore {
    constructor() {
        // Telemetry data (value + timestamp + unit info) hashed by path. Partial paths are also tracked and are the
        // aggregate value of all children (ANDed together for bools). Timestamps for intermediate nodes
        // are the most recent timestamp of their children.
        // Each entry is: { value: any, timestamp: Date, unitDisplayLabel: string|null, unitFullLabel: string|null }
        this._dataByPath = new Map();

        // Hierarchical map of paths, allowing us to navigate to subsequently deeper levels. Given
        // a path like foo/bar/baz, the first level of keys will include "foo", which will index a
        // map containing "bar", which will in turn index a map containing "baz". "baz" has no
        // children and its value is nil.
        this._pathChildren = new Map();    // hierarchical map of paths

        // Unit label maps: code -> display label and code -> full label
        this._unitDisplayLabelByCode = new Map();
        this._unitFullLabelByCode = new Map();
    }

    // Returns the aggregate "health" value and most recent timestamp of a path by looking at immediate
    // bool-valued children, otherwise looks at the node directly, assuming it is a leaf.
    // Returns { value: boolean, timestamp: Date }
    _getAggregateHealthAndTimestamp(path) {
        // First, we need to navigate to the correct position in the hierarchy
        const parts = path.split("/");
        let currentMap = this._pathChildren;

        // Navigate through the hierarchy to find the correct map
        for (let i = 0; i < parts.length; i++) {
            const part = parts[i];
            if (!currentMap || !currentMap.has(part)) {
                // Path doesn't exist in hierarchy, should be leaf node
                if (i != (parts.length - 1)) {
                    console.error(`[HierarchicalTelemetryStore] Path descendants are not stored in hierarchy: ${path} stored only up to ${parts.slice(0, i + 1)}`);
                }
                currentMap = null;
                break;
            }
            currentMap = currentMap.get(part);
        }

        // If no children, this is a leaf node
        if (!currentMap) {
            const data = this._dataByPath.get(path);
            if (!data) {
                return { value: false, timestamp: new Date(0) };
            }
            if (!isBool(data.value)) {
                console.error(`[HierarchicalTelemetryStore] Value of ${path} was expected to be bool but is ${typeof data.value}`);
            }
            return { value: data.value === true, timestamp: data.timestamp };
        }

        // Aggregate childrens' boolean values and find most recent timestamp
        let value = true;
        let mostRecentTimestamp = new Date(0);
        for (const nextPathComponent of currentMap.keys()) {
            const childPath = path + "/" + nextPathComponent;
            let childData = this._dataByPath.get(childPath);
            if (childData && isBool(childData.value)) {
                // Only bools are considered for aggregate health
                value = value && childData.value;
                if (childData.timestamp > mostRecentTimestamp) {
                    mostRecentTimestamp = childData.timestamp;
                }
            }
        }
        return { value, timestamp: mostRecentTimestamp };
    }

    // Updates telemetry state for a particular boolean-valued node. Will set the value directly
    // and then propagate upwards if changed. If the value did not already exist and this is the
    // first insertion, propagation will occur.
    updateBoolValue(path, value, timestamp = null) {
        const isBoolean = isBool(value) || value === 1 || value === 0;
        if (!isBoolean) {
            console.error(`[HierarchicalTelemetryStore] Value is not bool (${typeof value})`);
        }

        // Convert value to bool type if it is not
        value = value == true;

        // Convert timestamp to Date if provided, otherwise use current time
        const timestampDate = timestamp ? new Date(timestamp) : new Date();

        // Update value and timestamp, preserving unit information
        const oldData = this._dataByPath.get(path);
        const forceUpdate = oldData === undefined;
        let changed = !oldData || value !== oldData.value || timestampDate.getTime() !== oldData.timestamp.getTime();
        const newData = {
            value,
            timestamp: timestampDate,
            unitDisplayLabel: oldData ? oldData.unitDisplayLabel : null,
            unitFullLabel: oldData ? oldData.unitFullLabel : null
        };
        this._dataByPath.set(path, newData);
        console.log(`Set ${path} = ${value} at ${timestampDate.toISOString()}`);

        // No change? We are done.
        if (!changed && !forceUpdate) {
            return;
        }

        // Split path into components. Then move upwards to propagate the boolean value and timestamp.
        const parts = path.split("/");
        for (let i = parts.length; i > 0; i--) {
            const currentPath = parts.slice(0, i).join("/");
            if (currentPath !== path) {
                const aggregateData = this._getAggregateHealthAndTimestamp(currentPath);
                this._dataByPath.set(currentPath, aggregateData);
            }
        }
    }

    // Updates telemetry state for a particular uint-valued node. Will set the value directly
    // and then propagate upwards if changed. If the value did not already exist and this is the
    // first insertion, propagation will occur.
    updateUIntValue(path, value, timestamp = null) {
        if (!isInt(value)) {
            console.error(`[HierarchicalTelemetryStore] Value is not uint (${typeof value})`);
        }

        // Convert timestamp to Date if provided, otherwise use current time
        const timestampDate = timestamp ? new Date(timestamp) : new Date();

        // Update value and timestamp, preserving unit information
        const oldData = this._dataByPath.get(path);
        const forceUpdate = oldData === undefined;
        let changed = !oldData || value !== oldData.value || timestampDate.getTime() !== oldData.timestamp.getTime();
        const newData = {
            value,
            timestamp: timestampDate,
            unitDisplayLabel: oldData ? oldData.unitDisplayLabel : null,
            unitFullLabel: oldData ? oldData.unitFullLabel : null
        };
        this._dataByPath.set(path, newData);
        console.log(`Set ${path} = ${value} at ${timestampDate.toISOString()}`);

        // No change? We are done.
        if (!changed && !forceUpdate) {
            return;
        }

        // Split path into components. Then move upwards to propagate the boolean value and timestamp.
        const parts = path.split("/");
        for (let i = parts.length; i > 0; i--) {
            const currentPath = parts.slice(0, i).join("/");
            if (currentPath !== path) {
                // _getAggregateHealthAndTimestamp only supports bools at leaf level, so we don't want to
                // invoke it on our uint
                const aggregateData = this._getAggregateHealthAndTimestamp(currentPath);
                this._dataByPath.set(currentPath, aggregateData);
            }
        }
    }

    // Updates telemetry state for a particular double-valued node. Will set the value directly
    // and then propagate upwards if changed. If the value did not already exist and this is the
    // first insertion, propagation will occur.
    updateDoubleValue(path, value, timestamp = null) {
        if (typeof value !== "number") {
            console.error(`[HierarchicalTelemetryStore] Value is not double (${typeof value})`);
        }

        // Convert timestamp to Date if provided, otherwise use current time
        const timestampDate = timestamp ? new Date(timestamp) : new Date();

        // Update value and timestamp, preserving unit information
        const oldData = this._dataByPath.get(path);
        const forceUpdate = oldData === undefined;
        let changed = !oldData || value !== oldData.value || timestampDate.getTime() !== oldData.timestamp.getTime();
        const newData = {
            value,
            timestamp: timestampDate,
            unitDisplayLabel: oldData ? oldData.unitDisplayLabel : null,
            unitFullLabel: oldData ? oldData.unitFullLabel : null
        };
        this._dataByPath.set(path, newData);
        console.log(`Set ${path} = ${value} at ${timestampDate.toISOString()}`);

        // No change? We are done.
        if (!changed && !forceUpdate) {
            return;
        }

        // Split path into components. Then move upwards to propagate the boolean value and timestamp.
        const parts = path.split("/");
        for (let i = parts.length; i > 0; i--) {
            const currentPath = parts.slice(0, i).join("/");
            if (currentPath !== path) {
                // _getAggregateHealthAndTimestamp only supports bools at leaf level, so we don't want to
                // invoke it on our double
                const aggregateData = this._getAggregateHealthAndTimestamp(currentPath);
                this._dataByPath.set(currentPath, aggregateData);
            }
        }
    }

    // Updates telemetry state for a particular string-valued node. Will set the value directly
    // and then propagate upwards if changed. If the value did not already exist and this is the
    // first insertion, propagation will occur.
    updateStringValue(path, value, timestamp = null) {
        if (typeof value !== "string") {
            console.error(`[HierarchicalTelemetryStore] Value is not string (${typeof value})`);
        }

        // Convert timestamp to Date if provided, otherwise use current time
        const timestampDate = timestamp ? new Date(timestamp) : new Date();

        // Update value and timestamp, preserving unit information
        const oldData = this._dataByPath.get(path);
        const forceUpdate = oldData === undefined;
        let changed = !oldData || value !== oldData.value || timestampDate.getTime() !== oldData.timestamp.getTime();
        const newData = {
            value,
            timestamp: timestampDate,
            unitDisplayLabel: oldData ? oldData.unitDisplayLabel : null,
            unitFullLabel: oldData ? oldData.unitFullLabel : null
        };
        this._dataByPath.set(path, newData);
        console.log(`Set ${path} = ${value} at ${timestampDate.toISOString()}`);

        // No change? We are done.
        if (!changed && !forceUpdate) {
            return;
        }

        // Split path into components. Then move upwards to propagate the string value and timestamp.
        const parts = path.split("/");
        for (let i = parts.length; i > 0; i--) {
            const currentPath = parts.slice(0, i).join("/");
            if (currentPath !== path) {
                // _getAggregateHealthAndTimestamp only supports bools at leaf level, so we don't want to
                // invoke it on our string
                const aggregateData = this._getAggregateHealthAndTimestamp(currentPath);
                this._dataByPath.set(currentPath, aggregateData);
            }
        }
    }

    // Update function for unit label maps - iterates through the provided maps
    // and updates the internal ones (overwriting existing values and adding new ones)
    // Note: nlohmann::json serializes std::unordered_map<uint16_t, string> as array of pairs
    updateUnitLabelMaps(displayLabelMap, fullLabelMap) {
        if (displayLabelMap) {
            const displayMap = arrayOfPairsToMap(displayLabelMap);
            for (const [code, label] of displayMap) {
                this._unitDisplayLabelByCode.set(parseInt(code), label);
            }
        }

        if (fullLabelMap) {
            const fullMap = arrayOfPairsToMap(fullLabelMap);
            for (const [code, label] of fullMap) {
                this._unitFullLabelByCode.set(parseInt(code), label);
            }
        }

        console.log(`[HierarchicalTelemetryStore] Updated unit label maps`);
    }

    // Helper method to get unit labels for a given unit code
    _getUnitLabels(unitCode) {
        if (unitCode === null || unitCode === undefined) {
            return { unitDisplayLabel: null, unitFullLabel: null };
        }

        const displayLabel = this._unitDisplayLabelByCode.get(unitCode);
        const fullLabel = this._unitFullLabelByCode.get(unitCode);

        // Only include labels if they exist and are not empty/unknown
        const unitDisplayLabel = (displayLabel && displayLabel !== "" && displayLabel !== "<unknown>") ? displayLabel : null;
        const unitFullLabel = (fullLabel && fullLabel !== "" && fullLabel !== "<unknown>") ? fullLabel : null;

        return { unitDisplayLabel, unitFullLabel };
    }

    // Adds a new telemetry value to the store for the first time.
    addPath(path, initialValue, isBoolValue, timestamp = null, unitCode = null) {
        // Now update path component maps. Note that terminal part of path must have no children.
        const parts = path.split("/");
        let map = this._pathChildren;
        for (let i = 0; i < parts.length; i++) {
            const currentPart = parts[i];
            const reachedEnd = i == parts.length - 1;
            if (!map.has(currentPart)) {
                map.set(currentPart, reachedEnd ? null : new Map());
            }
            map = map.get(currentPart);
        }

        console.log(`[HierarchicalTelemetryStore] Added ${path} (value=${initialValue})`);

        // Get unit labels for this metric
        const { unitDisplayLabel, unitFullLabel } = this._getUnitLabels(unitCode);

        // Convert timestamp to Date if provided, otherwise use current time
        const timestampDate = timestamp ? new Date(timestamp) : new Date();

        // Store initial data with unit information
        const initialData = {
            value: isBoolValue ? (initialValue == true) : initialValue,
            timestamp: timestampDate,
            unitDisplayLabel,
            unitFullLabel
        };
        this._dataByPath.set(path, initialData);

        // Propagate changes upward (for bool metrics)
        if (isBoolValue) {
            const parts = path.split("/");
            for (let i = parts.length; i > 0; i--) {
                const currentPath = parts.slice(0, i).join("/");
                if (currentPath !== path) {
                    const aggregateData = this._getAggregateHealthAndTimestamp(currentPath);
                    this._dataByPath.set(currentPath, aggregateData);
                }
            }
        }
    }

    // Gets the names of immediate children, if any. For example, if the store contains a/b/c1 and
    // a/b/c2, getChildNames("a/b") will return [ "c1", "c2" ] and getChildNames("a") will return
    // [ "b" ].
    getChildNames(path) {
        // Special case: empty path, get first level
        if (path.length == 0) {
            return [ ...this._pathChildren.keys() ];
        }
        // Navigate through the hierarchy to find the correct map
        const parts = path.split("/");
        let currentMap = this._pathChildren;
        for (const part of parts) {
            if (!currentMap || !currentMap.has(part)) {
                return [];
            }
            currentMap = currentMap.get(part);
        }
        return currentMap ? [ ...currentMap.keys() ] : [];
    }

    // Gets telemetry data (value + timestamp + unit info) by path.
    // Returns { value: any, timestamp: Date, unitDisplayLabel: string|null, unitFullLabel: string|null } or null if not found.
    getData(path) {
        const data = this._dataByPath.get(path);
        if (!data) {
            return null;
        }

        console.log(`[HierarchicalTelemetryStore] Get data ${path} = ${data.value} at ${data.timestamp.toISOString()} (units: ${data.unitDisplayLabel}/${data.unitFullLabel})`);
        return data;
    }

    // Processes a telemetry snapshot in the new format (string-based paths)
    processSnapshot(snapshot) {
        let didUpdate = false;

        // Update unit label maps if present
        if (snapshot.metric_unit_display_label_by_code || snapshot.metric_unit_full_label_by_code) {
            this.updateUnitLabelMaps(
                snapshot.metric_unit_display_label_by_code,
                snapshot.metric_unit_full_label_by_code
            );
        }

        // Process bool metrics
        if (snapshot.bool_metrics) {
            for (const [path, value] of Object.entries(snapshot.bool_metrics)) {
                const timestamp = snapshot.bool_metric_timestamps ? snapshot.bool_metric_timestamps[path] : null;

                // Check if this is a new path
                if (!this._dataByPath.has(path)) {
                    this.addPath(path, value, true, timestamp);
                } else {
                    this.updateBoolValue(path, value, timestamp);
                }
                didUpdate = true;
            }
        }

        // Process uint metrics
        if (snapshot.uint_metrics) {
            for (const [path, value] of Object.entries(snapshot.uint_metrics)) {
                const timestamp = snapshot.uint_metric_timestamps ? snapshot.uint_metric_timestamps[path] : null;
                const unitCode = snapshot.uint_metric_units ? snapshot.uint_metric_units[path] : null;

                // Check if this is a new path
                if (!this._dataByPath.has(path)) {
                    this.addPath(path, value, false, timestamp, unitCode);
                } else {
                    this.updateUIntValue(path, value, timestamp);
                }
                didUpdate = true;
            }
        }

        // Process double metrics
        if (snapshot.double_metrics) {
            for (const [path, value] of Object.entries(snapshot.double_metrics)) {
                const timestamp = snapshot.double_metric_timestamps ? snapshot.double_metric_timestamps[path] : null;
                const unitCode = snapshot.double_metric_units ? snapshot.double_metric_units[path] : null;

                // Check if this is a new path
                if (!this._dataByPath.has(path)) {
                    this.addPath(path, value, false, timestamp, unitCode);
                } else {
                    this.updateDoubleValue(path, value, timestamp);
                }
                didUpdate = true;
            }
        }

        // Process string metrics
        if (snapshot.string_metrics) {
            for (const [path, value] of Object.entries(snapshot.string_metrics)) {
                const timestamp = snapshot.string_metric_timestamps ? snapshot.string_metric_timestamps[path] : null;
                const unitCode = snapshot.string_metric_units ? snapshot.string_metric_units[path] : null;

                // Check if this is a new path
                if (!this._dataByPath.has(path)) {
                    this.addPath(path, value, false, timestamp, unitCode);
                } else {
                    this.updateStringValue(path, value, timestamp);
                }
                didUpdate = true;
            }
        }

        return didUpdate;
    }
}

// Export the class as default for easier importing
export default HierarchicalTelemetryStore;
