// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * HierarchicalTelemetryStore.js
 *
 * Stores telemetry data, named by slash-delimited paths, in a hierarchical tree. Leaf nodes
 * contain actual reported telemetry data and intermediate nodes are aggregated from children.
 *
 * TODO:
 * -----
 * - Better handling of value types. The isBoolValue param on addPath() needs to go!
 * - Better detection of health vs. non-health metrics when rendering (don't just check value
 *   type, need some attribute along with them).
 */

export function isBool(value) {
    return typeof value === "boolean" || value instanceof Boolean;
}

function isInt(value) {
    return Number.isInteger(value);
}

export class HierarchicalTelemetryStore {
    constructor() {
        // Full path by ID. Telemetry updates are transmitted by ID number in order to keep
        // messages compact. This allows us to map back to path.
        this._pathById = new Map();

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
    updateBoolValue(id, value, timestamp = null) {
        const isBoolean = isBool(value) || value === 1 || value === 0;
        if (!isBoolean) {
            console.error(`[HierarchicalTelemetryStore] Value is not bool (${typeof value})`);
        }

        const path = this._pathById.get(id);
        if (!path) {
            // Invalid telemetry data, does not map to any known path
            console.error(`[HierarchicalTelemetryStore] Invalid id ${id}, cannot update bool value`);
            return;
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
    updateUIntValue(id, value, timestamp = null) {
        if (!isInt(value)) {
            console.error(`[HierarchicalTelemetryStore] Value is not uint (${typeof value})`);
        }

        const path = this._pathById.get(id);
        if (!path) {
            // Invalid telemetry data, does not map to any known path
            console.error(`[HierarchicalTelemetryStore] Invalid id ${id}, cannot update uint value`);
            return;
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
    updateDoubleValue(id, value, timestamp = null) {
        if (typeof value !== "number") {
            console.error(`[HierarchicalTelemetryStore] Value is not double (${typeof value})`);
        }

        const path = this._pathById.get(id);
        if (!path) {
            // Invalid telemetry data, does not map to any known path
            console.error(`[HierarchicalTelemetryStore] Invalid id ${id}, cannot update double value`);
            return;
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

    // Update function for unit label maps - iterates through the provided maps
    // and updates the internal ones (overwriting existing values and adding new ones)
    updateUnitLabelMaps(displayLabelMap, fullLabelMap) {
        let displayCount = 0;
        let fullCount = 0;

        if (displayLabelMap) {
            for (const [code, label] of Object.entries(displayLabelMap)) {
                this._unitDisplayLabelByCode.set(parseInt(code), label);
                displayCount++;
            }
        }

        if (fullLabelMap) {
            for (const [code, label] of Object.entries(fullLabelMap)) {
                this._unitFullLabelByCode.set(parseInt(code), label);
                fullCount++;
            }
        }

        console.log(`[HierarchicalTelemetryStore] Updated ${displayCount} display labels and ${fullCount} full labels`);
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
    addPath(path, id, initialValue, isBoolValue, timestamp = null, unitCode = null) {
        if (this._pathById.has(id)) {
            const existingPath = this._pathById.get(id);
            console.error(`[HierarchicalTelemetryStore] Cannot add (${id}, ${path}) to id -> path mapping because (${id}, ${existingPath}) already exists there`);
            return;
        }

        this._pathById.set(id, path);

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

        console.log(`[HierarchicalTelemetryStore] Added ${id}:${path} (value=${initialValue})`);

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

    // Gets telemetry data (value + timestamp + unit info). The path (a string) or ID (a number) may be supplied.
    // Returns { value: any, timestamp: Date, unitDisplayLabel: string|null, unitFullLabel: string|null } or null if not found.
    getData(idOrPath) {
        const isPath = typeof(idOrPath) === "string" || idOrPath instanceof String;
        const path = isPath ? idOrPath : this._pathById.get(idOrPath);
        if (!path) {
            // Invalid telemetry data, does not map to any known path
            console.error(`[HierarchicalTelemetryStore] Unknown id ${idOrPath}`);
            return null;
        }
        const data = this._dataByPath.get(path);
        if (!data) {
            return null;
        }

        console.log(`[HierarchicalTelemetryStore] Get data ${path} = ${data.value} at ${data.timestamp.toISOString()} (units: ${data.unitDisplayLabel}/${data.unitFullLabel})`);
        return data;
    }
}

// Export the class as default for easier importing
export default HierarchicalTelemetryStore;
