/*
 * HierarchicalTelemetryStore.js
 *
 * Stores telemetry data, named by underscore-delimited paths, in a hierarchical tree. Leaf nodes
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

        // Boolean telemetry values are hashed by path. Partial paths are also tracked and are the
        // aggregate value of all children (ANDed together). That is, "true" is good and if any
        // descendant is false (bad), all ancestors up the tree will be bad. Each time an actual
        // telemetry point (i.e., a leaf node) changes, the aggregate value is propagated upwards.
        // So e.g. if "host_data1" and "host_data2" are telemetry points, their value will be
        // stored directly while "host" will also be present in the map and computed from both.
        this._valueByPath = new Map();

        // Hierarchical map of paths, allowing us to navigate to subsequently deeper levels. Given
        // a path like foo_bar_baz, the first level of keys will include "foo", which will index a
        // map containing "bar", which will in turn index a map containing "baz". "baz" has no 
        // children and its value is nil.
        this._pathChildren = new Map();    // hierarchical map of paths
    }

    // Returns the aggregate "health" value of a path by looking at immediate bool-valued children,
    // otherwise looks at the node directly, assuming it is a leaf. If it cannot be found, false is
    // returned. Non-boolean children are ignored in aggregation.
    _getAggregateHealth(path) {
        // First, we need to navigate to the correct position in the hierarchy
        const parts = path.split("_");
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
            const value = this._valueByPath.get(path);
            if (!isBool(value)) {
                console.error(`[HierarchicalTelemetryStore] Value of ${path} was expected to be bool but is ${typeof value}`);
            }
            return value === true ? true : false;
        }
        
        // Aggregate childrens' boolean values. For now, we only look at children who are boolean.
        // Eventually, scalar values and others can be associated with conditions for "good" vs.
        // "bad" (e.g., numeric thresholds, etc.). Inner (non-leaf) nodes should always be boolean
        // to represent aggregate health of their descendants.
        let value = true;
        for (const nextPathComponent of currentMap.keys()) {
            const childPath = path + "_" + nextPathComponent;
            let childValue = this._valueByPath.get(childPath);
            if (isBool(childValue)) {
                // Only bools are considered for aggregate health
                value = value && childValue;
            }
        }
        return value;
    }

    // Updates telemetry state for a particular boolean-valued node. Will set the value directly
    // and then propagate upwards if changed. If the value did not already exist and this is the
    // first insertion, propagation will occur.
    updateBoolValue(id, value) {
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
        
        // Update value
        const oldValue = this._valueByPath.get(path);
        const forceUpdate = oldValue === undefined;
        let changed = value != oldValue;
        this._valueByPath.set(path, value);
        console.log(`Set ${path} = ${value}`);

        // No change? We are done.
        if (!changed && !forceUpdate) {
            return;
        }

        // Split path into components. Then move upwards to propagate the boolean value.
        const parts = path.split("_");
        for (let i = parts.length; i > 0; i--) {
            const currentPath = parts.slice(0, i).join("_");
            this._valueByPath.set(currentPath, this._getAggregateHealth(currentPath));
        }
    }

    // Updates telemetry state for a particular uint-valued node. Will set the value directly
    // and then propagate upwards if changed. If the value did not already exist and this is the
    // first insertion, propagation will occur.
    updateUIntValue(id, value) {
        if (!isInt(value)) {
            console.error(`[HierarchicalTelemetryStore] Value is not uint (${typeof value})`);
        }

        const path = this._pathById.get(id);
        if (!path) {
            // Invalid telemetry data, does not map to any known path
            console.error(`[HierarchicalTelemetryStore] Invalid id ${id}, cannot update uint value`);
            return;
        }

        // Update value
        const oldValue = this._valueByPath.get(path);
        const forceUpdate = oldValue === undefined;
        let changed = value != oldValue;
        this._valueByPath.set(path, value);
        console.log(`Set ${path} = ${value}`);

        // No change? We are done.
        if (!changed && !forceUpdate) {
            return;
        }

        // Split path into components. Then move upwards to propagate the boolean value.
        const parts = path.split("_");
        for (let i = parts.length; i > 0; i--) {
            const currentPath = parts.slice(0, i).join("_");
            if (currentPath != path) {
                // _getAggregateHealth only supports bools at leaf level, so we don't want to
                // invoke it on our uint
                this._valueByPath.set(currentPath, this._getAggregateHealth(currentPath));
            }
        }
    }

    // Adds a new telemetry value to the store for the first time.
    addPath(path, id, initialValue, isBoolValue) {
        if (this._pathById.has(id)) {
            const existingPath = this._pathById.get(id);
            console.error(`[HierarchicalTelemetryStore] Cannot add (${id}, ${path}) to id -> path mapping because (${id}, ${existingPath}) already exists there`);
            return;
        }

        this._pathById.set(id, path);
    
        // Now update path component maps. Note that terminal part of path must have no children.
        const parts = path.split("_");
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

        // Update value
        if (isBoolValue) {
            this.updateBoolValue(id, initialValue);
        } else {
            this.updateUIntValue(id, initialValue);
        }
    }

    // Gets the names of immediate children, if any. For example, if the store contains a_b_c1 and
    // a_b_c2, getChildNames("a_b") will return [ "c1", "c2" ] and getChildNames("a") will return
    // [ "b" ].
    getChildNames(path) {
        // Special case: empty path, get first level
        if (path.length == 0) {
            return [ ...this._pathChildren.keys() ];
        }
        // Navigate through the hierarchy to find the correct map
        const parts = path.split("_");
        let currentMap = this._pathChildren;
        for (const part of parts) {
            if (!currentMap || !currentMap.has(part)) {
                return [];
            }
            currentMap = currentMap.get(part);
        }
        return currentMap ? [ ...currentMap.keys() ] : [];
    }

    // Gets telemetry value. The path (a string) or ID (a number) may be supplied.
    getValue(idOrPath) {
        const isPath = typeof(idOrPath) === "string" || idOrPath instanceof String;
        const path = isPath ? idOrPath : this._pathById.get(idOrPath);
        if (!path) {
            // Invalid telemetry data, does not map to any known path
            console.error(`[HierarchicalTelemetryStore] Unknown id ${idOrPath}`);
            return false;
        }
        console.log(`[HierarchicalTelemetryStore] Get value ${path} = ${this._valueByPath.get(path)}`);
        return this._valueByPath.get(path);
    }
}

// Export the class as default for easier importing
export default HierarchicalTelemetryStore;
