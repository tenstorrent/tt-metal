/*
 * HierarchicalTelemetryStore.js
 *
 * Stores telemetry data, named by underscore-delimited paths, in a hierarchical tree. Leaf nodes
 * contain actual reported telemetry data and intermediate nodes are aggregated from children.
 */

export class HierarchicalTelemetryStore {
    constructor() {
        // Full path by ID. Telemetry updates are transmitted by ID number in order to keep
        // messages compact. This allows us to map back to path.
        this._pathById = new Map();

        // Telemetry data (state) is hashed by path. Partial paths are also tracked and are the
        // aggregate state of all children (ANDed together). That is, "true" is good and if any
        // descendant is false (bad), all ancestors up the tree will be bad. Each time an actual
        // telemetry point (i.e., a leaf node) changes, the aggregate state is propagated upwards.
        // So e.g. if "host_data1" and "host_data2" are telemetry points, their state will be
        // stored directly while "host" will also be present in the map and computed from both.
        this._stateByPath = new Map();

        // Hierarchical map of paths, allowing us to navigate to subsequently deeper levels. Given
        // a path like foo_bar_baz, the first level of keys will include "foo", which will index a
        // map containing "bar", which will in turn index a map containing "baz". "baz" has no 
        // children and its value is nil.
        this._pathChildren = new Map();    // hierarchical map of paths
    }

    // Returns the aggregate state of a path by looking at immediate children, otherwise looks at
    // the node directly, assuming it is a leaf. If it cannot be found, false is returned.
    _getAggregateState(path) {
        // First, we need to navigate to the correct position in the hierarchy
        const parts = path.split("_");
        let currentMap = this._pathChildren;
        
        // Navigate through the hierarchy to find the correct map
        for (const part of parts) {
            if (!currentMap || !currentMap.has(part)) {
                // Path doesn't exist in hierarchy, must be leaf node
                return this._stateByPath.get(path) === true ? true : false;
            }
            currentMap = currentMap.get(part);
        }
        
        // If currentMap is null, this is a leaf node
        if (!currentMap) {
            return this._stateByPath.get(path) === true ? true : false;
        }
        
        // Aggregate children states
        let state = true;
        for (const nextPathComponent of currentMap.keys()) {
            const childPath = path + "_" + nextPathComponent;
            const childState = this._stateByPath.get(childPath) === true ? true : false;
            state &= childState;
        }
        return state;
    }

    // Updates telemetry state for a particular node. Will set the state directly and then
    // propagate upwards if changed. If the state did not already exist and this is the first
    // insertion, propagation will occur.
    updateState(id, state) {
        const path = this._pathById.get(id);
        if (!path) {
            // Invalid telemetry data, does not map to any known path
            console.log(`Invalid id ${id}, cannot update state`);
            return;
        }

        // Convert state to bool if it is not
        state = state == true;
        
        // Update state
        const oldState = this._stateByPath.get(path);
        const forceUpdate = oldState === undefined;
        let changed = state != oldState;
        this._stateByPath.set(path, state);
        console.log(`Set ${path} = ${state}`);

        // No change? We are done.
        if (!changed && !forceUpdate) {
            return;
        }

        // Split path into components. Then move upwards to propagate the state.
        const parts = path.split("_");
        for (let i = parts.length; i > 0; i--) {
            const currentPath = parts.slice(0, i).join("_");
            this._stateByPath.set(currentPath, this._getAggregateState(currentPath));
        }
    }

    // Adds a new telemetry value to the store for the first time.
    addPath(path, id, initialState) {
        if (this._pathById.has(id)) {
            const existingPath = this._pathById.get(id);
            console.error(`Cannot add (${id}, ${path}) to id -> path mapping because (${id}, ${existingPath}) already exists there`);
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

        console.log(`[HierarchicalTelemetryStore] Added ${id}:${path} (state=${initialState})`);

        // Update state
        this.updateState(id, initialState);
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

    // Gets telemetry state for a particular value. The path (a string) or ID (a number) may be
    // supplied.
    getState(idOrPath) {
        const isPath = typeof(idOrPath) === "string" || idOrPath instanceof String;
        const path = isPath ? idOrPath : this._pathById.get(idOrPath);
        if (!path) {
            // Invalid telemetry data, does not map to any known path
            console.error(`Unknown id ${idOrPath}`);
            return false;
        }
        console.log(`Get state ${path} = ${this._stateByPath.get(path)}`);
        return this._stateByPath.get(path);
    }
}

// Export the class as default for easier importing
export default HierarchicalTelemetryStore;
