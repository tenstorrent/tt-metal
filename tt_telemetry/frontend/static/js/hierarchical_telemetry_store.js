class TelemetryPoint {
    constructor() {
        this.state = false;
    }
}

export class HierarchicalTelemetryStore {
    constructor() {
        this._path_by_id = new Map();       // full path by id
        this._state_by_path = new Map();    // state by path (including partial paths: part1, part1_part2, part1_part2_part3, etc.)
        this._path_children = new Map();    // hierarchical map of paths
    }

    _getAggregateState(path) {
        // First, we need to navigate to the correct position in the hierarchy
        const parts = path.split("_");
        let current_map = this._path_children;
        
        // Navigate through the hierarchy to find the correct map
        for (const part of parts) {
            if (!current_map || !current_map.has(part)) {
                // Path doesn't exist in hierarchy, must be leaf node
                return this._state_by_path.get(path) == true ? true : false;
            }
            current_map = current_map.get(part);
        }
        
        // If current_map is null, this is a leaf node
        if (!current_map) {
            return this._state_by_path.get(path) == true ? true : false;
        }
        
        // Aggregate children states
        let state = true;
        for (const next_path_component of current_map.keys()) {
            const child_path = path + "_" + next_path_component;
            const child_state = this._state_by_path.get(child_path) == true ? true : false;
            state &= child_state;
        }
        return state;
    }

    updateState(id, state, forceUpdate) {
        const path = this._path_by_id.get(id);
        if (!path) {
            // Invalid telemetry data, does not map to any known path
            return;
        }
        
        // Update state
        const old_state = this._state_by_path.get(path) == true ? true : false;
        let changed = state != old_state;
        this._state_by_path.set(path, state);

        // No change? We are done.
        if (!changed && !forceUpdate) {
            return;
        }

        // Split path into components. Then move upwards to propagate the state.
        const parts = path.split("_");
        for (let i = parts.length; i > 0; i--) {
            const current_path = parts.slice(0, i).join("_");
            this._state_by_path.set(current_path, this._getAggregateState(current_path));
        }
    }

    addPath(path, id, initialState) {
        if (this._path_by_id.has(id)) {
            const existing_path = this._path_by_id.get(id);
            console.error(`cannot add (${id}, ${path}) to id -> path mapping because (${id}, ${existing_path}) already exists there`);
            return;
        }

        this._path_by_id.set(id, path);
    
        // Now update path component maps. Note that terminal part of path must have no children.
        const parts = path.split("_");
        let map = this._path_children;
        for (let i = 0; i < parts.length; i++) {
            const current_part = parts[i];
            const reached_end = i == parts.length - 1;
            if (!map.has(current_part)) {
                map.set(current_part, reached_end ? null : new Map());
            }
            map = map.get(current_part);
        }

        // Update state
        this.updateState(id, initialState, true);
    }

    getState(id_or_path) {
        const is_path = typeof(id_or_path) === "string" || id_or_path instanceof String;
        const path = is_path ? id_or_path : this._path_by_id.get(id_or_path);
        if (!path) {
            // Invalid telemetry data, does not map to any known path
            console.error(`Unknown id ${id_or_path}`);
            return false;
        }
        return this._state_by_path.get(path) == true ? true : false;
    }
}

// Export the class as default for easier importing
export default HierarchicalTelemetryStore;
