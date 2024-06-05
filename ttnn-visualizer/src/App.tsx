import React from 'react';
import './App.scss';
import { FocusStyleManager } from "@blueprintjs/core";
import ApplicationList from "./components/ApplicationList.tsx";

function App() {
    FocusStyleManager.onlyShowFocusOnTabs();
    return (
        <>
            <ApplicationList/>
        </>
    )
}

export default App;
