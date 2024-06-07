import { FocusStyleManager } from '@blueprintjs/core';
import ApplicationList from './components/ApplicationList';
import TenstorrentLogo from './components/TenstorrentLogo';

function App() {
    FocusStyleManager.onlyShowFocusOnTabs();

    return (
        <>
            <header className='app-header'>
                <TenstorrentLogo />
            </header>
            <ApplicationList />;
        </>
    );
}

export default App;
