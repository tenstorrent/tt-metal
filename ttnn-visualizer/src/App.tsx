import React, { useEffect, useState } from 'react';
import { FocusStyleManager } from '@blueprintjs/core';
import axios from 'axios';
import ApplicationList from './components/ApplicationList';
import TenstorrentLogo from './components/TenstorrentLogo';

function App() {
    FocusStyleManager.onlyShowFocusOnTabs();

    const [message, setMessage] = useState('');
    const [operations, setOperations] = useState('');

    useEffect(() => {
        axios
            .get('http://localhost:8000/api')
            .then((response) => {
                setMessage(response.data.message);
            })
            .catch((error) => {
                console.error('There was an error!', error);
            });
    }, []);
    useEffect(() => {
        const fetchOperations = async () => {
            console.log('fetching operations');
            try {
                const response = await axios.get('/api/get-operations');
                console.log(response);
                setOperations(response.data);
            } catch (error) {
                console.error('Error fetching operations:', error);
            }
        };

        fetchOperations();
    }, []);

    return (
        <>
            {message && <h1>{message}</h1>}
            <header className='app-header'>
                <TenstorrentLogo />
            </header>
            <ApplicationList />;
        </>
    );
}

export default App;
