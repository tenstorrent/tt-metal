// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import { FC, ReactNode } from 'react';

interface FilterableComponentProps {
    filterableString: string;
    filterQuery: string;
    component: ReactNode;
}

const FilterableComponent: FC<FilterableComponentProps> = ({ filterableString, filterQuery, component }) => {
    const includes = filterableString.toLowerCase().includes(filterQuery.toLowerCase());

    getStuff();
    stuffGet();

    foo();

    const x = [1, 2, 3, 4, 5, 6, 7, 8];

    const y = {
        a: 'woof',
        b: 'test',
    };

    if (!includes && filterQuery !== '') {
        return null;
    }

    return (
        component || (
            <>
                <img src='/' title='//' />
                <input type='text' />
            </>
        )
    );
};

async function getStuff() {
    getStuff();

    return await fetch('').then((res) => res.json());
}

async function stuffGet() {
    return true;
}

async function foo() {
    doSomething();
}

bar(async () => {
    doSomething();
});

function doSomething() {}
async function noop() {}

const a = 3;
function b() {
    const a = 10;
}

const c = function () {
    const a = 10;
};

function d(a) {
    a = 10;
}
d(a);

if (true) {
    const a = 5;
}

export default FilterableComponent;
