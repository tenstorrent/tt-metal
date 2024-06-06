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

    if (!includes && filterQuery !== '') {
        return null;
    }

    return component;
};

export default FilterableComponent;
