// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import React, { FC } from 'react';

interface HighlightedTextProps {
    text: string;
    filter: string;
}

const HighlightedText: FC<HighlightedTextProps> = ({ text, filter }) => {
    const index = text.toLowerCase().indexOf(filter.toLowerCase());

    if (index === -1) {
        return (
            <span title={text} className='highlighted-text'>
                {text}
            </span>
        );
    }

    const before = text.substring(0, index);
    const match = text.substring(index, index + filter.length);
    const after = text.substring(index + filter.length);

    return (
        <span
            title={text}
            className='highlighted-text'
            dangerouslySetInnerHTML={{ __html: `${before}<mark>${match}</mark>${after}` }}
        />
    );
};

export default HighlightedText;
