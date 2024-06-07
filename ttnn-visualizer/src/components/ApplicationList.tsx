// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
import { useState } from 'react';

import { Button, Icon } from '@blueprintjs/core';
import { IconNames } from '@blueprintjs/icons';
import { Tooltip2 } from '@blueprintjs/popover2';
import SearchField from './SearchField';
import FilterableComponent from './FilterableComponent';
import Collapsible from './Collapsible';
import HighlightedText from './HighlightedText';
import GoldenTensorComparisonIndicator from './GoldenTensorComparisonIndicator';

const ApplicationList = () => {
    const ops = [
        { name: 'ttnn.from_torch', goldenGlobal: 1, goldenLocal: 1 },
        { name: 'ttnn.from_torch', goldenGlobal: 0.9999, goldenLocal: 0.9999 },
        { name: 'ttnn.add', goldenGlobal: 0.9888, goldenLocal: 1 },

        { name: 'ttnn.deallocate.buffer.opname', goldenGlobal: 1, goldenLocal: 0.988 },
        { name: 'ttnn.to_torch', goldenGlobal: 1, goldenLocal: 1 },
        { name: 'ttnn.compare', goldenGlobal: 0.8888, goldenLocal: 0.99 },
    ];

    const [filterQuery, setFilterQuery] = useState('');

    return (
        <div className='app'>
            <fieldset className='operations-wrap'>
                <legend>Operations</legend>

                <div className='ops'>
                    <SearchField
                        placeholder='Filter operations'
                        searchQuery={filterQuery}
                        onQueryChanged={setFilterQuery}
                        controls={
                            [
                                // <Tooltip2
                                //     content='Select all filtered operations'
                                //     position={PopoverPosition.RIGHT}
                                //     key='select-all-ops'
                                // >
                                //     <Button icon={IconNames.CUBE_ADD}/>
                                // </Tooltip2>,
                                // <Tooltip2
                                //     content='Deselect all filtered operations'
                                //     position={PopoverPosition.RIGHT}
                                //     key='deselect-all-ops'
                                // >
                                //     <Button
                                //         icon={IconNames.CUBE_REMOVE}
                                //
                                //     />
                                // </Tooltip2>,
                            ]
                        }
                    />
                    {ops.map((op, index: number) => {
                        const hasContent =
                            op.name.toLowerCase().includes('ttnn.add') ||
                            op.name.toLowerCase().includes('ttnn.to_torch');
                        return (
                            <FilterableComponent
                                key={index}
                                filterableString={op.name}
                                filterQuery={filterQuery}
                                component={
                                    <div className='op'>
                                        <Collapsible
                                            label={
                                                <>
                                                    <Icon size={20} icon={IconNames.CUBE} />
                                                    <span style={{ color: '#fff', fontSize: '20px' }}>
                                                        <HighlightedText text={op.name} filter={filterQuery} />
                                                    </span>
                                                    <Tooltip2 content='Operation tensor report'>
                                                        <Button minimal small icon={IconNames.GRAPH} />
                                                    </Tooltip2>
                                                    <Tooltip2 content='Stack trace'>
                                                        <Button minimal small icon={IconNames.CODE} />
                                                    </Tooltip2>
                                                    <Tooltip2 content='Buffer view'>
                                                        <Button minimal small icon={IconNames.SEGMENTED_CONTROL} />
                                                    </Tooltip2>
                                                    <GoldenTensorComparisonIndicator value={op.goldenGlobal} />
                                                    <GoldenTensorComparisonIndicator value={op.goldenLocal} />
                                                </>
                                            }
                                            isOpen={false}
                                        >
                                            {hasContent && (
                                                <div className='collapsible-content'>
                                                    <ul className='op-params'>
                                                        <li>
                                                            <strong>Shape: </strong>ttnn.Shape([1 [32], 64])
                                                        </li>
                                                        <li>
                                                            <strong>Dtype: </strong>DataType.BFLOAT16
                                                        </li>
                                                        <li>
                                                            <strong>Layout: </strong>Layout.TILE
                                                        </li>
                                                        <li>
                                                            <strong>Device: </strong>0
                                                        </li>
                                                        <li>
                                                            <strong>Memory: </strong>tt::tt_metal::MemoryConfig(
                                                            memory_layout=TensorMemoryLayout:: INTERLEAVED,
                                                            <br /> buffer_type=BufferpType::DRAM, shard_spec=std:
                                                            :nullopt)
                                                        </li>
                                                    </ul>
                                                </div>
                                            )}
                                        </Collapsible>
                                    </div>
                                }
                            />
                        );
                    })}
                </div>
            </fieldset>

            {/* <h2>Buffer report</h2> */}
            {/* <div className={'buffer-report'}> */}
            {/*    <h3>ttnn.add</h3> */}
            {/* </div> */}
        </div>
    );
};

export default ApplicationList;
