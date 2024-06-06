import React, { useState } from 'react';

import { Button, Icon } from '@blueprintjs/core';
import { IconNames } from '@blueprintjs/icons';
import { Tooltip2 } from '@blueprintjs/popover2';
import TenstorrentLogo from './TenstorrentLogo.tsx';
import SearchField from './SearchField.tsx';
import FilterableComponent from './FilterableComponent.tsx';
import Collapsible from './Collapsible.tsx';
import HighlightedText from './HighlightedText.tsx';

const ApplicationList = () => {
    const ops = [
        'ttnn.from_torch',
        'ttnn.from_torch',
        'ttnn.add',

        'ttnn.deallocate.buffer.opname',
        'ttnn.to_torch',
        'ttnn.compare',
    ];

    const goldenGlobal = [1, 0.9999, 0.9888, 1, 1, 0.8888];
    const goldenLocal = [1, 0.9999, 1, 0.9888, 1, 0.99];

    const calculateOpPerformanceColor = (value: number): string => {
        const min = 0.8;
        const ratio = (value - min) / (1 - min);
        const intensity = Math.round(ratio * 255);
        console.log(value, ratio, intensity);

        return `rgb(${255 - intensity}, ${intensity}, 0)`;
    };

    const [filterQuery, setFilterQuery] = useState('');

    return (
        <div className='app'>
            <TenstorrentLogo style={{ display: 'block', position: 'relative', marginBottom: '30px' }} />

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
                            op.toLowerCase().includes('ttnn.add') || op.toLowerCase().includes('ttnn.to_torch');
                        return (
                            <FilterableComponent
                                key={index}
                                filterableString={op}
                                filterQuery={filterQuery}
                                component={
                                    <div className='op'>
                                        <Collapsible
                                            label={
                                                <>
                                                    {' '}
                                                    <Icon size={20} icon={IconNames.CUBE} />
                                                    <span style={{ color: '#fff', fontSize: '20px' }}>
                                                        <HighlightedText text={op} filter={filterQuery} />
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
                                                    <div
                                                        style={{
                                                            width: '10px',
                                                            height: '10px',
                                                            margin: '0 -7px 0 0',
                                                            // padding: '0 5px',
                                                            backgroundColor: calculateOpPerformanceColor(
                                                                goldenGlobal[index],
                                                            ),
                                                        }}
                                                    />
                                                    <span>{goldenGlobal[index].toFixed(4)}</span>
                                                    <div
                                                        style={{
                                                            width: '10px',
                                                            height: '10px',
                                                            margin: '0 -7px 0 0',
                                                            backgroundColor: calculateOpPerformanceColor(
                                                                goldenLocal[index],
                                                            ),
                                                        }}
                                                    />
                                                    <span>{goldenLocal[index].toFixed(4)}</span>
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
