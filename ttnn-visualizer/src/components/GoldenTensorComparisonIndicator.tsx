import calculateOpPerformanceColor from '../functions/calculateOpPerformanceColor';
// TODO: look at making this import path nicer (alias throw a linting error in .tsx files)
import '../scss/components/GoldenTensorComparisonIndicator.scss';

interface GoldenTensorComparisonIndicatorProps {
    value: number;
}

function GoldenTensorComparisonIndicator({ value }: GoldenTensorComparisonIndicatorProps) {
    return (
        <>
            <div
                className='golden-tensor-comparison-square'
                style={{
                    backgroundColor: calculateOpPerformanceColor(value),
                }}
            />
            <span>t{value.toFixed(4)}</span>
        </>
    );
}

export default GoldenTensorComparisonIndicator;
