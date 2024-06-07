import calculateOpPerformanceColor from '../functions/calculateOpPerformanceColor';
// TODO: look at making this import path nicer (alias throw a linting error in .tsx files)
import '../scss/components/GoldenTensorComparisonIndicator.scss';

const GOLDEN_GLOBAL = [1, 0.9999, 0.9888, 1, 1, 0.8888];
const GOLDEN_LOCAL = [1, 0.9999, 1, 0.9888, 1, 0.99];

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
