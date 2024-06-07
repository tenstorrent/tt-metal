import './styles.scss';

const GOLDEN_GLOBAL = [1, 0.9999, 0.9888, 1, 1, 0.8888];
const GOLDEN_LOCAL = [1, 0.9999, 1, 0.9888, 1, 0.99];

type CalculationOptions = 'global' | 'local';

interface GoldenTensorComparisonIndicatorProps {
    index: number;
    calc: CalculationOptions;
}

function GoldenTensorComparisonIndicator({ index, calc }: GoldenTensorComparisonIndicatorProps) {
    const source = calc === 'local' ? GOLDEN_LOCAL : GOLDEN_GLOBAL;

    return (
        <>
            <div
                className='performance-square'
                style={{
                    backgroundColor: calculateOpPerformanceColor(source[index]),
                }}
            />
            <span>t{source[index].toFixed(4)}</span>
        </>
    );
}

const calculateOpPerformanceColor = (value: number): string => {
    const min = 0.8;
    const ratio = (value - min) / (1 - min);
    const intensity = Math.round(ratio * 255);
    console.log(value, ratio, intensity);

    return `rgb(${255 - intensity}, ${intensity}, 0)`;
};

export default GoldenTensorComparisonIndicator;
