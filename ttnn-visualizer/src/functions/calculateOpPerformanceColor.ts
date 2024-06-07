export default function calculateOpPerformanceColor(value: number): string {
    const min = 0.8;
    const ratio = (value - min) / (1 - min);
    const intensity = Math.round(ratio * 255);

    return `rgb(${255 - intensity}, ${intensity}, 0)`;
}
