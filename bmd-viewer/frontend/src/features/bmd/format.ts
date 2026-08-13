// Shared formatting helpers for cohort-percentile displays (anomaly badge, BHI chip).

// English ordinal suffix: 1st, 2nd, 3rd, 4th, 11th-13th, 21st, 22nd, ... 100th.
export function ordinalPct(value: number): string {
  const n = Math.round(value);
  const mod100 = n % 100;
  if (mod100 >= 11 && mod100 <= 13) return `${n}th pct`;
  switch (n % 10) {
    case 1:
      return `${n}st pct`;
    case 2:
      return `${n}nd pct`;
    case 3:
      return `${n}rd pct`;
    default:
      return `${n}th pct`;
  }
}
