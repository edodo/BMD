// BMD 추세 차트: 누적 데이터로 골밀도 변화량 표시.
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useBmdTrend } from "@/features/patients/api/queries";

export function BmdTrendChart({ patientId }: { patientId: string }) {
  const { data } = useBmdTrend(patientId);
  if (!data || data.points.length === 0) {
    return <div className="trend empty">No measurement data yet.</div>;
  }

  const chartData = data.points.map((p) => ({
    date: new Date(p.measured_at).toLocaleDateString("en-CA"),
    bmd: p.bmd_value,
    t: p.t_score,
  }));

  return (
    <div className="trend">
      <div className="trend-header">
        <h3>{data.target_vertebra} Bone Density Trend</h3>
        {data.delta_percent != null && (
          <span
            className={`delta ${data.delta_percent >= 0 ? "up" : "down"}`}
          >
            {data.delta_percent >= 0 ? "▲" : "▼"}{" "}
            {Math.abs(data.delta_percent).toFixed(1)}%
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="date" fontSize={12} />
          <YAxis domain={["auto", "auto"]} fontSize={12} />
          <Tooltip />
          {/* 골다공증 임계 참고선 (T-score -2.5 상당, 예시) */}
          <ReferenceLine y={0.7} stroke="#dc2626" strokeDasharray="4 4" />
          <Line
            type="monotone"
            dataKey="bmd"
            stroke="#2563eb"
            strokeWidth={2}
            dot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
