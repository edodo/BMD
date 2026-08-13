// BMD 추세 차트: 누적 데이터로 골밀도 변화량 표시 + 급격한 골손실 임상 경고.
import {
  CartesianGrid,
  Label,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  useBmdTrend,
  useCalibratePrecision,
} from "@/features/patients/api/queries";
import { config } from "@/lib/config";

export function BmdTrendChart({ patientId }: { patientId: string }) {
  const { data } = useBmdTrend(patientId);
  const calibrate = useCalibratePrecision();
  if (!data || data.points.length === 0) {
    return <div className="trend empty">No measurement data yet.</div>;
  }

  const chartData = data.points.map((p) => ({
    date: new Date(p.measured_at).toLocaleDateString("en-CA"),
    bmd: p.bmd_value,
    t: p.t_score,
  }));

  // 직전 검사 대비 마지막 검사의 급격한 하락 감지 (단기 골손실 경고).
  // 임계값은 실측 LSC%(재위치 노이즈의 95% 신뢰한계)가 있으면 그것을,
  // 아직 보정하지 않았으면 미검증 기본값(config.significantLossPct)을 쓴다.
  const calibrated = data.lsc_percent != null;
  const threshold = data.lsc_percent ?? config.significantLossPct;
  const n = chartData.length;
  const last = chartData[n - 1];
  const prev = n >= 2 ? chartData[n - 2] : null;
  const stepDropPct =
    prev && prev.bmd > 0 ? ((prev.bmd - last.bmd) / prev.bmd) * 100 : 0;
  const alert = stepDropPct >= threshold;

  return (
    <div className={`trend${alert ? " alert" : ""}`}>
      <div className="trend-header">
        <h3>{data.target_vertebra} Bone Density Trend</h3>
        {data.delta_percent != null && (
          <span
            className={`delta ${
              alert ? "danger" : data.delta_percent >= 0 ? "up" : "down"
            }`}
          >
            {data.delta_percent >= 0 ? "▲" : "▼"}{" "}
            {Math.abs(data.delta_percent).toFixed(1)}%
          </span>
        )}
      </div>

      <div className="trend-precision">
        {calibrated ? (
          <span className="precision-status calibrated">
            LSC {data.lsc_percent!.toFixed(1)}% (measured, calibrated{" "}
            {new Date(data.lsc_calibrated_at!).toLocaleDateString("en-CA")})
          </span>
        ) : (
          <span className="precision-status uncalibrated">
            LSC not yet calibrated — using unverified default (
            {config.significantLossPct}%)
          </span>
        )}
        <button
          type="button"
          className="calibrate-btn"
          onClick={() => calibrate.mutate()}
          disabled={calibrate.isPending}
        >
          {calibrate.isPending ? "Calibrating…" : "Calibrate precision"}
        </button>
        {calibrate.isError && (
          <span className="precision-error">
            {(calibrate.error as any)?.response?.data?.detail ??
              "Calibration failed."}
          </span>
        )}
      </div>

      {alert && (
        <div className="trend-alert" role="alert">
          <span className="flag" aria-hidden="true">
            ⚠
          </span>
          <span>
            <b>Significant Bone Loss</b> — {stepDropPct.toFixed(1)}% drop since
            the previous exam (≥ {threshold.toFixed(1)}%
            {calibrated ? " LSC" : ""}).
          </span>
        </div>
      )}

      <ResponsiveContainer width="100%" height={240}>
        <LineChart
          data={chartData}
          margin={{ top: 18, right: 16, bottom: 8, left: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="date" fontSize={12} />
          <YAxis domain={["auto", "auto"]} fontSize={12} />
          <Tooltip />
          {/* 골다공증 임계 참고선 (예시, 미보정) */}
          <ReferenceLine y={0.7} stroke="#dc2626" strokeDasharray="4 4" />
          <Line
            type="monotone"
            dataKey="bmd"
            stroke={alert ? "#dc2626" : "#2563eb"}
            strokeWidth={2}
            dot={{ r: 4 }}
          />
          {/* 급락 시 마지막 지점을 빨간 플래그로 강조 */}
          {alert && (
            <ReferenceDot
              x={last.date}
              y={last.bmd}
              r={6}
              fill="#dc2626"
              stroke="#fff"
              strokeWidth={2}
              isFront
            >
              <Label
                value="⚠ Significant Bone Loss"
                position="top"
                fill="#b91c1c"
                fontSize={11}
                fontWeight={700}
              />
            </ReferenceDot>
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
