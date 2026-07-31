// 종적 대조(Side-by-Side Longitudinal) 뷰.
// 왼쪽 리스트에서 기준(수술 전) 1건, 오른쪽 리스트에서 비교(수술 후) 여러 건을
// 고르면, 중앙에 각 검사의 Extracted L4 + Density basis가 좌우로 나란히 정렬된다.
// '대조 모드'에서는 post−pre 정규화 밀도 격자를 비교해
// 밀도 감소 부위는 파란 격자, 증가 부위는 붉은 격자로 플롯한다.
import { useMemo, useState } from "react";
import { useQueries } from "@tanstack/react-query";
import { useStudies } from "@/features/patients/api/queries";
import { getDataProvider } from "@/lib/data";
import { config } from "@/lib/config";
import type { XrayStudyDetail, XrayStudyListItem } from "@/lib/types";

const dp = getDataProvider();

// 시간 정렬 키 (취득일 우선, 없으면 업로드일)
function tkey(s: XrayStudyDetail): number {
  const iso = s.acquired_at ?? s.uploaded_at;
  return iso ? new Date(iso).getTime() : 0;
}
function shortDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-CA");
}

// post−pre 밀도 격자 → 감소(파랑)/증가(빨강) 격자 오버레이
function BoneLossGrid({
  pre,
  post,
}: {
  pre: (number | null)[][] | null;
  post: (number | null)[][] | null;
}) {
  if (!pre || !post) return null;
  const n = post.length;
  const cells: JSX.Element[] = [];
  let lossCells = 0;
  let gainCells = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const p = pre[i]?.[j];
      const q = post[i]?.[j];
      const d = p != null && q != null ? q - p : null;
      const loss = d != null && d < -config.compareBoneLossDelta;
      const gain = d != null && d > config.compareBoneLossDelta;
      if (loss) lossCells++;
      if (gain) gainCells++;
      const alpha =
        loss || gain ? Math.min(0.75, 0.25 + Math.abs(d!) * 1.5) : 0;
      const color = loss
        ? `rgba(37,99,235,${alpha})` // 밀도 감소 → 파랑
        : gain
          ? `rgba(220,38,38,${alpha})` // 밀도 증가 → 빨강
          : "transparent";
      const line = loss
        ? "1px solid rgba(37,99,235,0.45)"
        : gain
          ? "1px solid rgba(220,38,38,0.45)"
          : "none";
      cells.push(
        <span key={`${i}-${j}`} style={{ background: color, outline: line }} />
      );
    }
  }
  return (
    <div className="bone-loss">
      <div
        className="bone-loss-grid"
        style={{
          gridTemplateColumns: `repeat(${n}, 1fr)`,
          gridTemplateRows: `repeat(${n}, 1fr)`,
        }}
      >
        {cells}
      </div>
      <p className="bone-loss-cap">
        Density change vs pre-op — ↓ {lossCells} (blue) / ↑ {gainCells} (red)
        of {n * n} regions (|Δ| &gt; {config.compareBoneLossDelta})
      </p>
    </div>
  );
}

function CompareCard({
  study,
  role,
  preGrid,
  contrast,
}: {
  study: XrayStudyDetail;
  role: "pre" | "post";
  preGrid: (number | null)[][] | null;
  contrast: boolean;
}) {
  const m = study.measurement;
  const l4 = dp.assetUrl(m?.l4_crop_path ?? null);
  const dens = dp.assetUrl(m?.xai_overlay_path ?? null);
  return (
    <div className={`cmp-card ${role}`}>
      <div className="cmp-card-head">
        <span className={`cmp-tag ${role}`}>
          {role === "pre" ? "Pre-op" : "Post-op"}
        </span>
        <span className="cmp-date">{shortDate(study.acquired_at)}</span>
        <span className="cmp-bmd">
          BMD {m?.bmd_value != null ? m.bmd_value.toFixed(3) : "—"}
        </span>
      </div>
      {l4 ? (
        <img
          className="cmp-img"
          src={`${l4}?v=${study.id}`}
          alt="Extracted L4"
        />
      ) : (
        <div className="cmp-img placeholder">No L4 crop</div>
      )}
      {dens && (
        <img
          className="cmp-img"
          src={`${dens}?v=${study.id}`}
          alt="Density basis"
        />
      )}
      {role === "post" && contrast && (
        <BoneLossGrid pre={preGrid} post={m?.l4_density_grid ?? null} />
      )}
    </div>
  );
}

export function CompareView({ patientId }: { patientId: string }) {
  const { data: studies } = useStudies(patientId);
  const completed = (studies ?? []).filter(
    (s: XrayStudyListItem) => s.status === "completed"
  );

  const [preId, setPreId] = useState<string>();
  const [postIds, setPostIds] = useState<string[]>([]);
  const [contrast, setContrast] = useState(true);

  const ids = useMemo(() => {
    const set = new Set<string>();
    if (preId) set.add(preId);
    postIds.forEach((id) => set.add(id));
    return Array.from(set);
  }, [preId, postIds]);

  const results = useQueries({
    queries: ids.map((id) => ({
      queryKey: ["study", id],
      queryFn: () => dp.getStudy(id),
      enabled: !!id,
    })),
  });
  const detailById = new Map<string, XrayStudyDetail>();
  results.forEach((r, i) => {
    if (r.data) detailById.set(ids[i], r.data as XrayStudyDetail);
  });

  const pre = preId ? detailById.get(preId) : undefined;
  const preGrid = pre?.measurement?.l4_density_grid ?? null;
  const posts = postIds
    .filter((id) => id !== preId)
    .map((id) => detailById.get(id))
    .filter((d): d is XrayStudyDetail => !!d)
    .sort((a, b) => tkey(a) - tkey(b));

  const togglePost = (id: string) =>
    setPostIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );

  return (
    <div className="compare-view">
      <div className="cmp-toolbar">
        <h3>Side-by-side comparison</h3>
        <label className="cmp-contrast">
          <input
            type="checkbox"
            checked={contrast}
            onChange={(e) => setContrast(e.target.checked)}
          />
          Contrast mode (density-change grid)
        </label>
      </div>

      {/* 두 선택 리스트: 왼쪽=수술 전(1개), 오른쪽=수술 후(여러 개) */}
      <div className="cmp-pickers">
        <div className="cmp-picker">
          <h4>Pre-op (select 1)</h4>
          <ul>
            {completed.map((s) => (
              <li key={s.id}>
                <label>
                  <input
                    type="radio"
                    name="cmp-pre"
                    checked={preId === s.id}
                    onChange={() => setPreId(s.id)}
                  />
                  <span>{shortDate(s.acquired_at ?? s.uploaded_at)}</span>
                  <span className="muted">
                    BMD {s.bmd_value != null ? s.bmd_value.toFixed(3) : "—"}
                  </span>
                </label>
              </li>
            ))}
          </ul>
        </div>
        <div className="cmp-picker">
          <h4>Post-op (select many)</h4>
          <ul>
            {completed.map((s) => (
              <li key={s.id}>
                <label>
                  <input
                    type="checkbox"
                    checked={postIds.includes(s.id)}
                    disabled={preId === s.id}
                    onChange={() => togglePost(s.id)}
                  />
                  <span>{shortDate(s.acquired_at ?? s.uploaded_at)}</span>
                  <span className="muted">
                    BMD {s.bmd_value != null ? s.bmd_value.toFixed(3) : "—"}
                  </span>
                </label>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* 중앙: pre | post1 post2 ... 좌우 정렬 (post는 시간순) */}
      {pre || posts.length > 0 ? (
        <div className="cmp-strip">
          {pre && (
            <CompareCard
              study={pre}
              role="pre"
              preGrid={null}
              contrast={false}
            />
          )}
          {posts.map((s) => (
            <CompareCard
              key={s.id}
              study={s}
              role="post"
              preGrid={preGrid}
              contrast={contrast}
            />
          ))}
        </div>
      ) : (
        <div className="cmp-empty">
          Select a pre-op study and one or more post-op studies to compare.
        </div>
      )}
    </div>
  );
}
