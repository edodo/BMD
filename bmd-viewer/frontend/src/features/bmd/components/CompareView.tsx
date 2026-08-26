// 종적 대조(Side-by-Side Longitudinal) 뷰.
// 왼쪽 리스트에서 기준(수술 전) 1건, 오른쪽 리스트에서 비교(수술 후) 여러 건을
// 고르면, 중앙에 각 검사의 Extracted L4 + Density basis가 좌우로 나란히 정렬된다.
// '대조 모드'에서는 post−pre 정규화 밀도를 5구역(중앙+네 모서리)으로 나눠
// 구역별 증감을 색 + %로 표시한다 — 셀 단위 세밀 격자 대신 5구역인 이유는
// TextureCompareHardModal과 동일한 논리(작은 셀은 텍스처 특징이 무의미)를
// 밀도 화면에도 일관되게 적용하기 위함.
import { useMemo, useState } from "react";
import { useQueries } from "@tanstack/react-query";
import {
  useStudies,
  useCreateComparison,
  useListComparisons,
  useDeleteComparison,
} from "@/features/patients/api/queries";
import { getDataProvider } from "@/lib/data";
import { Modal } from "@/components/ui/Modal";
import { CompareCard, shortDate } from "./DensityCompareCard";
import { TextureCompareEasyModal } from "./TextureCompareEasyModal";
import { TextureCompareHardModal } from "./TextureCompareHardModal";
import type { XrayStudyDetail, XrayStudyListItem, SavedComparison } from "@/lib/types";

const dp = getDataProvider();

// 시간 정렬 키 (취득일 우선, 없으면 업로드일)
function tkey(s: XrayStudyDetail): number {
  const iso = s.acquired_at ?? s.uploaded_at;
  return iso ? new Date(iso).getTime() : 0;
}

function SaveComparisonModal({
  patientId,
  preId,
  postIds,
  contrast,
  onClose,
}: {
  patientId: string;
  preId: string;
  postIds: string[];
  contrast: boolean;
  onClose: () => void;
}) {
  const create = useCreateComparison();
  const [title, setTitle] = useState("");
  const submit = () => {
    if (!title.trim()) return;
    create.mutate(
      {
        type: "pre_post",
        title: title.trim(),
        patient_id: patientId,
        pre_study_id: preId,
        post_study_ids: postIds,
        config: { contrast },
      },
      { onSuccess: onClose }
    );
  };
  return (
    <Modal
      title="Save comparison"
      onClose={onClose}
      footer={
        <>
          <button className="btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button onClick={submit} disabled={create.isPending}>
            {create.isPending ? "Saving…" : "Save"}
          </button>
        </>
      }
    >
      <label>
        Title
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="e.g. 6-month post-op check"
          autoFocus
        />
      </label>
      {create.isError && <p className="auth-error">Failed to save comparison.</p>}
    </Modal>
  );
}

function SavedComparisonsModal({
  patientId,
  onClose,
  onLoad,
}: {
  patientId: string;
  onClose: () => void;
  onLoad: (c: SavedComparison) => void;
}) {
  const { data, isLoading } = useListComparisons({ patientId, type: "pre_post" });
  const del = useDeleteComparison();
  return (
    <Modal title="Saved comparisons" onClose={onClose} size="wide">
      {isLoading && <p className="muted">Loading…</p>}
      {data?.length === 0 && <p className="muted">No saved comparisons yet.</p>}
      <ul className="saved-cmp-list">
        {data?.map((c) => (
          <li key={c.id}>
            <div>
              <div className="saved-cmp-title">{c.title}</div>
              <div className="saved-cmp-meta">
                {shortDate(c.created_at)} · {c.post_study_ids?.length ?? 0} post-op
                {" "}study(ies)
              </div>
            </div>
            <div className="saved-cmp-actions">
              <button
                onClick={() => {
                  onLoad(c);
                  onClose();
                }}
              >
                Load
              </button>
              <button
                className="saved-cmp-delete"
                title="Delete this saved comparison"
                disabled={del.isPending}
                onClick={() => {
                  if (window.confirm(`Delete saved comparison "${c.title}"?`)) {
                    del.mutate(c.id);
                  }
                }}
              >
                🗑
              </button>
            </div>
          </li>
        ))}
      </ul>
    </Modal>
  );
}

export function CompareView({ patientId }: { patientId: string }) {
  const { data: studies } = useStudies(patientId);
  // useStudies는 최신순(내림차순)으로 오므로, 이 화면(pre/post 선택 목록)만
  // 오래된 날짜가 위 · 최신 날짜가 아래로 오도록 뒤집는다. History 등 다른
  // 화면의 정렬(최신순)은 그대로 둔다.
  const completed = (studies ?? [])
    .filter((s: XrayStudyListItem) => s.status === "completed")
    .reverse();

  const [preId, setPreId] = useState<string>();
  const [postIds, setPostIds] = useState<string[]>([]);
  const [contrast, setContrast] = useState(true);
  const [texturePopup, setTexturePopup] = useState<"easy" | "hard" | null>(null);
  const [showSave, setShowSave] = useState(false);
  const [showSaved, setShowSaved] = useState(false);

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
  const preBmd = pre?.measurement?.bmd_value ?? null;
  const posts = postIds
    .filter((id) => id !== preId)
    .map((id) => detailById.get(id))
    .filter((d): d is XrayStudyDetail => !!d)
    .sort((a, b) => tkey(a) - tkey(b));

  const togglePost = (id: string) =>
    setPostIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );

  const canCompare = !!preId && postIds.filter((id) => id !== preId).length > 0;

  return (
    <div className="compare-view">
      <div className="cmp-toolbar">
        <h3>Side-by-side comparison</h3>
        <div className="cmp-toolbar-actions">
          <label className="cmp-contrast">
            <input
              type="checkbox"
              checked={contrast}
              onChange={(e) => setContrast(e.target.checked)}
            />
            Contrast mode (density-change by region)
          </label>
          <button
            className="btn-secondary"
            disabled={!canCompare}
            onClick={() => setTexturePopup("easy")}
          >
            Texture (simple)
          </button>
          <button
            className="btn-secondary"
            disabled={!canCompare}
            onClick={() => setTexturePopup("hard")}
          >
            Texture (detailed)
          </button>
          <button disabled={!canCompare} onClick={() => setShowSave(true)}>
            💾 Save
          </button>
          <button className="btn-secondary" onClick={() => setShowSaved(true)}>
            📂 Saved
          </button>
        </div>
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
              preBmd={null}
              contrast={false}
            />
          )}
          {posts.map((s) => (
            <CompareCard
              key={s.id}
              study={s}
              role="post"
              preGrid={preGrid}
              preBmd={preBmd}
              preStudyId={preId}
              contrast={contrast}
            />
          ))}
        </div>
      ) : (
        <div className="cmp-empty">
          Select a pre-op study and one or more post-op studies to compare.
        </div>
      )}

      {texturePopup === "easy" && pre && posts.length > 0 && (
        <TextureCompareEasyModal
          pre={pre}
          posts={posts}
          onClose={() => setTexturePopup(null)}
        />
      )}
      {texturePopup === "hard" && pre && posts.length > 0 && (
        <TextureCompareHardModal
          pre={pre}
          posts={posts}
          onClose={() => setTexturePopup(null)}
        />
      )}
      {showSave && preId && (
        <SaveComparisonModal
          patientId={patientId}
          preId={preId}
          postIds={postIds.filter((id) => id !== preId)}
          contrast={contrast}
          onClose={() => setShowSave(false)}
        />
      )}
      {showSaved && (
        <SavedComparisonsModal
          patientId={patientId}
          onClose={() => setShowSaved(false)}
          onLoad={(c) => {
            if (c.pre_study_id) setPreId(c.pre_study_id);
            if (c.post_study_ids) setPostIds(c.post_study_ids);
            if (c.config && typeof c.config.contrast === "boolean") {
              setContrast(c.config.contrast);
            }
          }}
        />
      )}
    </div>
  );
}
