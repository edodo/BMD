// 공용 모달 껍데기 — PatientCreateModal이 만든 .modal-backdrop/.modal 마크업을
// 재사용 가능하게 뽑아낸 것. 기존 시각 스타일은 그대로, 새 팝업(텍스처 비교,
// 저장/불러오기, 다중환자 비교)이 각자 백드롭 마크업을 반복하지 않도록 한다.
import type { ReactNode } from "react";

interface Props {
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  size?: "default" | "wide" | "xwide";
}

export function Modal({ title, onClose, children, footer, size = "default" }: Props) {
  const sizeClass =
    size === "wide" ? " modal-wide" : size === "xwide" ? " modal-xwide" : "";
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className={`modal${sizeClass}`} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{title}</h3>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-footer">{footer}</div>}
      </div>
    </div>
  );
}
