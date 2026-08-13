// Click an image to view it enlarged in a full-screen lightbox.
// Backdrop click, close button, or Escape all dismiss it.
import { useEffect, useState } from "react";

export function ClickToZoom({
  src,
  alt,
  className,
}: {
  src: string;
  alt: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <>
      <img
        src={src}
        alt={alt}
        className={className}
        onClick={() => setOpen(true)}
        role="button"
        tabIndex={0}
        aria-label={`${alt} (click to enlarge)`}
      />
      {open && (
        <div className="lightbox-backdrop" onClick={() => setOpen(false)}>
          <button
            className="lightbox-close"
            onClick={() => setOpen(false)}
            aria-label="Close"
          >
            ✕
          </button>
          <img
            className="lightbox-img"
            src={src}
            alt={alt}
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  );
}
