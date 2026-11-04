// The browser tab's title for each page: "Scorecard · F1 Predictor".

import { useEffect } from "react";

export function usePageTitle(title: string | null) {
  useEffect(() => {
    document.title = title ? `${title} · F1 Predictor` : "F1 Predictor";
  }, [title]);
}
