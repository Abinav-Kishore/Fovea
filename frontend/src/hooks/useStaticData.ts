import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { BenchmarkResponse, DatasetName, FoveaConfig, StatusResponse } from "../api/types";

/** Data that doesn't change per-frame: backend status and config, fetched
 * once on mount (dataset-independent). Benchmark data IS dataset-specific
 * (each dataset has its own results/benchmark_result*.json), so it's
 * re-fetched whenever `dataset` changes. */
export function useStaticData(dataset: DatasetName) {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [config, setConfig] = useState<FoveaConfig | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkResponse | null>(null);
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .getStatus(controller.signal)
      .then((s) => {
        setStatus(s);
        setBackendConnected(true);
      })
      .catch(() => setBackendConnected(false));
    api.getConfig(controller.signal).then(setConfig).catch(() => undefined);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setBenchmark(null); // honest: don't show the previous dataset's numbers while loading
    api
      .getBenchmark(dataset, controller.signal)
      .then(setBenchmark)
      .catch(() => undefined);
    return () => controller.abort();
  }, [dataset]);

  return { status, config, benchmark, backendConnected };
}
