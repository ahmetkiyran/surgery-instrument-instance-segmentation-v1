import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, SurgicalApiClient } from "@surgical/api-client";
import type { Artifact, Doctor, Job, JobOptions, ModelStatus, PrivacyMode } from "@surgical/api-client";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { confirm } from "@tauri-apps/plugin-dialog";
import { StatusAnimation, type AnimationState } from "./lottie";
import { chooseDirectory, chooseVideo, revealDirectory, saveBlob, startManagedApi, stopManagedApi } from "./tauri";

const terminal = new Set(["completed", "failed", "cancelled"]);
const baseOptions: JobOptions = { privacy_mode: "skeleton-only", device: "auto", depth_enabled: true, tracker: "botsort", health_confidence: 0.35, instrument_confidence: 0.25, pose_confidence: 0.25, iou: 0.45 };
interface DesktopVideoDetails { filename: string; sizeBytes: number; durationSeconds: number; }

function seconds(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  const rounded = Math.max(0, Math.round(value));
  return `${String(Math.floor(rounded / 60)).padStart(2, "0")}:${String(rounded % 60).padStart(2, "0")}`;
}

function formatBytes(value: number): string {
  return `${(value / (1024 * 1024)).toFixed(value < 1024 * 1024 ? 1 : 0)} MB`;
}

export function App() {
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8765");
  const [token, setToken] = useState("");
  const [apiMode, setApiMode] = useState<"managed" | "external">("managed");
  const [connected, setConnected] = useState(false);
  const [doctor, setDoctor] = useState<Doctor | null>(null);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [videoDetails, setVideoDetails] = useState<DesktopVideoDetails | null>(null);
  const [outputDirectory, setOutputDirectory] = useState<string | null>(null);
  const [options, setOptions] = useState<JobOptions>(baseOptions);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("Özet");

  const client = useMemo(() => new SurgicalApiClient({ baseUrl, token: token || undefined }), [baseUrl, token]);
  const animation: AnimationState = error ? "error" : job?.status === "completed" ? "completed" : job?.status === "finalizing" ? "finalizing" : job && !terminal.has(job.status) ? "analyzing" : connected ? "idle" : "checking_models";

  const refreshSystem = useCallback(async () => {
    try {
      await client.checkHealth();
      const [nextDoctor, nextModels] = await Promise.all([client.runDoctor(), client.listModels()]);
      setDoctor(nextDoctor);
      setModels(nextModels);
      setConnected(true);
      setError(null);
    } catch (reason) {
      setConnected(false);
      setError(reason instanceof Error ? reason.message : "API bağlantısı kurulamadı.");
    }
  }, [client]);

  useEffect(() => {
    if (apiMode !== "managed") return;
    let disposed = false;
    void (async () => {
      try {
        const managed = await startManagedApi();
        if (disposed) { await stopManagedApi(); return; }
        setBaseUrl(managed.baseUrl);
        setToken(managed.token);
        const managedClient = new SurgicalApiClient({ baseUrl: managed.baseUrl, token: managed.token });
        for (let attempt = 0; attempt < 15 && !disposed; attempt += 1) {
          try {
            await managedClient.checkHealth();
            const [nextDoctor, nextModels] = await Promise.all([managedClient.runDoctor(), managedClient.listModels()]);
            if (!disposed) { setDoctor(nextDoctor); setModels(nextModels); setConnected(true); setError(null); }
            return;
          } catch {
            await new Promise((resolve) => window.setTimeout(resolve, 200));
          }
        }
        if (!disposed) setError("Yönetilen API zamanında hazır olmadı; .venv ve server bağımlılıklarını kontrol edin.");
      } catch {
        // Browser/Vite mode uses an already-running loopback API; connection remains editable.
      }
    })();
    return () => { disposed = true; void stopManagedApi().catch(() => undefined); };
  }, [apiMode]);

  useEffect(() => { void refreshSystem(); }, [refreshSystem]);

  useEffect(() => {
    if (!job || terminal.has(job.status)) return;
    const timer = window.setInterval(async () => {
      try { setJob(await client.getJob(job.job_id)); } catch (reason) { setError(reason instanceof Error ? reason.message : "İş durumu alınamadı."); }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [client, job]);

  useEffect(() => {
    const warning = (event: BeforeUnloadEvent) => {
      if (job && !terminal.has(job.status)) { event.preventDefault(); event.returnValue = ""; }
    };
    window.addEventListener("beforeunload", warning);
    return () => window.removeEventListener("beforeunload", warning);
  }, [job]);

  const setSelectedVideo = useCallback(async (path: string) => {
    setVideoPath(path);
    setVideoDetails(null);
    setError(null);
    try {
      const details = await client.inspectDesktopVideo(path);
      setVideoDetails({ filename: details.filename, sizeBytes: details.size_bytes, durationSeconds: details.duration_seconds });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Video bilgisi okunamadı.");
    }
  }, [client]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    void getCurrentWebview().onDragDropEvent((event) => {
      if (event.payload.type === "drop") {
        const path = event.payload.paths.find((item: string) => /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(item));
        if (path) void setSelectedVideo(path);
        else setError("Desteklenen bir video dosyası bırakın.");
      }
    }).then((stop) => { unlisten = stop; }).catch(() => undefined);
    return () => { unlisten?.(); };
  }, [setSelectedVideo]);

  async function selectVideo() {
    const selected = await chooseVideo();
    if (selected) await setSelectedVideo(selected);
  }

  async function runAnalysis() {
    if (!videoPath || !outputDirectory) { setError("Video ve sonuç klasörü seçin."); return; }
    try {
      setError(null);
      const next = await client.createJob({ local_path: videoPath, options: { ...options, output_directory: outputDirectory } });
      setJob(next);
      setActiveTab("Özet");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Analiz başlatılamadı."); }
  }

  async function download(artifact: Artifact) {
    const blob = await loadArtifact(artifact);
    if (!blob) return;
    try { await saveBlob(blob, artifact.filename); }
    catch (reason) { setError(reason instanceof ApiError ? reason.message : "Artifact indirilemedi."); }
  }

  async function loadArtifact(artifact: Artifact): Promise<Blob | null> {
    const accepted = !artifact.sensitive || await confirm("Blur çıktısı hassas görüntü içerebilir. İndirmek istediğinizi onaylıyor musunuz?", { title: "Hassas çıktı", kind: "warning" });
    if (!accepted || !job) return null;
    try { return await client.downloadArtifact(job.job_id, artifact.artifact_id, artifact.sensitive); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Artifact indirilemedi."); return null; }
  }

  const artifacts = job?.artifacts ?? [];
  const skeleton = artifacts.find((item) => item.kind === "skeleton_video");
  const blur = artifacts.find((item) => item.kind === "blur_video");
  const tabs = ["Özet", ...(skeleton ? ["Skeleton Video"] : []), ...(blur ? ["Blur Video"] : []), ...(artifacts.some((item) => item.kind === "instrument_usage") ? ["Alet Kullanımı"] : []), ...(artifacts.some((item) => item.kind === "personnel_motion") ? ["Personel Hareketi"] : []), ...(artifacts.some((item) => item.kind === "relative_4d_html") ? ["Göreli 4B Yörünge"] : []), ...(artifacts.length ? ["Dosyalar"] : [])];

  return <main className="shell">
    <header className="topbar"><div><p className="eyebrow">LOCAL · PRIVACY-FIRST</p><h1>Surgical Analysis Studio</h1></div><div className={`connection ${connected ? "ok" : ""}`}>{connected ? "API bağlı" : "API bekleniyor"}</div></header>
    <section className="connection-panel" aria-label="API bağlantısı"><label>API modu<select value={apiMode} onChange={(event) => { setApiMode(event.target.value as "managed" | "external"); setConnected(false); }}><option value="managed">Yönetilen yerel API</option><option value="external">Harici API</option></select></label><label>API adresi<input value={baseUrl} disabled={apiMode === "managed"} onChange={(event) => setBaseUrl(event.target.value)} /></label><label>Oturum tokenı<input type="password" value={token} disabled={apiMode === "managed"} onChange={(event) => setToken(event.target.value)} placeholder="Yalnızca yönetilen/LAN oturumlarında" /></label><button onClick={() => void refreshSystem()}>Bağlantıyı test et</button>{apiMode === "external" && <small>Harici API adresini yalnızca güvenilir yerel ağda kullanın.</small>}</section>
    {error && <div className="alert error" role="alert">{error}</div>}
    <section className="grid overview">
      <article className="card status-card"><StatusAnimation state={animation} /><div><h2>Sistem durumu</h2><p>{doctor?.ready ? "Analiz için hazır" : connected ? "Doctor uyarıları var" : "API bağlantısı bekleniyor"}</p></div></article>
      <article className="card"><h2>Modeller</h2><div className="chips">{models.map((model) => <span key={model.key} className={model.ready ? "chip ok" : "chip warn"}>{model.key}: {model.ready ? "hazır" : model.state}</span>)}</div></article>
      <article className="card"><h2>Donanım ve araçlar</h2><div className="check-grid">{doctor?.checks.filter((item) => ["Python", "CUDA", "GPU adı", "FFmpeg"].includes(item.label)).map((item) => <span key={item.label} className={item.ok ? "ok-text" : "warn-text"}>{item.label}: {item.ok ? "hazır" : "kontrol et"}</span>)}</div></article>
    </section>
    <section className="grid workspace">
      <article className="card input-card"><h2>Video seçimi</h2><button className="dropzone" onClick={() => void selectVideo()} aria-label="Video seç"><span>Video dosyasını sürükleyip bırakın veya seçmek için tıklayın</span><small>Kaynak video değiştirilmez; yalnızca yerel loopback API'ye gönderilen yol kullanılır.</small></button>{videoPath && <p className="selected-file">Seçildi: {videoDetails?.filename ?? videoPath.split(/[\\/]/).pop()} {videoDetails && <>· {formatBytes(videoDetails.sizeBytes)} · {seconds(videoDetails.durationSeconds)}</>}</p>}<button className="secondary" onClick={() => void chooseDirectory().then(setOutputDirectory)}>Sonuç klasörü seç</button>{outputDirectory && <p className="selected-file">Çıktı: {outputDirectory}</p>}</article>
      <article className="card settings"><h2>Analiz ayarları</h2><fieldset><legend>Gizlilik modu</legend><div className="segmented">{(["skeleton-only", "blur", "both"] as PrivacyMode[]).map((mode) => <button key={mode} className={options.privacy_mode === mode ? "selected" : ""} onClick={() => setOptions({ ...options, privacy_mode: mode })}>{mode}</button>)}</div></fieldset>{options.privacy_mode !== "skeleton-only" && <p className="alert warning">Blur veya both gerçek görüntüden türetilmiş hassas çıktı üretebilir.</p>}<div className="form-grid"><label>Cihaz<select value={options.device} onChange={(event) => setOptions({ ...options, device: event.target.value })}><option value="auto">auto</option><option value="cpu">cpu</option><option value="cuda:0">cuda:0</option></select></label><label>Takip<select value={options.tracker} onChange={(event) => setOptions({ ...options, tracker: event.target.value as "botsort" | "bytetrack" })}><option value="botsort">BoT-SORT</option><option value="bytetrack">ByteTrack</option></select></label><label>Health confidence<input type="number" min="0.01" max="1" step="0.05" value={options.health_confidence} onChange={(event) => setOptions({ ...options, health_confidence: Number(event.target.value) })} /></label><label>Alet confidence<input type="number" min="0.01" max="1" step="0.05" value={options.instrument_confidence} onChange={(event) => setOptions({ ...options, instrument_confidence: Number(event.target.value) })} /></label></div><label className="toggle"><input type="checkbox" checked={options.depth_enabled} onChange={(event) => setOptions({ ...options, depth_enabled: event.target.checked })} /> Göreli depth üret</label><button className="primary" disabled={!connected || !videoPath || !outputDirectory || Boolean(job && !terminal.has(job.status))} onClick={() => void runAnalysis()}>{job && !terminal.has(job.status) ? "Analiz sürüyor" : "Analizi başlat"}</button></article>
    </section>
    {job && <section className="card progress-card"><div className="progress-header"><div><p className="eyebrow">{job.status}</p><h2>{job.current_stage}</h2><p>{job.message}</p></div>{!terminal.has(job.status) && <button className="danger" onClick={() => void client.cancelJob(job.job_id).then(setJob)}>Güvenli iptal</button>}</div><progress value={job.progress_percent} max="100">{job.progress_percent}%</progress><div className="metrics"><span>{job.progress_percent}%</span><span>{job.processed_frames}/{job.total_frames} kare</span><span>Geçen {seconds(job.elapsed_seconds)}</span><span>Kalan {seconds(job.estimated_remaining_seconds)}</span></div>{job.safe_error_message && <div className="alert error">{job.safe_error_message}</div>}</section>}
    {job?.status === "completed" && <section className="card results"><div className="results-heading"><div><p className="eyebrow">ANALİZ SONUÇLARI</p><h2>Sonuçlar</h2></div>{outputDirectory && <button className="secondary" onClick={() => void revealDirectory(outputDirectory)}>Sonuç klasörünü aç</button>}</div><nav aria-label="Sonuç sekmeleri">{tabs.map((tab) => <button key={tab} className={activeTab === tab ? "tab active" : "tab"} onClick={() => setActiveTab(tab)}>{tab}</button>)}</nav><ResultPanel tab={activeTab} skeleton={skeleton} blur={blur} artifacts={artifacts} onDownload={download} onLoadVideo={loadArtifact} /></section>}
  </main>;
}

function ResultPanel({ tab, skeleton, blur, artifacts, onDownload, onLoadVideo }: { tab: string; skeleton?: Artifact; blur?: Artifact; artifacts: Artifact[]; onDownload: (artifact: Artifact) => Promise<void>; onLoadVideo: (artifact: Artifact) => Promise<Blob | null> }) {
  if (tab === "Özet") return <p>Sonuç hesaplamaları Python analiz çekirdeği tarafından üretildi. Kullanım süreleri algılanan görünürlük süresidir; fiziksel kullanım süresi değildir.</p>;
  if (tab === "Skeleton Video" && skeleton) return <ArtifactVideo artifact={skeleton} onLoadVideo={onLoadVideo} />;
  if (tab === "Blur Video" && blur) return <><div className="alert warning">Bu video hassas içerik barındırabilir. Otomatik paylaşım paketine dahil edilmez.</div><ArtifactVideo artifact={blur} onLoadVideo={onLoadVideo} /></>;
  if (tab === "Dosyalar") return <ul className="artifact-list">{artifacts.map((artifact) => <li key={artifact.artifact_id}><span><strong>{artifact.filename}</strong><small>{artifact.kind} · {formatBytes(artifact.size_bytes)}</small></span><button onClick={() => void onDownload(artifact)}>İndir</button></li>)}</ul>;
  const related = artifacts.filter((artifact) => tab === "Alet Kullanımı" ? artifact.kind === "instrument_usage" : tab === "Personel Hareketi" ? artifact.kind === "personnel_motion" : artifact.kind === "relative_4d_html");
  return <ul className="artifact-list">{related.map((artifact) => <li key={artifact.artifact_id}><span><strong>{artifact.filename}</strong><small>Python tarafından üretilen artifact</small></span><button onClick={() => void onDownload(artifact)}>İndir</button></li>)}</ul>;
}

function ArtifactVideo({ artifact, onLoadVideo }: { artifact: Artifact; onLoadVideo: (artifact: Artifact) => Promise<Blob | null> }) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => () => { if (src) URL.revokeObjectURL(src); }, [src]);
  async function play() {
    const blob = await onLoadVideo(artifact);
    if (blob) setSrc(URL.createObjectURL(blob));
  }
  return <div className="video-panel"><p>İçerik güvenli olarak bellekten uygulama içinde oynatılır.</p>{src ? <video controls src={src} /> : <button onClick={() => void play()}>Uygulama içinde oynat</button>}</div>;
}
