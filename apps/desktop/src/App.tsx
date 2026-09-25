import { useCallback, useEffect, useMemo, useState, type MouseEvent } from "react";
import { ApiError, normalizeContainedPoint, SurgicalApiClient } from "@surgical/api-client";
import type { AnalysisSession, Artifact, Doctor, Job, JobOptions, ModelStatus, PrivacyMode, RenderMode } from "@surgical/api-client";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { confirm } from "@tauri-apps/plugin-dialog";
import { StatusAnimation, type AnimationState } from "./lottie";
import { chooseDirectory, chooseVideo, revealDirectory, saveBlob, startManagedApi, stopManagedApi } from "./tauri";

const terminal = new Set(["completed", "failed", "cancelled"]);
const baseOptions: JobOptions = { privacy_mode: "skeleton-only", render_mode: "dual", enable_sam3: true, enable_xray_skeleton: true, device: "auto", depth_enabled: true, tracker: "botsort", health_confidence: 0.35, instrument_confidence: 0.25, pose_confidence: 0.25, iou: 0.45 };
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
        // Model discovery can initialize the local SAM3 runtime on first use;
        // allow that cold start to finish instead of surfacing a transient
        // WebView "Failed to fetch" error.
        for (let attempt = 0; attempt < 60 && !disposed; attempt += 1) {
          try {
            await managedClient.checkHealth();
            const [nextDoctor, nextModels] = await Promise.all([managedClient.runDoctor(), managedClient.listModels()]);
            if (!disposed) { setDoctor(nextDoctor); setModels(nextModels); setConnected(true); setError(null); }
            return;
          } catch {
            await new Promise((resolve) => window.setTimeout(resolve, 250));
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
  const inspection = artifacts.find((item) => item.kind === "inspection_video");
  const privacyXray = artifacts.find((item) => item.kind === "privacy_xray_video");
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
    <RenderSettings options={options} onOptions={setOptions} />
    {job && !terminal.has(job.status) && <JobLifecycleControls client={client} job={job} onJob={setJob} />}
    <InteractiveSelectionWorkspace client={client} connected={connected} videoPath={videoPath} options={options} outputDirectory={outputDirectory} onJobStarted={setJob} />
  </main>;
}

function JobLifecycleControls({ client, job, onJob }: { client: SurgicalApiClient; job: Job; onJob: (job: Job) => void }) {
  const paused = job.status === "paused";
  return <section className="card"><h2>Job lifecycle</h2><button onClick={() => void (paused ? client.resumeJob(job.job_id) : client.pauseJob(job.job_id)).then(onJob)}>{paused ? "Resume" : "Pause"}</button><button className="danger" onClick={() => void client.cancelJob(job.job_id).then(onJob)}>Cancel</button></section>;
}

function RenderSettings({ options, onOptions }: { options: JobOptions; onOptions: (value: JobOptions) => void }) {
  return <section className="card"><h2>Unified render controls</h2><div className="segmented">{(["legacy", "inspection", "privacy-xray", "dual"] as RenderMode[]).map((render_mode) => <button key={render_mode} className={options.render_mode === render_mode ? "selected" : ""} onClick={() => onOptions({ ...options, render_mode, enable_sam3: render_mode !== "legacy" })}>{render_mode}</button>)}</div><label className="toggle"><input type="checkbox" checked={Boolean(options.enable_sam3)} onChange={(event) => onOptions({ ...options, enable_sam3: event.target.checked })} /> Enable SAM3 propagation</label><label className="toggle"><input type="checkbox" checked={Boolean(options.enable_xray_skeleton)} onChange={(event) => onOptions({ ...options, enable_xray_skeleton: event.target.checked })} /> Enable synthetic X-ray skeleton</label></section>;
}

function SelectionWorkspace({ client, videoPath, options, outputDirectory, onJobStarted }: { client: SurgicalApiClient; videoPath: string | null; options: JobOptions; outputDirectory: string | null; onJobStarted: (job: Job) => void }) {
  const [session, setSession] = useState<AnalysisSession | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [notice, setNotice] = useState("Select a video, then create a metadata-backed selection session.");
  async function open() {
    if (!videoPath) return;
    try {
      const created = await client.createSession();
      const attached = await client.attachSessionVideo(created.session_id, { local_path: videoPath, options: { privacy_mode: "skeleton-only" } });
      setSession(attached); setPreview(null); setNotice("Clean preview ready. Click a target, then start the selected session job.");
    } catch (reason) { setNotice(reason instanceof Error ? reason.message : "Session could not be created."); }
  }
  async function select(event: MouseEvent<HTMLVideoElement>) {
    if (!session || !session.source_fps) { setNotice("Source FPS metadata is required before selection."); return; }
    const element = event.currentTarget; const rect = element.getBoundingClientRect();
    const point = normalizeContainedPoint(event.clientX - rect.left, event.clientY - rect.top, rect.width, rect.height, element.videoWidth, element.videoHeight);
    if (!point) { setNotice("Letterbox area cannot be selected."); return; }
    try {
      const frameIndex = Math.floor(element.currentTime * session.source_fps);
      const next = await client.addSelection(session.session_id, { ...point, displayed_width: Math.round(rect.width), displayed_height: Math.round(rect.height), source_width: element.videoWidth, source_height: element.videoHeight, frame_index: frameIndex, timestamp: element.currentTime, target_category: "object", action: "add" });
      setSession(next);
      const blob = await client.sessionPreview(next.session_id, frameIndex, "inspection");
      setPreview(URL.createObjectURL(blob)); setNotice("Selection propagated; it will be included in the session job.");
    } catch (reason) { setNotice(reason instanceof Error ? reason.message : "Selection failed."); }
  }
  async function startSelected() {
    if (!session) return;
    try {
      const started = await client.startSession(session.session_id, { local_path: videoPath ?? undefined, options: { ...options, output_directory: outputDirectory ?? undefined } });
      if (started.job_id) onJobStarted(await client.getJob(started.job_id));
      setSession(started); setNotice("Selected session job started.");
    } catch (reason) { setNotice(reason instanceof Error ? reason.message : "Selected job could not start."); }
  }
  return <section className="card" aria-label="Interactive target selection"><h2>Interactive target selection</h2><p>{notice}</p><button disabled={!videoPath} onClick={() => void open()}>Open clean preview session</button>{session && <><video className="selection-video" controls src={videoPath ?? undefined} onClick={(event) => void select(event)} /><p>FPS: {session.source_fps ?? "unavailable"} · frame {session.current_frame}/{Math.max(0, (session.frame_count ?? 1) - 1)}</p><p>Active targets: {session.active_tracks.map((track) => track.unified_track_id ?? `SAM3 ${track.sam3_track_id}`).join(", ") || "none"}</p>{session.active_tracks.map((track) => <button key={track.event_id} onClick={() => void client.removeSelection(session.session_id, track.event_id).then(setSession)}>Remove {track.unified_track_id ?? track.event_id}</button>)}{preview && <img className="selection-video" src={preview} alt="Selected SAM3 targets" />}<button className="primary" onClick={() => void startSelected()}>Start selected analysis</button></>}</section>;
}

function InteractiveSelectionWorkspace({ client, connected, videoPath, options, outputDirectory, onJobStarted }: { client: SurgicalApiClient; connected: boolean; videoPath: string | null; options: JobOptions; outputDirectory: string | null; onJobStarted: (job: Job) => void }) {
  const [session, setSession] = useState<AnalysisSession | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [overlay, setOverlay] = useState<"clean" | "inspection" | "privacy-xray">("clean");
  const [notice, setNotice] = useState("Choose a video, open a session, then click targets on any timeline frame.");
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  async function show(base: AnalysisSession, requested: number, mode: "clean" | "inspection" | "privacy-xray" = overlay) {
    const maximum = Math.max(0, (base.frame_count ?? 1) - 1);
    const safe = Math.min(maximum, Math.max(0, Math.floor(requested)));
    try {
      const blob = await client.sessionPreview(base.session_id, safe, mode);
      const nextUrl = URL.createObjectURL(blob);
      if (preview) URL.revokeObjectURL(preview);
      setPreview(nextUrl); setFrame(safe); setOverlay(mode);
      setSession({ ...base, current_frame: safe, current_timestamp: base.source_fps ? safe / base.source_fps : 0 });
    } catch (reason) { setNotice(reason instanceof Error ? reason.message : "Preview unavailable."); }
  }
  async function open() {
    if (!videoPath) return;
    if (!connected) { setNotice("Yerel API henüz hazır değil; bağlantı göstergesi hazır olduğunda tekrar deneyin."); return; }
    try {
      const created = await client.createSession();
      const attached = await client.attachSessionVideo(created.session_id, { local_path: videoPath, options: { privacy_mode: "skeleton-only" } });
      setSession(attached); setNotice("Clean frame ready. Use the timeline and click any target.");
      await show(attached, 0, "clean");
    } catch (reason) { setNotice(reason instanceof Error ? reason.message : "Interactive session could not be created."); }
  }
  async function select(event: MouseEvent<HTMLImageElement>) {
    if (!session || !session.source_fps || !session.video_width || !session.video_height) { setNotice("Video metadata is required before selection."); return; }
    const element = event.currentTarget; const rect = element.getBoundingClientRect();
    const point = normalizeContainedPoint(event.clientX - rect.left, event.clientY - rect.top, rect.width, rect.height, session.video_width, session.video_height);
    if (!point) { setNotice("Letterbox area cannot be selected."); return; }
    try {
      const next = await client.addSelection(session.session_id, { ...point, displayed_width: Math.round(rect.width), displayed_height: Math.round(rect.height), source_width: session.video_width, source_height: session.video_height, frame_index: frame, timestamp: frame / session.source_fps, target_category: "object", action: "add" });
      setNotice("Target recorded. Move to another frame and click to add another target.");
      await show(next, frame, "inspection");
    } catch (reason) { setNotice(reason instanceof Error ? reason.message : "Selection could not be recorded."); }
  }
  async function remove(eventId: string) {
    if (!session) return;
    const next = await client.removeSelection(session.session_id, eventId);
    await show(next, frame, overlay);
  }
  async function start() {
    if (!session || !session.active_tracks.length) { setNotice("Select at least one target first."); return; }
    try {
      const started = await client.startSession(session.session_id, { local_path: videoPath ?? undefined, options: { ...options, render_mode: "dual", enable_sam3: true, enable_xray_skeleton: true, output_directory: outputDirectory ?? undefined } });
      if (started.job_id) onJobStarted(await client.getJob(started.job_id));
      setSession(started); setNotice("Tracking started. RGB inspection and final security X-ray outputs are being produced.");
    } catch (reason) { setNotice(reason instanceof Error ? reason.message : "Tracking could not start."); }
  }
  const maximum = Math.max(0, (session?.frame_count ?? 1) - 1);
  return <section className="card" aria-label="Interactive SAM3 target selection"><h2>Interactive SAM3 tracking</h2><p>{notice}</p><button disabled={!videoPath} onClick={() => void open()}>Open video selection session</button>{session && <><div className="timeline-controls"><button disabled={frame <= 0} onClick={() => void show(session, frame - 1, overlay)}>Previous</button><input type="range" min="0" max={maximum} value={frame} onChange={(event) => setFrame(Number(event.target.value))} onMouseUp={() => void show(session, frame, overlay)} onKeyUp={() => void show(session, frame, overlay)} /><input type="number" min="0" max={maximum} value={frame} onChange={(event) => setFrame(Math.min(maximum, Math.max(0, Number(event.target.value) || 0)))} onBlur={() => void show(session, frame, overlay)} /><button disabled={frame >= maximum} onClick={() => void show(session, frame + 1, overlay)}>Next</button></div><p>FPS: {session.source_fps ?? "?"} · Frame: {frame}/{maximum} · Time: {(session.source_fps ? frame / session.source_fps : 0).toFixed(2)} s</p>{preview && <img className="selection-preview" src={preview} alt="Video frame; click to select a target" draggable={false} onClick={(event) => void select(event)} />}<div className="segmented"><button className={overlay === "clean" ? "selected" : ""} onClick={() => void show(session, frame, "clean")}>Clean</button><button className={overlay === "inspection" ? "selected" : ""} onClick={() => void show(session, frame, "inspection")}>RGB inspection</button><button className={overlay === "privacy-xray" ? "selected" : ""} onClick={() => void show(session, frame, "privacy-xray")}>Security X-ray</button></div><p>Active targets: {session.active_tracks.map((track) => track.unified_track_id ?? `SAM3 ${track.sam3_track_id}`).join(", ") || "none"}</p><div className="selection-list">{session.active_tracks.map((track) => <button key={track.event_id} className="danger" onClick={() => void remove(track.event_id)}>Remove {track.unified_track_id ?? track.event_id}</button>)}</div><button className="primary" disabled={!session.active_tracks.length || Boolean(session.job_id)} onClick={() => void start()}>Start tracking · save RGB + Security X-ray</button></>}</section>;
}

function LegacySelectionWorkspace({ client, videoPath }: { client: SurgicalApiClient; videoPath: string | null }) {
  const [session, setSession] = useState<import("@surgical/api-client").AnalysisSession | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [notice, setNotice] = useState("Video seçin; seçimden önce bu preview temiz kalır.");
  async function open() {
    if (!videoPath) return;
    const created = await client.createSession();
    const attached = await client.attachSessionVideo(created.session_id, { local_path: videoPath, options: { privacy_mode: "skeleton-only" } });
    setSession(attached); setPreview(null); setNotice("Temiz preview hazır. Video üzerine tıklayarak hedef ekleyin.");
  }
  async function click(event: MouseEvent<HTMLVideoElement>) {
    if (!session) return;
    const element = event.currentTarget; const rect = element.getBoundingClientRect();
    const point = normalizeContainedPoint(event.clientX - rect.left, event.clientY - rect.top, rect.width, rect.height, element.videoWidth, element.videoHeight);
    if (!point) { setNotice("Letterbox alanı seçilemez."); return; }
    try {
      if (!session.source_fps || session.source_fps <= 0) { setNotice("Video FPS metadata okunamadı; seçim gönderilmedi."); return; }
      const next = await client.addSelection(session.session_id, { ...point, displayed_width: Math.round(rect.width), displayed_height: Math.round(rect.height), source_width: element.videoWidth, source_height: element.videoHeight, frame_index: Math.floor(element.currentTime * session.source_fps), timestamp: element.currentTime, target_category: "object", action: "add" });
      setSession(next); const blob = await client.sessionPreview(next.session_id, next.current_frame, "inspection"); setPreview(URL.createObjectURL(blob)); setNotice(next.warnings[0] || "SAM3 inspection preview güncellendi.");
    } catch (reason) { setNotice(reason instanceof Error ? reason.message : "Seçim gönderilemedi."); }
  }
  return <section className="card" aria-label="İnteraktif hedef seçimi"><h2>İnteraktif hedef seçimi</h2><p>{notice}</p><button disabled={!videoPath} onClick={() => void open()}>Temiz preview oturumu aç</button>{session && <><video className="selection-video" controls src={videoPath ?? undefined} onClick={(event) => void click(event)} /><p>Aktif hedefler: {session.active_tracks.map((track) => track.unified_track_id ?? `SAM3 ${track.sam3_track_id}`).join(", ") || "yok"}</p>{session.active_tracks.map((track) => <button key={track.event_id} onClick={() => void client.removeSelection(session.session_id, track.event_id).then(setSession)}>Hedefi kaldır: {track.unified_track_id ?? track.event_id}</button>)}{preview && <img className="selection-video" src={preview} alt="Seçili SAM3 hedefleri" />}</>}</section>;
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
