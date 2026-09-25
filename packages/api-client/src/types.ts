export type JobStatus =
  | "queued"
  | "validating"
  | "running"
  | "paused"
  | "finalizing"
  | "completed"
  | "failed"
  | "cancelled";

export type PrivacyMode = "skeleton-only" | "blur" | "both";
export type RenderMode = "legacy" | "inspection" | "privacy-xray" | "dual";
export type Tracker = "botsort" | "bytetrack";

export interface Artifact {
  artifact_id: string;
  filename: string;
  kind: string;
  size_bytes: number;
  media_type: string;
  sensitive: boolean;
}

export interface JobOptions {
  privacy_mode: PrivacyMode;
  device?: string;
  depth_enabled?: boolean;
  health_confidence?: number;
  instrument_confidence?: number;
  pose_confidence?: number;
  iou?: number;
  tracker?: Tracker;
  /** Only accepted from a Tauri loopback connection, never mobile/LAN. */
  output_directory?: string;
  render_mode?: RenderMode;
  enable_sam3?: boolean;
  enable_xray_skeleton?: boolean;
  selection_events?: SelectionEvent[];
}

export interface SelectionEvent {
  normalized_x: number;
  normalized_y: number;
  displayed_width: number;
  displayed_height: number;
  source_width: number;
  source_height: number;
  frame_index: number;
  timestamp: number;
  target_category?: string;
  action?: "add" | "select" | "remove" | "pause" | "resume";
  unified_track_id?: string;
  sam3_track_id?: number;
  confidence?: number;
}

export interface SessionTrack { event_id: string; sam3_track_id: number | null; unified_track_id: string | null; category: string; }
export interface AnalysisSession {
  session_id: string; status: string; job_id: string | null;
  current_frame: number; current_timestamp: number; video_width: number | null; video_height: number | null;
  source_fps: number | null; frame_count: number | null;
  active_tracks: SessionTrack[]; selected_tracks: string[]; preview_url: string | null;
  inspection_preview_url: string | null; privacy_preview_url: string | null; progress: number; warnings: string[]; error: string | null;
}

export interface CreateJobInput {
  upload_id?: string;
  /** Desktop-only. Never send a local filesystem path from mobile. */
  local_path?: string;
  options: JobOptions;
}

export interface Job {
  job_id: string;
  status: JobStatus;
  progress_percent: number;
  processed_frames: number;
  total_frames: number;
  elapsed_seconds: number;
  estimated_remaining_seconds: number | null;
  current_stage: string;
  message: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  privacy_mode: PrivacyMode;
  artifacts: Artifact[];
  error_code: string | null;
  safe_error_message: string | null;
}

export interface Health {
  status: "ok";
  api_version: string;
  auth_required: boolean;
  lan_mode: boolean;
}

export interface DoctorCheck {
  label: string;
  ok: boolean;
  detail: string;
  severity: "critical" | "warning";
}

export interface Doctor {
  ready: boolean;
  checks: DoctorCheck[];
}

export interface ModelStatus {
  key: string;
  state: string;
  detail: string;
  ready: boolean;
}

export interface Defaults {
  privacy_mode: PrivacyMode;
  device: string;
  depth_enabled: boolean;
  health_confidence: number;
  instrument_confidence: number;
  iou: number;
  tracker: Tracker;
}

export interface UploadResult {
  upload_id: string;
  filename: string;
  size_bytes: number;
  duration_seconds: number;
}

/** Metadata from a desktop-local file; its path is intentionally never returned. */
export interface LocalVideoInspection {
  filename: string;
  size_bytes: number;
  duration_seconds: number;
}

export interface UploadInput {
  data: Blob | ArrayBuffer | Uint8Array;
  filename: string;
  onProgress?: (loadedBytes: number, totalBytes: number | null) => void;
  signal?: AbortSignal;
}
