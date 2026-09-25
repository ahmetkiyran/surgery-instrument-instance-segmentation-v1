import { ApiError } from "./errors";
import type {
  Artifact,
  AnalysisSession,
  CreateJobInput,
  Defaults,
  Doctor,
  Health,
  Job,
  LocalVideoInspection,
  ModelStatus,
  UploadInput,
  UploadResult,
  SelectionEvent,
} from "./types";

export interface ApiClientOptions {
  baseUrl: string;
  token?: string;
  fetchImpl?: typeof fetch;
}

type Json = Record<string, unknown> | unknown[];

export class SurgicalApiClient {
  private readonly baseUrl: string;
  private readonly token?: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ApiClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.token = options.token?.trim() || undefined;
    // WebView (including Tauri's Windows WebView2) requires fetch to retain
    // its global/window receiver. A detached `fetch` reference throws
    // "Illegal invocation" before the request reaches the local API.
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  withConnection(options: Pick<ApiClientOptions, "baseUrl" | "token">): SurgicalApiClient {
    return new SurgicalApiClient({ ...options, fetchImpl: this.fetchImpl });
  }

  async checkHealth(): Promise<Health> {
    return this.request<Health>("/api/v1/health");
  }

  async runDoctor(): Promise<Doctor> {
    return this.request<Doctor>("/api/v1/system/doctor");
  }

  async listModels(): Promise<ModelStatus[]> {
    return this.request<ModelStatus[]>("/api/v1/models");
  }

  async getDefaults(): Promise<Defaults> {
    return this.request<Defaults>("/api/v1/config/defaults");
  }

  async uploadVideo(input: UploadInput): Promise<UploadResult> {
    const total = input.data instanceof Blob ? input.data.size : input.data.byteLength;
    input.onProgress?.(0, total);
    if (typeof XMLHttpRequest !== "undefined" && input.data instanceof Blob) {
      return this.uploadWithXhr(input, total);
    }
    const result = await this.request<UploadResult>("/api/v1/uploads", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-File-Name": input.filename },
      body: input.data as unknown as BodyInit,
      signal: input.signal,
    });
    input.onProgress?.(total, total);
    return result;
  }

  async inspectDesktopVideo(localPath: string): Promise<LocalVideoInspection> {
    return this.request<LocalVideoInspection>("/api/v1/videos/inspect", {
      method: "POST",
      body: JSON.stringify({ local_path: localPath }),
    });
  }

  async createJob(input: CreateJobInput): Promise<Job> {
    return this.request<Job>("/api/v1/jobs", { method: "POST", body: JSON.stringify(input) });
  }

  async createSession(): Promise<AnalysisSession> {
    return this.request<AnalysisSession>("/api/v1/sessions", { method: "POST" });
  }

  async attachSessionVideo(sessionId: string, input: CreateJobInput): Promise<AnalysisSession> {
    return this.request<AnalysisSession>(`/api/v1/sessions/${encodeURIComponent(sessionId)}/video`, { method: "POST", body: JSON.stringify(input) });
  }

  async addSelection(sessionId: string, event: SelectionEvent): Promise<AnalysisSession> {
    return this.request<AnalysisSession>(`/api/v1/sessions/${encodeURIComponent(sessionId)}/selections`, { method: "POST", body: JSON.stringify(event) });
  }

  async startSession(sessionId: string, input: CreateJobInput): Promise<AnalysisSession> {
    return this.request<AnalysisSession>(`/api/v1/sessions/${encodeURIComponent(sessionId)}/start`, { method: "POST", body: JSON.stringify(input) });
  }

  async getSession(sessionId: string): Promise<AnalysisSession> {
    return this.request<AnalysisSession>(`/api/v1/sessions/${encodeURIComponent(sessionId)}/status`);
  }

  async removeSelection(sessionId: string, eventId: string): Promise<AnalysisSession> {
    return this.request<AnalysisSession>(`/api/v1/sessions/${encodeURIComponent(sessionId)}/selections/${encodeURIComponent(eventId)}/remove`, { method: "POST" });
  }

  async setSessionPaused(sessionId: string, paused: boolean): Promise<AnalysisSession> {
    return this.request<AnalysisSession>(`/api/v1/sessions/${encodeURIComponent(sessionId)}/${paused ? "pause" : "resume"}`, { method: "POST" });
  }

  async sessionPreview(sessionId: string, frameIndex: number, overlay: "clean" | "inspection" | "privacy-xray" = "clean"): Promise<Blob> {
    const response = await this.rawRequest(`/api/v1/sessions/${encodeURIComponent(sessionId)}/preview?frame_index=${Math.max(0, Math.floor(frameIndex))}&overlay=${encodeURIComponent(overlay)}`);
    if (!response.ok) throw await this.errorFrom(response);
    return response.blob();
  }

  async listJobs(): Promise<Job[]> {
    return this.request<Job[]>("/api/v1/jobs");
  }

  async getJob(jobId: string): Promise<Job> {
    return this.request<Job>(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
  }

  async cancelJob(jobId: string): Promise<Job> {
    return this.request<Job>(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
  }

  async pauseJob(jobId: string): Promise<Job> {
    return this.request<Job>(`/api/v1/jobs/${encodeURIComponent(jobId)}/pause`, { method: "POST" });
  }

  async resumeJob(jobId: string): Promise<Job> {
    return this.request<Job>(`/api/v1/jobs/${encodeURIComponent(jobId)}/resume`, { method: "POST" });
  }

  async listArtifacts(jobId: string): Promise<Artifact[]> {
    return this.request<Artifact[]>(`/api/v1/jobs/${encodeURIComponent(jobId)}/artifacts`);
  }

  async downloadArtifact(jobId: string, artifactId: string, sensitiveConfirmed = false): Promise<Blob> {
    const response = await this.rawRequest(`/api/v1/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(artifactId)}`, {
      headers: sensitiveConfirmed ? { "X-Sensitive-Artifact-Confirmed": "true" } : undefined,
    });
    if (!response.ok) throw await this.errorFrom(response);
    return response.blob();
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await this.rawRequest(path, init);
    if (!response.ok) throw await this.errorFrom(response);
    return (await response.json()) as T;
  }

  private rawRequest(path: string, init: RequestInit = {}): Promise<Response> {
    const headers = new Headers(init.headers);
    if (this.token) headers.set("Authorization", `Bearer ${this.token}`);
    if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    return this.fetchImpl(`${this.baseUrl}${path}`, { ...init, headers });
  }

  private async errorFrom(response: Response): Promise<ApiError> {
    let message = "Sunucu isteği tamamlayamadı.";
    try {
      const data = (await response.json()) as { detail?: string };
      if (data.detail) message = data.detail;
    } catch {
      // Never substitute a raw server body, which could contain implementation details.
    }
    return new ApiError(message, response.status);
  }

  private uploadWithXhr(input: UploadInput, total: number): Promise<UploadResult> {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open("POST", `${this.baseUrl}/api/v1/uploads`);
      if (this.token) request.setRequestHeader("Authorization", `Bearer ${this.token}`);
      request.setRequestHeader("Content-Type", "application/octet-stream");
      request.setRequestHeader("X-File-Name", input.filename);
      request.upload.onprogress = (event) => input.onProgress?.(event.loaded, event.lengthComputable ? event.total : total);
      request.onerror = () => reject(new ApiError("Yükleme bağlantısı kesildi.", 0));
      input.signal?.addEventListener("abort", () => request.abort(), { once: true });
      request.onabort = () => reject(new DOMException("Yükleme iptal edildi.", "AbortError"));
      request.onload = () => {
        if (request.status < 200 || request.status >= 300) {
          reject(new ApiError("Video yüklenemedi.", request.status));
          return;
        }
        try {
          resolve(JSON.parse(request.responseText) as UploadResult);
        } catch {
          reject(new ApiError("Sunucu geçersiz bir upload yanıtı verdi.", request.status));
        }
      };
      request.send(input.data as unknown as XMLHttpRequestBodyInit);
    });
  }
}
