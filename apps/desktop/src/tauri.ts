import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openPath } from "@tauri-apps/plugin-opener";

export interface ManagedConnection {
  baseUrl: string;
  token: string;
}

export async function startManagedApi(): Promise<ManagedConnection> {
  return invoke<ManagedConnection>("start_managed_api");
}

export async function stopManagedApi(): Promise<void> {
  await invoke("stop_managed_api");
}

export async function chooseVideo(): Promise<string | null> {
  const selected = await open({
    title: "Cerrahi video seç",
    multiple: false,
    filters: [{ name: "Video", extensions: ["mp4", "mov", "avi", "mkv", "webm", "m4v"] }],
  });
  return typeof selected === "string" ? selected : null;
}

export async function chooseDirectory(): Promise<string | null> {
  const selected = await open({ title: "Sonuç klasörünü seç", directory: true, multiple: false });
  return typeof selected === "string" ? selected : null;
}

export async function saveBlob(blob: Blob, filename: string): Promise<void> {
  const path = await save({ defaultPath: filename });
  if (!path) return;
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = path.split(/[\\/]/).pop() || filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export async function revealDirectory(path: string): Promise<void> {
  await openPath(path);
}
