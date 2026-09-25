#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::{net::TcpListener, path::PathBuf, process::{Child, Command}, sync::Mutex};
use tauri::{Manager, State};

struct ManagedApi(Mutex<Option<Child>>);

impl Drop for ManagedApi {
    fn drop(&mut self) {
        if let Ok(slot) = self.0.get_mut() {
            if let Some(mut child) = slot.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ManagedConnection {
    base_url: String,
    token: String,
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..").canonicalize().unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."))
}

#[tauri::command]
fn start_managed_api(state: State<'_, ManagedApi>) -> Result<ManagedConnection, String> {
    let mut child_slot = state.0.lock().map_err(|_| "API durumu kilitlenemedi.")?;
    if child_slot.as_mut().is_some_and(|child| child.try_wait().ok().flatten().is_none()) {
        return Err("Yönetilen API zaten çalışıyor; mevcut oturum bağlantısını kullanın.".into());
    }
    *child_slot = None;
    let root = project_root();
    let configured = std::env::var_os("SAM3_PYTHON_PATH").map(PathBuf::from);
    let sibling = root.parent()
        .map(|parent| parent.join("sam3tracking").join(".venv").join("Scripts").join("python.exe"));
    let local = root.join(".venv").join("Scripts").join("python.exe");
    let executable = configured
        .filter(|path| path.is_file())
        .or_else(|| sibling.filter(|path| path.is_file()))
        .or_else(|| local.is_file().then_some(local))
        .unwrap_or_else(|| PathBuf::from("python"));
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|_| "Yerel API için bir loopback port ayrılamadı.")?;
    let port = listener.local_addr().map_err(|_| "Yerel port okunamadı.")?.port();
    drop(listener);
    let token = uuid::Uuid::new_v4().simple().to_string() + &uuid::Uuid::new_v4().simple().to_string();
    let child = Command::new(executable)
        .current_dir(&root)
        .args(["-m", "surgical_pipeline.server", "--host", "127.0.0.1", "--port", &port.to_string(), "--token", &token, "--log-level", "warning"])
        .spawn()
        .map_err(|_| "Python API başlatılamadı. .venv ve Python server bağımlılıklarını kontrol edin.")?;
    *child_slot = Some(child);
    Ok(ManagedConnection { base_url: format!("http://127.0.0.1:{port}"), token })
}

#[tauri::command]
fn stop_managed_api(state: State<'_, ManagedApi>) -> Result<(), String> {
    if let Some(mut child) = state.0.lock().map_err(|_| "API durumu kilitlenemedi.")?.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .manage(ManagedApi(Mutex::new(None)))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![start_managed_api, stop_managed_api])
        .run(tauri::generate_context!())
        .expect("Tauri uygulaması başlatılamadı");
}
