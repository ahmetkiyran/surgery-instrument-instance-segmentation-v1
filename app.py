"""Local-only Gradio interface. Run with: python app.py"""

from __future__ import annotations

import html
import json
import threading
from dataclasses import replace
from pathlib import Path

import gradio as gr

from surgical_pipeline.config import load_config
from surgical_pipeline.doctor import run_doctor
from surgical_pipeline.model_manager import ModelManager, ModelManagerError
from surgical_pipeline.pipeline import AnalysisCancelled, AnalysisPipeline

ROOT = Path(__file__).resolve().parent
RUN_LOCK = threading.Lock()
CANCEL_EVENT = threading.Event()


def _checks_markdown() -> str:
    config = load_config(ROOT)
    checks = run_doctor(ROOT, config)
    rows = "\n".join(f"| {'🟢' if item.ok else '🔴'} | {item.label} | {item.detail.replace('|', '/')[:180]} |" for item in checks if item.label in {"Sağlık modeli", "Alet modeli", "CUDA", "Derinlik modeli/cache"})
    return "### Yerel sistem durumu\n\n| Durum | Bileşen | Ayrıntı |\n|---|---|---|\n" + rows


def _models_markdown() -> str:
    checks = run_doctor(ROOT, load_config(ROOT))
    health_ok = next(item.ok for item in checks if item.label == "Sağlık modeli")
    instrument_ok = next(item.ok for item in checks if item.label == "Alet modeli")
    if health_ok and instrument_ok:
        return "### Modeller\n\n✓ Sağlık personeli modeli hazır  \n✓ Cerrahi alet modeli hazır"
    try:
        statuses = ModelManager(ROOT).statuses()
        details = "\n".join(f"- {status.key}: {status.detail}" for status in statuses if not status.ready)
    except ModelManagerError as error:
        details = str(error)
    return "### Modeller\n\nModel ağırlıkları henüz indirilmedi veya doğrulanamadı.\n" + details


def download_models(progress=gr.Progress(track_tqdm=False)) -> str:
    def update(key: str, current: int, total: int | None) -> None:
        ratio = current / total if total else None
        progress(ratio, desc=f"{key} indiriliyor") if ratio is not None else progress(desc=f"{key} indiriliyor")
    try:
        ModelManager(ROOT).download_all(progress=update)
    except ModelManagerError as error:
        return f"### Modeller\n\n❌ {html.escape(str(error))}"
    return _models_markdown()


def _summary_html(summary_path: Path) -> str:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    health = payload["health_person_statistics"]
    instruments = payload.get("instrument_usage", {})
    cards = "".join(f"<div class='metric'><strong>{html.escape(name)}</strong><br>{values['union_usage_seconds']:.2f} sn birleşik süre<br>{values['relative_3d_motion']:.4f} relative motion units</div>" for name, values in instruments.items()) or "<div class='metric'>Alet algılanmadı.</div>"
    return f"<div class='summary'><h3>Analiz özeti</h3><p>Personel — min: {health['minimum']}, max: {health['maximum']}, ort.: {health['average']:.2f}</p><div class='metrics'>{cards}</div><p class='notice'>Kullanım süreleri, modelin aleti görünür olarak algıladığı sürelerdir. 3B değerleri göreli birimlerdir.</p></div>"


def _trajectory_iframe(path: Path) -> str:
    content = html.escape(path.read_text(encoding="utf-8"), quote=True)
    return f"<iframe title='Göreli 3B yörüngeler' srcdoc=\"{content}\" style='width:100%;height:620px;border:0;border-radius:10px'></iframe>"


def analyse(video: str | None, blur_kernel: int, confidence: float, iou: float, tracker: str, depth_stride: int, depth_enabled: bool, progress=gr.Progress(track_tqdm=False)):
    if not video:
        raise gr.Error("Lütfen önce bir video seçin.")
    if not RUN_LOCK.acquire(blocking=False):
        raise gr.Error("Başka bir analiz çalışıyor. Tamamlanmasını veya iptal edilmesini bekleyin.")
    CANCEL_EVENT.clear()
    try:
        base = load_config(ROOT)
        config = replace(
            base,
            blur_kernel=int(blur_kernel),
            confidence=float(confidence),
            iou=float(iou),
            health_confidence=float(confidence),
            health_iou=float(iou),
            instrument_confidence=float(confidence),
            instrument_iou=float(iou),
            tracker_algorithm=tracker,
            depth_stride=int(depth_stride),
            depth_enabled=bool(depth_enabled),
        ).validate()
        def update(current: int, total: int, eta: float, stage: str) -> None:
            progress((current, max(total, 1)), desc=f"{stage} — {current}/{total}, yaklaşık {eta:.0f} sn kaldı")
        result = AnalysisPipeline(ROOT, config).run(Path(video), progress=update, cancel_event=CANCEL_EVENT)
        file_map = {path.name: path for path in result.files}
        files = [str(path) for path in result.files if path.is_file()]
        return (
            "✅ Analiz tamamlandı. Sonuçlar yalnızca bu yerel çalışma klasöründe üretildi.",
            str(result.processed_video),
            _summary_html(result.summary),
            str(file_map["usage_duration_chart.png"]),
            str(file_map["health_person_count_chart.png"]),
            str(file_map["relative_3d_motion_chart.png"]),
            _trajectory_iframe(file_map["trajectories_3d.html"]),
            files,
        )
    except AnalysisCancelled:
        return "⚠️ Analiz iptal edildi; tamamlanmış bir sonuç paketi oluşturulmadı.", None, "", None, None, None, "", []
    except Exception as error:
        # Technical details are in a run-local log if a run directory was created.
        return f"❌ Analiz başlatılamadı: {html.escape(str(error))}", None, "", None, None, None, "", []
    finally:
        RUN_LOCK.release()


def cancel() -> str:
    CANCEL_EVENT.set()
    return "İptal isteği gönderildi; mevcut kare güvenli biçimde tamamlandıktan sonra analiz duracak."


CSS = """
body, .gradio-container {background:#050b17 !important; color:#e8fbff !important}
.gradio-container {max-width:1450px !important}
#title {background:linear-gradient(100deg,#071a31,#0a3440); border:1px solid #16c9d5; border-radius:16px; padding:20px}
.summary,.metric {background:#0b1e31;border:1px solid #1f6372;border-radius:12px;padding:14px;margin:8px 0}.metrics{display:flex;gap:10px;flex-wrap:wrap}.metric{min-width:200px}.notice{color:#94dce4}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Cerrahi Video Analitiği") as demo:
        gr.HTML("<div id='title'><h1>◈ Cerrahi Video Analitiği</h1><p>Yerel çalışır · Kimliksizleştirme öncelikli · Göreli 3B takip</p></div>")
        gr.Markdown(_checks_markdown())
        model_status = gr.Markdown(_models_markdown())
        download_button = gr.Button("Modelleri İndir", variant="secondary")
        with gr.Row():
            with gr.Column(scale=2):
                video = gr.Video(label="İncelenecek cerrahi video", sources=["upload"])
                start = gr.Button("Analizi Başlat", variant="primary", size="lg")
                stop = gr.Button("İptal Et", variant="stop")
                message = gr.Markdown("Video seçin ve analizi başlatın.")
            with gr.Column():
                blur = gr.Slider(15, 121, value=51, step=2, label="Blur seviyesi (Gaussian kernel)")
                confidence = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="Confidence")
                iou = gr.Slider(0.05, 0.95, value=0.45, step=0.05, label="IoU")
                tracker = gr.Dropdown(["botsort", "bytetrack"], value="botsort", label="Tracker")
                depth_stride = gr.Slider(1, 12, value=3, step=1, label="Depth stride")
                depth_enabled = gr.Checkbox(value=True, label="Göreli 3B tracking etkin")
        with gr.Tabs():
            with gr.Tab("İşlenmiş video"):
                output_video = gr.Video(label="Kimliksizleştirilmiş çıktı")
            with gr.Tab("Özet"):
                summary = gr.HTML()
            with gr.Tab("Alet kullanım grafiği"):
                usage_chart = gr.Image(type="filepath")
            with gr.Tab("Personel sayısı grafiği"):
                health_chart = gr.Image(type="filepath")
            with gr.Tab("Göreli 3B hareket"):
                motion_chart = gr.Image(type="filepath")
            with gr.Tab("Etkileşimli 3B yörünge"):
                trajectory = gr.HTML()
            with gr.Tab("Dosyaları indir"):
                downloads = gr.File(label="Tekil dosyalar ve results.zip", file_count="multiple")
        start.click(analyse, [video, blur, confidence, iou, tracker, depth_stride, depth_enabled], [message, output_video, summary, usage_chart, health_chart, motion_chart, trajectory, downloads])
        stop.click(cancel, outputs=message)
        download_button.click(download_models, outputs=model_status)
    return demo.queue(default_concurrency_limit=1, max_size=4)


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False, inbrowser=False, show_error=False, css=CSS)
