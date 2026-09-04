# Terminal CLI

Kurulumdan sonra CLI, mevcut Python çekirdeğini çağırır; ayrı bir analiz motoru içermez.

```powershell
python -m pip install -e .
python -m surgical_pipeline doctor
surgery-analyze --help
```

Analiz için örnek:

```powershell
python -m surgical_pipeline analyze `
  --input ".\video.mp4" `
  --health-model ".\models\weights\health_personnel_segmentation.pt" `
  --instrument-model ".\models\weights\surgical_instrument_segmentation.pt" `
  --pose-model ".\models\weights\pose_model.pt" `
  --privacy-mode skeleton-only `
  --output-dir ".\outputs"
```

`skeleton-only`, kaynak RGB görüntüyü ve sesi çıktı videosuna taşımaz. `blur`, health-personnel maskelerini anonimleştiren mevcut pipeline çıktısını üretir. `both` iki mevcut pipeline çıktısını ayrı çalışma dizinlerinde üretir.

Ayar önceliği CLI argümanı, `--config` YAML, ortam değişkeni ve varsayılan config sırasındadır. Model yolları için de aynı sıra kullanılır; son çare olarak doğrulanmış `models/weights` ağırlıkları denenir. Ortam değişkenleri: `HEALTH_MODEL_PATH`, `INSTRUMENT_MODEL_PATH`, `POSE_MODEL_PATH`, `SURGICAL_OUTPUT_DIR`, `SURGICAL_DEVICE`.

`doctor`, Python/işletim sistemi, GPU/CUDA, gerekli paketler, FFmpeg/FFprobe, model rolleri, keypoint şekli, depth durumu ve çıktı dizini yazılabilirliğini kontrol eder. Skeleton hedeflendiğinde pose modeli kritik gereksinimdir.

| Kod | Anlam |
| --- | --- |
| 0 | Başarılı |
| 2 | CLI kullanımı veya config hatası |
| 3 | Girdi video ya da model dosyası bulunamadı |
| 4 | Ortam veya model doğrulama hatası |
| 5 | Analiz/inference hatası |
| 6 | Çıktı yazma/doğrulama hatası |
| 130 | Kullanıcı iptali |

Sık sorunlar: `doctor` içindeki model rolü hatası yanlış `.pt` dosyasının seçildiğini gösterir; `--device cuda` hatası CUDA'nın kullanılabilir olmadığını gösterir; `--no-depth` göreli `z_rel` verisini kasıtlı olarak üretmez. Çıktı dizini hiçbir zaman kaynak video veya model dosyası olmamalıdır.
