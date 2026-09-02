# Katkıda bulunma

Katkı göndermeden önce video, `.env`, `outputs/`, log veya kişisel yol eklemediğinizi doğrulayın. Yalnızca `models/weights/` altındaki iki açıkça yetkilendirilmiş bundle ağırlık istisnadır; başka model dosyası eklemeyin.

1. Python 3.11 ortamını kurun.
2. `python -m pytest -v` ve `python -m ruff check .` çalıştırın.
3. Değişikliğin gizlilik, model sınıfı çözümleme ve çıktı şemasına etkisini açıklayın.
4. Küçük, odaklı pull request gönderin.

Hasta/ameliyat görüntüleri, `.env` veya yetkilendirilmemiş model ağırlıkları içeren issue ya da PR kabul edilmez.
