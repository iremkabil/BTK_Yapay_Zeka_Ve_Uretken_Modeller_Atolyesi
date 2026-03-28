# BTK Yapay Zeka ve Üretken Modeller Atölyesi

Bu repo, **BTK Akademi — Yapay Zeka ve Üretken Modeller Atölyesi** kapsamında gerçekleştirilen haftalık çalışmaları içermektedir.

## İçindekiler

| Dosya                                             | Açıklama                                         |
| ------------------------------------------------- | ------------------------------------------------ |
| `ai_atolye2_2026.ipynb`                           | Hafta 2 — Atölye çalışması (Notebook)            |
| `ai_atolye2_2026.pdf`                             | Hafta 2 — Atölye Notları (PDF)                   |
| `ai_atolye3_2026.ipynb`                           | Hafta 3 — Atölye çalışması (Notebook)            |
| `ai_atolye3_2026.pdf`                             | Hafta 3 — Atölye Notları (PDF)                   |
| `ai_atolye4_2026.ipynb`                           | Hafta 4 — Atölye çalışması (Notebook)            |
| `ai_atolye4_2026.pdf`                             | Hafta 4 — Atölye Notları (PDF)                   |
| `ai_atolye5_2026.ipynb`                           | Hafta 5 — Atölye çalışması (Notebook)            |
| `ai_atolye5_2026.pdf`                             | Hafta 5 — Atölye Notları (PDF)                   |
| `ai_atolye6_2026.ipynb`                           | Hafta 6 — FastText ile Metin Sınıflandırma       |
| `FastText_RandomForest_Metin_Siniflandirma.ipynb` | FastText + Random Forest ile Metin Sınıflandırma |
| `Tomato_leaf_disease_detection.ipynb`             | Domates Yaprak Hastalığı Tespiti                 |
| `ogrenci_notlari.csv`                             | Öğrenci notları veri seti                        |
| `satislar.csv`                                    | Satış verileri veri seti                         |
| `titanic.csv`                                     | Titanic veri seti                                |

## FastText ile Metin Sınıflandırma

Bu projede **20 Newsgroups** veri seti üzerinde, FastText word embedding'leri ve Random Forest sınıflandırıcı kullanılarak metin sınıflandırma yapılmıştır.

### Seçilen Kategoriler

- `sci.electronics`
- `sci.space`
- `comp.windows.x`
- `rec.motorcycles`

### Pipeline

```
Metin → Preprocessing (lowercase, regex, tokenize, stem, stopword)
      → FastText Embedding (kelime vektörleri)
      → Mean Pooling (doküman vektörü)
      → Random Forest → Tahmin
```

### Kullanılan Teknolojiler

- **FastText** — Facebook'un pre-trained `cc.en.300.bin` modeli ile kelime embedding'leri
- **Random Forest** — Ensemble sınıflandırma modeli
- **NLTK** — Metin ön işleme (tokenization, stemming, stopword removal)
- **scikit-learn** — Model eğitimi, değerlendirme metrikleri, train-test split

### Çalıştırma

1. [Google Colab](https://colab.research.google.com/) üzerinde `.ipynb` dosyasını açın
2. Hücreleri sırasıyla çalıştırın
3. FastText modeli (~4.5 GB) ilk hücrede otomatik indirilir

## Gereksinimler

```
numpy
scikit-learn
nltk
fasttext
```

## Yazar

**İrem Kabil**

---

> Bu repo, BTK Akademi eğitim programı süresince güncellenmektedir.
