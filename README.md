,# Steam Oyun Yorum Analizi

## Proje Amacı

Bu proje, Steam platformundaki oyun yorumlarını otomatik olarak toplayıp analiz eden, şikayet kategorilerini tespit eden ve oyun stüdyolarına veri odaklı öneriler sunan büyük ölçekli bir analiz sistemidir.

## Temel Özellikler

- **Veri Toplama:** Selenium ile Steam yorumlarının otomatik scraping'i
- **Dil İşleme:** Çok dilli yorumların tespit edilip Türkçe/İngilizce'ye çevrilmesi
- **Duygu Analizi:** VADER ve NLTK tabanlı pozitif/negatif yorum sınıflandırması
- **Şikayet Kategorileri:** Performans, hikaye, fiyat, teknik sorunlar gibi kategorilerin otomatik tespiti
- **Görselleştirme:** Plotly ve Matplotlib ile interaktif grafikler, kelime bulutları
- **Dashboard:** Streamlit tabanlı canlı analiz paneli

## Klasör Yapısı

```
steam_analiz/
├── data/
│   ├── raw/          # Ham, işlenmemiş yorum verileri
│   └── processed/    # Temizlenmiş ve dönüştürülmüş veriler
├── scripts/          # Veri toplama ve analiz betikleri
├── dashboard/        # Streamlit arayüz dosyaları
└── reports/
    └── figures/      # Üretilen grafikler ve görseller
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
# Dashboard'u başlat
streamlit run dashboard/app.py
```
