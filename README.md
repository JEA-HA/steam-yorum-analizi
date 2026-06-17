# Steam Oyun Yorum Analizi

Bursa Uludağ Üniversitesi, İnegöl İşletme Fakültesi, Yönetim Bilişim Sistemleri bölümü bitirme projesi.

## Proje Amacı

Steam platformundaki oyun yorumlarını büyük veri analizi yöntemleriyle inceleyen, kullanıcı şikayetlerini kategorize eden ve oyun stüdyolarına veri odaklı stratejik öneriler sunan bir sistem.

## Veri Seti

25 farklı oyundan (5 oyun türü × 5 oyun) Steam Web API aracılığıyla toplanan 12.285 gerçek kullanıcı yorumu. Veri seti `data/processed/` klasöründe bulunmaktadır.

## Temel Özellikler

- **Veri Toplama:** Steam Web API ile yorum toplama
- **Dil İşleme:** Çok dilli yorumların tespiti ve İngilizceye çevrilmesi
- **Duygu Analizi:** VADER algoritmasıyla yorum sınıflandırması
- **Şikayet Kategorileri:** 8 kategoride otomatik şikayet tespiti
- **Büyük Veri:** Apache Spark, Hadoop HDFS ve Hive Metastore entegrasyonu
- **Dashboard:** Streamlit tabanlı interaktif analiz paneli
- **SHAP Analizi:** Açıklanabilir yapay zeka ile özellik önem sıralaması

## Klasör Yapısı
steam_analiz/

├── data/raw/          # Ham yorum verileri

├── data/processed/    # İşlenmiş veri seti

├── scripts/           # Veri toplama ve analiz betikleri

├── dashboard/          # Streamlit dashboard

├── notebook/           # Jupyter Notebook (EDA + SHAP)

└── reports/            # Analiz raporları ve görseller

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
streamlit run dashboard/app.py
```

## Geliştirici

Mehmet Mıdıkoğlu — Danışman: Dr. Öğr. Üyesi Melih Engin
