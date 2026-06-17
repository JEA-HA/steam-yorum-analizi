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

## Dashboard Görselleri
<img width="1920" height="970" alt="1" src="https://github.com/user-attachments/assets/a86f2f50-2e5c-458f-b109-b8afad92eaa0" />
<img width="1920" height="3677" alt="2" src="https://github.com/user-attachments/assets/17739c9c-4f62-4117-a78b-36db43447880" />
<img width="1920" height="2370" alt="3" src="https://github.com/user-attachments/assets/f9057b7e-e3c1-479b-a145-3a80ad658901" />
<img width="1920" height="2063" alt="4" src="https://github.com/user-attachments/assets/0c9d2318-1ad6-48be-b407-11d2355e66b4" />
<img width="1920" height="2483" alt="5" src="https://github.com/user-attachments/assets/9a607c01-426b-4712-8cd2-fa2ae43d91c0" />
<img width="1920" height="2420" alt="6" src="https://github.com/user-attachments/assets/4038fbf8-67ce-4bd3-b346-612fde3bcc84" />
<img width="1920" height="2606" alt="7" src="https://github.com/user-attachments/assets/a937d3f0-7ad5-4063-afd5-3f9434c9f0ce" />
<img width="1920" height="2435" alt="8" src="https://github.com/user-attachments/assets/b950d8c6-1f47-47fb-aa13-d4a1c4ac74a0" />













## Geliştirici

Mehmet Mıdıkoğlu — Danışman: Prof. Dr. Melih Engin
