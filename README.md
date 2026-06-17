# 🎮 Steam Oyun Yorum Analizi

> Bursa Uludağ Üniversitesi – İnegöl İşletme Fakültesi
> Yönetim Bilişim Sistemleri Bölümü Bitirme Projesi

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Apache Spark](https://img.shields.io/badge/Apache-Spark-orange.svg)
![Hadoop](https://img.shields.io/badge/Hadoop-HDFS-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)

---

## 📋 İçindekiler

* [Proje Amacı](#-proje-amacı)
* [Veri Seti](#-veri-seti)
* [Temel Özellikler](#-temel-özellikler)
* [Klasör Yapısı](#-klasör-yapısı)
* [Kurulum](#-kurulum)
* [Kullanım](#-kullanım)
* [Dashboard Görselleri](#-dashboard-görselleri)
* [Geliştirici](#-geliştirici)

---

## 🎯 Proje Amacı

Steam platformundaki oyun yorumlarını büyük veri analizi yöntemleriyle inceleyen, kullanıcı şikayetlerini kategorize eden ve oyun stüdyolarına veri odaklı stratejik öneriler sunan bir sistem.

---

## 📊 Veri Seti

25 farklı oyundan (**5 oyun türü × 5 oyun**) Steam Web API aracılığıyla toplanan **12.285 gerçek kullanıcı yorumu**.

Veri seti:

```text
data/processed/
```

klasöründe bulunmaktadır.

---

## 🚀 Temel Özellikler

### 📥 Veri Toplama

* Steam Web API ile yorum toplama

### 🌍 Dil İşleme

* Çok dilli yorumların tespiti
* İngilizceye otomatik çeviri

### 😊 Duygu Analizi

* VADER algoritmasıyla yorum sınıflandırması

### 🏷️ Şikayet Kategorileri

* 8 kategoride otomatik şikayet tespiti

### ⚡ Büyük Veri Altyapısı

* Apache Spark
* Hadoop HDFS
* Hive Metastore entegrasyonu

### 📈 Dashboard

* Streamlit tabanlı interaktif analiz paneli

### 🤖 Açıklanabilir Yapay Zeka

* SHAP analizi ile özellik önem sıralaması

---

## 📁 Klasör Yapısı

```text
steam_analiz/

├── data/
│   ├── raw/              # Ham yorum verileri
│   └── processed/        # İşlenmiş veri seti
│
├── scripts/              # Veri toplama ve analiz betikleri
│
├── dashboard/            # Streamlit dashboard
│
├── notebook/             # Jupyter Notebook (EDA + SHAP)
│
└── reports/              # Analiz raporları ve görseller
```

---

## ⚙️ Kurulum

```bash
pip install -r requirements.txt
```

---

## ▶️ Kullanım

```bash
streamlit run dashboard/app.py
```

---

## 📸 Dashboard Görselleri

### Genel Dashboard Görünümü

<p align="center">
  <img src="https://github.com/user-attachments/assets/a86f2f50-2e5c-458f-b109-b8afad92eaa0" width="100%">
</p>

### Analiz Ekranları

<p align="center">
  <img src="https://github.com/user-attachments/assets/17739c9c-4f62-4117-a78b-36db43447880" width="100%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/f9057b7e-e3c1-479b-a145-3a80ad658901" width="100%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/0c9d2318-1ad6-48be-b407-11d2355e66b4" width="100%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/9a607c01-426b-4712-8cd2-fa2ae43d91c0" width="100%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/4038fbf8-67ce-4bd3-b346-612fde3bcc84" width="100%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/a937d3f0-7ad5-4063-afd5-3f9434c9f0ce" width="100%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b950d8c6-1f47-47fb-aa13-d4a1c4ac74a0" width="100%">
</p>

---

## 👨‍💻 Geliştirici

**Mehmet Mıdıkoğlu**

**Danışman:** Prof. Dr. Melih Engin

Bursa Uludağ Üniversitesi
İnegöl İşletme Fakültesi
Yönetim Bilişim Sistemleri Bölümü
