"""
Ön İşleme (Preprocessing) Pipeline
data/raw/ altındaki CSV'leri temizler, dili tespit eder,
İngilizce'ye çevirir ve data/processed/ altına kaydeder.
"""

import os
import re
import time
import pandas as pd
from tqdm import tqdm
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
import nltk

nltk.download("stopwords",  quiet=True)
nltk.download("punkt",      quiet=True)
nltk.download("punkt_tab",  quiet=True)

DATA_RAW_DIR       = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Çeviri için kaynak → hedef dil  (deep-translator dil kodları)
CEVIRI_DILI = "en"

# Bu dillerdeki yorumlar çevrilmez (zaten İngilizce ya da desteklenmiyor)
CEVIRME = {"en", "english"}


# ---------------------------------------------------------------------------
# Metin temizleme
# ---------------------------------------------------------------------------

def metni_temizle(metin: str) -> str:
    """
    Ham yorum metninden HTML artıklarını, özel karakterleri,
    fazla boşlukları ve satır sonlarını temizler.
    """
    if not isinstance(metin, str):
        return ""
    metin = re.sub(r"<[^>]+>",    " ", metin)   # HTML tag'leri kaldır
    metin = re.sub(r"http\S+",    " ", metin)   # URL'leri kaldır
    metin = re.sub(r"[^\w\s.,!?'\"()-]", " ", metin)  # Gereksiz karakterler
    metin = re.sub(r"\s+",        " ", metin)   # Çoklu boşluk → tek boşluk
    return metin.strip()


# ---------------------------------------------------------------------------
# Dil tespiti
# ---------------------------------------------------------------------------

def dil_tespit_et(metin: str) -> str:
    """
    langdetect ile metnin dilini tahmin eder.
    Kısa veya belirsiz metinlerde 'unknown' döner.
    """
    if not metin or len(metin) < 10:
        return "unknown"
    try:
        return detect(metin)
    except LangDetectException:
        return "unknown"


# ---------------------------------------------------------------------------
# Çeviri
# ---------------------------------------------------------------------------

def ingilizceye_cevir(metin: str, kaynak_dil: str) -> str:
    """
    deep-translator / GoogleTranslator ile metni İngilizce'ye çevirir.
    500 karakterden uzun metinleri parçalara böler (API sınırı).
    Hata durumunda orijinal metni döndürür.
    """
    if kaynak_dil in CEVIRME or kaynak_dil == "unknown":
        return metin

    try:
        translator = GoogleTranslator(source="auto", target=CEVIRI_DILI)
        if len(metin) <= 4900:
            return translator.translate(metin)

        # Uzun metni cümle cümle çevir
        parcalar = [metin[i:i+4900] for i in range(0, len(metin), 4900)]
        ceviriler = [translator.translate(p) for p in parcalar]
        return " ".join(ceviriler)
    except Exception:
        return metin   # Hata → orijinalini koru


# ---------------------------------------------------------------------------
# Tek CSV dosyasını işleme
# ---------------------------------------------------------------------------

def dosyayi_isle(dosya_yolu: str) -> pd.DataFrame:
    """
    Ham CSV'yi okur; temizleme, dil tespiti ve çeviri işlemlerini uygular.
    Sonucu DataFrame olarak döndürür.
    """
    df = pd.read_csv(dosya_yolu, encoding="utf-8-sig")

    # Hangi sütun yorum metnini içeriyor?
    metin_sutun = "review_text" if "review_text" in df.columns else "review"
    if metin_sutun not in df.columns:
        print(f"   [UYARI] Metin sütunu bulunamadı: {dosya_yolu}")
        return pd.DataFrame()

    # Boş yorumları düşür
    df = df.dropna(subset=[metin_sutun]).reset_index(drop=True)
    df = df[df[metin_sutun].astype(str).str.len() > 5].reset_index(drop=True)

    temiz_metinler   = []
    tespit_diller    = []
    cevrilmis_metinler = []

    for _, satir in tqdm(df.iterrows(), total=len(df), desc=f"  {os.path.basename(dosya_yolu)}", leave=False):
        ham = str(satir[metin_sutun])
        temiz = metni_temizle(ham)

        # Dil tespiti: önce mevcut 'language' sütununa bak, yoksa tespit et
        mevcut_dil = str(satir.get("language", "")).lower()
        if mevcut_dil and mevcut_dil not in ("nan", "unknown", ""):
            dil = mevcut_dil
        else:
            dil = dil_tespit_et(temiz)

        ceviri = ingilizceye_cevir(temiz, dil)
        time.sleep(0.05)   # API'ye nazik olmak için mini bekleme

        temiz_metinler.append(temiz)
        tespit_diller.append(dil)
        cevrilmis_metinler.append(ceviri)

    df["review_clean"]      = temiz_metinler
    df["detected_language"] = tespit_diller
    df["review_english"]    = cevrilmis_metinler

    return df


# ---------------------------------------------------------------------------
# Tüm ham verileri işle
# ---------------------------------------------------------------------------

def tum_verileri_isle() -> None:
    """
    data/raw/ altındaki tüm CSV'leri okuyup işler
    ve data/processed/ altına kaydeder.
    """
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    dosyalar = [
        os.path.join(DATA_RAW_DIR, f)
        for f in os.listdir(DATA_RAW_DIR)
        if f.endswith(".csv")
    ]

    if not dosyalar:
        print("data/raw/ altında CSV bulunamadı. Önce 1_scraper.py çalıştırın.")
        return

    print(f"{len(dosyalar)} CSV bulundu, işleniyor...\n")
    basarili = 0

    for dosya in dosyalar:
        df = dosyayi_isle(dosya)
        if df.empty:
            continue

        cikis_adi = os.path.basename(dosya).replace(".csv", "_processed.csv")
        cikis_yolu = os.path.join(DATA_PROCESSED_DIR, cikis_adi)
        df.to_csv(cikis_yolu, index=False, encoding="utf-8-sig")
        print(f"  Kaydedildi: {cikis_yolu}  ({len(df)} satır)")
        basarili += 1

    print(f"\nTamamlandı. {basarili}/{len(dosyalar)} dosya işlendi.")


if __name__ == "__main__":
    tum_verileri_isle()
