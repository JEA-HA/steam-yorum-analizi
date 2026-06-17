"""
Duygu Analizi — VADER
data/processed/*_processed.csv → data/processed/*_sentiment.csv

Her yoruma dört VADER skoru atar:
  neg / neu / pos  : 0–1 arası oran
  compound         : -1 (çok negatif) → +1 (çok pozitif)
Eşik değerlerine göre 'positive' / 'negative' / 'neutral' etiketi verir.
"""

import os
import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

POZITIF_ESIK =  0.05   # compound >= +0.05 → positive
NEGATIF_ESIK = -0.05   # compound <= -0.05 → negative

analyzer = SentimentIntensityAnalyzer()

# ---------------------------------------------------------------------------
# Skor & sınıflandırma
# ---------------------------------------------------------------------------

def vader_skoru(metin: str) -> dict:
    """
    Metni VADER analizörüne gönderir; neg/neu/pos/compound skorlarını döndürür.
    Boş veya geçersiz metin için sıfır sözlüğü döner.
    """
    if not isinstance(metin, str) or not metin.strip():
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    return analyzer.polarity_scores(metin)


def etiketle(compound: float) -> str:
    """compound skoruna göre duygu etiketi döndürür."""
    if compound >= POZITIF_ESIK:
        return "positive"
    if compound <= NEGATIF_ESIK:
        return "negative"
    return "neutral"


def voted_up_uyumu(etiket: str, voted_up) -> int:
    """
    Steam'in kendi tavsiye bilgisi (voted_up) ile VADER tahmini örtüşüyor mu?
    1 = örtüşüyor, 0 = örtüşmüyor, -1 = karşılaştırılamaz (eksik veri).
    """
    if pd.isna(voted_up):
        return -1
    steam_pozitif = bool(voted_up)
    vader_pozitif = etiket == "positive"
    return 1 if steam_pozitif == vader_pozitif else 0

# ---------------------------------------------------------------------------
# Tek dosya işleme
# ---------------------------------------------------------------------------

def sentiment_ekle(dosya_yolu: str) -> pd.DataFrame:
    """
    Bir *_processed.csv okur, her satıra VADER skorları ve duygu etiketi ekler.
    Çevrilmiş metni (review_english) tercih eder; yoksa review_clean veya review kullanır.
    """
    df = pd.read_csv(dosya_yolu, encoding="utf-8-sig", on_bad_lines='skip')

    # Hangi metin sütununu kullanacağız?
    metin_sutun = next(
        (s for s in ("review_english", "review_clean", "review_text", "review")
         if s in df.columns),
        None,
    )
    if metin_sutun is None:
        print(f"   [UYARI] Metin sütunu bulunamadı: {dosya_yolu}")
        return pd.DataFrame()

    neg_list = []
    neu_list = []
    pos_list = []
    compound_list = []
    etiket_list = []
    uyum_list = []

    for _, satir in tqdm(df.iterrows(), total=len(df),
                         desc=f"  {os.path.basename(dosya_yolu)}", leave=False):
        skor   = vader_skoru(str(satir[metin_sutun]))
        etiket = etiketle(skor["compound"])
        uyum   = voted_up_uyumu(etiket, satir.get("voted_up"))

        neg_list.append(round(skor["neg"],      4))
        neu_list.append(round(skor["neu"],      4))
        pos_list.append(round(skor["pos"],      4))
        compound_list.append(round(skor["compound"], 4))
        etiket_list.append(etiket)
        uyum_list.append(uyum)

    df["vader_neg"]          = neg_list
    df["vader_neu"]          = neu_list
    df["vader_pos"]          = pos_list
    df["vader_compound"]     = compound_list
    df["sentiment"]          = etiket_list
    df["vader_voted_match"]  = uyum_list

    return df

# ---------------------------------------------------------------------------
# Tüm dosyaları işle
# ---------------------------------------------------------------------------

def tum_sentiment_analizi() -> None:
    """
    data/processed/ altındaki *_processed.csv dosyalarının tamamını işler.
    Her dosya için *_sentiment.csv çıkarır ve kısa bir özet basar.
    """
    dosyalar = sorted([
        os.path.join(DATA_PROCESSED_DIR, f)
        for f in os.listdir(DATA_PROCESSED_DIR)
        if f.endswith("_processed.csv")
    ])

    if not dosyalar:
        print("İşlenecek dosya bulunamadı. Önce 2_preprocess.py çalıştırın.")
        return

    print(f"{len(dosyalar)} dosya bulundu, duygu analizi başlıyor...\n")

    for dosya in dosyalar:
        df = sentiment_ekle(dosya)
        if df.empty:
            continue

        cikis = os.path.basename(dosya).replace("_processed.csv", "_sentiment.csv")
        cikis_yolu = os.path.join(DATA_PROCESSED_DIR, cikis)
        df.to_csv(cikis_yolu, index=False, encoding="utf-8-sig")

        dagılım = df["sentiment"].value_counts().to_dict()
        uyum_df  = df[df["vader_voted_match"] >= 0]
        uyum_oran = uyum_df["vader_voted_match"].mean() * 100 if len(uyum_df) else 0
        print(f"  {cikis}")
        print(f"    Dağılım        : {dagılım}")
        print(f"    Steam↔VADER uyumu: %{uyum_oran:.1f}\n")

    print("Tamamlandı.")


if __name__ == "__main__":
    tum_sentiment_analizi()
