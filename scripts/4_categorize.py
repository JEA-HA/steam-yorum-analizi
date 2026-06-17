"""
Şikayet Kategorizasyonu — Anahtar Kelime Eşleştirme
data/processed/*_sentiment.csv → data/processed/*_categorized.csv

Her yoruma çoklu kategori atanır (birden fazla etiket mümkün).
Hiçbir kategoriyle eşleşmeyen yorumlar 'diger' olarak işaretlenir.
Sonunda oyun bazlı özet rapor reports/ altına kaydedilir.
"""

import os
import re
import json
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dizin yolları
# ---------------------------------------------------------------------------

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
REPORTS_DIR        = os.path.join(os.path.dirname(__file__), "..", "reports")

# ---------------------------------------------------------------------------
# Kategori → anahtar kelime sözlüğü
# ---------------------------------------------------------------------------

KATEGORILER: dict[str, list[str]] = {
    "performans": [
        "fps", "lag", "stutter", "stuttering", "freeze", "freezing",
        "performance", "optimization", "unoptimized", "frame rate",
        "framerate", "low fps", "frame drop", "slow", "loading",
        "load time", "memory leak", "cpu", "gpu", "ram",
    ],
    "teknik_sorun": [
        "bug", "bugs", "glitch", "glitches", "broken", "error",
        "crash", "crashing", "crashes", "fix", "patch", "issue",
        "not working", "doesn't work", "won't launch", "won't start",
        "black screen", "corrupted", "save file", "progress lost",
    ],
    "fiyat": [
        "price", "expensive", "overpriced", "cheap", "worth",
        "value for money", "not worth", "cost", "pay", "paid",
        "dlc", "microtransaction", "microtransactions", "pay to win",
        "p2w", "season pass", "battle pass", "paywall", "monetization",
    ],
    "hikaye_icerik": [
        "story", "plot", "narrative", "ending", "character", "characters",
        "writing", "dialogue", "lore", "world building", "boring story",
        "generic", "cliche", "shallow", "repetitive", "no content",
        "empty", "lifeless", "hollow",
    ],
    "oynanabilirlik": [
        "gameplay", "mechanic", "mechanics", "controls", "control",
        "clunky", "combat", "difficulty", "too easy", "too hard",
        "grind", "grinding", "tedious", "balance", "unbalanced",
        "satisfying", "fun", "addictive", "skill",
    ],
    "cok_oyunculu": [
        "multiplayer", "online", "server", "servers", "matchmaking",
        "queue", "toxic", "toxicity", "cheater", "cheaters", "hacker",
        "hackers", "anti-cheat", "community", "co-op", "coop",
        "pvp", "ranked", "ping", "latency", "disconnect",
    ],
    "destek_guncelleme": [
        "developer", "developers", "devs", "support", "update", "updates",
        "abandoned", "dead game", "no updates", "communication",
        "early access", "unfinished", "incomplete", "roadmap",
        "promised", "broken promise", "listen to community",
    ],
    "gorsel_ses": [
        "graphics", "graphic", "visuals", "visual", "art style",
        "beautiful", "ugly", "texture", "textures", "resolution",
        "sound", "music", "audio", "voice acting", "soundtrack",
        "atmosphere", "immersive", "immersion",
    ],
}

# ---------------------------------------------------------------------------
# Eşleştirme
# ---------------------------------------------------------------------------

def kategorileri_bul(metin: str) -> list[str]:
    """
    Metni küçük harfe çevirip her kategorinin anahtar kelimeleriyle karşılaştırır.
    Eşleşen kategori adlarını liste olarak döndürür; eşleşme yoksa ['diger'].
    """
    if not isinstance(metin, str) or not metin.strip():
        return ["diger"]

    kucuk = metin.lower()
    bulunan = []

    for kategori, kelimeler in KATEGORILER.items():
        for kelime in kelimeler:
            # Kelime sınırlarını koruyarak ara (örn. "bug" → "bugs" de yakalansın)
            if re.search(r"\b" + re.escape(kelime) + r"\b", kucuk):
                bulunan.append(kategori)
                break

    return bulunan if bulunan else ["diger"]


def ikili_sutunlar_olustur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her kategori için 0/1 ikili sütun ekler.
    Makine öğrenmesi modellerine hazır çoklu etiket formatı üretir.
    """
    for kat in list(KATEGORILER.keys()) + ["diger"]:
        df[f"cat_{kat}"] = df["categories"].apply(lambda lst: int(kat in lst))
    return df

# ---------------------------------------------------------------------------
# Tek dosya işleme
# ---------------------------------------------------------------------------

def dosyayi_kategorize_et(dosya_yolu: str) -> pd.DataFrame:
    """
    Bir *_sentiment.csv okur, her yoruma kategori listesi atar,
    ikili sütunlar ekler ve döndürür.
    """
    df = pd.read_csv(dosya_yolu, encoding="utf-8-sig")

    metin_sutun = next(
        (s for s in ("review_english", "review_clean", "review_text", "review")
         if s in df.columns),
        None,
    )
    if metin_sutun is None:
        print(f"   [UYARI] Metin sütunu bulunamadı: {dosya_yolu}")
        return pd.DataFrame()

    kategoriler = []
    for _, satir in tqdm(df.iterrows(), total=len(df),
                         desc=f"  {os.path.basename(dosya_yolu)}", leave=False):
        kategoriler.append(kategorileri_bul(str(satir[metin_sutun])))

    df["categories"]     = kategoriler
    df["categories_str"] = df["categories"].apply("|".join)
    df["category_count"] = df["categories"].apply(len)

    df = ikili_sutunlar_olustur(df)
    return df

# ---------------------------------------------------------------------------
# Özet istatistik
# ---------------------------------------------------------------------------

def ozet_hesapla(df: pd.DataFrame, oyun_adi: str) -> dict:
    """Her kategorinin görülme sayısını ve toplam içindeki yüzdesini döndürür."""
    toplam = len(df)
    ozet = {"oyun": oyun_adi, "toplam_yorum": toplam}
    for kat in list(KATEGORILER.keys()) + ["diger"]:
        sutun = f"cat_{kat}"
        if sutun in df.columns:
            n = int(df[sutun].sum())
            ozet[kat]            = n
            ozet[f"{kat}_pct"]   = round(n / toplam * 100, 1) if toplam else 0
    return ozet

# ---------------------------------------------------------------------------
# Tüm dosyaları işle
# ---------------------------------------------------------------------------

def tum_kategorileri_isle() -> None:
    """
    data/processed/ altındaki *_sentiment.csv dosyalarını sırayla kategorize eder.
    Her oyun için *_categorized.csv kaydeder; sonunda özet JSON ve CSV üretir.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    dosyalar = sorted([
        os.path.join(DATA_PROCESSED_DIR, f)
        for f in os.listdir(DATA_PROCESSED_DIR)
        if f.endswith("_sentiment.csv")
    ])

    if not dosyalar:
        print("İşlenecek dosya bulunamadı. Önce 3_sentiment.py çalıştırın.")
        return

    print(f"{len(dosyalar)} dosya bulundu, kategorizasyon başlıyor...\n")
    tum_ozetler = []

    for dosya in dosyalar:
        df = dosyayi_kategorize_et(dosya)
        if df.empty:
            continue

        cikis_adi  = os.path.basename(dosya).replace("_sentiment.csv", "_categorized.csv")
        cikis_yolu = os.path.join(DATA_PROCESSED_DIR, cikis_adi)
        df.to_csv(cikis_yolu, index=False, encoding="utf-8-sig")

        oyun_adi = cikis_adi.replace("_categorized.csv", "")
        ozet     = ozet_hesapla(df, oyun_adi)
        tum_ozetler.append(ozet)

        # En sık 3 kategoriyi göster
        en_sik = sorted(
            [(k, ozet[k]) for k in KATEGORILER if k in ozet],
            key=lambda x: x[1], reverse=True
        )[:3]
        print(f"  {cikis_adi}  →  {len(df)} satır")
        print(f"    En sık: {en_sik}\n")

    # Özet raporları kaydet
    json_yolu = os.path.join(REPORTS_DIR, "kategori_raporu.json")
    with open(json_yolu, "w", encoding="utf-8") as f:
        json.dump(tum_ozetler, f, ensure_ascii=False, indent=2)

    ozet_df = pd.DataFrame(tum_ozetler)
    csv_yolu = os.path.join(REPORTS_DIR, "kategori_ozet.csv")
    ozet_df.to_csv(csv_yolu, index=False, encoding="utf-8-sig")

    print(f"Özet JSON : {json_yolu}")
    print(f"Özet CSV  : {csv_yolu}")
    print("\nTamamlandı.")


if __name__ == "__main__":
    tum_kategorileri_isle()
