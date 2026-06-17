"""
Steam Oyun Yorum Scraper — Steam Web API tabanlı
Her oyun için 500 yorum çeker ve data/raw/ altına CSV olarak kaydeder.
Selenium kullanılmaz; yalnızca requests ile HTTP istekleri yapılır.
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

API_URL            = "https://store.steampowered.com/appreviews/{app_id}"
DATA_RAW_DIR       = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
HEDEF_YORUM_SAYISI = 500    # Oyun başına hedeflenen yorum adedi
SAYFA_BOYUTU       = 100    # Her API çağrısında gelen maksimum yorum sayısı
ISTEK_ARASI_BEKLE  = 1.5    # İstekler arası bekleme süresi (saniye) — API'ye nazik ol

# ---------------------------------------------------------------------------
# Oyun listesi  {tür: [{ad, app_id, dosya}]}
# ---------------------------------------------------------------------------

OYUNLAR = {
    "RPG": [
        {"ad": "Cyberpunk 2077",         "app_id": 1091500, "dosya": "cyberpunk_2077"},
        {"ad": "Baldur's Gate 3",         "app_id": 1086940, "dosya": "baldurs_gate_3"},
        {"ad": "Elden Ring",              "app_id": 1245620, "dosya": "elden_ring"},
        {"ad": "The Witcher 3",           "app_id":  292030, "dosya": "the_witcher_3"},
        {"ad": "Dragon Age Inquisition",  "app_id": 1222690, "dosya": "dragon_age_inquisition"},
    ],
    "FPS": [
        {"ad": "Counter-Strike 2",  "app_id":  730,    "dosya": "counter_strike_2"},
        {"ad": "Apex Legends",      "app_id": 1172470, "dosya": "apex_legends"},
        {"ad": "Doom Eternal",      "app_id":  782330, "dosya": "doom_eternal"},
        {"ad": "Battlefield 2042",  "app_id": 1517290, "dosya": "battlefield_2042"},
        {"ad": "Titanfall 2",       "app_id": 1237970, "dosya": "titanfall_2"},
    ],
    "Strateji": [
        {"ad": "Civilization VI",            "app_id":  289070, "dosya": "civilization_vi"},
        {"ad": "Total War Warhammer III",     "app_id": 1142710, "dosya": "total_war_warhammer_3"},
        {"ad": "Stellaris",                  "app_id":  281990, "dosya": "stellaris"},
        {"ad": "Age of Empires IV",          "app_id": 1466860, "dosya": "age_of_empires_iv"},
        {"ad": "XCOM 2",                     "app_id":  268500, "dosya": "xcom_2"},
    ],
    "Simulasyon": [
        {"ad": "Cities Skylines",            "app_id":  255710, "dosya": "cities_skylines"},
        {"ad": "Planet Zoo",                 "app_id":  703080, "dosya": "planet_zoo"},
        {"ad": "Microsoft Flight Simulator", "app_id": 1250410, "dosya": "ms_flight_simulator"},
        {"ad": "Stardew Valley",             "app_id":  413150, "dosya": "stardew_valley"},
        {"ad": "Euro Truck Simulator 2",     "app_id":  227300, "dosya": "euro_truck_simulator_2"},
    ],
    "Cok_Oyunculu": [
        {"ad": "PUBG",              "app_id":  578080, "dosya": "pubg"},
        {"ad": "Among Us",          "app_id":  945360, "dosya": "among_us"},
        {"ad": "Fall Guys",         "app_id": 1097150, "dosya": "fall_guys"},
        {"ad": "Deep Rock Galactic","app_id":  548430, "dosya": "deep_rock_galactic"},
        {"ad": "Warframe",          "app_id":  230410, "dosya": "warframe"},
    ],
}

# ---------------------------------------------------------------------------
# API isteği
# ---------------------------------------------------------------------------

def yorumlari_getir(app_id: int, cursor: str = "*") -> dict | None:
    """
    Steam API'sine tek bir istek yapar ve ham JSON yanıtı döndürür.
    Hata durumunda None döner; program çökmez.
    """
    params = {
        "json":         1,
        "filter":       "all",
        "language":     "all",
        "num_per_page": SAYFA_BOYUTU,
        "cursor":       cursor,
        "review_type":  "all",
        "purchase_type":"all",
    }
    try:
        yanit = requests.get(
            API_URL.format(app_id=app_id),
            params=params,
            timeout=15,
        )
        yanit.raise_for_status()
        return yanit.json()
    except requests.RequestException as hata:
        print(f"\n   [HATA] API isteği başarısız: {hata}")
        return None


# ---------------------------------------------------------------------------
# Yorum ayrıştırma
# ---------------------------------------------------------------------------

def yorumu_ayristir(ham: dict, oyun_adi: str, oyun_turu: str) -> dict:
    """
    Ham API yanıtından tek bir yorum sözlüğü çıkarır.
    Eksik alanlar için güvenli varsayılan değerler kullanılır.
    """
    yazar = ham.get("author", {})
    ts    = ham.get("timestamp_created", 0)

    return {
        "game_name":          oyun_adi,
        "game_genre":         oyun_turu,
        "review":             ham.get("review", "").strip(),
        "voted_up":           ham.get("voted_up"),
        "timestamp_created":  (
            datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if ts else None
        ),
        "language":           ham.get("language", ""),
        "playtime_forever":   yazar.get("playtime_forever"),   # dakika cinsinden
    }


# ---------------------------------------------------------------------------
# Tek oyun için döngü
# ---------------------------------------------------------------------------

def oyun_yorumlarini_cek(app_id: int, oyun_adi: str, oyun_turu: str) -> list[dict]:
    """
    Cursor tabanlı sayfalama ile HEDEF_YORUM_SAYISI kadar yorum çeker.
    Her sayfa isteğinden sonra ISTEK_ARASI_BEKLE kadar uyur.
    """
    yorumlar = []
    cursor   = "*"

    with tqdm(total=HEDEF_YORUM_SAYISI, desc=f"  {oyun_adi}", unit="yorum", leave=False) as bar:
        while len(yorumlar) < HEDEF_YORUM_SAYISI:
            veri = yorumlari_getir(app_id, cursor)

            # API yanıt vermediyse ya da başarısız döndüyse dur
            if not veri or veri.get("success") != 1:
                break

            yeni_yorumlar = veri.get("reviews", [])
            if not yeni_yorumlar:
                break   # Daha fazla yorum kalmadı

            for ham in yeni_yorumlar:
                if len(yorumlar) >= HEDEF_YORUM_SAYISI:
                    break
                kayit = yorumu_ayristir(ham, oyun_adi, oyun_turu)
                if kayit["review"]:   # Boş yorumları atla
                    yorumlar.append(kayit)

            bar.update(len(yeni_yorumlar))

            # Yeni cursor yoksa ya da değişmediyse sayfalama bitti demektir
            yeni_cursor = veri.get("cursor", "")
            if not yeni_cursor or yeni_cursor == cursor:
                break
            cursor = yeni_cursor

            time.sleep(ISTEK_ARASI_BEKLE)

    return yorumlar


# ---------------------------------------------------------------------------
# CSV kaydetme
# ---------------------------------------------------------------------------

def csv_kaydet(kayitlar: list[dict], dosya_adi: str) -> str:
    """Kayıt listesini data/raw/{dosya_adi}.csv olarak kaydeder."""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    yol = os.path.join(DATA_RAW_DIR, f"{dosya_adi}.csv")
    pd.DataFrame(kayitlar).to_csv(yol, index=False, encoding="utf-8-sig")
    return yol


# ---------------------------------------------------------------------------
# Tüm oyunları çek — ana orkestrasyon fonksiyonu
# ---------------------------------------------------------------------------

def tum_oyunlari_cek() -> None:
    """
    OYUNLAR sözlüğündeki tüm oyunları sırayla işler.
    Her oyunun sonucunu data/raw/ altına kaydeder ve özet tablo basar.
    """
    toplam_oyun = sum(len(v) for v in OYUNLAR.values())
    ozet        = []

    print(f"Toplam {toplam_oyun} oyun, oyun başına {HEDEF_YORUM_SAYISI} yorum hedefleniyor.\n")

    with tqdm(total=toplam_oyun, desc="Genel ilerleme", unit="oyun") as genel:
        for tur, oyun_listesi in OYUNLAR.items():
            for oyun in oyun_listesi:
                tqdm.write(f"\n[{tur}] {oyun['ad']}  (App ID: {oyun['app_id']})")

                kayitlar = oyun_yorumlarini_cek(oyun["app_id"], oyun["ad"], tur)

                if kayitlar:
                    yol = csv_kaydet(kayitlar, oyun["dosya"])
                    tqdm.write(f"  → {len(kayitlar)} yorum kaydedildi: {yol}")
                    ozet.append({"oyun": oyun["ad"], "tür": tur,
                                 "yorum_sayisi": len(kayitlar), "durum": "OK"})
                else:
                    tqdm.write(f"  → Veri alınamadı, atlandı.")
                    ozet.append({"oyun": oyun["ad"], "tür": tur,
                                 "yorum_sayisi": 0, "durum": "HATA"})

                genel.update(1)

    # --------------- Özet tablosu ---------------
    print(f"\n{'='*55}")
    print("  TAMAMLANDI")
    print(f"{'='*55}")
    ozet_df = pd.DataFrame(ozet)
    print(ozet_df.to_string(index=False))
    toplam = ozet_df["yorum_sayisi"].sum()
    print(f"\nToplam çekilen yorum: {toplam:,}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tum_oyunlari_cek()
