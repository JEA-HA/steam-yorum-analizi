"""
Steam Oyun Yorum Analizi — Streamlit Dashboard
Çalıştırmak için: streamlit run dashboard/app.py
"""

import io
import os
import re
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Sayfa yapılandırması ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Steam Yorum Analizi",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Oyun / Tür sabitleri ────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

TURLER = {
    "RPG":          ["cyberpunk_2077", "baldurs_gate_3", "elden_ring", "the_witcher_3", "dragon_age_inquisition"],
    "FPS":          ["counter_strike_2", "apex_legends", "doom_eternal", "battlefield_2042", "titanfall_2"],
    "Strateji":     ["civilization_vi", "total_war_warhammer_3", "stellaris", "age_of_empires_iv", "xcom_2"],
    "Simulasyon":   ["cities_skylines", "planet_zoo", "ms_flight_simulator", "stardew_valley", "euro_truck_simulator_2"],
    "Cok_Oyunculu": ["pubg", "among_us", "fall_guys", "deep_rock_galactic", "warframe"],
}

TUR_IKONLARI = {
    "RPG": "⚔️", "FPS": "🔫", "Strateji": "♟️",
    "Simulasyon": "🏗️", "Cok_Oyunculu": "🎮",
}

# Oyun adları — hiçbir zaman çevrilmez
OYUN_ADLARI = {
    "cyberpunk_2077":         "Cyberpunk 2077",
    "baldurs_gate_3":         "Baldur's Gate 3",
    "elden_ring":              "Elden Ring",
    "the_witcher_3":           "The Witcher 3",
    "dragon_age_inquisition":  "Dragon Age: Inquisition",
    "counter_strike_2":        "Counter-Strike 2",
    "apex_legends":            "Apex Legends",
    "doom_eternal":            "DOOM Eternal",
    "battlefield_2042":        "Battlefield 2042",
    "titanfall_2":             "Titanfall 2",
    "civilization_vi":         "Civilization VI",
    "total_war_warhammer_3":   "Total War: Warhammer III",
    "stellaris":               "Stellaris",
    "age_of_empires_iv":       "Age of Empires IV",
    "xcom_2":                  "XCOM 2",
    "cities_skylines":         "Cities: Skylines",
    "planet_zoo":              "Planet Zoo",
    "ms_flight_simulator":     "Microsoft Flight Simulator",
    "stardew_valley":          "Stardew Valley",
    "euro_truck_simulator_2":  "Euro Truck Simulator 2",
    "pubg":                    "PUBG",
    "among_us":                "Among Us",
    "fall_guys":               "Fall Guys",
    "deep_rock_galactic":      "Deep Rock Galactic",
    "warframe":                "Warframe",
}

# ─── Kategori etiketleri (dil bazlı) ─────────────────────────────────────────

_KAT_TR = {
    "cat_performans":        "Performans",
    "cat_teknik_sorun":      "Teknik Sorun",
    "cat_fiyat":             "Fiyat",
    "cat_hikaye_icerik":     "Hikaye/İçerik",
    "cat_oynanabilirlik":    "Oynanabilirlik",
    "cat_cok_oyunculu":      "Çok Oyunculu",
    "cat_destek_guncelleme": "Destek/Güncelleme",
    "cat_gorsel_ses":        "Görsel/Ses",
}

_KAT_EN = {
    "cat_performans":        "Performance",
    "cat_teknik_sorun":      "Technical Issues",
    "cat_fiyat":             "Pricing",
    "cat_hikaye_icerik":     "Story/Content",
    "cat_oynanabilirlik":    "Gameplay",
    "cat_cok_oyunculu":      "Multiplayer",
    "cat_destek_guncelleme": "Support/Updates",
    "cat_gorsel_ses":        "Visuals/Audio",
}

def kat_etiketler(lang: str) -> dict:
    return _KAT_TR if lang == "TR" else _KAT_EN

KATEGORI_RENKLERI_TR = {
    "Performans": "#EF4444", "Teknik Sorun": "#F97316", "Fiyat": "#EAB308",
    "Hikaye/İçerik": "#22C55E", "Oynanabilirlik": "#3B82F6",
    "Çok Oyunculu": "#8B5CF6", "Destek/Güncelleme": "#EC4899", "Görsel/Ses": "#14B8A6",
}
KATEGORI_RENKLERI_EN = {
    "Performance": "#EF4444", "Technical Issues": "#F97316", "Pricing": "#EAB308",
    "Story/Content": "#22C55E", "Gameplay": "#3B82F6",
    "Multiplayer": "#8B5CF6", "Support/Updates": "#EC4899", "Visuals/Audio": "#14B8A6",
}

def kat_renkleri(lang: str) -> dict:
    return KATEGORI_RENKLERI_TR if lang == "TR" else KATEGORI_RENKLERI_EN

# ─── Arayüz metinleri ────────────────────────────────────────────────────────

STRINGS = {
    "TR": {
        "caption":         "Steam'den toplanan yorumların duygu analizi ve şikayet kategorizasyonu",
        "back_btn":        "← Oyun Listesine Dön",
        "explore_btn":     "İncele →",
        "genre_lbl":       "Tür",
        "total_reviews":   "Toplam Yorum",
        "pos_reviews":     "Olumlu Yorum",
        "neg_reviews":     "Olumsuz Yorum",
        "active_cats":     "Aktif Kategori",
        "sec_complaints":  "🗂️ Şikayet Kategorileri",
        "sec_sentiment":   "💬 Duygu Dağılımı (VADER)",
        "sec_comparison":  "📊 Türler Arası Karşılaştırma",
        "sec_actions":     "💡 Aksiyon Önerileri",
        "no_actions":      "Aksiyon önerisi üretmek için kategori verisi bulunamadı.",
        "games_suffix":    "Oyunları",
        "review_count":    "yorum",
    },
    "EN": {
        "caption":         "Sentiment analysis and complaint categorization of Steam reviews",
        "back_btn":        "← Back to Game List",
        "explore_btn":     "Explore →",
        "genre_lbl":       "Genre",
        "total_reviews":   "Total Reviews",
        "pos_reviews":     "Positive Reviews",
        "neg_reviews":     "Negative Reviews",
        "active_cats":     "Active Categories",
        "sec_complaints":  "🗂️ Complaint Categories",
        "sec_sentiment":   "💬 Sentiment Distribution (VADER)",
        "sec_comparison":  "📊 Cross-Genre Comparison",
        "sec_actions":     "💡 Action Recommendations",
        "no_actions":      "No category data available for action recommendations.",
        "games_suffix":    "Games",
        "review_count":    "reviews",
    },
}

# Aksiyon şablonları
AKSIYON_SABLONLARI = {
    "TR": {
        "cat_performans":        "Performans optimizasyonu önceliklendirilmeli: FPS düşmeleri ve donma sorunları için acil patch yayınlanması gerekiyor.",
        "cat_teknik_sorun":      "Teknik sorunlar hızla çözülmeli: bug fix süreci hızlandırılmalı ve bilinen hatalar için hotfix takvimi oluşturulmalı.",
        "cat_fiyat":             "Fiyatlandırma stratejisi gözden geçirilmeli: bölgesel fiyatlandırma veya daha uygun DLC paketleri değerlendirilebilir.",
        "cat_hikaye_icerik":     "İçerik zenginliği artırılmalı: yeni senaryo DLC'leri veya ücretsiz içerik güncellemeleri planlanabilir.",
        "cat_oynanabilirlik":    "Oyun dengesi iyileştirilmeli: mekanik ayarlar ve kullanıcı geri bildirimleri doğrultusunda denge güncellemesi yapılmalı.",
        "cat_cok_oyunculu":      "Çok oyunculu altyapı güçlendirilmeli: sunucu kararlılığı artırılmalı ve matchmaking sistemi yeniden düzenlenmeli.",
        "cat_destek_guncelleme": "Geliştirici desteği artırılmalı: daha sık güncelleme takvimi belirlenmeli ve toplulukla şeffaf iletişim kurulmalı.",
        "cat_gorsel_ses":        "Görsel ve ses kalitesi iyileştirilmeli: grafik optimizasyonu ve ses tasarımı revizyonu için yol haritası paylaşılmalı.",
    },
    "EN": {
        "cat_performans":        "Performance optimization is a priority: an emergency patch is needed to address FPS drops and freezing issues.",
        "cat_teknik_sorun":      "Technical issues must be resolved quickly: accelerate the bug fix pipeline and establish a hotfix schedule for known issues.",
        "cat_fiyat":             "Pricing strategy should be reconsidered: regional pricing or more affordable DLC bundles may increase satisfaction.",
        "cat_hikaye_icerik":     "Content depth should be improved: new story DLC or free content updates can address player demand.",
        "cat_oynanabilirlik":    "Game balance needs improvement: mechanical tuning and player feedback-driven balance patches are recommended.",
        "cat_cok_oyunculu":      "Multiplayer infrastructure needs strengthening: improve server stability and rework the matchmaking system.",
        "cat_destek_guncelleme": "Developer support should increase: establish a regular update cadence and maintain transparent community communication.",
        "cat_gorsel_ses":        "Visual and audio quality should be improved: a graphics update and audio design revision roadmap is recommended.",
    },
}

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    /* Başlık banner */
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #1e293b 100%);
        border-radius: 14px;
        padding: 1.6rem 2rem 1.4rem 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .header-title {
        font-size: 2.1rem; font-weight: 700;
        color: #f1f5f9; margin-bottom: 0.25rem; letter-spacing: -0.5px;
    }
    .header-sub { font-size: 0.9rem; color: #94a3b8; }

    /* Oyun kartı */
    .game-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1rem 1rem 0.7rem 1rem;
        min-height: 130px;
    }
    .game-title {
        font-size: 0.93rem; font-weight: 600; color: #1e293b;
        margin-bottom: 0.45rem; line-height: 1.3;
        display: -webkit-box; -webkit-line-clamp: 2;
        -webkit-box-orient: vertical; overflow: hidden;
    }
    .progress-wrap {
        display: flex; height: 8px; border-radius: 4px;
        overflow: hidden; margin-bottom: 0.3rem;
    }
    .pct-row { font-size: 0.78rem; display: flex; justify-content: space-between; }
    .review-count { font-size: 0.73rem; color: #94a3b8; margin-top: 0.2rem; }

    /* Metrik kart */
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 0.9rem 1rem; text-align: center;
    }
    .metric-value { font-size: 1.7rem; font-weight: 700; }
    .metric-label { font-size: 0.78rem; color: #64748b; margin-top: 0.15rem; }

    /* Bölüm başlığı */
    .sec-title {
        font-size: 1rem; font-weight: 600; color: #334155;
        margin: 1.2rem 0 0.4rem 0;
        border-left: 3px solid #3b82f6; padding-left: 0.5rem;
    }

    /* Top 10 kelime listesi */
    .word-list-item {
        display: flex; align-items: center; padding: 0.45rem 0.6rem;
        border-radius: 6px; margin-bottom: 0.3rem;
        background: #fef2f2; border-left: 3px solid #ef4444;
    }
    .word-rank {
        font-size: 0.7rem; font-weight: 700; color: #9f1239;
        min-width: 18px; margin-right: 0.4rem;
    }
    .word-key {
        font-family: monospace; font-size: 0.85rem; font-weight: 600;
        color: #1e293b; min-width: 110px;
    }
    .word-arrow { color: #94a3b8; margin: 0 0.4rem; font-size: 0.8rem; }
    .word-meaning { font-size: 0.82rem; color: #475569; }

    /* Aksiyon kartı */
    .action-card {
        background: #fffbeb; border-left: 4px solid #f59e0b;
        border-radius: 6px; padding: 0.65rem 1rem;
        margin-bottom: 0.45rem; font-size: 0.875rem; color: #78350f; line-height: 1.5;
    }
    .action-rank {
        display: inline-block; background: #f59e0b; color: white;
        border-radius: 50%; width: 20px; height: 20px; text-align: center;
        line-height: 20px; font-size: 0.75rem; font-weight: 700; margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Veri yükleme ────────────────────────────────────────────────────────────

@st.cache_data
def oyun_yukle(oyun_adi: str) -> pd.DataFrame:
    dosya = os.path.join(DATA_DIR, f"{oyun_adi}_categorized.csv")
    if not os.path.exists(dosya):
        return pd.DataFrame()
    try:
        df = pd.read_csv(dosya, encoding="utf-8-sig")
        if "voted_up" in df.columns:
            df["voted_up_bool"] = df["voted_up"].apply(
                lambda x: True if str(x).strip().lower() in ["true", "1", "yes", "1.0"] else False
            )
        else:
            df["voted_up_bool"] = False
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data
def tur_ozet_hesapla(tur_adi: str) -> dict:
    sonuc = {}
    for oyun in TURLER[tur_adi]:
        df = oyun_yukle(oyun)
        toplam = len(df)
        if df.empty:
            sonuc[oyun] = {"toplam": 0, "pos_pct": 0.0, "neg_pct": 0.0}
            continue
        pos_pct = (int(df["voted_up"].sum()) / toplam * 100) if "voted_up" in df.columns and toplam else 0.0
        sonuc[oyun] = {"toplam": toplam, "pos_pct": pos_pct, "neg_pct": 100.0 - pos_pct}
    return sonuc

# ─── Grafik fonksiyonları (lang parametreli) ─────────────────────────────────

def grafik_olumlu_olumsuz(df: pd.DataFrame, lang: str) -> go.Figure:
    if "voted_up" not in df.columns:
        return go.Figure()
    pos = int(df["voted_up"].sum())
    neg = len(df) - pos
    toplam = pos + neg

    if lang == "TR":
        y_labels  = ["Olumlu", "Olumsuz"]
        title     = "Olumlu / Olumsuz Yorum Dağılımı"
        hover_tmpl = (
            "<b>%{y}</b><br>"
            "Yorum Sayısı: <b>%{x:,}</b><br>"
            "Oran: <b>%{customdata:.1f}%</b><extra></extra>"
        )
    else:
        y_labels  = ["Positive", "Negative"]
        title     = "Positive / Negative Review Distribution"
        hover_tmpl = (
            "<b>%{y}</b><br>"
            "Review Count: <b>%{x:,}</b><br>"
            "Share: <b>%{customdata:.1f}%</b><extra></extra>"
        )

    pcts = [pos / toplam * 100 if toplam else 0, neg / toplam * 100 if toplam else 0]
    cmap = {y_labels[0]: "#22c55e", y_labels[1]: "#ef4444"}

    fig = px.bar(
        x=[pos, neg], y=y_labels, orientation="h",
        color=y_labels, color_discrete_map=cmap,
        labels={"x": "", "y": ""},
        title=title,
        text=[f"{pos:,}", f"{neg:,}"],
    )
    fig.update_traces(
        textposition="outside",
        customdata=pcts,
        hovertemplate=hover_tmpl,
    )
    fig.update_layout(showlegend=False, height=200, margin=dict(t=40, b=10, l=0, r=30))
    return fig


def grafik_dil_pasta(df: pd.DataFrame, lang: str, neg_only: bool = False) -> go.Figure:
    col = next((c for c in ("language", "detected_language") if c in df.columns), None)
    if col is None:
        return go.Figure()
    dagılım = df[col].value_counts().head(8)
    if lang == "TR":
        title = "Olumsuz Yorumların Dil Dağılımı" if neg_only else "Dil Dağılımı"
        hover_tmpl = "<b>%{label}</b><br>Yorum Sayısı: <b>%{value:,}</b><br>Oran: <b>%{percent:.1%}</b><extra></extra>"
    else:
        title = "Negative Review Language Distribution" if neg_only else "Language Distribution"
        hover_tmpl = "<b>%{label}</b><br>Review Count: <b>%{value:,}</b><br>Share: <b>%{percent:.1%}</b><extra></extra>"
    fig = px.pie(values=dagılım.values, names=dagılım.index, hole=0.42, title=title)
    fig.update_traces(hovertemplate=hover_tmpl)
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=0, r=0))
    return fig


def grafik_kategori_bar(df: pd.DataFrame, lang: str, neg_only: bool = False, neg_toplam: int = 0) -> go.Figure:
    etiketler = kat_etiketler(lang)
    kat_col = [c for c in df.columns if c in etiketler]
    if not kat_col:
        return go.Figure()
    toplamlar = {etiketler[c]: int(df[c].sum()) for c in kat_col}
    siralı = sorted(toplamlar.items(), key=lambda x: x[1], reverse=True)
    isimler, sayilar = zip(*siralı)
    renkler = kat_renkleri(lang)
    if neg_only:
        title = "Olumsuz Yorum Şikayet Kategorileri" if lang == "TR" else "Negative Review Complaint Categories"
        x_lbl  = "Olumsuz Yorum Sayısı" if lang == "TR" else "Negative Review Count"
        if neg_toplam > 0:
            text_vals = [
                f"{s:,} (%{s / neg_toplam * 100:.1f})" for s in sayilar
            ]
        else:
            text_vals = [f"{s:,}" for s in sayilar]
    else:
        title = "Şikayet Kategorileri" if lang == "TR" else "Complaint Categories"
        x_lbl  = ""
        text_vals = [f"{s:,}" for s in sayilar]
    if lang == "TR":
        hover_tmpl = "Kategori: <b>%{y}</b><br>Yorum Sayısı: <b>%{x:,}</b><extra></extra>"
    else:
        hover_tmpl = "Category: <b>%{y}</b><br>Review Count: <b>%{x:,}</b><extra></extra>"
    fig = px.bar(
        x=list(sayilar), y=list(isimler), orientation="h",
        color=list(isimler), color_discrete_map=renkler,
        labels={"x": x_lbl, "y": ""}, title=title,
        text=text_vals,
    )
    fig.update_traces(textposition="outside", hovertemplate=hover_tmpl)
    fig.update_layout(
        showlegend=False, height=340,
        margin=dict(t=40, b=10, l=0, r=120),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def render_kategori_yorumu(neg_df: pd.DataFrame, kat_col: list, etiketler: dict, neg_sayi: int):
    """Şikayet kategorileri grafiğinin altına otomatik Türkçe yorum kutusu ekler."""
    if not kat_col or neg_sayi == 0:
        return
    # Top 3 kategori (olumsuz yorum sayısına göre)
    sıralı = sorted(kat_col, key=lambda c: int(neg_df[c].sum()) if c in neg_df.columns else 0, reverse=True)
    top3 = sıralı[:3]
    if not top3:
        return

    def _etiket(c):
        return etiketler.get(c, c)

    def _pct(c):
        sayi = int(neg_df[c].sum()) if c in neg_df.columns else 0
        return sayi / max(1, neg_sayi) * 100

    kat1, kat2, kat3 = (top3 + ["", ""])[:3]
    l1, l2, l3 = _etiket(kat1), (_etiket(kat2) if kat2 else ""), (_etiket(kat3) if kat3 else "")
    p1, p2, p3 = _pct(kat1), (_pct(kat2) if kat2 else 0), (_pct(kat3) if kat3 else 0)

    # Kategori-özel detay yorumu
    _ozel = {
        "Teknik Sorun": (
            "Oyuncular oyunun teknik kararlılığından ciddi şekilde şikayet etmektedir. "
            "Bu durum oyun deneyimini doğrudan olumsuz etkilemekte ve acil teknik iyileştirme gerektirmektedir."
        ),
        "Hikaye/İçerik": (
            "Oyuncular hikaye derinliği ve içerik zenginliği konusunda beklentilerinin "
            "karşılanmadığını ifade etmektedir. Bu bulgu RPG türü için özellikle kritiktir."
        ),
        "Oynanabilirlik": (
            "Temel oyun mekaniği ve kontrol sistemi oyuncular tarafından yetersiz bulunmaktadır. "
            "Bu kategori tüm oyun türlerinde en sık karşılaşılan şikayet olmaya devam etmektedir."
        ),
        "Performans": (
            "Oyunun teknik optimizasyonu yetersiz bulunmaktadır. "
            "FPS düşüşleri ve yavaş yükleme süreleri en sık dile getirilen sorunlar arasındadır."
        ),
        "Fiyat": (
            "Fiyatlandırma politikası ve DLC yapısı oyuncular tarafından sorgulanmaktadır. "
            "Fiyat/değer dengesi konusunda ciddi memnuniyetsizlik gözlemlenmektedir."
        ),
        "Çok Oyunculu": (
            "Çevrimiçi oyun altyapısı yetersiz bulunmaktadır. "
            "Sunucu kararlılığı, matchmaking kalitesi ve hile sorunu öne çıkan konulardır."
        ),
        "Görsel/Ses": (
            "Grafiksel kalite ve ses tasarımı oyuncu beklentilerini karşılamamaktadır. "
            "Görsel optimizasyon ve ses dengesi iyileştirme gerektirmektedir."
        ),
        "Destek/Güncelleme": (
            "Geliştirici desteği ve güncelleme sıklığı yetersiz bulunmaktadır. "
            "Oyuncular daha aktif geliştirici iletişimi talep etmektedir."
        ),
    }
    detay = _ozel.get(l1, f"Bu kategori oyuncuların öncelikli şikayet konusudur ve dikkat gerektirmektedir.")

    kat2_str = f", <strong>{l2}</strong> (%{p2:.1f})" if l2 else ""
    kat3_str = f" ve <strong>{l3}</strong> (%{p3:.1f})" if l3 else ""

    yorum = (
        f"Bu oyunun olumsuz yorumları incelendiğinde en belirgin şikayet "
        f"<strong>{l1}</strong> alanında gözlemlenmektedir "
        f"(olumsuz yorumların %{p1:.1f}'i). "
        f"Bunu{kat2_str}{kat3_str} kategorileri izlemektedir."
        f"<br><br>"
        f"{detay}"
        f"<br><br>"
        f"Bu bulgular doğrultusunda geliştirici ekibin öncelikle "
        f"<strong>{l1}</strong> alanına odaklanması önerilmektedir."
    )

    st.markdown(
        f'<div style="background:#0f172a;border-left:4px solid #3b82f6;border-radius:8px;'
        f'padding:1rem 1.2rem;margin-top:0.8rem;font-size:0.88rem;color:#cbd5e1;line-height:1.6;">'
        f'{yorum}'
        f'</div>',
        unsafe_allow_html=True,
    )


def grafik_duygu_bar(df: pd.DataFrame, lang: str) -> go.Figure:
    col = next((c for c in ("sentiment", "vader_label") if c in df.columns), None)
    if col is None:
        return go.Figure()
    dagılım = df[col].value_counts()
    toplam = len(df)
    renk_map = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#94a3b8"}

    duygu_label = {
        "positive": {"TR": "Olumlu", "EN": "Positive"},
        "negative": {"TR": "Olumsuz", "EN": "Negative"},
        "neutral":  {"TR": "Tarafsız", "EN": "Neutral"},
    }

    title = "Duygu Dağılımı (VADER)" if lang == "TR" else "Sentiment Distribution (VADER)"
    if lang == "TR":
        hover_tmpl = "%{y}<br>Yorum Sayısı: <b>%{x:,}</b><br>Oran: <b>%{customdata:.1f}%</b><extra></extra>"
    else:
        hover_tmpl = "%{y}<br>Review Count: <b>%{x:,}</b><br>Share: <b>%{customdata:.1f}%</b><extra></extra>"

    labels, values, renkler, pcts = [], [], [], []
    for d in ["positive", "negative", "neutral"]:
        v = int(dagılım.get(d, 0))
        pct = v / toplam * 100 if toplam else 0
        labels.append(f"{duygu_label[d][lang]}  ({pct:.1f}%)")
        values.append(v)
        renkler.append(renk_map[d])
        pcts.append(pct)

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=renkler,
        text=[f"{v:,}" for v in values],
        textposition="outside",
        customdata=pcts,
        hovertemplate=hover_tmpl,
    ))
    fig.update_layout(title=title, showlegend=False, height=220, margin=dict(t=40, b=10, l=0, r=50))
    return fig


def grafik_tur_heatmap(tur_adi: str, lang: str) -> go.Figure:
    etiketler = kat_etiketler(lang)
    kat_col = list(etiketler.keys())
    satirlar, oyun_etiketler = [], []
    for oyun in TURLER[tur_adi]:
        df = oyun_yukle(oyun)
        if df.empty:
            continue
        neg_df = df[df["voted_up_bool"] == False]
        neg_n = max(1, len(neg_df))
        satirlar.append([round(neg_df[c].sum() / neg_n * 100, 1) if c in neg_df.columns else 0 for c in kat_col])
        oyun_etiketler.append(OYUN_ADLARI.get(oyun, oyun))
    if not satirlar:
        return go.Figure()
    matris = pd.DataFrame(
        satirlar, index=oyun_etiketler,
        columns=[etiketler[k] for k in kat_col],
    )
    if lang == "TR":
        title = f"{tur_adi} — Türler Arası Kategori Karşılaştırması (Olumsuz Yorum Yüzdesi %)"
        hover_tmpl = "<b>%{y}</b><br>%{x}<br>Oran: <b>%{z:.1f}%</b><extra></extra>"
    else:
        title = f"{tur_adi} — Cross-Genre Category Comparison (%)"
        hover_tmpl = "<b>%{y}</b><br>%{x}<br>Rate: <b>%{z:.1f}%</b><extra></extra>"
    fig = px.imshow(
        matris, text_auto=".1f",
        color_continuous_scale="YlOrRd",
        title=title,
        labels={"color": "%"},
        aspect="auto",
    )
    fig.update_traces(hovertemplate=hover_tmpl)
    fig.update_layout(height=300, margin=dict(t=50, b=10, l=0, r=0))
    return fig

# ─── VADER duygu analizi (detay sayfası) ─────────────────────────────────────

def render_vader_analizi(df: pd.DataFrame, lang: str):
    """Oyun detay sayfası için VADER duygu analizi kutucukları."""
    if lang == "TR":
        st.markdown("<div class='sec-title'>💬 Duygu Analizi Sonuçları</div>", unsafe_allow_html=True)
        st.markdown(
            "**VADER algoritması** her yorumu **−1** ile **+1** arasında bir duygu skoru ile "
            "değerlendirir.  \n"
            "+1'e yakın = çok olumlu &nbsp;|&nbsp; −1'e yakın = çok olumsuz &nbsp;|&nbsp; "
            "0'a yakın = nötr"
        )
    else:
        st.markdown("<div class='sec-title'>💬 Sentiment Analysis Results</div>", unsafe_allow_html=True)
        st.markdown(
            "**VADER** scores each review from **−1** (very negative) to **+1** (very positive).  \n"
            "Near +1 = very positive &nbsp;|&nbsp; Near −1 = very negative &nbsp;|&nbsp; "
            "Near 0 = neutral"
        )

    has_compound = "vader_compound" in df.columns
    has_voted    = "voted_up_bool" in df.columns
    scol = next((c for c in ("sentiment", "vader_label") if c in df.columns), None)

    if has_compound and has_voted:
        pos_df_v = df[df["voted_up_bool"] == True]
        neg_df_v = df[df["voted_up_bool"] == False]
        pos_compound = pd.to_numeric(pos_df_v["vader_compound"], errors="coerce").mean()
        neg_compound = pd.to_numeric(neg_df_v["vader_compound"], errors="coerce").mean()

        if scol:
            voted_bool = df["voted_up_bool"]
            sent_bool  = df[scol] == "positive"
            match_pct  = float((voted_bool == sent_bool).mean() * 100)
        else:
            match_pct = 0.0

        r1, r2, r3 = st.columns(3)
        if lang == "TR":
            if not pd.isna(pos_compound):
                r1.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:#22c55e">+{pos_compound:.2f}</div>'
                    f'<div class="metric-label">Olumlu Yorum Ort. VADER</div>'
                    f'<div style="font-size:0.70rem;color:#94a3b8;margin-top:0.2rem;">güçlü pozitif duygu</div>'
                    f'</div>', unsafe_allow_html=True,
                )
            if not pd.isna(neg_compound):
                r2.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:#ef4444">{neg_compound:.2f}</div>'
                    f'<div class="metric-label">Olumsuz Yorum Ort. VADER</div>'
                    f'<div style="font-size:0.70rem;color:#94a3b8;margin-top:0.2rem;">güçlü negatif duygu</div>'
                    f'</div>', unsafe_allow_html=True,
                )
            if scol:
                renk_uyum = "#22c55e" if match_pct >= 80 else ("#f59e0b" if match_pct >= 65 else "#ef4444")
                r3.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:{renk_uyum}">%{match_pct:.1f}</div>'
                    f'<div class="metric-label">VADER–Steam Uyumu</div>'
                    f'<div style="font-size:0.70rem;color:#94a3b8;margin-top:0.2rem;">etiket örtüşme oranı</div>'
                    f'</div>', unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)
            if scol:
                if match_pct >= 80:
                    uyum_yorum = "✅ Yüksek uyum — Yorumlar tutarlı duygu içeriyor"
                elif match_pct >= 65:
                    uyum_yorum = "⚠️ Orta uyum — Bazı yorumlar karma duygu içeriyor"
                else:
                    uyum_yorum = "❗ Düşük uyum — Yorumlar güçlü mixed sentiment içeriyor"
                st.info(
                    f"**VADER algoritması** bu oyun için Steam değerlendirmeleriyle "
                    f"**%{match_pct:.1f}** oranında örtüştü  \n{uyum_yorum}"
                )
        else:
            if not pd.isna(pos_compound):
                r1.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:#22c55e">+{pos_compound:.2f}</div>'
                    f'<div class="metric-label">Positive Avg. VADER</div>'
                    f'<div style="font-size:0.70rem;color:#94a3b8;margin-top:0.2rem;">strong positive sentiment</div>'
                    f'</div>', unsafe_allow_html=True,
                )
            if not pd.isna(neg_compound):
                r2.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:#ef4444">{neg_compound:.2f}</div>'
                    f'<div class="metric-label">Negative Avg. VADER</div>'
                    f'<div style="font-size:0.70rem;color:#94a3b8;margin-top:0.2rem;">strong negative sentiment</div>'
                    f'</div>', unsafe_allow_html=True,
                )
            if scol:
                renk_uyum = "#22c55e" if match_pct >= 80 else ("#f59e0b" if match_pct >= 65 else "#ef4444")
                r3.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:{renk_uyum}">{match_pct:.1f}%</div>'
                    f'<div class="metric-label">VADER–Steam Match</div>'
                    f'<div style="font-size:0.70rem;color:#94a3b8;margin-top:0.2rem;">label agreement rate</div>'
                    f'</div>', unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)
            if scol:
                if match_pct >= 80:
                    uyum_yorum = "✅ High agreement — Reviews contain consistent sentiment"
                elif match_pct >= 65:
                    uyum_yorum = "⚠️ Medium agreement — Some reviews contain mixed sentiment"
                else:
                    uyum_yorum = "❗ Low agreement — Reviews contain strong mixed sentiment"
                st.info(
                    f"**VADER algorithm** agreed with Steam labels at a rate of "
                    f"**{match_pct:.1f}%**  \n{uyum_yorum}"
                )
    else:
        # vader_compound kolonu yoksa mevcut bar grafik göster
        st.plotly_chart(grafik_duygu_bar(df, lang), use_container_width=True)


# ─── Aksiyon önerileri ────────────────────────────────────────────────────────

def aksiyon_uret(df: pd.DataFrame, lang: str) -> list:
    sablonlar = AKSIYON_SABLONLARI[lang]
    kat_col = [c for c in df.columns if c in sablonlar]
    if not kat_col:
        return []
    neg_df = df[df["voted_up_bool"] == False] if "voted_up_bool" in df.columns else df
    toplamlar = {c: int(neg_df[c].sum()) for c in kat_col if c in neg_df.columns}
    en_yuksek = sorted(toplamlar.items(), key=lambda x: x[1], reverse=True)[:3]
    return [sablonlar[k] for k, _ in en_yuksek]

# ─── Navigasyon: tür butonları ───────────────────────────────────────────────

def render_tur_nav():
    cols = st.columns(5)
    for i, (tur, ikon) in enumerate(TUR_IKONLARI.items()):
        with cols[i]:
            aktif = st.session_state.secili_tur == tur
            if st.button(
                f"{ikon} {tur}",
                key=f"nav_{tur}",
                use_container_width=True,
                type="primary" if aktif else "secondary",
            ):
                st.session_state.secili_tur = tur
                st.session_state.secili_oyun = None
                st.rerun()

# ─── Görünüm: oyun listesi ───────────────────────────────────────────────────

def render_oyun_listesi(lang: str):
    s   = STRINGS[lang]
    tur  = st.session_state.secili_tur
    ikon = TUR_IKONLARI[tur]
    st.markdown(
        f"<div class='sec-title'>{ikon} {tur} {s['games_suffix']}</div>",
        unsafe_allow_html=True,
    )

    ozet = tur_ozet_hesapla(tur)
    cols = st.columns(5)

    for i, oyun in enumerate(TURLER[tur]):
        ist    = ozet.get(oyun, {"toplam": 0, "pos_pct": 0.0, "neg_pct": 0.0})
        ad     = OYUN_ADLARI[oyun]          # her zaman İngilizce orijinal ad
        pos    = ist["pos_pct"]
        neg    = ist["neg_pct"]
        toplam = ist["toplam"]

        with cols[i]:
            st.markdown(
                f"""
                <div class="game-card">
                  <div class="game-title">{ad}</div>
                  <div class="progress-wrap">
                    <div style="background:#22c55e;width:{pos:.1f}%;height:100%;"></div>
                    <div style="background:#ef4444;width:{neg:.1f}%;height:100%;"></div>
                  </div>
                  <div class="pct-row">
                    <span style="color:#22c55e;font-weight:600;">▲ {pos:.1f}%</span>
                    <span style="color:#ef4444;font-weight:600;">▼ {neg:.1f}%</span>
                  </div>
                  <div class="review-count">{toplam:,} {s['review_count']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Buton her zaman görünür; oyun ismi değişmez
            if st.button(s["explore_btn"], key=f"sec_{oyun}", use_container_width=True):
                st.session_state.secili_oyun = oyun
                st.rerun()

# ─── Görünüm: oyun detayı ────────────────────────────────────────────────────

def render_detay(lang: str):
    s    = STRINGS[lang]
    oyun = st.session_state.secili_oyun
    tur  = st.session_state.secili_tur
    ad   = OYUN_ADLARI[oyun]               # her zaman İngilizce orijinal ad

    if st.button(s["back_btn"], key="geri_btn"):
        st.session_state.secili_oyun = None
        st.rerun()

    st.markdown(f"## 🎮 {ad}")
    st.caption(f"{s['genre_lbl']}: {tur}")

    df = oyun_yukle(oyun)
    if df.empty:
        st.warning("Bu oyun için işlenmiş veri bulunamadı." if lang == "TR" else "No processed data found for this game.")
        return

    # voted_up_bool: tüm format varyantlarını güvenli şekilde boolean'a çevir
    if "voted_up" in df.columns:
        df["voted_up_bool"] = df["voted_up"].apply(
            lambda x: True if str(x).strip().lower() in ["true", "1", "yes"] else False
        )
    else:
        df["voted_up_bool"] = False

    # ── 1. Metrik kartları ────────────────────────────────────────────────
    toplam       = len(df)
    pos_sayi     = int(df["voted_up_bool"].sum())
    neg_sayi     = toplam - pos_sayi
    etiketler    = kat_etiketler(lang)
    kat_col      = [c for c in df.columns if c in etiketler]
    neg_df_detay = df[df["voted_up_bool"] == False]
    aktif_kat    = sum(1 for c in kat_col if neg_df_detay[c].sum() > 0)

    m1, m2, m3, m4 = st.columns(4)
    if lang == "TR":
        metrik_info = [
            (m1, "Toplam Yorum",       f"{toplam:,}",   "#3b82f6", "Steam'den çekilen toplam yorum sayısı"),
            (m2, "Olumlu Yorum",       f"{pos_sayi:,}", "#22c55e", "voted_up=True olan yorumlar"),
            (m3, "Olumsuz Yorum",      f"{neg_sayi:,}", "#ef4444", "voted_up=False olan yorumlar"),
            (m4, "Şikayet Kategorisi", str(aktif_kat),  "#8b5cf6",
             "Olumsuz yorumlarda tespit edilen şikayet kategorisi sayısı"),
        ]
    else:
        metrik_info = [
            (m1, "Total Reviews",        f"{toplam:,}",   "#3b82f6", "Total reviews collected from Steam"),
            (m2, "Positive Reviews",     f"{pos_sayi:,}", "#22c55e", "Reviews with voted_up=True"),
            (m3, "Negative Reviews",     f"{neg_sayi:,}", "#ef4444", "Reviews with voted_up=False"),
            (m4, "Complaint Categories", str(aktif_kat),  "#8b5cf6",
             "Complaint categories detected in negative reviews"),
        ]
    for w, label, val, renk, aciklama in metrik_info:
        w.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{renk}">{val}</div>'
            f'<div class="metric-label">{label}</div>'
            f'<div style="font-size:0.70rem;color:#94a3b8;margin-top:0.2rem;">{aciklama}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2. Olumlu/olumsuz bar  +  3. Dil dağılımı (sadece olumsuz) ───────
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(grafik_olumlu_olumsuz(df, lang), use_container_width=True)
    with g2:
        st.plotly_chart(grafik_dil_pasta(neg_df_detay, lang, neg_only=True), use_container_width=True)
        if lang == "TR":
            st.caption("Bu grafik yalnızca olumsuz yorumların hangi dillerde yazıldığını göstermektedir.")
        else:
            st.caption("This chart shows the language distribution of negative reviews only.")

    # ── 4. Şikayet kategorileri (sadece olumsuz yorumlar) ─────────────────
    st.markdown(f"<div class='sec-title'>{s['sec_complaints']}</div>", unsafe_allow_html=True)
    if lang == "TR":
        st.markdown(
            "Aşağıdaki grafik, bu oyunun **OLUMSUZ** yorumları içinde her şikayet kategorisinin "
            "kaç yorumda geçtiğini göstermektedir.  \n"
            "*Örnek: Oynanabilirlik: 85 → Olumsuz yorumların 85 tanesinde "
            "oynanabilirlik şikayeti tespit edilmiştir.*"
        )
    else:
        st.markdown(
            "The chart below shows how many **negative** reviews contain each complaint category.  \n"
            "*Example: Gameplay: 85 → 85 negative reviews contained a gameplay complaint.*"
        )
    st.plotly_chart(
        grafik_kategori_bar(neg_df_detay, lang, neg_only=True, neg_toplam=neg_sayi),
        use_container_width=True,
    )
    render_kategori_yorumu(neg_df_detay, kat_col, etiketler, neg_sayi)

    # ── 5. VADER duygu analizi ────────────────────────────────────────────
    render_vader_analizi(df, lang)

    # ── 6. Türler arası karşılaştırma heat map ────────────────────────────
    st.markdown(f"<div class='sec-title'>{s['sec_comparison']}</div>", unsafe_allow_html=True)
    st.plotly_chart(grafik_tur_heatmap(tur, lang), use_container_width=True)
    st.caption("Her değer ilgili oyunun OLUMSUZ yorumları içindeki kategori yüzdesini gösterir.")

    # ── 8. Aksiyon önerileri ──────────────────────────────────────────────
    st.markdown(f"<div class='sec-title'>{s['sec_actions']}</div>", unsafe_allow_html=True)
    oneriler = aksiyon_uret(df, lang)
    if oneriler:
        for i, oneri in enumerate(oneriler, 1):
            st.markdown(
                f'<div class="action-card">'
                f'<span class="action-rank">{i}</span>{oneri}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info(s["no_actions"])

    # ── 9. Gizli Şikayetler (Mixed Sentiment) ────────────────────────────
    if "vader_compound" in df.columns:
        pos_df_gs   = df[df["voted_up_bool"] == True]
        vader_gs    = pd.to_numeric(pos_df_gs["vader_compound"], errors="coerce")
        mixed_sayi  = int((vader_gs < -0.05).sum())
        mixed_pct_g = mixed_sayi / max(1, len(pos_df_gs)) * 100
        if lang == "TR":
            st.markdown(
                "<div class='sec-title'>💡 Gizli Şikayetler (Mixed Sentiment)</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "Aşağıdaki sayı, oyunu tavsiye eden (olumlu yorum bırakan) ancak yorum metninde "
                "olumsuz ifadeler kullanan oyuncu sayısını göstermektedir.  \n"
                "*Bu yorumlar 'oyun iyi ama şu eksik' yapısında cümleler içermektedir.*"
            )
            st.info(
                f"**{mixed_sayi:,}** olumlu yorumun içinde gizli şikayet tespit edildi "
                f"*(olumlu yorumların %{mixed_pct_g:.1f}'si)*"
            )
        else:
            st.markdown(
                "<div class='sec-title'>💡 Hidden Complaints (Mixed Sentiment)</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "The number below shows players who recommended the game (positive review) "
                "but used negative language in their review text.  \n"
                "*These reviews contain sentences like 'the game is good but this is missing'.*"
            )
            st.info(
                f"**{mixed_sayi:,}** positive reviews contain hidden complaints "
                f"*({mixed_pct_g:.1f}% of positive reviews)*"
            )

    st.divider()
    _render_pdf_butonu(oyun, lang, "detay")

# ─── Yardımcı: hex → rgba ────────────────────────────────────────────────────

def _hex_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _parse_timestamp(df: pd.DataFrame) -> pd.Series:
    ts_raw = df["timestamp_created"]
    ts_num = pd.to_numeric(ts_raw, errors="coerce")
    if ts_num.notna().sum() > len(df) * 0.4:
        return pd.to_datetime(ts_num, unit="s", errors="coerce")
    return pd.to_datetime(ts_raw, errors="coerce")


# ─── MODÜL 1: Rakip Karşılaştırma Motoru ─────────────────────────────────────

def _rakip_veri_hesapla(tur_adi: str) -> dict:
    kat_keys = list(_KAT_TR.keys())
    sonuc = {}
    for oyun in TURLER[tur_adi]:
        df = oyun_yukle(oyun)
        if df.empty:
            continue
        # voted_up_bool normalizasyonu
        if "voted_up" in df.columns:
            df["voted_up_bool"] = df["voted_up"].apply(
                lambda x: True if str(x).strip().lower() in ["true", "1", "yes"] else False
            )
        else:
            df["voted_up_bool"] = False

        toplam = len(df)
        pos_sayi_r = int(df["voted_up_bool"].sum())
        pos_pct = pos_sayi_r / toplam * 100 if toplam else 0.0

        scol = next((c for c in ("sentiment", "vader_label") if c in df.columns), None)
        if scol:
            voted_bool = df["voted_up_bool"]
            sent_bool = df[scol] == "positive"
            vader_uyum = float((voted_bool == sent_bool).mean() * 100)
        else:
            vader_uyum = 0.0

        # Şikayet oranları: sadece voted_up_bool=False olan olumsuz yorumlardan
        neg_df = df[df["voted_up_bool"] == False]
        neg_toplam = len(neg_df)
        kat_pcts = {}
        for k in kat_keys:
            kat_pcts[k] = float(neg_df[k].sum() / max(1, neg_toplam) * 100) if k in neg_df.columns else 0.0

        # Gizli şikayet: voted_up_bool=True + negatif vader_compound
        if "vader_compound" in df.columns:
            pos_df_m = df[df["voted_up_bool"] == True]
            vader_num = pd.to_numeric(pos_df_m["vader_compound"], errors="coerce")
            mixed_pct = float((vader_num < -0.05).sum() / max(1, len(pos_df_m)) * 100)
        else:
            mixed_pct = 0.0

        sonuc[oyun] = {
            "toplam": toplam,
            "neg_sayi": neg_toplam,
            "pos_pct": pos_pct,
            "neg_pct": 100.0 - pos_pct,
            "vader_uyum": vader_uyum,
            "kat_pcts": kat_pcts,
            "mixed_pct": mixed_pct,
        }
    return sonuc


def _grafik_radar(veri: dict, secili_oyun: str, lang: str) -> go.Figure:
    etiketler = kat_etiketler(lang)
    kat_keys = list(etiketler.keys())
    kat_labels = [etiketler[k] for k in kat_keys]
    renk_listesi = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6"]
    fig = go.Figure()
    for i, (oyun, metr) in enumerate(veri.items()):
        vals = [metr["kat_pcts"].get(k, 0) for k in kat_keys]
        vals_c = vals + [vals[0]]
        lbls_c = kat_labels + [kat_labels[0]]
        renk = renk_listesi[i % len(renk_listesi)]
        kalınlık = 3 if oyun == secili_oyun else 1.5
        opaklık = 1.0 if oyun == secili_oyun else 0.65
        fig.add_trace(go.Scatterpolar(
            r=vals_c, theta=lbls_c, fill="toself",
            fillcolor=_hex_rgba(renk, 0.18),
            name=OYUN_ADLARI.get(oyun, oyun),
            line=dict(color=renk, width=kalınlık),
            opacity=opaklık,
        ))
    title = "Kategori Karşılaştırması — Şikayet Oranı (%)" if lang == "TR" else "Category Comparison — Complaint Rate (%)"
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 50], color="#94a3b8"), bgcolor="#0f172a"),
        showlegend=True, title=title, template="plotly_dark",
        height=500, margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def _otomatik_karsilastirma_metni(veri: dict, secili_oyun: str, lang: str) -> str:
    if secili_oyun not in veri or len(veri) < 2:
        return ""
    etiketler = kat_etiketler(lang)
    secili = veri[secili_oyun]
    kat_pcts = secili["kat_pcts"]
    en_kotu_kat = max(kat_pcts, key=lambda k: kat_pcts[k])
    en_kotu_pct = kat_pcts[en_kotu_kat]
    ort_pct = sum(v["kat_pcts"].get(en_kotu_kat, 0) for v in veri.values()) / len(veri)
    oran = en_kotu_pct / max(0.01, ort_pct)
    rakipler = {o: m for o, m in veri.items() if o != secili_oyun}
    en_iyi_rakip = min(rakipler, key=lambda o: rakipler[o]["kat_pcts"].get(en_kotu_kat, 999))
    en_iyi_pct = rakipler[en_iyi_rakip]["kat_pcts"].get(en_kotu_kat, 0)
    en_iyi_ad = OYUN_ADLARI.get(en_iyi_rakip, en_iyi_rakip)
    secili_ad = OYUN_ADLARI.get(secili_oyun, secili_oyun)
    kat_et = etiketler.get(en_kotu_kat, en_kotu_kat)
    en_iyi_genel = max(veri, key=lambda o: veri[o]["pos_pct"])
    en_iyi_genel_ad = OYUN_ADLARI.get(en_iyi_genel, en_iyi_genel)
    en_iyi_genel_pct = veri[en_iyi_genel]["pos_pct"]
    en_kotu_genel = max(veri, key=lambda o: veri[o]["neg_pct"])
    en_kotu_genel_ad = OYUN_ADLARI.get(en_kotu_genel, en_kotu_genel)
    if lang == "TR":
        return (
            f"**🔍 Otomatik Analiz:**\n\n"
            f"1. **{secili_ad}**'nin en yüksek şikayet kategorisi **{kat_et}** olup oran **%{en_kotu_pct:.1f}**. "
            f"Bu, tür ortalamasının (**%{ort_pct:.1f}**) **{oran:.1f}x** katıdır. "
            f"Bu kategoride en iyi performans **{en_iyi_ad}** oyununa ait (**%{en_iyi_pct:.1f}**).\n\n"
            f"2. Tür genelinde en yüksek olumlu yorum oranı **{en_iyi_genel_ad}** oyununa ait "
            f"(**%{en_iyi_genel_pct:.1f}** olumlu). Bu oyunun başarı faktörleri örnek alınmalıdır.\n\n"
            f"3. **{secili_ad}** için öncelikli iyileştirme alanı **{kat_et}** kategorisidir. "
            f"Şikayet oranını **%{en_iyi_pct:.1f}** seviyesine indirmek, "
            f"oyunun tür sıralamasında anlamlı bir yükseliş sağlayabilir. "
            f"En fazla olumsuz yorum oranına sahip oyun ise **{en_kotu_genel_ad}**'dır."
        )
    else:
        return (
            f"**🔍 Automated Analysis:**\n\n"
            f"1. **{secili_ad}**'s highest complaint category is **{kat_et}** at **{en_kotu_pct:.1f}%**. "
            f"This is **{oran:.1f}x** the genre average (**{ort_pct:.1f}%**). "
            f"The best performer in this category is **{en_iyi_ad}** (**{en_iyi_pct:.1f}%**).\n\n"
            f"2. The game with the highest positive review rate in this genre is **{en_iyi_genel_ad}** "
            f"(**{en_iyi_genel_pct:.1f}%** positive). Its success factors should be studied as a benchmark.\n\n"
            f"3. The primary improvement area for **{secili_ad}** is the **{kat_et}** category. "
            f"Reducing the complaint rate to **{en_iyi_pct:.1f}%** could meaningfully improve its genre ranking. "
            f"The game with the most negative reviews overall is **{en_kotu_genel_ad}**."
        )


def render_rakip_analizi(lang: str):
    if lang == "TR":
        st.markdown("## 🏆 Rakip Karşılaştırma Motoru")
        st.caption("Seçtiğiniz oyunu aynı türdeki rakiplerle karşılaştırın")
        tur_lbl = "Oyun Türü Seçin"
        oyun_lbl = "Karşılaştırılacak Oyunu Seçin"
        tablo_lbl = "📊 Karşılaştırma Tablosu"
        radar_lbl = "🕸️ Radar Grafiği"
        analiz_lbl = "📝 Otomatik Karşılaştırma"
        pos_lbl, neg_lbl, vader_lbl, sira_lbl = "Olumlu %", "Olumsuz %", "VADER Uyum %", "Sıra"
        no_data = "Yeterli veri bulunamadı."
    else:
        st.markdown("## 🏆 Competitor Analysis Engine")
        st.caption("Compare the selected game against rivals in the same genre")
        tur_lbl = "Select Genre"
        oyun_lbl = "Select Game to Compare"
        tablo_lbl = "📊 Comparison Table"
        radar_lbl = "🕸️ Radar Chart"
        analiz_lbl = "📝 Automated Comparison"
        pos_lbl, neg_lbl, vader_lbl, sira_lbl = "Positive %", "Negative %", "VADER Match %", "Rank"
        no_data = "Not enough data available."

    etiketler = kat_etiketler(lang)
    c1, c2 = st.columns([1, 2])
    with c1:
        secili_tur = st.selectbox(tur_lbl, list(TURLER.keys()), key="rakip_tur")
    with c2:
        oyun_list = TURLER[secili_tur]
        secili_idx = st.selectbox(
            oyun_lbl, range(len(oyun_list)),
            format_func=lambda i: OYUN_ADLARI.get(oyun_list[i], oyun_list[i]),
            key="rakip_oyun",
        )
        secili_oyun = oyun_list[secili_idx]

    try:
        veri = _rakip_veri_hesapla(secili_tur)
        if not veri:
            st.warning(no_data)
            return

        # ── Karşılaştırma tablosu ──────────────────────────────────────────
        st.markdown(f"<div class='sec-title'>{tablo_lbl}</div>", unsafe_allow_html=True)

        neg_sayi_lbl = "Olumsuz Yorum Sayısı" if lang == "TR" else "Negative Review Count"
        satirlar = []
        for oyun, metr in veri.items():
            neg_s = metr.get("neg_sayi", 0)
            neg_label = f"⚠️ {neg_s}" if neg_s < 30 else str(neg_s)
            satir = {
                "Oyun": OYUN_ADLARI.get(oyun, oyun),
                pos_lbl: metr["pos_pct"],
                neg_lbl: metr["neg_pct"],
                neg_sayi_lbl: neg_label,
            }
            for kat_key, kat_et in etiketler.items():
                satir[kat_et] = metr["kat_pcts"].get(kat_key, 0.0)
            satir[vader_lbl] = metr["vader_uyum"]
            satir["_oyun_key"] = oyun
            satirlar.append(satir)

        tablo_df = pd.DataFrame(satirlar).sort_values(pos_lbl, ascending=False).reset_index(drop=True)
        tablo_df.insert(0, sira_lbl, range(1, len(tablo_df) + 1))

        def _stil(satir):
            oyun_key = tablo_df.at[satir.name, "_oyun_key"]
            if oyun_key == secili_oyun:
                return ["background-color:#1e3a5f; color:white;"] * len(satir)
            return [""] * len(satir)

        pct_cols = [pos_lbl, neg_lbl, vader_lbl] + list(etiketler.values())
        display_df = tablo_df.drop(columns=["_oyun_key"])
        fmt = {c: "{:.1f}%" for c in pct_cols if c in display_df.columns}
        st.dataframe(display_df.style.apply(_stil, axis=1).format(fmt), use_container_width=True, hide_index=True)

        # Olumsuz yorum bazlı hesaplama notu
        if lang == "TR":
            st.caption(
                "ℹ️ **Olumsuz Yorumlardaki Şikayet Oranı (%)** — Bu yüzdeler yalnızca olumsuz yorumlar baz alınarak hesaplanmıştır.  \n"
                "⚠️ Not: Olumsuz yorum sayısı 30'un altında olan oyunlarda yüzdeler istatistiksel olarak güvenilir olmayabilir."
            )
        else:
            st.caption(
                "ℹ️ **Complaint Rate in Negative Reviews (%)** — These percentages are calculated based on negative reviews only.  \n"
                "⚠️ Note: Percentages may not be statistically reliable for games with fewer than 30 negative reviews."
            )

        # Gizli şikayet (mixed sentiment) bilgisi
        mixed = veri.get(secili_oyun, {}).get("mixed_pct", 0.0)
        secili_ad_rakip = OYUN_ADLARI.get(secili_oyun, secili_oyun)
        if lang == "TR":
            st.info(
                f"🔍 **Gizli Şikayet Tespiti:** **{secili_ad_rakip}**'in olumlu yorumlarının "
                f"**%{mixed:.1f}**'i gizli şikayet içeriyor.\n\n"
                f"*Oyunu tavsiye eden ancak olumsuz ifadeler kullanan yorumlar*"
            )
        else:
            st.info(
                f"🔍 **Hidden Complaint Detection:** **{mixed:.1f}%** of **{secili_ad_rakip}**'s "
                f"positive reviews contain hidden complaints.\n\n"
                f"*Reviews that recommend the game but use negative sentiment language*"
            )

        # ── Radar grafiği ──────────────────────────────────────────────────
        st.markdown(f"<div class='sec-title'>{radar_lbl}</div>", unsafe_allow_html=True)
        st.plotly_chart(_grafik_radar(veri, secili_oyun, lang), use_container_width=True)

        # ── Otomatik metin ─────────────────────────────────────────────────
        metin = _otomatik_karsilastirma_metni(veri, secili_oyun, lang)
        if metin:
            st.markdown(f"<div class='sec-title'>{analiz_lbl}</div>", unsafe_allow_html=True)
            st.markdown(metin)
    except Exception:
        st.warning(no_data)

    st.divider()
    _render_pdf_butonu(secili_oyun, lang, "rakip")


# ─── MODÜL 2: Şikayet Trendi Tahmini ─────────────────────────────────────────

def _trend_hesapla(df: pd.DataFrame, kat: str):
    if "timestamp_created" not in df.columns or kat not in df.columns:
        return None, None, None
    df2 = df[["timestamp_created", kat]].copy()
    df2["_ts"] = _parse_timestamp(df2)
    df2 = df2.dropna(subset=["_ts"])
    if len(df2) == 0:
        return None, None, None
    df2["_hafta"] = df2["_ts"].dt.to_period("W")
    aylik = df2.groupby("_hafta")[kat].sum().reset_index()
    aylik.columns = ["hafta", kat]
    aylik = aylik.sort_values("hafta").reset_index(drop=True)
    aylik["hafta_str"] = aylik["hafta"].apply(lambda p: p.start_time.strftime("%d %b"))
    aylik["hafta_idx"] = np.arange(len(aylik))
    aylik[kat] = aylik[kat].astype(int)

    if len(aylik) < 2:
        return None, None, None

    x = aylik["hafta_idx"].values.astype(float)
    y = aylik[kat].values.astype(float)
    katsayi = np.polyfit(x, y, 1)

    son_idx = int(x[-1])
    son_hafta = aylik["hafta"].iloc[-1]
    proj_rows = []
    for i in range(1, 4):
        val = max(0, int(round(np.polyval(katsayi, son_idx + i))))
        proj_rows.append({"hafta_str": (son_hafta + i).start_time.strftime("%d %b"), "hafta_idx": son_idx + i, kat: val})
    projeksiyon = pd.DataFrame(proj_rows)

    # Trend yüzdesi: son yarı vs ilk yarı ortalaması
    yarı = max(2, len(aylik) // 2)
    ilk_ort = aylik[kat].iloc[:yarı].mean()
    son_ort = aylik[kat].iloc[-yarı:].mean()
    trend_pct = (son_ort - ilk_ort) / max(1.0, ilk_ort) * 100

    return aylik, projeksiyon, trend_pct


def _grafik_trend(aylik: pd.DataFrame, projeksiyon, kat: str, lang: str) -> go.Figure:
    gercek_lbl = "Gerçek Veri" if lang == "TR" else "Actual Data"
    tahmin_lbl = "3 Haftalık Tahmin" if lang == "TR" else "3-Week Forecast"
    y_lbl = "Şikayet Sayısı" if lang == "TR" else "Complaint Count"
    title = "Haftalık Şikayet Trendi + Projeksiyon" if lang == "TR" else "Weekly Complaint Trend + Projection"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=aylik["hafta_str"], y=aylik[kat],
        mode="lines+markers", name=gercek_lbl,
        line=dict(color="#3b82f6", width=2.5), marker=dict(size=6),
    ))
    if projeksiyon is not None and not projeksiyon.empty:
        proj_x = [aylik["hafta_str"].iloc[-1]] + list(projeksiyon["hafta_str"])
        proj_y = [int(aylik[kat].iloc[-1])] + list(projeksiyon[kat])
        fig.add_trace(go.Scatter(
            x=proj_x, y=proj_y,
            mode="lines+markers", name=tahmin_lbl,
            line=dict(color="#ef4444", width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        ))
    fig.update_layout(
        title=title, template="plotly_dark", height=380,
        margin=dict(t=50, b=60, l=0, r=0),
        xaxis_title="", yaxis_title=y_lbl,
        legend=dict(x=0.01, y=0.99),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def render_trend_tahmini(lang: str):
    if lang == "TR":
        st.markdown("## 📈 Şikayet Trendi Tahmini")
        st.caption("Seçili oyun ve kategorideki şikayet trendini analiz edin, gelecek 3 ayı tahmin edin")
        oyun_lbl  = "Oyun Seçin"
        kat_lbl   = "Şikayet Kategorisi Seçin"
        trend_lbl = "📉 Haftalık Trend + 3 Haftalık Projeksiyon"
        ozet_lbl  = "📋 Tüm Kategoriler — Haftalık Trend Özeti"
        kat_col_lbl, trend_col_lbl, degisim_col_lbl = "Kategori", "Son 3 Hafta Trendi", "Değişim %"
        artis, azalis, stabil = "↑ Artıyor", "↓ Azalıyor", "→ Stabil"
        no_data   = "Yeterli veri bulunamadı."
        neg_suffix = "olumsuz yorum"
        az_veri_msg = (
            "Bu oyunun olumsuz yorum sayısı ({n}) anlamlı bir trend analizi için yetersizdir.\n\n"
            "Trend analizi en az 50 olumsuz yorum gerektirmektedir.\n\n"
            "Bu oyun için Oyun Analizi sekmesindeki şikayet kategorisi dağılımını incelemeniz önerilir."
        )
    else:
        st.markdown("## 📈 Complaint Trend Prediction")
        st.caption("Analyze complaint trends for a selected game and forecast the next 3 months")
        oyun_lbl  = "Select Game"
        kat_lbl   = "Select Complaint Category"
        trend_lbl = "📉 Weekly Trend + 3-Week Projection"
        ozet_lbl  = "📋 All Categories — Weekly Trend Summary"
        kat_col_lbl, trend_col_lbl, degisim_col_lbl = "Category", "Last 3 Weeks Trend", "Change %"
        artis, azalis, stabil = "↑ Rising", "↓ Declining", "→ Stable"
        no_data   = "Not enough data available."
        neg_suffix = "negative reviews"
        az_veri_msg = (
            "This game's negative review count ({n}) is insufficient for a meaningful trend analysis.\n\n"
            "Trend analysis requires at least 50 negative reviews.\n\n"
            "It is recommended to check the complaint category distribution in the Game Analysis tab."
        )

    etiketler = kat_etiketler(lang)

    # Sadece 50+ olumsuz yorumu olan oyunları listele
    oyun_listesi = []
    for tur in TURLER.values():
        for oyun in tur:
            df_tmp = oyun_yukle(oyun)
            if df_tmp.empty:
                continue
            neg_n = int((df_tmp["voted_up_bool"] == False).sum())
            if neg_n >= 50:
                etiket = f"{OYUN_ADLARI.get(oyun, oyun)} ({neg_n} {neg_suffix})"
                oyun_listesi.append((oyun, etiket, neg_n))

    if not oyun_listesi:
        st.warning(no_data)
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        oyun_idx = st.selectbox(oyun_lbl, range(len(oyun_listesi)),
                                format_func=lambda i: oyun_listesi[i][1], key="trend_oyun")
        secili_oyun = oyun_listesi[oyun_idx][0]
    with c2:
        kat_listesi = list(etiketler.items())
        kat_idx = st.selectbox(kat_lbl, range(len(kat_listesi)),
                               format_func=lambda i: kat_listesi[i][1], key="trend_kat")
        secili_kat_key, secili_kat_label = kat_listesi[kat_idx]

    try:
        df = oyun_yukle(secili_oyun)
        if df.empty:
            st.warning(no_data)
            return

        neg_df_trend = df[df["voted_up_bool"] == False]
        neg_sayi_trend = len(neg_df_trend)

        # ── Yetersiz veri kontrolü ────────────────────────────────────────
        if neg_sayi_trend < 50:
            st.warning(f"⚠️ {az_veri_msg.format(n=neg_sayi_trend)}")
            st.divider()
            _render_pdf_butonu(secili_oyun, lang, "trend")
            return

        aylik, projeksiyon, trend_pct = _trend_hesapla(neg_df_trend, secili_kat_key)

        if aylik is None:
            st.warning(no_data)
            return

        # Başlangıç referans değeri yetersizse uyarıları bastır
        yarı_uw = max(2, len(aylik) // 2)
        baz_deger = float(aylik[secili_kat_key].iloc[:yarı_uw].mean())
        veri_yeterli = baz_deger >= 3

        # ── Uyarı sistemi (yalnızca yeterli veri varsa) ────────────────────
        if trend_pct is not None and veri_yeterli:
            if trend_pct > 20:
                if lang == "TR":
                    st.error(f"🔴 KRİTİK: {secili_kat_label} şikayetleri hızla artıyor "
                             f"(%{trend_pct:.1f}). 3 ay içinde en yüksek seviyeye ulaşabilir!")
                else:
                    st.error(f"🔴 CRITICAL: {secili_kat_label} complaints are rising rapidly "
                             f"({trend_pct:.1f}%). Could peak within 3 months!")
            elif trend_pct > 10:
                if lang == "TR":
                    st.warning(f"🟡 UYARI: {secili_kat_label} kategorisinde artış trendi var "
                               f"(%{trend_pct:.1f}). Yakından takip edilmeli.")
                else:
                    st.warning(f"🟡 WARNING: Rising trend in {secili_kat_label} "
                               f"({trend_pct:.1f}%). Needs close monitoring.")
            else:
                if lang == "TR":
                    st.success(f"🟢 NORMAL: {secili_kat_label} şikayet trendi kontrol altında "
                               f"(%{trend_pct:.1f}).")
                else:
                    st.success(f"🟢 NORMAL: {secili_kat_label} complaint trend is under control "
                               f"({trend_pct:.1f}%).")

        # ── Trend grafiği ──────────────────────────────────────────────────
        st.markdown(f"<div class='sec-title'>{trend_lbl}</div>", unsafe_allow_html=True)
        st.plotly_chart(_grafik_trend(aylik, projeksiyon, secili_kat_key, lang), use_container_width=True)

        if projeksiyon is not None and not projeksiyon.empty and trend_pct is not None:
            if not veri_yeterli:
                if lang == "TR":
                    st.info("📊 Trend: Yetersiz veri — Analiz için yeterli haftalık veri bulunmamaktadır.")
                else:
                    st.info("📊 Trend: Insufficient data — Not enough weekly data for analysis.")
            else:
                trend_et = artis if trend_pct > 10 else (azalis if trend_pct < -5 else stabil)
                proj_val = int(projeksiyon[secili_kat_key].iloc[-1])
                if lang == "TR":
                    st.info(f"📊 Trend: {trend_et} | Dönem değişimi: %{trend_pct:.1f} | "
                            f"Tahmini 3. hafta değeri: {proj_val:,} şikayet")
                else:
                    st.info(f"📊 Trend: {trend_et} | Period change: {trend_pct:.1f}% | "
                            f"Estimated week-3 value: {proj_val:,} complaints")

        # ── Otomatik trend yorumu (düz metin, bold işareti yok) ───────────
        if aylik is not None and len(aylik) > 0:
            if not veri_yeterli:
                auto_yorum = ("Bu kategori için yeterli veri bulunmamaktadır. "
                              "Anlamlı bir trend analizi yapılamamaktadır."
                              if lang == "TR" else
                              "There is not enough data for this category. "
                              "A meaningful trend analysis cannot be performed.")
            else:
                if len(aylik) >= 6:
                    son_3_ort  = aylik[secili_kat_key].iloc[-3:].mean()
                    once_3_ort = aylik[secili_kat_key].iloc[-6:-3].mean()
                elif len(aylik) >= 3:
                    son_3_ort  = aylik[secili_kat_key].iloc[-3:].mean()
                    once_3_ort = aylik[secili_kat_key].iloc[:max(1, len(aylik) - 3)].mean()
                else:
                    son_3_ort  = aylik[secili_kat_key].mean()
                    once_3_ort = son_3_ort
                son_3_degisim = (son_3_ort - once_3_ort) / max(1.0, once_3_ort) * 100

                if lang == "TR":
                    if son_3_degisim > 5:
                        auto_yorum = (
                            f"📈 Bu kategorideki şikayetler artış trendinde. "
                            f"Son 3 haftada %{abs(son_3_degisim):.1f} artış gözlemlendi. "
                            f"Geliştiricilerin bu konuya öncelik vermesi önerilir."
                        )
                    elif son_3_degisim < -5:
                        auto_yorum = (
                            f"📉 Bu kategorideki şikayetler azalış trendinde. "
                            f"Son 3 haftada %{abs(son_3_degisim):.1f} düşüş gözlemlendi. "
                            f"Mevcut iyileştirmeler olumlu etki yaratıyor."
                        )
                    else:
                        auto_yorum = (
                            "➡️ Bu kategorideki şikayetler stabil seyrediyor. "
                            "Belirgin bir artış veya azalış gözlemlenmedi."
                        )
                else:
                    if son_3_degisim > 5:
                        auto_yorum = (
                            f"📈 Complaints in this category are on an upward trend. "
                            f"A {abs(son_3_degisim):.1f}% increase was observed in the last 3 weeks. "
                            f"Developers are recommended to prioritize this issue."
                        )
                    elif son_3_degisim < -5:
                        auto_yorum = (
                            f"📉 Complaints in this category are on a downward trend. "
                            f"A {abs(son_3_degisim):.1f}% decrease was observed in the last 3 weeks. "
                            f"Current improvements are having a positive effect."
                        )
                    else:
                        auto_yorum = (
                            "➡️ Complaints in this category are stable. "
                            "No significant increase or decrease was observed."
                        )
            st.markdown(
                f'<div style="background:#1e293b;border-left:4px solid #3b82f6;'
                f'border-radius:6px;padding:0.7rem 1rem;margin-top:0.5rem;'
                f'font-size:0.88rem;color:#cbd5e1;">{auto_yorum}</div>',
                unsafe_allow_html=True,
            )

        # ── Tüm kategoriler — son 3 ay özet tablosu ───────────────────────
        st.markdown(f"<div class='sec-title'>{ozet_lbl}</div>", unsafe_allow_html=True)
        yetersiz_lbl = "Yetersiz veri" if lang == "TR" else "Insufficient data"
        ozet_rows = []
        for kk, kl in etiketler.items():
            try:
                aylik_k, _, _ = _trend_hesapla(neg_df_trend, kk)
                if aylik_k is None or len(aylik_k) < 2:
                    ozet_rows.append({
                        kat_col_lbl:     kl,
                        trend_col_lbl:   "—",
                        degisim_col_lbl: yetersiz_lbl,
                    })
                    continue
                # Son 3 ay vs öncesi değişimi
                if len(aylik_k) >= 6:
                    s3 = aylik_k[kk].iloc[-3:].mean()
                    o3 = aylik_k[kk].iloc[-6:-3].mean()
                else:
                    s3 = aylik_k[kk].iloc[-3:].mean()
                    o3 = aylik_k[kk].iloc[:max(1, len(aylik_k) - 3)].mean()

                # Başlangıç değeri çok düşükse güvenilmez
                if o3 < 3:
                    ozet_rows.append({
                        kat_col_lbl:     kl,
                        trend_col_lbl:   "—",
                        degisim_col_lbl: yetersiz_lbl,
                    })
                    continue

                degisim = (s3 - o3) / o3 * 100

                if degisim > 10:
                    trend_icon = f"🔴 {artis}"
                elif degisim > 5:
                    trend_icon = f"🟡 {artis}"
                elif degisim < -5:
                    trend_icon = f"🟢 {azalis}"
                else:
                    trend_icon = f"🟢 {stabil}"

                degisim_str = f"+%{degisim:.1f}" if degisim >= 0 else f"-%{abs(degisim):.1f}"
                if lang != "TR":
                    degisim_str = f"+{degisim:.1f}%" if degisim >= 0 else f"{degisim:.1f}%"

                ozet_rows.append({
                    kat_col_lbl:     kl,
                    trend_col_lbl:   trend_icon,
                    degisim_col_lbl: degisim_str,
                })
            except Exception:
                pass
        if ozet_rows:
            st.dataframe(pd.DataFrame(ozet_rows), use_container_width=True, hide_index=True)
        if lang == "TR":
            st.caption(
                "ℹ️ Kategori yüzdeleri yalnızca olumsuz yorumlar baz alınarak hesaplanmıştır.  \n"
                "Not: Az veri içeren kategorilerde yüzde değişimleri yanıltıcı olabilir."
            )
        else:
            st.caption(
                "ℹ️ Category percentages are calculated based on negative reviews only.  \n"
                "Note: Percentage changes may be misleading for categories with little data."
            )
    except Exception:
        st.warning(no_data)

    st.divider()
    _render_pdf_butonu(secili_oyun, lang, "trend")


# ─── MODÜL 3: ROI Simülatörü ─────────────────────────────────────────────────

def _roi_hesapla(neg_yorum: int, toplam: int, basari: float,
                 aktif_oyuncu: int, gelir_per: float, maliyet: float) -> dict:
    neg_pct = neg_yorum / max(1, toplam) * 100
    donusen = int(neg_yorum * basari)
    etkilenen = int(aktif_oyuncu * neg_pct / 100)
    kazanilan = int(etkilenen * basari)
    aylik_ek = kazanilan * gelir_per
    yillik_ek = aylik_ek * 12
    roi = (yillik_ek - maliyet) / max(1.0, maliyet) * 100
    geri_don = maliyet / max(0.01, aylik_ek) if aylik_ek > 0 else float("inf")
    return dict(neg_pct=neg_pct, neg_yorum=neg_yorum, donusen=donusen,
                kazanilan=kazanilan, aylik_ek=aylik_ek, yillik_ek=yillik_ek,
                roi=roi, geri_don=geri_don)


def _grafik_gauge(roi: float, lang: str) -> go.Figure:
    title = "Net ROI Göstergesi" if lang == "TR" else "Net ROI Gauge"
    renk = "#22c55e" if roi >= 0 else "#ef4444"
    roi_display = max(-200.0, min(500.0, roi))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=roi_display,
        number={"suffix": "%", "font": {"size": 30, "color": renk}},
        title={"text": title, "font": {"size": 15}},
        gauge={
            "axis": {"range": [-200, 500], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": renk, "thickness": 0.25},
            "bgcolor": "#0f172a",
            "borderwidth": 1, "bordercolor": "#334155",
            "steps": [
                {"range": [-200, 0], "color": "#450a0a"},
                {"range": [0, 200], "color": "#052e16"},
                {"range": [200, 500], "color": "#14532d"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": roi_display},
        },
    ))
    fig.update_layout(template="plotly_dark", height=260, margin=dict(t=60, b=10, l=30, r=30))
    return fig


def render_roi_simulator(lang: str):
    if lang == "TR":
        st.markdown("## 💰 ROI Simülatörü")
        st.caption("Şikayet kategorisi çözümünün finansal etkisini anlık hesaplayın")
        sub1_lbl = "📊 Tek Senaryo"
        sub2_lbl = "⚔️ Senaryo Karşılaştırma"
        param_lbl = "### ⚙️ Parametreler"
        sonuc_lbl = "### 📈 Sonuçlar"
        oyun_lbl = "Oyun Seçin"
        kat_lbl = "Çözülecek Şikayet Kategorisi"
        maliyet_lbl = "Tahmini Geliştirme Maliyeti ($)"
        gelir_lbl = "Oyun Başına Ortalama Gelir ($)"
        oyuncu_lbl = "Mevcut Aylık Aktif Oyuncu"
        basari_lbl = "Şikayet Çözüm Başarı Oranı (%)"
        metrik_lbl = "📊 Temel Metrikler"
        finans_lbl = "💵 Finansal Projeksiyon"
        oneri_lbl = "💡 Otomatik Öneri"
        oncelik_lbl = "🏆 Kategori Öncelik Sıralaması"
        oncelik_cap = "Aynı parametrelerle tüm kategorilerin tahmini ROI değerleri"
        no_data = "Yeterli veri bulunamadı."
        karli, kayip = "Karlı Yatırım ✓", "Maliyet Yüksek ✗"
        neg_lbl, donusen_lbl, kazanilan_lbl = "Olumsuz Yorum", "Olumluya Dönecek", "Geri Kazanılan Oyuncu"
        aylik_lbl, yillik_lbl, geri_don_lbl = "Aylık Ek Gelir", "Yıllık Ek Gelir", "Geri Dönüş Süresi"
        kat_col, neg_col, geri_col = "Kategori", "Olumsuz", "Geri Dönüş (ay)"
        sen_a_lbl  = "🟦 Senaryo A"
        sen_b_lbl  = "🟨 Senaryo B"
        kat_a_lbl  = "Çözülecek Kategori — Senaryo A"
        kat_b_lbl  = "Çözülecek Kategori — Senaryo B"
        mal_a_lbl  = "Maliyet A ($)"
        mal_b_lbl  = "Maliyet B ($)"
        bas_a_lbl  = "Başarı Oranı A (%)"
        bas_b_lbl  = "Başarı Oranı B (%)"
        kars_tablo = "📊 Karşılaştırma Tablosu"
        karar_lbl  = "💡 Otomatik Karar"
        kazan_oyuncu_lbl = "Geri Kazanılan Oyuncu"
        ek_gelir_lbl = "Tahmini Ek Gelir ($)"
        net_roi_lbl = "Net ROI %"
        geri_sure_lbl = "Geri Dönüş (ay)"
    else:
        st.markdown("## 💰 ROI Simulator")
        st.caption("Calculate the financial impact of resolving a complaint category in real time")
        sub1_lbl = "📊 Single Scenario"
        sub2_lbl = "⚔️ Scenario Comparison"
        param_lbl = "### ⚙️ Parameters"
        sonuc_lbl = "### 📈 Results"
        oyun_lbl = "Select Game"
        kat_lbl = "Complaint Category to Resolve"
        maliyet_lbl = "Estimated Development Cost ($)"
        gelir_lbl = "Average Revenue per Player ($)"
        oyuncu_lbl = "Current Monthly Active Players"
        basari_lbl = "Complaint Resolution Success Rate (%)"
        metrik_lbl = "📊 Key Metrics"
        finans_lbl = "💵 Financial Projection"
        oneri_lbl = "💡 Automated Recommendation"
        oncelik_lbl = "🏆 Category Priority Ranking"
        oncelik_cap = "Estimated ROI for all categories using the same parameters"
        no_data = "Not enough data available."
        karli, kayip = "Profitable Investment ✓", "High Cost ✗"
        neg_lbl, donusen_lbl, kazanilan_lbl = "Negative Reviews", "Recoverable Reviews", "Players Recovered"
        aylik_lbl, yillik_lbl, geri_don_lbl = "Monthly Extra Revenue", "Annual Extra Revenue", "Payback Period"
        kat_col, neg_col, geri_col = "Category", "Negatives", "Payback (mo)"
        sen_a_lbl  = "🟦 Scenario A"
        sen_b_lbl  = "🟨 Scenario B"
        kat_a_lbl  = "Category — Scenario A"
        kat_b_lbl  = "Category — Scenario B"
        mal_a_lbl  = "Cost A ($)"
        mal_b_lbl  = "Cost B ($)"
        bas_a_lbl  = "Success Rate A (%)"
        bas_b_lbl  = "Success Rate B (%)"
        kars_tablo = "📊 Comparison Table"
        karar_lbl  = "💡 Auto Decision"
        kazan_oyuncu_lbl = "Players Recovered"
        ek_gelir_lbl = "Est. Extra Revenue ($)"
        net_roi_lbl = "Net ROI %"
        geri_sure_lbl = "Payback (mo)"

    etiketler   = kat_etiketler(lang)
    tum_oyunlar = [(o, OYUN_ADLARI.get(o, o)) for tur in TURLER.values() for o in tur]
    roi_sub1, roi_sub2 = st.tabs([sub1_lbl, sub2_lbl])

    # ── Sub-tab 1: Tek Senaryo ─────────────────────────────────────────────
    with roi_sub1:
        sol, sag = st.columns([1, 1.4])

        with sol:
            st.markdown(param_lbl)
            oyun_idx = st.selectbox(oyun_lbl, range(len(tum_oyunlar)),
                                    format_func=lambda i: tum_oyunlar[i][1], key="roi_oyun")
            secili_oyun = tum_oyunlar[oyun_idx][0]

            kat_listesi = list(etiketler.items())
            kat_idx = st.selectbox(kat_lbl, range(len(kat_listesi)),
                                   format_func=lambda i: kat_listesi[i][1], key="roi_kat")
            secili_kat_key, secili_kat_label = kat_listesi[kat_idx]

            maliyet = st.slider(maliyet_lbl, 10_000, 5_000_000, 500_000, 10_000,
                                format="$%d", key="roi_maliyet")
            gelir_per = st.slider(gelir_lbl, 5, 70, 30, 1,
                                  format="$%d", key="roi_gelir")
            aktif_oyuncu = st.number_input(oyuncu_lbl, min_value=1_000, max_value=10_000_000,
                                           value=100_000, step=1_000, key="roi_oyuncu")
            basari_pct = st.slider(basari_lbl, 50, 95, 70, 5,
                                   format="%d%%", key="roi_basari")
            basari = basari_pct / 100.0

        with sag:
            st.markdown(sonuc_lbl)
            try:
                df = oyun_yukle(secili_oyun)
                if df.empty:
                    st.warning(no_data)
                else:
                    toplam = len(df)
                    scol = next((c for c in ("sentiment", "vader_label") if c in df.columns), None)
                    if scol and secili_kat_key in df.columns:
                        neg_sayi = int(df[(df[scol] == "negative") & (df[secili_kat_key] == 1)].shape[0])
                    elif secili_kat_key in df.columns:
                        neg_sayi = int(df[secili_kat_key].sum())
                    else:
                        neg_sayi = 0

                    h = _roi_hesapla(neg_sayi, toplam, basari, int(aktif_oyuncu), float(gelir_per), float(maliyet))

                    st.markdown(f"<div class='sec-title'>{metrik_lbl}</div>", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric(neg_lbl, f"{h['neg_yorum']:,}", f"{h['neg_pct']:.1f}%")
                    m2.metric(donusen_lbl, f"{h['donusen']:,}")
                    m3.metric(kazanilan_lbl, f"{h['kazanilan']:,}")

                    st.markdown(f"<div class='sec-title'>{finans_lbl}</div>", unsafe_allow_html=True)
                    f1, f2, f3 = st.columns(3)
                    f1.metric(aylik_lbl, f"${h['aylik_ek']:,.0f}")
                    f2.metric(yillik_lbl, f"${h['yillik_ek']:,.0f}")
                    gd_str = (f"{h['geri_don']:.1f} " + ("ay" if lang == "TR" else "mo")
                              if h["geri_don"] != float("inf") else "∞")
                    f3.metric(geri_don_lbl, gd_str)

                    roi = h["roi"]
                    if roi >= 0:
                        st.success(f"✅ **{karli}** — Net ROI: **{roi:.1f}%**")
                    else:
                        st.error(f"❌ **{kayip}** — Net ROI: **{roi:.1f}%**")

                    st.plotly_chart(_grafik_gauge(roi, lang), use_container_width=True)

                    st.markdown(f"<div class='sec-title'>{oneri_lbl}</div>", unsafe_allow_html=True)
                    oyun_ad = OYUN_ADLARI.get(secili_oyun, secili_oyun)
                    gd_str2 = f"{h['geri_don']:.0f}" if h["geri_don"] != float("inf") else "∞"
                    if lang == "TR":
                        oncelik_metin = ("✅ Öncelikli Yatırım: Yüksek ROI potansiyeli, hemen harekete geçin."
                                         if roi >= 50 else
                                         "⚠️ Orta Öncelik: Makul yatırım; daha yüksek ROI'li kategorileri de değerlendirin."
                                         if roi >= 0 else
                                         "❌ Düşük Öncelik: Mevcut parametrelerle karlı görünmüyor. "
                                         "Maliyeti azaltmayı veya başka bir kategoriyi deneyin.")
                        st.write(
                            f"{oyun_ad} — {secili_kat_label} kategorisi analizi:\n\n"
                            f"- Bu kategoriye {maliyet:,.0f} $ yatırım yapıldığında tahminen "
                            f"{h['yillik_ek']:,.0f} $ yıllık ek gelir elde edilebilir.\n"
                            f"- Yatırım yaklaşık {gd_str2} ayda geri dönebilir.\n"
                            f"- Tahminen {h['kazanilan']:,} oyuncu geri kazanılabilir.\n\n"
                            f"{oncelik_metin}"
                        )
                    else:
                        priority = ("✅ Priority Investment: High ROI potential — act now."
                                    if roi >= 50 else
                                    "⚠️ Medium Priority: Reasonable return; also evaluate higher-ROI categories."
                                    if roi >= 0 else
                                    "❌ Low Priority: Not profitable with current parameters. "
                                    "Consider reducing cost or targeting a different category.")
                        st.write(
                            f"{oyun_ad} — {secili_kat_label} category analysis:\n\n"
                            f"- Investing {maliyet:,.0f} $ could generate approximately "
                            f"{h['yillik_ek']:,.0f} $ in annual extra revenue.\n"
                            f"- The investment could pay back in approximately {gd_str2} months.\n"
                            f"- An estimated {h['kazanilan']:,} players could be recovered.\n\n"
                            f"{priority}"
                        )

                    st.markdown(f"<div class='sec-title'>{oncelik_lbl}</div>", unsafe_allow_html=True)
                    st.caption(oncelik_cap)
                    oncelik_rows = []
                    for kk, kl in etiketler.items():
                        try:
                            if kk not in df.columns:
                                continue
                            if scol:
                                nk = int(df[(df[scol] == "negative") & (df[kk] == 1)].shape[0])
                            else:
                                nk = int(df[kk].sum())
                            hh = _roi_hesapla(nk, toplam, basari, int(aktif_oyuncu), float(gelir_per), float(maliyet))
                            gd = (f"{hh['geri_don']:.1f}" if hh["geri_don"] != float("inf") else "∞")
                            oncelik_rows.append({
                                kat_col: kl, neg_col: nk,
                                "Net ROI %": hh["roi"],
                                geri_col: gd,
                            })
                        except Exception:
                            pass
                    if oncelik_rows:
                        onc_df = (pd.DataFrame(oncelik_rows)
                                  .sort_values("Net ROI %", ascending=False)
                                  .reset_index(drop=True))
                        onc_df["Net ROI %"] = onc_df["Net ROI %"].map("{:.1f}%".format)
                        st.dataframe(onc_df, use_container_width=True, hide_index=True)

                    st.divider()
                    _render_pdf_butonu(secili_oyun, lang, "roi_tek")
            except Exception:
                st.warning(no_data)

    # ── Sub-tab 2: Senaryo Karşılaştırma ──────────────────────────────────
    with roi_sub2:
        kat_listesi2 = list(etiketler.items())

        sc_oyun_idx = st.selectbox(oyun_lbl, range(len(tum_oyunlar)),
                                    format_func=lambda i: tum_oyunlar[i][1], key="roi_sc_oyun")
        sc_oyun = tum_oyunlar[sc_oyun_idx][0]

        sc_gelir  = st.slider(gelir_lbl, 5, 70, 30, 1, format="$%d", key="roi_sc_gelir")
        sc_oyuncu = st.number_input(oyuncu_lbl, min_value=1_000, max_value=10_000_000,
                                     value=100_000, step=1_000, key="roi_sc_oyuncu")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f'<div style="background:#1e3a5f;border-radius:10px;padding:1rem;border:2px solid #3b82f6;">'
                f'<div style="font-size:1rem;font-weight:700;color:#93c5fd;margin-bottom:0.5rem;">{sen_a_lbl}</div>',
                unsafe_allow_html=True,
            )
            kat_a_idx = st.selectbox(kat_a_lbl, range(len(kat_listesi2)),
                                      format_func=lambda i: kat_listesi2[i][1], key="roi_kat_a")
            kat_a_key, kat_a_label = kat_listesi2[kat_a_idx]
            mal_a = st.slider(mal_a_lbl, 10_000, 5_000_000, 300_000, 10_000, format="$%d", key="roi_mal_a")
            bas_a = st.slider(bas_a_lbl, 50, 95, 70, 5, format="%d%%", key="roi_bas_a")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown(
                f'<div style="background:#2d1b4e;border-radius:10px;padding:1rem;border:2px solid #a78bfa;">'
                f'<div style="font-size:1rem;font-weight:700;color:#c4b5fd;margin-bottom:0.5rem;">{sen_b_lbl}</div>',
                unsafe_allow_html=True,
            )
            kat_b_idx = st.selectbox(kat_b_lbl, range(len(kat_listesi2)),
                                      format_func=lambda i: kat_listesi2[i][1],
                                      index=min(1, len(kat_listesi2) - 1), key="roi_kat_b")
            kat_b_key, kat_b_label = kat_listesi2[kat_b_idx]
            mal_b = st.slider(mal_b_lbl, 10_000, 5_000_000, 500_000, 10_000, format="$%d", key="roi_mal_b")
            bas_b = st.slider(bas_b_lbl, 50, 95, 75, 5, format="%d%%", key="roi_bas_b")
            st.markdown("</div>", unsafe_allow_html=True)

        try:
            df_sc = oyun_yukle(sc_oyun)
            if df_sc.empty:
                st.warning(no_data)
            else:
                toplam_sc = len(df_sc)
                scol_sc   = next((c for c in ("sentiment", "vader_label") if c in df_sc.columns), None)

                def _get_neg_sc(kat_key):
                    if scol_sc and kat_key in df_sc.columns:
                        return int(df_sc[(df_sc[scol_sc] == "negative") & (df_sc[kat_key] == 1)].shape[0])
                    return int(df_sc[kat_key].sum()) if kat_key in df_sc.columns else 0

                h_a = _roi_hesapla(_get_neg_sc(kat_a_key), toplam_sc, bas_a / 100,
                                   int(sc_oyuncu), float(sc_gelir), float(mal_a))
                h_b = _roi_hesapla(_get_neg_sc(kat_b_key), toplam_sc, bas_b / 100,
                                   int(sc_oyuncu), float(sc_gelir), float(mal_b))

                def _gd_fmt(h):
                    return (f"{h['geri_don']:.1f}" + (" ay" if lang == "TR" else " mo")
                            if h["geri_don"] != float("inf") else "∞")

                # ── Karşılaştırma tablosu ───────────────────────────────────
                st.markdown(f"<div class='sec-title'>{kars_tablo}</div>", unsafe_allow_html=True)
                kazanan = "A" if h_a["roi"] >= h_b["roi"] else "B"
                tablo_sc = pd.DataFrame({
                    "":        [kazan_oyuncu_lbl, ek_gelir_lbl, net_roi_lbl, geri_sure_lbl],
                    sen_a_lbl: [f"{h_a['kazanilan']:,}", f"${h_a['yillik_ek']:,.0f}",
                                f"{h_a['roi']:.1f}%", _gd_fmt(h_a)],
                    sen_b_lbl: [f"{h_b['kazanilan']:,}", f"${h_b['yillik_ek']:,.0f}",
                                f"{h_b['roi']:.1f}%", _gd_fmt(h_b)],
                })

                def _hl_winner(df_style):
                    styles = pd.DataFrame("", index=df_style.index, columns=df_style.columns)
                    win_col = sen_a_lbl if kazanan == "A" else sen_b_lbl
                    styles[win_col] = "background-color:rgba(34,197,94,0.15);font-weight:600;color:#4ade80;"
                    return styles

                st.dataframe(tablo_sc.style.apply(_hl_winner, axis=None),
                             use_container_width=True, hide_index=True)

                # ── Gauge charts yan yana ────────────────────────────────────
                g1, g2 = st.columns(2)
                with g1:
                    st.plotly_chart(_grafik_gauge(h_a["roi"], lang), use_container_width=True)
                    st.caption(f"**{sen_a_lbl}** — {kat_a_label}")
                with g2:
                    st.plotly_chart(_grafik_gauge(h_b["roi"], lang), use_container_width=True)
                    st.caption(f"**{sen_b_lbl}** — {kat_b_label}")

                # ── Otomatik karar metni ─────────────────────────────────────
                st.markdown(f"<div class='sec-title'>{karar_lbl}</div>", unsafe_allow_html=True)
                h_w  = h_a if kazanan == "A" else h_b
                h_l  = h_b if kazanan == "A" else h_a
                kw_l = kat_a_label if kazanan == "A" else kat_b_label
                kl_l = kat_b_label if kazanan == "A" else kat_a_label
                toplam_maliyet = mal_a + mal_b
                ikisi_ay = max(
                    h_a["geri_don"] if h_a["geri_don"] != float("inf") else 99,
                    h_b["geri_don"] if h_b["geri_don"] != float("inf") else 99,
                )
                if lang == "TR":
                    st.info(
                        f"- **{kw_l}** kategorisi **%{h_w['roi']:.1f} ROI** ile "
                        f"**{kl_l}** kategorisinden daha karlı.\n"
                        f"- Önce **{kw_l}** kategorisine odaklanmanızı öneririz.\n"
                        f"- **${toplam_maliyet:,.0f}** toplam bütçe ile her iki sorunu da "
                        f"yaklaşık **{ikisi_ay:.0f} ayda** çözebilirsiniz."
                    )
                else:
                    st.info(
                        f"- **{kw_l}** delivers **{h_w['roi']:.1f}% ROI**, outperforming **{kl_l}**.\n"
                        f"- We recommend focusing on **{kw_l}** first.\n"
                        f"- With a total budget of **${toplam_maliyet:,.0f}**, "
                        f"you can resolve both issues in approximately **{ikisi_ay:.0f} months**."
                    )

                st.divider()
                _render_pdf_butonu(sc_oyun, lang, "roi_sc")
        except Exception:
            st.warning(no_data)


# ─── Dil görüntüleme eşlemesi (Modül 4-6 paylaşımlı) ────────────────────────

_DIL_GORUNUM = {
    "english": "English", "russian": "Русский", "turkish": "Türkçe",
    "german": "Deutsch", "french": "Français", "spanish": "Español",
    "portuguese": "Português", "schinese": "中文(简)", "tchinese": "中文(繁)",
    "japanese": "日本語", "korean": "한국어", "arabic": "العربية",
    "polish": "Polski", "dutch": "Nederlands", "italian": "Italiano",
    "swedish": "Svenska", "ukrainian": "Українська", "czech": "Čeština",
    "hungarian": "Magyar", "finnish": "Suomi", "romanian": "Română",
    "danish": "Dansk", "norwegian": "Norsk", "thai": "ภาษาไทย",
    "indonesian": "Indonesia", "vietnamese": "Tiếng Việt",
    "brazilian": "Português(BR)",
    "en": "English", "tr": "Türkçe", "ru": "Русский", "de": "Deutsch",
    "fr": "Français", "es": "Español", "pt": "Português", "zh": "中文",
    "ja": "日本語", "ko": "한국어", "ar": "العربية", "pl": "Polski",
    "nl": "Nederlands", "it": "Italiano", "sv": "Svenska",
    "uk": "Українська", "cs": "Çeştçe", "hu": "Magyar",
}

# ─── MODÜL 4: Çapraz Oyun Şikayet Matrisi ────────────────────────────────────

@st.cache_data
def _matris_tum_veri() -> pd.DataFrame:
    """Tüm oyunlar × 8 kategori şikayet % ve sayı tablosunu hesaplar (cached)."""
    kat_keys = list(_KAT_TR.keys())
    satirlar = []
    for tur, oyunlar in TURLER.items():
        for oyun in oyunlar:
            df = oyun_yukle(oyun)
            if df.empty:
                continue
            toplam = len(df)
            neg_df_m = df[df["voted_up_bool"] == False]
            neg_toplam_m = len(neg_df_m)
            satir = {
                "oyun_ad":   OYUN_ADLARI.get(oyun, oyun),
                "oyun_key":  oyun,
                "tur":       tur,
                "toplam":    toplam,
                "neg_toplam": neg_toplam_m,
            }
            for k in kat_keys:
                if k in df.columns:
                    sayi = int(neg_df_m[k].sum())
                    satir[f"{k}_pct"]  = sayi / max(1, neg_toplam_m) * 100 if neg_toplam_m else 0.0
                    satir[f"{k}_sayi"] = sayi
                else:
                    satir[f"{k}_pct"]  = 0.0
                    satir[f"{k}_sayi"] = 0
            satirlar.append(satir)
    return pd.DataFrame(satirlar)


def _grafik_matris(tablo: pd.DataFrame, lang: str) -> go.Figure:
    kat_keys   = list(_KAT_TR.keys())
    etiketler  = kat_etiketler(lang)
    kat_labels = [etiketler[k] for k in kat_keys]
    oyun_labels = list(tablo["oyun_ad"])

    z_vals, hover_texts = [], []
    for _, row in tablo.iterrows():
        z_row, h_row = [], []
        for k in kat_keys:
            pct  = row.get(f"{k}_pct",  0.0)
            sayi = int(row.get(f"{k}_sayi", 0))
            kat_et = etiketler[k]
            z_row.append(round(pct, 1))
            if lang == "TR":
                h_row.append(f"<b>{row['oyun_ad']}</b><br>{kat_et}: %{pct:.1f}<br>({sayi:,} yorum)")
            else:
                h_row.append(f"<b>{row['oyun_ad']}</b><br>{kat_et}: {pct:.1f}%<br>({sayi:,} reviews)")
        z_vals.append(z_row)
        hover_texts.append(h_row)

    title     = "Oyun × Kategori Şikayet Matrisi (%)" if lang == "TR" else "Game × Category Complaint Matrix (%)"
    color_lbl = "Şikayet %" if lang == "TR" else "Complaint %"

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=kat_labels,
        y=oyun_labels,
        customdata=hover_texts,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale=[
            [0.00, "#f8fafc"], [0.10, "#fecaca"],
            [0.30, "#f87171"], [0.60, "#dc2626"], [1.00, "#450a0a"],
        ],
        zmin=0, zmax=50,
        showscale=True,
        colorbar=dict(title=color_lbl, ticksuffix="%"),
    ))
    x_baslik = "Şikayet Kategorisi" if lang == "TR" else "Complaint Category"
    y_baslik = "Oyun" if lang == "TR" else "Game"
    height = max(460, len(oyun_labels) * 36 + 130)
    fig.update_layout(
        title=title, template="plotly_dark", height=height,
        margin=dict(t=80, b=40, l=200, r=20),
        xaxis=dict(side="top", title=dict(text=x_baslik, font=dict(size=12))),
        yaxis=dict(title=dict(text=y_baslik, font=dict(size=12))),
    )
    return fig


def render_sikayet_matrisi(lang: str):
    if lang == "TR":
        st.markdown("## 🗺️ Çapraz Oyun Şikayet Matrisi")
        st.caption("Tüm oyunları 8 şikayet kategorisinde aynı anda karşılaştırın")
        filtre_lbl  = "Tür Filtresi"
        tumu        = "Tümü"
        otomatik_lbl = "📝 Otomatik Analiz"
        no_data     = "Yeterli veri bulunamadı."
    else:
        st.markdown("## 🗺️ Cross-Game Complaint Matrix")
        st.caption("Compare all games across 8 complaint categories at once")
        filtre_lbl  = "Genre Filter"
        tumu        = "All"
        otomatik_lbl = "📝 Automated Analysis"
        no_data     = "Not enough data available."

    secili_tur = st.radio(
        filtre_lbl, [tumu] + list(TURLER.keys()),
        horizontal=True, key="matris_tur",
    )

    try:
        tum_tablo = _matris_tum_veri()
        if tum_tablo.empty:
            st.warning(no_data)
            return

        tablo = (tum_tablo if secili_tur == tumu
                 else tum_tablo[tum_tablo["tur"] == secili_tur].reset_index(drop=True))
        if tablo.empty:
            st.warning(no_data)
            return

        # Matris açıklama notu
        if lang == "TR":
            st.info(
                "**Her hücre**, ilgili oyunun **OLUMSUZ** yorumları içinde o şikayet kategorisinin "
                "yüzdesini göstermektedir.\n\n"
                "**Örnek:** Cyberpunk 2077 — Performans %23,8 ifadesi, bu oyunun olumsuz yorumlarının "
                "%23,8'inin performans şikayeti içerdiği anlamına gelir."
            )
        else:
            st.info(
                "**Each cell** shows the percentage of that complaint category among the game's "
                "**negative reviews only**.\n\n"
                "**Example:** Cyberpunk 2077 — Performance 23.8% means 23.8% of the game's negative "
                "reviews contain a performance complaint."
            )

        st.plotly_chart(_grafik_matris(tablo, lang), use_container_width=True)

        # ── Otomatik analiz ────────────────────────────────────────────────
        st.markdown(f"<div class='sec-title'>{otomatik_lbl}</div>", unsafe_allow_html=True)

        etiketler = kat_etiketler(lang)
        kat_keys  = list(_KAT_TR.keys())
        pct_cols  = [f"{k}_pct" for k in kat_keys if f"{k}_pct" in tablo.columns]

        # En fazla şikayet kategorisi (seçili grubun ortalaması)
        kat_ortalar = {etiketler[k]: tablo[f"{k}_pct"].mean()
                       for k in kat_keys if f"{k}_pct" in tablo.columns}
        en_fazla_kat = max(kat_ortalar, key=kat_ortalar.get)
        en_fazla_pct = kat_ortalar[en_fazla_kat]

        # En sorunlu oyun — sadece 50+ olumsuz yorumu olan oyunlar
        tablo2 = tablo.copy()
        tablo2["_ort"] = tablo2[pct_cols].mean(axis=1)
        yeterli_veri = tablo2[tablo2["neg_toplam"] >= 50] if "neg_toplam" in tablo2.columns else tablo2
        if yeterli_veri.empty:
            yeterli_veri = tablo2
        en_sorunlu_idx  = yeterli_veri["_ort"].idxmax()
        en_sorunlu_oyun = tablo2.at[en_sorunlu_idx, "oyun_ad"]
        en_sorunlu_pct  = tablo2.at[en_sorunlu_idx, "_ort"]

        # Türler arası en az şikayet
        tur_ortalar = {}
        for tur in TURLER:
            tur_alt = tum_tablo[tum_tablo["tur"] == tur]
            if not tur_alt.empty:
                tur_ortalar[tur] = tur_alt[pct_cols].values.mean()
        en_az_tur = min(tur_ortalar, key=tur_ortalar.get) if tur_ortalar else "-"
        en_az_pct = tur_ortalar.get(en_az_tur, 0.0)

        # Matristeki en yüksek tek hücre
        max_pct, max_game_nm, max_cat_nm = 0.0, "", ""
        for _, row in tablo.iterrows():
            for k in kat_keys:
                v = row.get(f"{k}_pct", 0.0)
                if v > max_pct:
                    max_pct     = v
                    max_game_nm = row["oyun_ad"]
                    max_cat_nm  = etiketler.get(k, k)

        c1, c2, c3 = st.columns(3)
        if lang == "TR":
            c1.metric("En Fazla Şikayet Kategorisi", en_fazla_kat, f"Ort. %{en_fazla_pct:.1f}")
            c2.metric("En Sorunlu Oyun",              en_sorunlu_oyun, f"Ort. %{en_sorunlu_pct:.1f}")
            c3.metric("En Az Şikayet Türü",           en_az_tur, f"Ort. %{en_az_pct:.1f}")
            st.markdown(
                f"- **En fazla şikayet alan kategori:** {en_fazla_kat} "
                f"(seçili oyunlar ortalaması %{en_fazla_pct:.1f})\n"
                f"- **En sorunlu oyun:** {en_sorunlu_oyun} "
                f"(ortalama şikayet oranı %{en_sorunlu_pct:.1f}) "
                f"(50+ olumsuz yorumu olan oyunlar arasında)\n"
                f"- **En az şikayet alan tür:** {en_az_tur} "
                f"(ortalama %{en_az_pct:.1f})\n"
                f"- **Matristeki en yüksek değer:** {max_game_nm} → {max_cat_nm}: "
                f"%{max_pct:.1f}"
            )
        else:
            c1.metric("Highest Complaint Category", en_fazla_kat,    f"Avg {en_fazla_pct:.1f}%")
            c2.metric("Most Problematic Game",       en_sorunlu_oyun, f"Avg {en_sorunlu_pct:.1f}%")
            c3.metric("Least Complaints Genre",      en_az_tur,       f"Avg {en_az_pct:.1f}%")
            st.markdown(
                f"- **Most complained category:** {en_fazla_kat} "
                f"(avg across selected games {en_fazla_pct:.1f}%)\n"
                f"- **Most problematic game:** {en_sorunlu_oyun} "
                f"(avg complaint rate {en_sorunlu_pct:.1f}%) "
                f"(among games with 50+ negative reviews)\n"
                f"- **Least complaining genre:** {en_az_tur} "
                f"(avg {en_az_pct:.1f}%)\n"
                f"- **Matrix peak cell:** {max_game_nm} → {max_cat_nm}: "
                f"{max_pct:.1f}%"
            )
    except Exception:
        st.warning(no_data)

    st.divider()
    _matris_oyunlar = [o for tur in TURLER.values() for o in tur]
    _render_pdf_butonu(_matris_oyunlar[0] if _matris_oyunlar else "", lang, "matris")


# ─── MODÜL 7: Şikayet Ağırlık Skoru ─────────────────────────────────────────

def _agirlik_hesapla(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "playtime_forever" in df.columns:
        pt = pd.to_numeric(df["playtime_forever"], errors="coerce").fillna(0)
    else:
        pt = pd.Series(0.0, index=df.index)
    df["_pt"] = pt
    conds  = [pt < 60, (pt >= 60) & (pt < 600), (pt >= 600) & (pt < 3000), pt >= 3000]
    df["_agirlik"] = np.select(conds, [0.5, 1.0, 1.5, 2.0], default=1.0)
    return df


def render_agirlikli_analiz(lang: str):
    st.markdown("## ⚖️ Şikayet Ağırlık Skoru")
    st.caption("Deneyimli oyuncuların şikayetlerine daha fazla ağırlık vererek gerçek etki skorunu hesaplayın")

    if lang == "TR":
        st.markdown(
            "**Ham Şikayet Sayısı Nedir?**  \n"
            "O kategoride kaç olumsuz yorum olduğunu gösterir. Tüm yorumlar eşit kabul edilir."
        )
        st.markdown(
            "**Ağırlıklı Skor Nasıl Hesaplanır?**  \n"
            "Her oyuncunun şikayetine, o oyunu ne kadar oynadığına göre farklı ağırlık verilir:\n"
            "- **0–60 dakika** oynayan → 0.5 ağırlık *(az deneyim)*\n"
            "- **60–600 dakika** oynayan → 1.0 ağırlık *(normal)*\n"
            "- **600–3000 dakika** oynayan → 1.5 ağırlık *(deneyimli)*\n"
            "- **3000+ dakika** oynayan → 2.0 ağırlık *(çok deneyimli)*\n\n"
            "*Örnek: 100 saatlik oyuncunun performans şikayeti, "
            "5 saatlik oyuncunun aynı şikayetinden 3 kat daha fazla ağırlık taşır.*"
        )
    no_data  = "Yeterli veri bulunamadı."
    etiketler = kat_etiketler(lang)
    kat_keys  = list(_KAT_TR.keys())
    tum_oyunlar = [(o, OYUN_ADLARI.get(o, o)) for tur in TURLER.values() for o in tur]

    oyun_idx = st.selectbox("Oyun Seçin", range(len(tum_oyunlar)),
                            format_func=lambda i: tum_oyunlar[i][1], key="agirlik_oyun")
    secili_oyun = tum_oyunlar[oyun_idx][0]

    try:
        df = oyun_yukle(secili_oyun)
        if df.empty:
            st.warning(no_data)
            return

        df = _agirlik_hesapla(df)
        neg_df_aw = df[df["voted_up_bool"] == False]
        oyun_ad = OYUN_ADLARI.get(secili_oyun, secili_oyun)

        # ── Karşılaştırma tablosu ──────────────────────────────────────────
        st.markdown("<div class='sec-title'>📊 Ağırlıklı vs Ham Karşılaştırma</div>", unsafe_allow_html=True)

        tablo_rows = []
        for k in kat_keys:
            if k not in neg_df_aw.columns:
                continue
            ham       = int(neg_df_aw[k].sum())
            agirlikli = float((neg_df_aw[k] * neg_df_aw["_agirlik"]).sum())
            fark_pct  = (agirlikli - ham) / max(1.0, ham) * 100
            fark_metin = ("Deneyimli oyuncular daha çok şikayet ediyor ↑" if agirlikli > ham
                          else "Yeni oyuncular daha çok şikayet ediyor ↓" if agirlikli < ham
                          else "Eşit")
            tablo_rows.append({
                "Kategori":           etiketler[k],
                "Ham Şikayet Sayısı": ham,
                "Ağırlıklı Skor":     round(agirlikli, 1),
                "Fark":               fark_metin,
                "_aw":                agirlikli,
                "_fark_pct":          round(fark_pct, 1),
            })

        if not tablo_rows:
            st.warning(no_data)
            return

        tablo_df = (pd.DataFrame(tablo_rows)
                    .sort_values("_aw", ascending=False)
                    .reset_index(drop=True))

        display_df = tablo_df.drop(columns=["_aw", "_fark_pct"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        if lang == "TR":
            st.caption("ℹ️ Kategori yüzdeleri yalnızca olumsuz yorumlar baz alınarak hesaplanmıştır.")
        else:
            st.caption("ℹ️ Category percentages are calculated based on negative reviews only.")

        # ── Bar chart + Pasta ──────────────────────────────────────────────
        col_bar, col_pasta = st.columns([1.3, 0.9])

        with col_bar:
            st.markdown("<div class='sec-title'>📊 Ağırlıklı Şikayet Sıralaması</div>", unsafe_allow_html=True)
            fig_bar = go.Figure(go.Bar(
                x=tablo_df["Ağırlıklı Skor"],
                y=tablo_df["Kategori"],
                orientation="h",
                marker=dict(
                    color=tablo_df["Ağırlıklı Skor"].values,
                    colorscale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1.0, "#ef4444"]],
                    showscale=False,
                ),
                text=tablo_df["Ağırlıklı Skor"].map("{:.1f}".format),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Ağırlıklı Skor: %{x:.1f}<extra></extra>",
            ))
            fig_bar.update_layout(
                template="plotly_dark", height=340,
                margin=dict(t=10, b=10, l=0, r=70),
                xaxis_title="Ağırlıklı Şikayet Skoru",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_pasta:
            st.markdown("<div class='sec-title'>🎮 Oyuncu Deneyim Dağılımı</div>", unsafe_allow_html=True)
            pt_col   = neg_df_aw["_pt"]
            sinirlar = [(0, 60, "0–1 saat"), (60, 600, "1–10 saat"),
                        (600, 3000, "10–50 saat"), (3000, float("inf"), "50+ saat")]
            pasta_labels, pasta_values = [], []
            for alt, ust, etiket in sinirlar:
                if ust < float("inf"):
                    n = int(((pt_col >= alt) & (pt_col < ust)).sum())
                else:
                    n = int((pt_col >= alt).sum())
                pasta_labels.append(etiket)
                pasta_values.append(n)
            fig_pasta = go.Figure(go.Pie(
                labels=pasta_labels, values=pasta_values, hole=0.42,
                marker_colors=["#94a3b8", "#3b82f6", "#f59e0b", "#ef4444"],
                hovertemplate="<b>%{label}</b><br>Oyuncu: %{value:,}<br>Oran: %{percent}<extra></extra>",
            ))
            fig_pasta.update_layout(
                template="plotly_dark", height=340, margin=dict(t=10, b=10, l=0, r=0),
            )
            st.plotly_chart(fig_pasta, use_container_width=True)

        # ── Otomatik içgörü ────────────────────────────────────────────────
        st.markdown("<div class='sec-title'>📝 Otomatik İçgörü</div>", unsafe_allow_html=True)

        en_agirlikli_kat = tablo_df.iloc[0]["Kategori"]
        en_ham_kat = tablo_df.sort_values("Ham Şikayet Sayısı", ascending=False).iloc[0]["Kategori"]
        max_fark_idx = tablo_df["_fark_pct"].abs().idxmax()
        max_fark_kat = tablo_df.at[max_fark_idx, "Kategori"]
        max_fark_val = tablo_df.at[max_fark_idx, "_fark_pct"]

        deneyimli = df[df["_pt"] >= 600]
        if not deneyimli.empty:
            den_sk  = {etiketler[k]: float((deneyimli[k] * deneyimli["_agirlik"]).sum())
                       for k in kat_keys if k in deneyimli.columns}
            en_den_kat = max(den_sk, key=den_sk.get) if den_sk else "-"
        else:
            en_den_kat = "-"

        st.info(
            f"- **{oyun_ad}**'in en deneyimli oyuncuları ağırlıklı olarak **{en_den_kat}** "
            f"kategorisini şikayet ediyor — bu kritik bir sinyal.\n"
            f"- Ham veriye göre en büyük şikayet **{en_ham_kat}** iken, "
            f"ağırlıklı analizde **{en_agirlikli_kat}** öne çıkıyor.\n"
            f"- **{max_fark_kat}** kategorisinde ham ve ağırlıklı skor arasındaki fark "
            f"en yüksek (**{max_fark_val:+.1f}%**)."
        )
    except Exception:
        st.warning(no_data)

    st.divider()
    _render_pdf_butonu(secili_oyun, lang, "agirlik")


# ─── PDF Rapor ────────────────────────────────────────────────────────────────

@st.cache_data
def _pdf_rapor_olustur(oyun_key: str, lang: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4 as _A4
        from reportlab.lib import colors as _rc
        from reportlab.lib.styles import ParagraphStyle as _PS
        from reportlab.platypus import (
            SimpleDocTemplate as _Doc, Table as _T, TableStyle as _TS,
            Paragraph as _P, Spacer as _SP, PageBreak as _PB, HRFlowable as _HR,
        )
        from reportlab.lib.units import cm as _cm
        from reportlab.pdfbase import pdfmetrics as _pm
        from reportlab.pdfbase.ttfonts import TTFont as _TTF
    except ImportError:
        return b""

    fn, fb = "Helvetica", "Helvetica-Bold"
    try:
        _wd = os.environ.get("WINDIR", "C:/Windows")
        _ap = os.path.join(_wd, "Fonts", "arial.ttf")
        _ab = os.path.join(_wd, "Fonts", "arialbd.ttf")
        if os.path.exists(_ap):
            _pm.registerFont(_TTF("ArR", _ap))
            fn = "ArR"
        if os.path.exists(_ab):
            _pm.registerFont(_TTF("ArB", _ab))
            fb = "ArB"
    except Exception:
        pass

    MAVI   = _rc.HexColor("#1a1a2e")
    ACIK_M = _rc.HexColor("#2563eb")
    GRI_BG = _rc.HexColor("#f8fafc")
    KOYU_G = _rc.HexColor("#334155")

    df       = oyun_yukle(oyun_key)
    oyun_ad  = OYUN_ADLARI.get(oyun_key, oyun_key)
    tur_adi  = next((t for t, ol in TURLER.items() if oyun_key in ol), "")
    tarih    = pd.Timestamp.now().strftime("%d.%m.%Y")
    toplam   = len(df)
    pos_sayi = int(df["voted_up"].astype(bool).sum()) if "voted_up" in df.columns and toplam > 0 else 0
    neg_sayi = toplam - pos_sayi
    pos_pct  = pos_sayi / max(1, toplam) * 100

    buf = io.BytesIO()
    doc = _Doc(buf, pagesize=_A4,
               rightMargin=2*_cm, leftMargin=2*_cm,
               topMargin=3*_cm, bottomMargin=2.5*_cm,
               title=f"{oyun_ad} — Steam Raporu")

    def _H1(t): return _P(t, _PS("h1", fontName=fb, fontSize=18, textColor=MAVI, spaceAfter=8))
    def _H2(t): return _P(t, _PS("h2", fontName=fb, fontSize=12, textColor=ACIK_M, spaceAfter=5, spaceBefore=10))
    def _BD(t): return _P(t, _PS("bd", fontName=fn, fontSize=9.5, textColor=KOYU_G, leading=13, spaceAfter=4))
    def _SEP(): return _HR(width="100%", thickness=0.8, color=_rc.HexColor("#e2e8f0"), spaceAfter=8)
    def _SPC(h=0.3): return _SP(1, h * _cm)

    def _tbl(data, col_w=None):
        t = _T(data, colWidths=col_w, repeatRows=1)
        t.setStyle(_TS([
            ("BACKGROUND",    (0, 0), (-1, 0), MAVI),
            ("TEXTCOLOR",     (0, 0), (-1, 0), _rc.white),
            ("FONTNAME",      (0, 0), (-1, 0), fb),
            ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
            ("FONTNAME",      (0, 1), (-1, -1), fn),
            ("FONTSIZE",      (0, 1), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_rc.white, GRI_BG]),
            ("GRID",          (0, 0), (-1, -1), 0.4, _rc.HexColor("#d1d5db")),
            ("ALIGN",         (1, 0), (-1, -1), "RIGHT"),
            ("ALIGN",         (0, 0), (0, -1),  "LEFT"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    def _on_page(canvas, doc_obj):
        canvas.saveState()
        w, h = _A4
        canvas.setFillColor(MAVI)
        canvas.setFont(fb, 8)
        canvas.drawString(2 * _cm, h - 1.7 * _cm, oyun_ad)
        canvas.drawRightString(w - 2 * _cm, h - 1.7 * _cm, f"Steam Raporu — {tarih}")
        canvas.setStrokeColor(_rc.HexColor("#e2e8f0"))
        canvas.line(2 * _cm, h - 2 * _cm, w - 2 * _cm, h - 2 * _cm)
        canvas.setFont(fn, 7.5)
        canvas.setFillColor(_rc.HexColor("#94a3b8"))
        canvas.drawCentredString(w / 2, 1.2 * _cm, f"Sayfa {doc_obj.page}")
        canvas.line(2 * _cm, 1.6 * _cm, w - 2 * _cm, 1.6 * _cm)
        canvas.restoreState()

    _KK   = list(_KAT_TR.keys())
    story = []

    # ── Sayfa 1: Kapak ────────────────────────────────────────────────────
    story += [_SPC(2), _H1("Steam Oyun Yorum Analizi Sistemi"), _SEP(), _SPC(0.5),
              _P(oyun_ad, _PS("gn", fontName=fb, fontSize=22, textColor=ACIK_M, spaceAfter=5)),
              _BD(f"Tür: {tur_adi}"), _BD(f"Rapor Tarihi: {tarih}"), _SPC(0.8)]
    story.append(_tbl(
        [["Metrik", "Değer"],
         ["Toplam Yorum",  f"{toplam:,}"],
         ["Olumlu Yorum",  f"{pos_sayi:,} (%{pos_pct:.1f})"],
         ["Olumsuz Yorum", f"{neg_sayi:,} (%{100 - pos_pct:.1f})"]],
        col_w=[9 * _cm, 6 * _cm],
    ))
    story.append(_PB())

    # ── Sayfa 2: Genel Analiz ─────────────────────────────────────────────
    story.append(_H2("Şikayet Kategorileri"))
    if not df.empty:
        df_w = _agirlik_hesapla(df)
        kat_data = [["Kategori", "Yorum Sayısı", "Yüzde (%)", "Ağırlıklı Skor"]]
        for k in _KK:
            if k not in df.columns:
                continue
            ham = int(df[k].sum())
            agw = float((df_w[k] * df_w["_agirlik"]).sum())
            pct_v = ham / max(1, toplam) * 100
            kat_data.append([_KAT_TR[k], str(ham), f"{pct_v:.1f}%", f"{agw:.1f}"])
        story.append(_tbl(kat_data, col_w=[5.5 * _cm, 3.5 * _cm, 3.5 * _cm, 3.5 * _cm]))
    story.append(_SPC(0.4))

    story.append(_H2("Duygu Dağılımı (VADER)"))
    scol_pdf = next((c for c in ("sentiment", "vader_label") if c in df.columns), None)
    sent_data = [["Duygu", "Sayı", "Oran (%)"]]
    if scol_pdf and not df.empty:
        for label_v, tr_l in [("positive", "Olumlu"), ("negative", "Olumsuz"), ("neutral", "Nötr")]:
            n = int((df[scol_pdf] == label_v).sum())
            sent_data.append([tr_l, str(n), f"{n / max(1, toplam) * 100:.1f}%"])
    story.append(_tbl(sent_data, col_w=[6 * _cm, 4 * _cm, 4 * _cm]))
    story.append(_SPC(0.4))

    story.append(_H2("Dil Dağılımı (Top 5)"))
    dil_col_pdf = next((c for c in ("detected_language", "language") if c in df.columns), None)
    dil_data = [["Dil", "Yorum Sayısı", "Oran (%)"]]
    if dil_col_pdf and not df.empty:
        for dil, n in df[dil_col_pdf].value_counts().head(5).items():
            gos = _DIL_GORUNUM.get(str(dil).lower(), str(dil).capitalize())
            dil_data.append([gos, str(n), f"{n / max(1, toplam) * 100:.1f}%"])
    story.append(_tbl(dil_data, col_w=[6 * _cm, 4 * _cm, 4 * _cm]))
    story.append(_PB())

    # ── Sayfa 3: Karşılaştırmalı Analiz ──────────────────────────────────
    story.append(_H2("Tür İçi Karşılaştırma"))
    tur_oyunlar_pdf = TURLER.get(tur_adi, [])
    if tur_oyunlar_pdf:
        karsil_data = [["Oyun"] + [_KAT_TR.get(k, k) for k in _KK]]
        for o in tur_oyunlar_pdf:
            df_o = oyun_yukle(o)
            if df_o.empty:
                continue
            row_o = [OYUN_ADLARI.get(o, o)]
            for k in _KK:
                pct_v = float(df_o[k].mean() * 100) if k in df_o.columns else 0.0
                row_o.append(f"*{pct_v:.0f}%*" if o == oyun_key else f"{pct_v:.0f}%")
            karsil_data.append(row_o)
        cw = [4.5 * _cm] + [13 * _cm / max(1, len(_KK))] * len(_KK)
        story.append(_tbl(karsil_data, col_w=cw))
    story.append(_SPC(0.4))
    if not df.empty:
        kp = {_KAT_TR[k]: float(df[k].mean() * 100) for k in _KK if k in df.columns}
        if kp:
            story += [_BD(f"En Az Şikayet: {min(kp, key=kp.get)} — %{min(kp.values()):.1f}"),
                      _BD(f"En Fazla Şikayet: {max(kp, key=kp.get)} — %{max(kp.values()):.1f}")]
    story.append(_PB())

    # ── Sayfa 4: Aksiyon Önerileri ────────────────────────────────────────
    story.append(_H2("Aksiyon Önerileri"))
    oneriler_pdf = aksiyon_uret(df, "TR") if not df.empty else []
    for i_o, oneri_o in enumerate(oneriler_pdf[:3], 1):
        clean_o = re.sub(r"<[^>]+>", "", oneri_o)
        story += [_BD(f"{i_o}. {clean_o}"), _SPC(0.2)]
    story += [_SPC(0.4), _SEP(), _H2("ROI Simülasyon Özeti (Varsayılan Parametreler)")]
    if not df.empty:
        scol4 = next((c for c in ("sentiment", "vader_label") if c in df.columns), None)
        roi_rows_pdf = []
        for k in _KK:
            if k not in df.columns:
                continue
            nk = int(df[(df[scol4] == "negative") & (df[k] == 1)].shape[0]) if scol4 else int(df[k].sum())
            hh = _roi_hesapla(nk, toplam, 0.70, 100_000, 30.0, 500_000.0)
            gd = f"{hh['geri_don']:.1f}" if hh["geri_don"] != float("inf") else "∞"
            roi_rows_pdf.append([_KAT_TR[k], str(nk), f"{hh['roi']:.1f}%", gd])
        roi_rows_pdf.sort(key=lambda r: float(r[2].replace("%", "")) if r[2] != "∞" else -999, reverse=True)
        story.append(_tbl(
            [["Kategori", "Olumsuz", "Net ROI %", "Geri Dönüş (ay)"]] + roi_rows_pdf,
            col_w=[5.5 * _cm, 3.5 * _cm, 3.5 * _cm, 4 * _cm],
        ))
    story += [_SPC(1), _SEP(),
              _BD("Bu rapor Steam Oyun Yorum Analizi Sistemi tarafından otomatik olarak üretilmiştir."),
              _BD(f"Oluşturulma tarihi: {tarih}")]

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return buf.getvalue()


def _render_pdf_butonu(oyun_key: str, lang: str, key_suffix: str = ""):
    if not oyun_key:
        return
    try:
        import reportlab  # noqa
        rl_ok = True
    except ImportError:
        rl_ok = False

    btn_lbl = "📄 PDF Rapor İndir" if lang == "TR" else "📄 Download PDF Report"
    if not rl_ok:
        st.warning("PDF için terminalde: pip install reportlab" if lang == "TR"
                   else "Run 'pip install reportlab' to enable PDF export.")
        return

    pdf_bytes = _pdf_rapor_olustur(oyun_key, lang)
    if pdf_bytes:
        safe_name = OYUN_ADLARI.get(oyun_key, oyun_key).replace(" ", "_").replace(":", "").replace("/", "")
        st.download_button(
            label=btn_lbl,
            data=pdf_bytes,
            file_name=f"{safe_name}_rapor.pdf",
            mime="application/pdf",
            key=f"pdf_dl_{key_suffix}_{oyun_key}",
        )
    else:
        st.warning("PDF oluşturulamadı." if lang == "TR" else "PDF generation failed.")



def main():
    # Session state başlangıcı
    for key, default in [("secili_tur", "RPG"), ("secili_oyun", None), ("ui_lang", "TR")]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Sidebar — önbellek temizleme
    with st.sidebar:
        st.markdown("### ⚙️ Ayarlar")
        if st.button("🗑️ Önbelleği Temizle", use_container_width=True):
            st.cache_data.clear()
            st.success("Önbellek temizlendi.")
        st.caption("Veriler güncellendiyse önbelleği temizleyin.")

    # Dil seçici (sağ üst)
    _, lang_col = st.columns([7, 1])
    with lang_col:
        st.radio("", ["TR", "EN"], horizontal=True, key="ui_lang", label_visibility="collapsed")
    lang = st.session_state.ui_lang
    s    = STRINGS[lang]

    # Başlık banner
    st.markdown(
        f'<div class="header-banner">'
        f'<div class="header-title">🎮 Steam Oyun Yorum Analizi</div>'
        f'<div class="header-sub">{s["caption"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Ana sekmeler ──────────────────────────────────────────────────────
    if lang == "TR":
        tab_labels = [
            "🎮 Oyun Analizi", "🏆 Rakip Analizi", "📈 Trend Tahmini", "💰 ROI Simülatörü",
            "🗺️ Şikayet Matrisi", "⚖️ Ağırlıklı Analiz",
        ]
    else:
        tab_labels = [
            "🎮 Game Analysis", "🏆 Competitor Analysis", "📈 Trend Forecast", "💰 ROI Simulator",
            "🗺️ Complaint Matrix", "⚖️ Weighted Analysis",
        ]

    (tab_ana, tab_rakip, tab_trend, tab_roi,
     tab_matris, tab_agirlik) = st.tabs(tab_labels)

    with tab_ana:
        render_tur_nav()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.secili_oyun:
            render_detay(lang)
        else:
            render_oyun_listesi(lang)

    with tab_rakip:
        render_rakip_analizi(lang)

    with tab_trend:
        render_trend_tahmini(lang)

    with tab_roi:
        render_roi_simulator(lang)

    with tab_matris:
        render_sikayet_matrisi(lang)

    with tab_agirlik:
        render_agirlikli_analiz(lang)


if __name__ == "__main__":
    main()