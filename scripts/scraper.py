import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

REVIEWS_URL = "https://store.steampowered.com/appreviews/{app_id}"
APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"
DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def get_game_name(app_id: int) -> str:
    try:
        resp = requests.get(
            APPDETAILS_URL,
            params={"appids": app_id, "filters": "basic"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get(str(app_id), {}).get("success"):
            return data[str(app_id)]["data"].get("name", str(app_id))
    except Exception:
        pass
    return str(app_id)


def fetch_reviews(app_id: int, max_reviews: int = 1000, language: str = "all") -> list:
    reviews = []
    cursor = "*"
    base_params = {
        "json": 1,
        "language": language,
        "num_per_page": 100,
        "review_type": "all",
        "purchase_type": "all",
        "filter": "recent",
    }

    with tqdm(total=max_reviews, desc=f"Yorumlar çekiliyor (App ID: {app_id})") as pbar:
        while len(reviews) < max_reviews:
            try:
                resp = requests.get(
                    REVIEWS_URL.format(app_id=app_id),
                    params={**base_params, "cursor": cursor},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"\nIstek hatası: {e}")
                break

            if data.get("success") != 1:
                print("\nAPI başarısız yanıt döndürdü.")
                break

            batch = data.get("reviews", [])
            if not batch:
                print("\nDaha fazla yorum yok.")
                break

            reviews.extend(batch)
            pbar.update(len(batch))

            new_cursor = data.get("cursor")
            if not new_cursor or new_cursor == cursor:
                break
            cursor = new_cursor

            time.sleep(1.0)

    return reviews[:max_reviews]


def parse_reviews(raw: list, app_id: int) -> list:
    records = []
    for r in raw:
        author = r.get("author", {})
        ts_created = r.get("timestamp_created", 0)
        ts_updated = r.get("timestamp_updated", 0)
        records.append({
            "app_id": app_id,
            "review_id": r.get("recommendationid"),
            "author_steamid": author.get("steamid"),
            "author_num_reviews": author.get("num_reviews"),
            "playtime_forever_min": author.get("playtime_forever"),
            "playtime_last_2weeks_min": author.get("playtime_last_two_weeks"),
            "language": r.get("language"),
            "review": r.get("review"),
            "date_created": datetime.fromtimestamp(ts_created, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if ts_created else None,
            "date_updated": datetime.fromtimestamp(ts_updated, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if ts_updated else None,
            "voted_up": r.get("voted_up"),
            "votes_up": r.get("votes_up"),
            "votes_funny": r.get("votes_funny"),
            "weighted_vote_score": r.get("weighted_vote_score"),
            "comment_count": r.get("comment_count"),
            "steam_purchase": r.get("steam_purchase"),
            "received_for_free": r.get("received_for_free"),
            "written_during_early_access": r.get("written_during_early_access"),
        })
    return records


def scrape(app_id: int, max_reviews: int = 1000, language: str = "all") -> pd.DataFrame:
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    game_name = get_game_name(app_id)
    print(f"Oyun: {game_name}  |  App ID: {app_id}")

    raw = fetch_reviews(app_id, max_reviews=max_reviews, language=language)
    print(f"\nToplam {len(raw)} yorum çekildi.")

    records = parse_reviews(raw, app_id)
    df = pd.DataFrame(records)

    safe_name = "".join(c if c.isalnum() or c in "_- " else "" for c in game_name)
    safe_name = safe_name.replace(" ", "_")[:40]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{app_id}_{timestamp}.csv"
    filepath = os.path.join(DATA_RAW_DIR, filename)

    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"Kaydedildi : {filepath}")
    print(f"Boyut      : {df.shape[0]} satır × {df.shape[1]} sütun")
    print(f"Dil dağılımı:\n{df['language'].value_counts().head(10).to_string()}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Steam yorum scraper")
    parser.add_argument("app_id", type=int, help="Steam App ID  (örn. 570 = Dota 2, 730 = CS2)")
    parser.add_argument("--max", type=int, default=1000, metavar="N", help="Maksimum yorum sayısı (varsayılan: 1000)")
    parser.add_argument("--lang", type=str, default="all", metavar="LANG", help="Dil filtresi (varsayılan: all)")
    args = parser.parse_args()

    scrape(args.app_id, max_reviews=args.max, language=args.lang)
