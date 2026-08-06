"""
Onion Price Forecaster — Step 1: Fetch Historical Mandi Data
==============================================================
Fetches onion price data from data.gov.in / Agmarknet API.
Downloads daily modal prices for key mandis (Lasalgaon, Nashik, Azadpur).

Data columns:
  - date, mandi, state, variety, min_price, max_price, modal_price (₹/quintal)

Usage:
    python fetch_mandi_data.py
"""

import csv
import json
import os
import time
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

# data.gov.in API (free, no auth needed for this dataset)
# Resource ID for daily market prices of agricultural commodities
DATA_GOV_API = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"  # Public demo key

# Target mandis for onion
TARGET_MANDIS = [
    "Lasalgaon",
    "Pimpalgaon",
    "Nashik",
    "Manmad",
    "Azadpur",
    "Indore",
    "Rajkot",
    "Kurnool",
]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "data", "onion_prices_historical.csv")


def fetch_from_data_gov(commodity="Onion", from_date="2020-01-01", to_date="2026-08-05"):
    """
    Fetch commodity prices from data.gov.in API.
    The API returns paginated results (max 1000 per page).
    """
    all_records = []
    offset = 0
    limit = 1000

    print(f"Fetching onion prices from {from_date} to {to_date}...")

    while True:
        params = {
            "api-key": API_KEY,
            "format": "json",
            "filters[commodity]": commodity,
            "offset": str(offset),
            "limit": str(limit),
        }

        url = f"{DATA_GOV_API}?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            response = urllib.request.urlopen(req, timeout=30)
            data = json.loads(response.read())

            records = data.get("records", [])
            if not records:
                break

            all_records.extend(records)
            print(f"  Fetched {len(all_records)} records so far...")

            if len(records) < limit:
                break  # Last page

            offset += limit
            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"  Error at offset {offset}: {e}")
            break

    return all_records


def generate_synthetic_data():
    """
    Generate realistic synthetic onion price data for model training.
    Based on known patterns:
    - Kharif onion (Oct-Dec): Prices typically HIGH (supply low)
    - Rabi onion (Jan-May): Prices typically LOW (supply high, Nashik belt harvest)
    - Late Kharif (Jun-Sep): Prices MODERATE to HIGH
    - Festive season (Oct-Nov): Price spikes
    - Monsoon storage: Prices increase (storage losses reduce supply)
    """
    import random

    print("Generating synthetic training data based on known market patterns...")

    # Base price patterns (₹/quintal) by month for Lasalgaon
    monthly_base = {
        1: 1200, 2: 1000, 3: 900, 4: 1100, 5: 1400,
        6: 1800, 7: 2200, 8: 2500, 9: 2800, 10: 3000,
        11: 2600, 12: 2000,
    }

    # Mandi-specific multipliers
    mandi_multiplier = {
        "Lasalgaon": 1.0,
        "Pimpalgaon": 0.95,
        "Nashik": 1.05,
        "Manmad": 0.92,
        "Azadpur": 1.35,  # Delhi always higher (transport cost + demand)
        "Indore": 1.10,
        "Rajkot": 0.98,
        "Kurnool": 1.02,
    }

    records = []
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2026, 8, 5)

    current_date = start_date
    while current_date <= end_date:
        # Skip Sundays (mandis closed)
        if current_date.weekday() == 6:
            current_date += timedelta(days=1)
            continue

        month = current_date.month
        base_price = monthly_base[month]

        # Year-over-year inflation (5-10% per year)
        year_factor = 1 + (current_date.year - 2020) * 0.07

        # Random daily variation (±15%)
        daily_noise = random.uniform(0.85, 1.15)

        # Trend within month (gradual increase/decrease)
        day_of_month = current_date.day
        intra_month_trend = 1 + (day_of_month - 15) * 0.003  # slight trend

        for mandi, multiplier in mandi_multiplier.items():
            modal_price = int(base_price * year_factor * daily_noise * intra_month_trend * multiplier)
            min_price = int(modal_price * random.uniform(0.80, 0.92))
            max_price = int(modal_price * random.uniform(1.08, 1.20))

            # Arrivals (quintals) — inversely correlated with price
            base_arrival = 15000 if mandi == "Lasalgaon" else 5000
            arrivals = int(base_arrival * random.uniform(0.5, 1.5) * (1.5 - (modal_price / 5000)))

            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "mandi": mandi,
                "state": {
                    "Lasalgaon": "Maharashtra", "Pimpalgaon": "Maharashtra",
                    "Nashik": "Maharashtra", "Manmad": "Maharashtra",
                    "Azadpur": "Delhi", "Indore": "Madhya Pradesh",
                    "Rajkot": "Gujarat", "Kurnool": "Andhra Pradesh",
                }[mandi],
                "variety": "Onion",
                "min_price": max(500, min_price),
                "max_price": max_price,
                "modal_price": max(600, modal_price),
                "arrivals_tonnes": max(100, arrivals),
            })

        current_date += timedelta(days=1)

    return records


def save_to_csv(records: list, output_file: str):
    """Save records to CSV."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fieldnames = ["date", "mandi", "state", "variety", "min_price", "max_price", "modal_price", "arrivals_tonnes"]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print(f"\n✅ Saved {len(records)} records to {output_file}")
    print(f"   Date range: {records[0]['date']} to {records[-1]['date']}")
    print(f"   Mandis: {len(set(r['mandi'] for r in records))}")


def main():
    print("=" * 60)
    print("  ONION PRICE FORECASTER — Data Collection")
    print("=" * 60)

    # Try fetching from data.gov.in first
    records = fetch_from_data_gov()

    if len(records) < 100:
        print("\n⚠️ Insufficient data from API. Using synthetic training data.")
        print("   (This produces realistic patterns for model training)")
        records = generate_synthetic_data()

    save_to_csv(records, OUTPUT_FILE)

    # Print sample
    print("\n📊 Sample data (last 5 records for Lasalgaon):")
    lasalgaon = [r for r in records if r["mandi"] == "Lasalgaon"]
    for r in lasalgaon[-5:]:
        print(f"   {r['date']} | ₹{r['modal_price']}/q | Arrivals: {r['arrivals_tonnes']}t")


if __name__ == "__main__":
    main()
