# PyaazAI — Smart Onion Farming Platform
## AWS Architecture Document

---

## 1. Overview

PyaazAI is an IoT + AI-powered precision agriculture platform for Indian onion farmers. It combines real-time field sensors, ML-based price forecasting, weather intelligence, and WhatsApp-native advisory delivery to maximize yield, reduce post-harvest losses, and help farmers sell at optimal prices.

**Target Users:**
- Primary: Onion farmers (Nashik, Indore, Kurnool belt)
- Secondary: FPO (Farmer Producer Organization) managers
- Tertiary: Government agriculture departments, researchers

**Key Outcomes:**
- 7-day onion price prediction (R² = 0.88, MAPE = 10.1%)
- Organic pest management advisories (zero chemical recommendations)
- Irrigation scheduling based on weather + crop stage
- Sell/Hold signals that can save ₹764/quintal per farmer

---

## 2. Architecture Layers

### Layer 1: Data Sources (Field + External)

```
┌────────────────────────────────────────────────────────────────┐
│  DATA SOURCES                                                   │
│                                                                  │
│  [1] IoT Sensors (Phase 2)      [2] WhatsApp (Farmer Input)    │
│  • Soil moisture (10cm, 30cm)   • Planting date reports         │
│  • Soil temperature             • Pest sighting descriptions    │
│  • Air temp + humidity (DHT22)  • Harvest notifications         │
│  • Rainfall gauge               • Market queries                │
│  • Light intensity (LDR)        • Photo uploads (pest ID)       │
│                                                                  │
│  [3] Weather APIs               [4] Market APIs                 │
│  • IMD (India Met Dept)         • Agmarknet (data.gov.in)       │
│  • OpenWeatherMap               • 8 mandis, 6 years history     │
│  • 5-day forecast               • Daily modal/min/max prices    │
│  • Hourly temp/humidity/rain    • Arrival quantities            │
└────────────────────────────────────────────────────────────────┘
```

**AWS Services:**
| Source | Service | Protocol |
|--------|---------|----------|
| IoT Sensors | AWS IoT Core | MQTT over TLS |
| WhatsApp | API Gateway + Lambda | HTTPS (Twilio webhook) |
| Weather | EventBridge + Lambda | REST API (scheduled) |
| Mandi Prices | EventBridge + Lambda | REST API (daily cron) |

---

### Layer 2: Ingestion & Processing

```
┌────────────────────────────────────────────────────────────────┐
│  INGESTION & PROCESSING                                         │
│                                                                  │
│  [5] AWS IoT Core              [6] API Gateway                  │
│  • Device registry             • Twilio WhatsApp webhook        │
│  • MQTT message broker         • POST /incoming                 │
│  • IoT Rules Engine            • Routes to Router Lambda        │
│       │                              │                          │
│       ▼                              ▼                          │
│  [7] IoT Rules → Lambda       [8] Router Lambda                │
│  • Transform sensor data       • Keyword-based intent routing   │
│  • Anomaly detection           • /help, /switch commands        │
│  • Write to S3 + DynamoDB      • Forwards to Farm/Legal bot     │
│       │                              │                          │
│       ▼                              ▼                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  [9] Amazon S3 — Data Lake                               │   │
│  │  s3://onion-farm-data/                                    │   │
│  │  ├── raw/prices/         (Agmarknet daily CSV)            │   │
│  │  ├── raw/weather/        (IMD/OWM JSON)                   │   │
│  │  ├── raw/iot-readings/   (sensor MQTT payloads)           │   │
│  │  ├── processed/          (feature-engineered datasets)    │   │
│  │  ├── models/             (price_model.pkl, forecast.json) │   │
│  │  └── dashboards/         (Tableau-ready CSVs)             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  [10] DynamoDB (7 tables)                                       │
│  • onion-farmers-prod (profiles, planting dates, region)        │
│  • onion-chat-history-prod (conversation log)                   │
│  • whatsapp-router-prod (user preferences, default bot)         │
│  • legal-cases-prod, legal-research-prod, legal-drafts-prod     │
│  • legal-chat-history-prod                                      │
└────────────────────────────────────────────────────────────────┘
```

---

### Layer 3: ML & AI Models

```
┌────────────────────────────────────────────────────────────────┐
│  ML & AI ENGINE                                                  │
│                                                                  │
│  [11] PRICE FORECASTER (GradientBoosting)                       │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Algorithm: sklearn GradientBoostingRegressor        │        │
│  │  Features: 41 (lag prices, rolling stats, season,    │        │
│  │            momentum, arrivals, calendar features)     │        │
│  │  Training: 16,520 records, 8 mandis, 6 years        │        │
│  │  Performance: R² = 0.88 | MAPE = 10.1% | RMSE = ₹294│        │
│  │  Output: 7-day price forecast + SELL/HOLD signal     │        │
│  │  Retrain: Daily via EventBridge cron                 │        │
│  │  Artifact: s3://…/models/price_model.pkl (1.5MB)     │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  [12] DISEASE RISK SCORER (Phase 1)                             │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Algorithm: Logistic Regression / Random Forest      │        │
│  │  Input: humidity + temp + leaf wetness + crop stage  │        │
│  │  Output: Risk score 0-100 per disease (thrips,       │        │
│  │          purple blotch, basal rot, stemphylium)      │        │
│  │  Trigger: Hourly (with sensor data) or on-demand     │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  [13] NLP ORCHESTRATOR (Amazon Bedrock)                         │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Model: Amazon Nova Pro v1                           │        │
│  │  Tasks:                                              │        │
│  │  • Intent detection (Hindi/English/Marathi)          │        │
│  │  • Crop advisory generation                          │        │
│  │  • Pest identification from description              │        │
│  │  • Legal research synthesis (FAISS RAG)              │        │
│  │  • Document drafting (Indian court format)           │        │
│  │  Cost: ~$0.01/query                                  │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  [14] LEGAL RESEARCH (FAISS Vector Search)                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Embedding: Amazon Titan Embed Text V2 (1024-dim)    │        │
│  │  Index: FAISS IndexFlatIP (cosine similarity)        │        │
│  │  Corpus: 273 chunks from 5 Indian court judgments    │        │
│  │  Retrieval: Top-5 chunks → Bedrock synthesis         │        │
│  │  Artifact: s3://…/faiss-index/ (index + metadata)    │        │
│  └─────────────────────────────────────────────────────┘        │
└────────────────────────────────────────────────────────────────┘
```

---

### Layer 4: Application Layer (Bot Logic)

```
┌────────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER (Lambda Functions)                            │
│                                                                  │
│  [15] WhatsApp Router Lambda (128MB, 15s timeout)               │
│  • Receives all Twilio webhooks                                 │
│  • Keyword detection (Hindi + English)                          │
│  • Routes to Farm Bot or Legal Bot (async invocation)           │
│  • Handles /help, /switch commands directly                     │
│  • Stores user preference in DynamoDB                           │
│                                                                  │
│  [16] Farm Bot Lambda (256MB, 60s timeout)                      │
│  • 6 tools: crop_calendar, weather_advisory, mandi_prices,     │
│    pest_management, irrigation_advisor, register_farmer         │
│  • Reads ML forecast from S3 (latest_forecast.json)            │
│  • Bedrock Nova Pro for intent detection + advisory gen         │
│  • Bilingual responses (Hindi primary, English secondary)       │
│                                                                  │
│  [17] Legal Bot Lambda (512MB, 60s timeout)                     │
│  • 3 tools: research_judgments, draft_document, devils_advocate │
│  • FAISS index loaded from S3 (cached per cold start)          │
│  • Bedrock Nova Pro for synthesis + drafting                    │
│  • Indian court citation format                                 │
│                                                                  │
│  [18] ML Training Lambda / Local Script                         │
│  • fetch_mandi_data.py (data collection)                       │
│  • train_price_model.py (model training)                       │
│  • Triggered by EventBridge (daily) or manual                  │
│  • Outputs: model.pkl + forecast.json → S3                     │
└────────────────────────────────────────────────────────────────┘
```

---

### Layer 5: Delivery & Visualization

```
┌────────────────────────────────────────────────────────────────┐
│  DELIVERY & DASHBOARDS                                          │
│                                                                  │
│  [19] WhatsApp (Twilio Sandbox)                                 │
│  • Farmer-facing: Hindi/Marathi advisories                     │
│  • ML predictions: "₹3663 → ₹3456 in 7 days. अभी बेचें!"     │
│  • Organic pest remedies                                        │
│  • Irrigation scheduling                                        │
│  • Legal research + document drafts                             │
│                                                                  │
│  [20] Tableau Public (Dashboard)                                │
│  • Price Trend (line chart with ML forecast overlay)           │
│  • Mandi Comparison (bar chart, 8 mandis, seasonal stacking)  │
│  • Seasonal Heatmap (month × year price intensity)            │
│  • Arrivals vs Price (scatter — supply/demand correlation)     │
│  • Forecast with Confidence Bands                              │
│                                                                  │
│  [21] SNS / Email Alerts (Phase 1)                             │
│  • "SELL NOW" push notification when peak predicted            │
│  • Disease outbreak warning to FPO managers                    │
│  • Weather emergency (heavy rain before harvest)               │
│  • Harvest window reminder                                      │
└────────────────────────────────────────────────────────────────┘
```

---

### Layer 6: IoT Field Infrastructure (Phase 2)

```
┌────────────────────────────────────────────────────────────────┐
│  IoT FIELD LAYER (Phase 2 — Hardware)                           │
│                                                                  │
│  [22] Sensor Node (per 2 acres)                                 │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Hardware: ESP32-S3 + Solar Panel (5W)               │        │
│  │  Sensors:                                            │        │
│  │  • Capacitive soil moisture × 2 (10cm + 30cm)       │        │
│  │  • DHT22 (air temp + humidity)                       │        │
│  │  • BH1750 (light intensity lux)                      │        │
│  │  • Tipping bucket rain gauge                         │        │
│  │  Communication: LoRa SX1276 (868 MHz, 5km range)    │        │
│  │  Power: 18650 Li-ion + solar (6-month battery)       │        │
│  │  Enclosure: IP67 weatherproof                        │        │
│  │  Cost: ₹3,000-5,000 per unit                        │        │
│  │  Transmission: Every 15 minutes                      │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  [23] LoRa Gateway (per 5km radius / village)                   │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Hardware: Raspberry Pi 4 + LoRa HAT + 4G SIM       │        │
│  │  Software: AWS IoT Greengrass                        │        │
│  │  Functions:                                          │        │
│  │  • Receive LoRa packets from field sensors           │        │
│  │  • Edge ML inference (anomaly detection)             │        │
│  │  • Buffer data during connectivity loss              │        │
│  │  • Forward to AWS IoT Core via MQTT/4G               │        │
│  │  • Local alerts (buzzer if critical threshold)       │        │
│  │  Cost: ₹8,000-12,000 per gateway                    │        │
│  │  Coverage: 20-50 sensor nodes per gateway            │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  [24] AWS IoT Greengrass (Edge)                                 │
│  • Runs ML model locally on gateway                            │
│  • Detects anomalies without cloud roundtrip                   │
│  • Syncs with IoT Core when connected                          │
│  • OTA updates for sensor firmware                             │
│                                                                  │
│  [25] AWS IoT Core (Cloud)                                      │
│  • MQTT broker for all sensor data                             │
│  • Device shadows (last known state)                           │
│  • Rules Engine → S3, DynamoDB, Lambda                         │
│  • Device management (fleet provisioning)                      │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow Diagram

```
Farmer WhatsApp msg
    │
    ▼
Twilio → API Gateway → Router Lambda
    │                       │
    │            ┌──────────┼──────────┐
    │            ▼                     ▼
    │     Farm Bot Lambda       Legal Bot Lambda
    │            │                     │
    │     ┌──────┤                     ├── FAISS (S3)
    │     │      │                     ├── Bedrock Nova Pro
    │     │      ├── ML Forecast (S3)  └── DynamoDB
    │     │      ├── Bedrock Nova Pro
    │     │      ├── Weather API
    │     │      └── DynamoDB
    │     │
    │     ▼
    │   Response (Hindi/English)
    │     │
    ▼     ▼
Twilio → WhatsApp → Farmer
```

```
IoT Sensor (Phase 2)
    │ (LoRa, every 15 min)
    ▼
LoRa Gateway (Greengrass)
    │ (MQTT over 4G)
    ▼
AWS IoT Core → Rules Engine
    │
    ├──→ S3 (raw time-series storage)
    ├──→ DynamoDB (latest reading per farm)
    ├──→ Lambda (threshold alerts)
    │         │
    │         ▼
    │    SNS → WhatsApp/SMS alert
    │    "⚠️ Soil moisture 25% — irrigate today!"
    │
    └──→ Lambda (daily aggregation)
              │
              ▼
         S3 (processed) → ML retrain
```

---

## 4. AWS Services Summary

| # | Service | Purpose | Status | Monthly Cost |
|---|---------|---------|--------|-------------|
| 1 | API Gateway (HTTP) | Twilio webhook, bot APIs | ✅ LIVE | $0 (free tier) |
| 2 | Lambda (5 functions) | Router, Farm Bot, Legal Bot, ETL, ML | ✅ LIVE | $0 (free tier) |
| 3 | DynamoDB (7 tables) | Profiles, chat, preferences, research | ✅ LIVE | ~$0.25 |
| 4 | S3 | Data lake, models, FAISS index | ✅ LIVE | ~$0.05 |
| 5 | Amazon Bedrock (Nova Pro) | NLU, advisory gen, drafting | ✅ LIVE | ~$2-30 (usage) |
| 6 | Amazon Bedrock (Titan Embed) | Legal document embeddings | ✅ LIVE | ~$0.01 |
| 7 | EventBridge | Daily ML retrain scheduler | 🔜 Phase 1 | $0 |
| 8 | SNS | Push alerts (sell signals, disease) | 🔜 Phase 1 | $0.50/100K |
| 9 | AWS IoT Core | Sensor data ingestion (MQTT) | 🔮 Phase 2 | $1/M msgs |
| 10 | AWS IoT Greengrass | Edge ML on LoRa gateway | 🔮 Phase 2 | $0 (3 devices free) |
| 11 | Tableau Public | Dashboards (FPO demo) | ✅ LIVE | $0 |
| 12 | QuickSight | Paid dashboards (FPO production) | 🔮 Phase 2 | $24+/user |
| 13 | SageMaker | Model training at scale | 🔮 Phase 2 | $0.05/hr |

---

## 5. Security & Compliance

| Control | Implementation |
|---------|---------------|
| Data in transit | TLS 1.2+ (MQTT, HTTPS) |
| Data at rest | S3 SSE-S3, DynamoDB encryption |
| IAM | Least-privilege roles per Lambda |
| API Auth | Twilio webhook signature validation |
| PII | Farmer phone numbers hashed in logs |
| Data retention | 30-day TTL on chat history, 1-year on profiles |
| Secrets | Lambda environment variables (future: Secrets Manager) |

---

## 6. Cost Analysis

| Scale | Users | Monthly AWS | Revenue | Margin |
|-------|-------|-------------|---------|--------|
| Testing | 1-10 | ₹300 | ₹0 | — |
| Early | 50 farmers + 5 lawyers | ₹3,800 | ₹19,945 | 81% |
| Growth | 500 farmers + 20 lawyers | ₹34,000 | ₹1,69,480 | 80% |
| Scale | 2000 farmers + 50 lawyers | ₹1,00,000 | ₹6,47,950 | 85% |

---

## 7. Deployment Status

| Component | Deployed | Endpoint |
|-----------|----------|----------|
| WhatsApp Router | ✅ | https://6mjfaec7l3.execute-api.us-east-1.amazonaws.com/ |
| Farm Bot | ✅ | Lambda: onion-farm-bot-prod |
| Legal Bot | ✅ | Lambda: legal-ai-bot-prod |
| ML Price Model | ✅ | S3: onion-ml-models/price_model.pkl |
| Tableau Dashboard | ✅ | Tableau Public (local) |
| IoT Core | 🔮 | Phase 2 |
| Greengrass | 🔮 | Phase 2 |

---

## 8. Roadmap

| Phase | Timeline | Deliverable |
|-------|----------|-------------|
| Phase 0 (DONE) | Aug 2026 | WhatsApp bot + ML forecast + Tableau |
| Phase 1 | Sep 2026 | Real Agmarknet data, daily retrain, SNS alerts, disease model |
| Phase 2 | Oct-Dec 2026 | IoT sensors (10 pilot units), Greengrass edge, real-time dashboard |
| Phase 3 | Q1 2027 | FPO partnerships, government integration, SageMaker at scale |

---

*Document Version: 1.0 | Date: 5 August 2026 | Author: Saurabh Mukherjee*
