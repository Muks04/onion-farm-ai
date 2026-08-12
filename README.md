# Onion Farm AI Bot — WhatsApp Advisory for Indian Farmers

A WhatsApp-based AI advisory system for Indian onion farmers that provides **ML-powered price forecasts**, weather advisories, crop calendar guidance, pest management, and irrigation scheduling — all in **Hindi/Hinglish**.

## Live System

- **WhatsApp:** Active via Twilio sandbox
- **Lambda:** `onion-farm-bot-prod` (us-east-1)
- **API Gateway:** https://swhfcy5v1c.execute-api.us-east-1.amazonaws.com/
- **ML Model:** GradientBoosting price forecaster (R² = 0.88)

## Features

| Feature | What it does |
|---------|-------------|
| Mandi Price Forecast | ML-predicted onion prices 7 days ahead with buy/sell/hold recommendation |
| Weather Advisory | Localized weather impact on crop + actionable farming advice |
| Crop Calendar | What to do today based on planting date (nursery → harvest) |
| Pest Management | Identifies pest/disease from farmer's description, gives treatment |
| Irrigation Advisor | When to water, how much, based on crop stage + weather |
| Farmer Registration | Tracks planting date for personalized calendar advice |

## ML Model — Price Forecaster

| Metric | Value |
|--------|-------|
| Algorithm | GradientBoostingRegressor |
| Training Data | 16,520 records (Lasalgaon mandi) |
| Features | 41 (lags, rolling stats, seasonality, momentum) |
| R² Score | 0.88 |
| Validation | 5-fold Time Series Cross-Validation |
| Forecast Window | 7 days ahead |

**Key features used:**
- Lag prices (1, 3, 7, 14, 30 days)
- Rolling averages & standard deviations (7, 14, 30 days)
- Price momentum (7d, 14d, 30d change)
- Calendar features (month, day, week, quarter)
- Indian agricultural seasons (Kharif, Rabi, Summer)
- Arrival volumes at mandi

## Sample Interaction (Hindi)

```
Farmer: मंडी भाव क्या है?
Bot:    📊 लासलगांव मंडी (प्याज):
        आज: ₹2,450/क्विंटल
        7-दिन पूर्वानुमान: ₹2,600/क्विंटल (📈 +6%)
        💡 सुझाव: HOLD — अगले 7 दिनों में भाव बढ़ने की संभावना

Farmer: कीड़ा लगा है, पत्ती पर धब्बे दिख रहे
Bot:    🦠 Purple Blotch (बैंगनी धब्बा रोग)
        उपचार: Mancozeb 75% WP (2.5 g/L) का छिड़काव करें
        सावधानी: सुबह 7-9 बजे छिड़काव करें, बारिश से पहले नहीं
        अगला छिड़काव: 10 दिन बाद
```

## Architecture

```
Farmer (WhatsApp, Hindi/Hinglish)
       │
       ▼
Twilio Webhook → API Gateway
       │
       ▼
Lambda: onion-farm-bot-prod
       │
       ├── Intent Detection (Bedrock Nova Pro)
       │         │
       │    ┌────┼────────────┬──────────────┐
       │    ▼    ▼            ▼              ▼
       │  Crop   Weather    Mandi Price    Pest
       │  Calendar Advisory  (ML Model)   Management
       │
       ▼
DynamoDB (Chat History + Farmer Profiles)
       │
       ▼
S3 (ML Model + Training Data)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Runtime | AWS Lambda (us-east-1) |
| Model (NLP) | Amazon Nova Pro v1.0 (intent detection + response generation) |
| Model (ML) | GradientBoostingRegressor (scikit-learn, price forecasting) |
| Messaging | Twilio WhatsApp API |
| Storage | DynamoDB (chat history), S3 (ML models, training data) |
| API | API Gateway (HTTP API) |
| Language | Hindi, Hinglish, Marathi, English (auto-detected) |

## Project Structure

```
onion-farm-bot/
├── functions/
│   ├── onion_orchestrator.py   # Lambda handler + intent routing
│   └── onion_tools.py          # Tool implementations (6 tools)
├── ml/
│   ├── train_price_model.py    # Model training pipeline
│   ├── fetch_mandi_data.py     # Data collection from AgMarkNet
│   ├── data/                   # Historical price CSVs
│   └── models/                 # Trained model + metrics + forecast
├── docs/                       # Documentation
├── infra/                      # IaC templates
└── README.md
```

## Training the ML Model

```bash
# Fetch latest mandi data
python ml/fetch_mandi_data.py

# Train model (outputs model.pkl + metrics + 7-day forecast)
python ml/train_price_model.py
```

Output:
```
  Model Performance (5-fold Time Series CV):
     RMSE:  ₹142 ± 23
     MAPE:  5.8% ± 1.2%
     R²:    0.88 ± 0.03

  Recommendation: HOLD — Price expected to rise to ₹2,600/q in next 7 days
```

## Why This Matters

- 75% of Indian onion farmers have no access to market intelligence
- Price volatility causes 30-40% income loss (sell too early or too late)
- Language barrier: existing advisory services are English-only
- This bot runs on WhatsApp (95%+ rural smartphone penetration) in Hindi

## Author

**Saurabh Mukherjee** — AWS Solutions Architect Professional | GenAI Professional | ML Engineer Associate

Built for Indian onion farmers in Nashik/Lasalgaon belt (Maharashtra).
