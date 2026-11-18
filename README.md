# Intelligent Port Logistics AI

AI-powered logistics platform for intelligent port cargo scheduling, customs coordination, truck routing, and congestion prediction using IBM Granite LLM and IBM Cloud.

This repository holds the first working piece: a **port congestion predictor** so cargo, truck and customs teams can plan ahead instead of reacting to long queues.

> **Status:** early prototype built on **synthetic data**. It is a learning and portfolio project, not a production system. No real port data is used and no real-world accuracy is claimed.

## The problem
Ports lose time and money when too many trucks and vessels arrive at once. Queues build up at gates and berths, customs checks run late, and trucks sit idle. Planners often learn about congestion only after it has already happened.

## The idea
Use past arrival patterns to estimate the chance of congestion for a given hour, so a planner can shift truck slots or customs staffing before the queue forms.

## What works today
- Synthetic data generator for port activity (arrivals, vessels in port, customs delay, queue length)
- Congestion prediction model (Random Forest, scikit-learn) with a basic evaluation report
- Streamlit dashboard: enter conditions for an hour and see the predicted congestion risk, plus average queue by hour of day

## Planned (not built yet)
- Truck scheduling and routing optimisation
- Customs coordination view
- Natural-language assistant (IBM Granite LLM on IBM Cloud) for questions like "which slot is least busy tomorrow?"
- Replace synthetic data with public or partner port datasets

## Architecture
```
data/generate_synthetic_data.py  ->  data/port_data.csv
                                        |
                              src/predict_congestion.py  (train + evaluate model)
                                        |
                                     app.py  (Streamlit dashboard)
```

## Tech stack
Python, pandas, NumPy, scikit-learn, Streamlit

## Quickstart
```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py
python src/predict_congestion.py
streamlit run app.py
```

## About
Built by a third-year B.Tech student exploring how AI can be applied to real logistics problems. Feedback and suggestions are welcome through issues.
