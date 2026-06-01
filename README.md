# AI-Driven Hardware Manufacturing QA

Modern hardware manufacturing (NPI and production) generates massive amounts of fleet-scale test data. Traditional threshold-based testing often misses subtle drift or complex multivariate anomalies. 

This repository demonstrates an AI approach to manufacturing test data:
1. Data Engineering: Simulates ETL of hardware functional test logs (e.g., Voltage, Temperature, Resistance).
2. Anomaly Detection: Uses an Isolation Forest (Machine Learning) to detect subtle, multivariate test anomalies before they cause critical field failures.
3. Agentic Root Cause Analysis (RCA): Simulates an LLM recommendation system (like Vertex AI) that translates the statistical anomaly into an actionable hardware troubleshooting step for the engineering team.

