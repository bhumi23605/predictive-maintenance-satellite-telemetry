Overview:

This is a satellite predictive maintenance application devekloped by taking dataset from NASA telemetry.

Key Points:

LightGBM uses histogram-based algorithms, which bucket continuous feature (attribute) values into discrete bins. This speeds up training and reduces memory usage. LightGBM grows trees leaf-wise. It will choose the leaf with max delta loss to grow.
Leaf-wise may cause over-fitting when data is small, therefore LightGBM includes max_depth parameter to limit tree depth.

Trained on models like LightGBM,XGBoost and used performance indicators like:
1) Confusion Matrix
2) Accuracy
3) Mathew Coorelation Coefficient: TP,TN,FP,FN

Data Pipeline Flow:

<img width="1266" height="335" alt="image" src="https://github.com/user-attachments/assets/5d49c8e7-7465-4226-a87c-ba3203792022" />

Tech Stack:

Languages: Python

Model building : LightGBM , XGBoost

Development tools: VS code and Jupyter notebook
 
Deployment/APIs: Streamlit, FastAPI
