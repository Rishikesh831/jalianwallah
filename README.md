# Jallianwala Bagh Public Interest Predictor

🕯️ **A Machine Learning Tribute to the Martyrs of 1919**

This project predicts public interest around the anniversary of the Jallianwala Bagh massacre using Multiple Linear Regression (implemented from scratch with gradient descent). It honors the memory of those who lost their lives and aims to help plan commemorative events and awareness campaigns.

## Features
- Predicts public interest based on news, YouTube uploads, and event data
- All ML math implemented from scratch (no scikit-learn)
- Streamlit web app for interactive predictions
- Modular, clean Python project structure
- Tribute UI with themed colors and historical context

## File Structure
```
MLV-Jalianwala/
│
├── data/
│   ├── final_jallianwala_dataset.csv
│   └── jallianwala_trends.csv
│
├── models/
│   ├── jallianwala_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── streamlit_app.py
├── README.md
└── requirements.txt
```

## Setup & Usage
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Preprocess data:**
   ```bash
   python src/preprocessing.py
   ```
3. **Train model:**
   ```bash
   python src/train.py
   ```
4. **Predict from CLI:**
   ```bash
   python src/predict.py <NewsArticles> <YouTubeUploads> <WeightedEvents> <InverseDays> <DaysSquared>
   ```
5. **Run Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Tribute
> "Never Forget. Never Again."
>
> This project is dedicated to the memory of the innocent lives lost at Jallianwala Bagh, Amritsar, 1919. May we always remember their sacrifice and strive for peace and justice.

---
Created with ❤️ by Rishikesh Bhatt "# jalianwallah" 


![Screenshot 2025-06-22 000052](https://github.com/user-attachments/assets/67b516f9-5c7c-4eb6-9c97-42ba08da3784)
![Screenshot 2025-06-22 000110](https://github.com/user-attachments/assets/c058eec4-3074-4d40-9f0e-38630a99cd5b)


