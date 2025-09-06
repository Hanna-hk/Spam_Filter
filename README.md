# SPAM FILTER (RNN Project)

## Description
This is a Flask web application that uses RNN to filter messages into spam or not-spam.
Users input the message, and the trained model outputs if the message is spam.

---

## Technologies
- Python 3.9+
- Flask (web application framework)
- pandas, numpy, nltk (data handling)
- tensorflow, keras-tuner (RNN)
- matplotlib, seaborn, wordcloud (for data analysis)
- dill (for saving)
- kaggle (for dataset download)

---

## Features
- Data Ingestion - load and split dataset
- Data Transformation - preprocessiong and feature engineering
- Model Training - train and evaluate RNN models
- Prediction Pipeline - reusable pipeline for inference
- Flask Web App - interactive UI for predictions

---

## Installation and Running

1. Clone the repository:

git clone https://github.com/Hanna-hk/Spam_Filter.git
cd Spam_Filter

2. Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

3. Install dependencies:

pip install -r requirements.txt

4. Run the Flask server:

python app.py

5. Open in your browser:

http://127.0.0.1:5000/