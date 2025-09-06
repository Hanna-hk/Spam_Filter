# Spam Filter: RNN-Powered Message Classification

## Description
This is a Flask web application that uses RNN to filter messages into spam or not-spam. Users input the message, and the trained model outputs if the message is spam.

## Technologies
- **Python 3.9+**
- Flask (web application framework)
- pandas, numpy, nltk (data handling)
- tensorflow, keras-tuner (RNN)
- matplotlib, seaborn, wordcloud (for data analysis)
- dill (for saving)
- kaggle (for dataset download)

## Features
- **Data Ingestion** - load and split dataset
- **Data Transformation** - preprocessing and feature engineering
- **Model Training** - train and evaluate RNN models
- **Prediction Pipeline** - reusable pipeline for inference
- **Flask Web App** - interactive UI for predictions

## Installation and Running

### 1. Clone the repository:
```bash
git clone https://github.com/Hanna-hk/Spam_Filter.git
cd Spam_Filter
```

### 2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the Flask server:
```bash
python app.py
```

### 5. Open in your browser:
```
http://127.0.0.1:5000/
```

## Usage
1. Enter your message in the text box
2. Click the "Classify" button
3. View the results (Spam or Not Spam)
4. See the confidence level of the prediction

## Model Performance
The RNN model achieves high accuracy in classifying messages with:
- Loss: 12.17%
- Accuracy: 98.58%
- Precision: 95.70%
- Recall: 92.71%
