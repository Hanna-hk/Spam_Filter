import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import NumberPreprocessor
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.components.data_transformation import TextPreprocessor

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = r'artifacts\model.pkl'
            preprocessor_path = r'artifacts\preprocessor.pkl'
            tokenizer_path=r'artifacts\tokenizer.pkl'
            scaler_path = r'artifacts\scaler.pkl'

            model = load_object(file_path=model_path)
            text_preprocessor = load_object(file_path=preprocessor_path)
            tokenizer = load_object(file_path=tokenizer_path)
            scaler = load_object(file_path=scaler_path)
            numerical_extractor = NumberPreprocessor()

            max_length = 100

            preprocessed_texts = text_preprocessor.transform(features['Message'])
            X_text_seq = tokenizer.texts_to_sequences(preprocessed_texts)
            X_text = pad_sequences(X_text_seq, maxlen=max_length, padding='post', truncating='post')

            X_numerical = numerical_extractor.transform(features)
            X_numerical_scaled=scaler.transform(X_numerical)
            
            predictions = model.predict([X_text, X_numerical_scaled])
            print(f"Raw prediction: {predictions}")
            if predictions < 0.5:
                return "Not Spam"
            else:
                return "Spam"

        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, message:str):
        if message is None:
            self.message = ""
        elif not isinstance(message, str):
            self.message = str(message)
        else:
            self.message = message
    def get_data_as_data_frame(self):
        try:
            custom_data_input = {"Message": [self.message]}
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomException(e, sys)