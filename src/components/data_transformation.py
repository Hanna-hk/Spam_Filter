import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from typing import Tuple, Optional
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    tokenizer_obj_file_path = os.path.join('artifacts', "tokenizer.pkl")
    label_encoder_file_path = os.path.join('artifacts', "label_encoder.pkl")
    scaler_file_path = os.path.join('artifacts', "scaler.pkl")

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        try:
            if isinstance(X, pd.Series):
                X = X.values
                return np.array([self._preprocess_text(text) for text in X])
        except Exception as e:
            raise CustomException(e, sys)
    
    def _preprocess_text(self, text: str) -> str:

        if not isinstance(text, str):
            return ""
        text = text.lower()
        tokens = word_tokenize(text)
        punctuation = set(string.punctuation)
        filtered_tokens = []
        for token in tokens:
            if token not in self.stop_words and token not in punctuation:
                filtered_tokens.append(token)

        stemmed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]

        return ' '.join(stemmed_tokens)
    
class NumberPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        exclamation_ratio = X['Message'].apply(lambda x: x.count('!') / len(x) if len(x) > 0 else 0)
        uppercase_ratio = X['Message'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
        digit_ratio = X['Message'].apply(lambda x: sum(1 for c in x if c.isdigit()) / len(x) if len(x) > 0 else 0)
        num_characters = X['Message'].apply(len)
        num_words = X['Message'].apply(lambda x:len(word_tokenize(text=x, preserve_line=True)))
        num_sentences = X['Message'].apply(lambda x:len(nltk.sent_tokenize(x)))
        numerical_features = np.column_stack([
                num_characters, num_words, num_sentences, 
                exclamation_ratio, uppercase_ratio, digit_ratio
            ])
            
        return numerical_features

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.tokenizer = None
        self.scaler = StandardScaler()
        self.numerical_extractor = NumberPreprocessor()
        self.label_encoder = LabelEncoder()
        self.text_preprocessor = TextPreprocessor()
    def get_data_transformer_obj(self, df: Optional[pd.DataFrame] = None):
        try:
            logging.info("Getting data transformer objects")
            self.text_preprocessor.fit(df['Message'])
            preprocessed_texts = self.text_preprocessor.transform(df['Message'])
            vocab_size = 10000

            self.tokenizer = Tokenizer(num_words=vocab_size)
            self.tokenizer.fit_on_texts(preprocessed_texts)

            self.numerical_extractor.fit(df)

            numerical_features = self.numerical_extractor.transform(df)
            self.scaler.fit(numerical_features)

            self.label_encoder.fit(df['Category'])
            return (self.text_preprocessor, self.tokenizer, self.scaler, self.label_encoder)
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path:str, test_path:str, val_path:str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            val_df = pd.read_csv(val_path)

            logging.info("Read train, test and validation data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj, tokenizer_obj, scaler_obj, label_encoder_obj = self.get_data_transformer_obj(train_df)
            X_text_train, X_numerical_train, y_train = self._transform_features(train_df)
            X_text_test, X_numerical_test, y_test = self._transform_features(test_df)
            X_text_val, X_numerical_val, y_val = self._transform_features(val_df)
            logging.info("Data transformation completed successfully")
            logging.info(f"Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            save_object(
                file_path = self.data_transformation_config.tokenizer_obj_file_path,
                obj=tokenizer_obj
            )
            save_object(
                file_path = self.data_transformation_config.label_encoder_file_path,
                obj=label_encoder_obj
            )
            save_object(
                file_path = self.data_transformation_config.scaler_file_path,
                obj=scaler_obj
            )
            return (X_text_train, X_numerical_train, y_train, 
                    X_text_test, X_numerical_test, y_test, 
                    X_text_val, X_numerical_val, y_val,
                   self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)
        
    def _transform_features(self, df: pd.DataFrame):
        try:
            max_length = 100
            preprocessed_texts = self.text_preprocessor.transform(df['Message'])

            sequences = self.tokenizer.texts_to_sequences(preprocessed_texts)
            X_text_pad = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

            X_numerical = self.numerical_extractor.transform(df)
            X_numerical = self.scaler.transform(X_numerical)

            y = self.label_encoder.transform(df['Category'])

            return X_text_pad, X_numerical, y
        except Exception as e:
            raise CustomException(e, sys)