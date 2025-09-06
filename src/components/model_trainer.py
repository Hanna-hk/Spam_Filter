import os 
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Embedding, LSTM, Dense, Dropout, 
                                     Bidirectional, Input, Concatenate)
import pickle

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def build_model(self, hp):
        vocab_size = 10000
        max_length = 100
        num_feature_count = 6
        embedding_dim = hp.Int("embedding_dim", min_value=50, max_value=300, step=50)
        lstm_units = hp.Int("lstm_units", min_value=32, max_value=256, step=32)
        use_bidirectional = hp.Boolean("bidirectional")

        numerical_dense_units = hp.Int("num_dense_units", min_value = 8, max_value = 64, step=8)

        n_dense = hp.Int("n_dense", min_value=0, max_value=3, default=1)
        dense_units = hp.Int("dense_units", min_value = 16, max_value = 128, step=16)
        dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)

        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        optimizer_choice = hp.Choice("optimizer", values=["adam", "rmsprop", "nadam"])

        if optimizer_choice == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        
        text_input = Input(shape=(max_length,), name='text_input')
        x = Embedding(input_dim = vocab_size, output_dim=embedding_dim)(text_input)

        if use_bidirectional:
            x = Bidirectional(LSTM(lstm_units, dropout=dropout_rate))(x)
        else:
            x = LSTM(lstm_units, dropout=dropout_rate)(x)

        numerical_input = Input(shape=(num_feature_count,), name='numerical_input')
        y = Dense(numerical_dense_units, activation="relu")(numerical_input)
        y = Dropout(dropout_rate)(y)

        combined = Concatenate()([x,y])

        z = combined
        for _ in range(n_dense):
            z = Dense(dense_units, activation="relu")(z)
            z = Dropout(dropout_rate)(z)

        output = Dense(1, activation="sigmoid")(z)

        model = Model(inputs=[text_input, numerical_input], outputs=output)

        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        return model
    def initiate_model_trainer(self,X_text_train, X_numerical_train, y_train_arr, X_text_test, X_numerical_test, y_test_arr, X_text_val, X_numerical_val, y_val_arr):
        try:
            tuner = kt.RandomSearch(
                self.build_model,
                objective="val_accuracy",
                max_trials=5,
                executions_per_trial=1,
                directory="spam_tuning_multi_input",
                project_name="spam_classification",
                overwrite=True
            )

            tuner.search([X_text_train, X_numerical_train], y_train_arr, epochs=20, 
                        validation_data=([X_text_val, X_numerical_val], y_val_arr),
                        callbacks = [tf.keras.callbacks.EarlyStopping(patience=3)])
            
            best_models=tuner.get_best_models(num_models=3)
            best_model = best_models[0]

            history = best_model.fit(
                [X_text_train, X_numerical_train], 
                y_train_arr, 
                epochs=20,
                validation_data=([X_text_val, X_numerical_val], y_val_arr),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1)
                ],
                verbose=1
            )
            test_loss, test_accuracy, test_precision, test_recall = best_model.evaluate(
                [X_text_test, X_numerical_test], 
                y_test_arr,
                verbose=1
            )
            if test_accuracy<0.6:
                raise CustomException("No best model found")
            logging.info(f"Found best model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict([X_text_test, X_numerical_test])
            return test_loss, test_accuracy, test_precision, test_recall
        except Exception as e:
            raise CustomException(e, sys)
