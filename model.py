import numpy as np
import pandas as pd
import os
import logging
import shutil
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import optuna
import fire

logging.basicConfig(
    filename='./data/output/log_file.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

os.makedirs('./data/output', exist_ok=True)
os.makedirs('./model', exist_ok=True)

class SpaceshipTitanicModel:
    def __init__(self):
        self.categorical_features_indices = [
            'HomePlanet', 
            'CryoSleep', 
            'Destination', 
            'VIP', 
            'Deck', 
            'CabinNumber',
            'Side', 
            'Group'
        ]
        
        self.model_path = './model/catboost_model.cbm'
    
    def _preprocess_data(self, dataset):
        logging.info("Начало предобработки данных")
        
        df = dataset.copy()
        
        df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
        df['NumberInGroup'] = df['PassengerId'].apply(lambda x: x.split('_')[1])
            
        df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notnull(x) else 'Unknown')
        df['CabinNumber'] = df['Cabin'].apply(lambda x: x.split('/')[1] if pd.notnull(x) else 'Unknown')
        df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notnull(x) else 'Unknown')
            
        numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for col in numerical_columns:
            df[col].fillna(df[col].mean(), inplace=True)
            
        categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
        for col in categorical_columns:
            df[col].fillna('Unknown', inplace=True)
            
        df.drop(['Name', 'Cabin'], axis=1, inplace=True)
        
        logging.info("Предобработка данных завершена")
        return df
    
    def _optimize_hyperparameters(self, X, y, n_trials=60, timeout=360):
        def objective(trial):
            X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)

            param = {
                "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                "depth": trial.suggest_int("depth", 1, 12),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                )
            }

            if param["bootstrap_type"] == "Bayesian":
                param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif param["bootstrap_type"] == "Bernoulli":
                param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

            cat_cls = CatBoostClassifier(**param)
            cat_cls.fit(
                X_train, 
                y_train, 
                eval_set=[(X_validation, y_validation)], 
                cat_features=self.categorical_features_indices,
                verbose=0, 
                early_stopping_rounds=100
            )

            preds = cat_cls.predict(X_validation)
            pred_labels = np.rint(preds)
            accuracy = accuracy_score(y_validation, pred_labels)
            return accuracy

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            
        return trial.params
    
    def train(self, dataset):
        logging.info(f"Начало обучения модели с датасетом: {dataset}")
        
        train_data = pd.read_csv(dataset)
        train_data = self._preprocess_data(train_data)
        
        nulls = train_data.isnull().sum(axis=0)
        print('Train nulls:', nulls[nulls > 0])
        
        X = train_data.drop(['PassengerId', 'Transported'], axis=1)
        y = train_data['Transported']
        
        logging.info("Начало оптимизации гиперпараметров")
        best_params = self._optimize_hyperparameters(X, y)
        logging.info("Оптимизация гиперпараметров завершена")
        
        logging.info("Обучение финальной модели")
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)
        
        model = CatBoostClassifier(
            verbose=False,
            random_state=42,
            **best_params
        )
        
        model.fit(
            X_train, 
            y_train,
            cat_features=self.categorical_features_indices,
            eval_set=(X_validation, y_validation)
        )
        
        y_pred = model.predict(X_validation)
        accuracy = round(accuracy_score(y_validation, y_pred), 4)
        logging.info(f"Точность модели на валидационной выборке: {accuracy}")
        print(classification_report(y_validation, y_pred))
        
        model.save_model(self.model_path)
        logging.info(f"Модель сохранена в файл {self.model_path}")
        
        return f"Модель успешно обучена и сохранена. Точность: {accuracy}"
    
    def predict(self, dataset):
        logging.info(f"Начало предсказания с датасетом: {dataset}")
        
        if not os.path.exists(self.model_path):
            error_msg = f"Модель не найдена по пути {self.model_path}. Сначала выполните обучение."
            logging.error(error_msg)
            return error_msg
        
        test_data = pd.read_csv(dataset)
        test_data = self._preprocess_data(test_data)
        
        nulls = test_data.isnull().sum(axis=0)
        print('Test nulls:', nulls[nulls > 0])
        
        model = CatBoostClassifier()
        model.load_model(self.model_path)
        logging.info(f"Модель загружена из файла {self.model_path}")
        
        passenger_ids = test_data['PassengerId']
        X_test = test_data.drop(['PassengerId'], axis=1)
        
        predictions = model.predict(X_test)
        
        submission = pd.DataFrame()
        submission['PassengerId'] = passenger_ids
        submission['Transported'] = [bool(pred) for pred in predictions]
        
        output_path = './data/output/submission.csv'
        submission.to_csv(output_path, index=False)
        logging.info(f"Файл предсказаний сохранён: {output_path}")
        
        return f"Предсказания выполнены и сохранены в {output_path}"

if __name__ == "__main__":
    fire.Fire(SpaceshipTitanicModel)

