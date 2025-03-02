import numpy as np
import pandas as pd
import os
import logging
import shutil
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import optuna
import fire
from catboost.utils import get_gpu_device_count


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

    def _check_gpu_support(self):
        """Проверяет доступность GPU через CatBoost."""
        logging.info("Проверка поддержки GPU...")
        try:
            from catboost.utils import get_gpu_device_count
            if (get_gpu_device_count() == 0):
                logging.info("GPU не поддерживается, --use_gpu=False")
            return get_gpu_device_count() > 0
        except:
            logging.info("GPU не поддерживается, --use_gpu=False")
            return False

    def _preprocess_data(self, dataset):
        logging.info("Начало предобработки данных")
        
        df = dataset.copy()
        
        df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
        df['NumberInGroup'] = df['PassengerId'].apply(lambda x: x.split('_')[1])
            
        df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notnull(x) else 'Unknown')
        df['CabinNumber'] = df['Cabin'].apply(lambda x: x.split('/')[1] if pd.notnull(x) else 'Unknown')
        df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notnull(x) else 'Unknown')
            
        numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
            
        categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
        df[categorical_columns] = df[categorical_columns].fillna('Unknown')
            
        df.drop(['Name', 'Cabin'], axis=1, inplace=True)
        
        logging.info("Предобработка данных завершена")
        return df
    
    def _optimize_hyperparameters(self, X, y, use_gpu, n_trials=10, timeout=3600):
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
                'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 0, 8)
            }


            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                train_data = Pool(data=X_train, label=y_train, cat_features=self.categorical_features_indices)
                val_data = Pool(data=X_val, label=y_val, cat_features=self.categorical_features_indices)

                model = CatBoostClassifier(**params, verbose=0)
                model.fit(train_data, eval_set=val_data, early_stopping_rounds=100, verbose=0)

                preds = model.predict_proba(X_val)[:, 1]
                cv_scores.append(roc_auc_score(y_val, preds))

            return np.mean(cv_scores)

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
    
    def train(self, dataset, use_gpu=False):
        if use_gpu:
            use_gpu = self._check_gpu_support()
        logging.info(f"Начало обучения модели с датасетом: {dataset}")
        logging.info(f"Использование GPU: {use_gpu}")
        
        train_data = pd.read_csv(dataset)
        train_data = self._preprocess_data(train_data)
        
        nulls = train_data.isnull().sum(axis=0)
        print('Train nulls:', nulls[nulls > 0])
        
        X = train_data.drop(['PassengerId', 'Transported'], axis=1)
        y = train_data['Transported']
        
        logging.info("Начало оптимизации гиперпараметров")
        best_params = self._optimize_hyperparameters(X, y, use_gpu)
        logging.info("Оптимизация гиперпараметров завершена")
        
        logging.info("Обучение финальной модели")
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        train_pool = Pool(data=X_train, label=y_train, cat_features=self.categorical_features_indices)
        val_pool = Pool(data=X_validation, label=y_validation, cat_features=self.categorical_features_indices)
        
        model = CatBoostClassifier(
            iterations=10000,
            early_stopping_rounds=100,
            verbose=100,
            task_type="GPU" if use_gpu else "CPU",
            devices='0' if use_gpu else None,
            **best_params
        )
        
        model.fit(train_pool, eval_set=val_pool)
        
        y_pred = model.predict(X_validation)
        y_pred_proba = model.predict_proba(X_validation)[:, 1]
        accuracy = round(accuracy_score(y_validation, y_pred), 4)
        roc_auc = round(roc_auc_score(y_validation, y_pred_proba), 4)
        logging.info(f"Точность модели на валидационной выборке: {accuracy}")
        logging.info(f"ROC AUC модели на валидационной выборке: {roc_auc}")
        print(classification_report(y_validation, y_pred))
        
        model.save_model(self.model_path)
        logging.info(f"Модель сохранена в файл {self.model_path}")
        
        return f"Модель успешно обучена и сохранена. Точность: {accuracy}, ROC AUC: {roc_auc}"
    
    def predict(self, dataset, use_gpu=False):
        logging.info(f"Начало предсказания с датасетом: {dataset}")
        logging.info(f"Использование GPU: {use_gpu}")
        
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
        submission['Transported'] = predictions
        
        output_path = './data/output/submission.csv'
        submission.to_csv(output_path, index=False)
        logging.info(f"Файл предсказаний сохранён: {output_path}")
        
        return f"Предсказания выполнены и сохранены в {output_path}"

if __name__ == "__main__":
    fire.Fire(SpaceshipTitanicModel)

