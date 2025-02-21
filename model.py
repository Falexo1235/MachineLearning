import numpy as np
import pandas as pd
import os
import logging
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import optuna

# Настройка логирования
logging.basicConfig(filename='./data/output/log_file.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка данных
train = pd.read_csv('./data/input/train.csv')
test = pd.read_csv('./data/input/test.csv')
datasets =[train,test]

logging.info("Начало предобработки данных")
for dataset in datasets:
    dataset['Group'] = dataset['PassengerId'].apply(lambda x: x.split('_')[0])
    dataset['NumberInGroup'] = dataset['PassengerId'].apply(lambda x: x.split('_')[1])
        
    dataset['Deck'] = dataset['Cabin'].apply(lambda x: x.split('/')[0] if pd.notnull(x) else 'Unknown')
    dataset['CabinNumber'] = dataset['Cabin'].apply(lambda x: x.split('/')[1] if pd.notnull(x) else 'Unknown')
    dataset['Side'] = dataset['Cabin'].apply(lambda x: x.split('/')[2] if pd.notnull(x) else 'Unknown')
        
    numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in numerical_columns:
        dataset[col].fillna(dataset[col].mean(), inplace=True)
        
    categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
    for col in categorical_columns:
        dataset[col].fillna('Unknown', inplace=True)
        
    dataset.drop(['Name', 'Cabin'], axis=1, inplace=True)
        
    dataset = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True)
logging.info("Предобработка данных завершена")

nulls = train.isnull().sum(axis=0)
print('Train:', nulls[nulls > 0])
nulls = test.isnull().sum(axis=0)
print('Test:', nulls[nulls > 0])

categorical_features_indices = [
    'HomePlanet', 
    'CryoSleep', 
    'Destination', 
    'VIP', 
    'Deck', 
    'CabinNumber',
    'Side', 
    'Group', 
    'Prefix'
]

def objective(trial):
    X=train.drop('Transported', axis=1)
    y=train['Transported']
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)
    X_test=test

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

    cat_cls.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], cat_features=categorical_features_indices,verbose=0, early_stopping_rounds=100)

    preds = cat_cls.predict(X_validation)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(y_validation, pred_labels)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=360)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

accuracy =[]
model_names =[]


X=train.drop('Transported', axis=1)
y=train['Transported']
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75, random_state=42)
X_test=test

model = CatBoostClassifier(verbose=False,random_state=42,
    objective= 'CrossEntropy',
    colsample_bylevel= 0.06695759969519763,
    depth= 12,
    boosting_type= 'Ordered',
    bootstrap_type= 'MVS')

model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation))
y_pred = model.predict(X_validation)
accuracy.append(round(accuracy_score(y_validation, y_pred),4))
print(classification_report(y_validation, y_pred))

model_names = ['Catboost_tuned']
result_df6 = pd.DataFrame({'Accuracy':accuracy}, index=model_names)
result_df6

submission = pd.DataFrame()
submission['PassengerId'] = X_test['PassengerId']
submission['Transported'] = model.predict(X_test)
submission.to_csv('submission.csv', index=False)
