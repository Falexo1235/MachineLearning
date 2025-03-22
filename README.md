# Classic ML with Cats

## Author
Anisimov Aleksey, 972303

## Description
This project is a solution for the Classic ML with Cats problem on Kaggle. The goal is to predict which passengers were transported to an alternate dimension during a space voyage.

## Project Structure
```
| MachineLearning/
|
|--| machinelearning/
|  |-- model.py
|  |--| data/
|     |--| model/
|
|--| dist/
|  |-- machinelearning-0.1.0-py3-none-any.whl
|
|-- Dockerfile
|-- pyproject.toml
|-- poetry.lock
|-- README.md
|
|--| notebooks/
|  |--notebook.ipynb
```

## How to Use

### Installing Dependencies

1\. Ensure you have Python 3.11 or higher installed.
2\. Install .whl file:
```
pip install machinelearning-0.1.0-py3-none-any.whl
```
### Training the Model
Open your terminal in /MachineLearning/machinelearning and run command:
```bash
python model.py train --dataset=/path/to/train.csv
```
There are additional options you can use to train your model:
### Training options
1\. `--use_gpu=0/1` can be used to determine if you want to use GPU or CPU The default is 0 (CPU).
2\. `--n_trials=10` can be used to determine if you want to do specific amount of trials. The default is 10.
### Using Docker
1\. Build the Docker image:
```bash
docker build -t machinelearning .
```
2\. Run the container for training:
```bash
docker run -v /path/to/your/data:/app/machinelearning/data machinelearning train --dataset=/app/machinelearning/data/train.csv
```
3\. Run the container for prediction:
```bash
docker run -v /path/to/your/data:/app/machinelearning/data -v /path/to/save/results:/app/machinelearning/data machinelearning predict --dataset=/app/machinelearning/data/test.csv
```
## My result progress:
|  First model | After tuning args  | After 30 train iterations  |
| ------------ | ------------ | ------------ |
|  0.79892 |  0.80126 |  0.80360 |

## Resources used:
1\. Code with comments and tutorials from: ``` https://www.kaggle.com/c/titanic/code ```
2\. Catboost hyperparameters tutorial from: ``` https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning_using_optuna_and_hyperopt.ipynb ```
