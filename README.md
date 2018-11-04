# Quora Duplicated Question Pair Detection
## Dev Environment Setup
`cd` into the working directory.
```
cd Duplicate_Question_Detection
```
Setup `virtualenv`.
```
python3 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt
```
Setup `PYTHONPATH` so that `python` is aware of modules in the working directory.
```
export PYTHONPATH=.
```
## Evaluation
```
python3 bin/eval.py -h
```
## Feature Extraction
```
python3 bin/extract_features.py -h
```
## Gradient Boosted Decision Tree
Training.
```
python3 bin/boosted_decision_tree/train.py -h
```
Prediction.
```
python3 bin/boosted_decision_tree/predict.py -h
```