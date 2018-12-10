# Quora Duplicated Question Pair Detection


## Prerequisites

We recommend you to use python3.
```
pip install -r requirements.txt
```
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
Setup `PYTHONPATH` so that `python` is aware of the modules in the working directory.
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


## BERT pretrained models

Since I have implemented the model with Texar, and it has been merged into the main repository, it becomes very easy to run the experiments.

```
cd src/texar
pip install -e .
cd examples/bert
sh bert_pretrained_models/download_model.sh
python bert_classifier_main.py --do_train --do_eval
python bert_classifier_main.py --do_test
```
```

