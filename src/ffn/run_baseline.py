import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier
RANDOM_STATE = 0
TEST_SIZE = 0.1
VAL_SIZE = 0.1

def main(train_path, val_path, test_path):
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(['question1', 'question2', 'is_duplicate'], axis=1).values
    y_train = df_train['is_duplicate'].values

    X_val = df_val.drop(['question1', 'question2', 'is_duplicate'], axis=1).values
    y_val = df_val['is_duplicate'].values

    X_test = df_test.drop(['question1', 'question2', 'is_duplicate'], axis=1).values
    y_test = df_test['is_duplicate'].values

    model = RandomForestClassifier(n_estimators=500, criterion='entropy',
                                   max_depth=10, min_samples_leaf=1,
                                   max_features=0.4, n_jobs=3)
    model.fit(X_train, y_train)
    preds_val = model.predict_proba(X_val)[:, 1]
    val_log_loss_score = log_loss(y_val, preds_val)
    print('Validation log loss = {}'.format(val_log_loss_score))
    preds_val[preds_val > 0.5] = 1
    preds_val[preds_val <= 0.5] = 0
    print('y_val', y_val)
    print('pred_val', preds_val)
    print('Predicting...')
    accuracy = accuracy_score(y_val, preds_val)
    print('Validation accuracy = {}\n'.format(accuracy))

    # testing
    print('Predicting...')
    preds_test = model.predict_proba(X_test)[:, 1]

    # evaluate testing set using log loss and accuracy metrics
    test_log_loss_score = log_loss(y_test, preds_test)
    print('Testing log loss = {}'.format(test_log_loss_score))
    preds_test[preds_test > 0.5] = 1
    preds_test[preds_test <= 0.5] = 0
    accuracy = accuracy_score(y_test, preds_test)
    print('Testing accuracy = {}\n'.format(accuracy))

if __name__ == '__main__':
    train_path = '../../data/train_10features.csv'
    val_path = '../../data/val_10features.csv'
    test_path = '../../data/test_10features.csv'
    main(train_path, val_path, test_path)
