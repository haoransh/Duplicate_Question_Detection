import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier
RANDOM_STATE = 0
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# hand-crafted features
train_hf_path = '../data/train_10features.csv'
val_hf_path = '../data/val_10features.csv'
test_hf_path = '../data/test_10features.csv'

dev_bertf_path = '../data/dev_bert_features.csv'
test_bertf_path = '../data/test_bert_features.csv'
USE_BERT_FEATURES = True

dev_mt_path = '../data/dev_matchtensor_features.csv'
test_mt_path = '../data/test_matchtensor_features.csv'
USE_MT_FEATURES = True
MODEL = 'random_forest'
#MODEl = 'lasso'
#MODEL = 'svm'

def main():
    df_train = pd.read_csv(train_hf_path)
    df_val = pd.read_csv(val_hf_path)
    df_test = pd.read_csv(test_hf_path)

    X_train = df_train.drop(['question1', 'question2', 'is_duplicate'], axis=1)
    y_train = df_train['is_duplicate'].values

    X_val = df_val.drop(['question1', 'question2', 'is_duplicate'], axis=1)
    y_val = df_val['is_duplicate'].values

    X_test = df_test.drop(['question1', 'question2', 'is_duplicate'], axis=1)
    y_test = df_test['is_duplicate']

    if USE_BERT_FEATURES:
        df_bert_dev = pd.read_csv(dev_bertf_path).drop(['bert_is_duplicate'], axis=1)
        df_bert_test = pd.read_csv(test_bertf_path).drop(['bert_is_duplicate'], axis=1)

        X_train = X_train.merge(df_bert_dev, on='id')
        X_val = X_val.merge(df_bert_dev, on='id')
        X_test = X_test.merge(df_bert_test, on='id')

    if USE_MT_FEATURES:
        df_mt_dev = pd.read_csv(dev_mt_path).drop(['bi_gru_is_duplicate'], axis=1)
        df_mt_test = pd.read_csv(test_mt_path).drop(['bi_gru_is_duplicate'], axis=1)

        X_train = X_train.merge(df_mt_dev, on='id')
        X_val = X_val.merge(df_mt_dev, on='id')
        X_test = X_test.merge(df_mt_test, on='id')

    test_ids = X_test['id'].values
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values
    print('begin building the model...')
    print('train size: {} val size:{} test size:{}'.format(
        len(X_train), len(X_val), len(X_test)))
    model = RandomForestClassifier(n_estimators=500, criterion='entropy',
                                   max_depth=10, min_samples_leaf=1,
                                   max_features=0.4, n_jobs=3)

    print('begin training...')
    model.fit(X_train, y_train)

    print('begin validation...')
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

    _dict = {
        'id': test_ids,
        'is_duplicate': preds_test
    }
    output_file = "random_forest_test_results.csv"
    pred_df = pd.DataFrame.from_dict(_dict).set_index('id')
    pred_df.to_csv(output_file, index_label='id')

if __name__ == '__main__':
    main()
