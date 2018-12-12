import pandas as pd
import utils
import sys

RANDOM_STATE = 0


def main(train_data_path, val_data_path, test_data_path):
    df_train = pd.read_csv(train_data_path)
    df_train = utils.feature_eng(df_train)
    train_10features_path = '../../data/train_10features.csv'
    df_train.to_csv(train_10features_path, index=False)
    print('Finish engineering 10 HCFs for training data and save in ' + \
          train_10features_path + '\n')

    df_val = pd.read_csv(val_data_path)
    df_val = utils.feature_eng(df_val)
    val_10features_path = '../../data/val_10features.csv'
    df_val.to_csv(val_10features_path, index=False)
    print('Finish engineering 10 HCFs for validation data and save in ' + \
          val_10features_path + '\n')

    df_test = pd.read_csv(test_data_path)
    df_test = utils.feature_eng(df_test)
    test_10features_path = '../../data/test_10features.csv'
    df_test.to_csv(test_10features_path, index=False)
    print('Finish engineering 10 HCFs for Kaggle testing data and save in ' + \
          test_10features_path + '\n')

if __name__ == '__main__':
    train_data_path = "../../data/qpairs_train.csv"
    val_data_path = "../../data/qpairs_val.csv"
    test_data_path = "../../data/qpairs_test.csv"
    main(train_data_path, val_data_path, test_data_path)
