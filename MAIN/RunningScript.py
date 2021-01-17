from datetime import datetime
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import pickle
from sklearn.metrics import confusion_matrix
path = r'C:\DATA\Projects\PortFolioProjects\ClassificationProject\Code'
sys.path.insert(0, path)
from UTIL import FileIO
from PREPROCESS.preprocessing_funcs import get_df, obtain_and_append_fin_data
from NLP.perform_NLP import perform_NLP
from ML.build_models import build_SVM
from RETURN_CALCULATIONS.funcs import get_daily_positions, get_total_return

config = FileIO.read_yaml(r'CONFIG\config_data.yml')

startTime = datetime.now()

if __name__ == "__main__":
    # Get data: read in raw data, extract relevent information, and
    # calculate/append the trading dates that will be influenced by the tweet
    if not config['splits_already_avail']:

        df = get_df()

        # Lemmatize and calculate sentiment scores
        df = perform_NLP(df)
        fin_data, training_df = obtain_and_append_fin_data(df)

        #####
        target = training_df[["buy_sell"]]
        test_size = config['test_size']

        X_train, X_test, y_train, y_test = train_test_split(
                                        training_df.drop(["buy_sell"], axis=1),
                                        target, test_size=test_size,
                                        random_state=1,
                                        shuffle=False)

        X_train.to_pickle(r'DATA\X_train.pkl')
        X_test.to_pickle(r'DATA\X_test.pkl')
        y_train.to_pickle(r'DATA\y_train.pkl')
        y_test.to_pickle(r'DATA\y_test.pkl')
        exit('Wrote split files to disk. Now exiting.')

    else:
        X_train = pd.read_pickle(r'DATA\X_train.pkl')
        X_test = pd.read_pickle(r'DATA\X_test.pkl')
        y_train = pd.read_pickle(r'DATA\y_train.pkl')
        y_test = pd.read_pickle(r'DATA\y_test.pkl')

    #####

    scaler = StandardScaler()

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.fit_transform(X_test)

    plt.scatter(X_train_std, y_train, label='train', alpha=0.5, marker="s")
    # plt.scatter(X_test_std, y_test, label='test', alpha=0.7, marker="o")
    plt.show()

    exit()
    kernel_list = ['linear', 'poly']
    for curr_kernel in kernel_list:

        SVM = build_SVM(X_train, y_train, curr_kernel)
        dump(SVM, r'SAVED_MODELS\SVM.joblib')
        SVM_upload = load('SVM.joblib')
        y_train_pred = SVM_upload.predict(X_train_std)
        y_test_pred = SVM_upload.predict(X_test_std)
        print(curr_kernel)
        print(f"train score {SVM_upload.score(X_train_std, y_train)}")
        print(f"train confusison matrix \n {confusion_matrix(y_train,
                                            y_train_pred)}")

        print(f"test score {SVM_upload.score(X_test_std, y_test)}")
        print(f"test confusison matrix \n {confusion_matrix(y_test,
                                           y_test_pred)}")
        print('----------')

    # SVM.score(X_train_std, y_train)

    fin_data_test = fin_data.tail(len(y_test_pred))

    buy_hold_positions = np.ones(len(y_test_pred))
    buy_hold_positions[-1] = 0
    buy_hold_return = get_total_return(buy_hold_positions, fin_data_test)

    sentiment_positions = get_daily_positions(y_test_pred)
    sentiment_return = get_total_return(sentiment_positions, fin_data_test)


print('CPU Time = ', datetime.now() - startTime)
