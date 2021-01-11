from datetime import datetime
import pandas as pd
import sys
path = r'C:\DATA\Projects\PortFolioProjects\ClassificationProject\Code'
sys.path.insert(0, path)
from UTIL import FileIO
from PREPROCESS.preprocessing_funcs import get_df, obtain_and_append_fin_data
from NLP.perform_NLP import perform_NLP
from ML.build_models import build_SVM
from sklearn.model_selection import train_test_split

config = FileIO.read_yaml(r'CONFIG\config_data.yml')

startTime = datetime.now()

if __name__ == "__main__":
    # Get data: read in raw data, extract relevent information, and
    # calculate/append the trading dates that will be influenced by the tweet
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
    #####

    SVM = build_SVM(X_train, y_train, 'linear')
    y_test_pred = SVM.predict(X_test)

    bk_tst_start, bk_tst_end = min(X_test.index), max(X_test.index)    
    buy_hold_return = fin_data.loc[bk_tst_end]['Close'] - fin_data.loc[bk_tst_start]['Close']
    position = []
    position.append(1)
    for i in range(1, len(y_test_pred)):
        if y_test_pred[i] == 1:
            if (position[i-1] == 1) or (position[i-1] == 0):
                position.append(1)
            else:
                position.append(0)
        elif y_test_pred[i] == -1:
            if (position[i-1] == -1) or (position[i-1] == 0):
                position.append(-1)
            else:
                position.append(0)

    
print('CPU Time = ', datetime.now() - startTime)
