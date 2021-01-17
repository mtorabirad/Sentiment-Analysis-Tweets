import numpy as np

def get_daily_positions(y_test_pred):
    position = np.zeros(len(y_test_pred))
    position[0] = 0
    position_open_price = np.empty(len(y_test_pred))
    position_open_price[:] = np.NaN

    for i in range(1, len(y_test_pred)):
        if y_test_pred[i] == 1:
            if (position[i-1] == 1) or (position[i-1] == 0):
                position[i] = 1
        elif y_test_pred[i] == -1:
            if (position[i-1] == -1) or (position[i-1] == 0):
                position[i] = -1

    position[len(y_test_pred)-1] = 0
    return position

def get_total_return(positions, fin_data_test):
    daily_return = np.zeros(len(positions))
    for i in range(1, len(positions)):
        if (positions[i] == 0 and positions[i-1] != 0):
            # You are closing a position.
            # Find when it was opened.
            for j in range(i, -1, -1):
                if (positions[j] != 0 and positions[j-1] == 0):
                    was_opened_at = j
                    break
            # It was opened at j.
            daily_return[i] = positions[i-1]*(fin_data_test.iloc[i]['Open'] - fin_data_test.iloc[was_opened_at]['Open'])
    return sum(daily_return)