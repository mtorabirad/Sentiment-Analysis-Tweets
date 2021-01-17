from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from UTIL import FileIO
config = FileIO.read_yaml(r'CONFIG\config_data.yml')


def build_SVM(X_train, y_train, kernel_):
    """
    The function builds SVM
    """

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)

    """ for i in range(1, 5):
        svc = SVC(kernel=kernel_, random_state=0, gamma=i)
        model = svc.fit(X_train_std, y_train.values.ravel())
        scores = cross_val_score(estimator=model, X=X_train_std,
                                 y=y_train.values.ravel(), cv=5)
        print(i, ':', np.average(scores)) """
    svc = SVC(kernel=kernel_, random_state=0)
    model = svc.fit(X_train_std, y_train.values.ravel())
    # y_test_pred = model.predict(X_test)

    return model
