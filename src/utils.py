import os
import sys
from xml.parsers.expat import model
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException

from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Train and evaluate multiple ML models using GridSearchCV
    """
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = params[model_name]

            gs = GridSearchCV(
                estimator=model,      # ✅ ACTUAL MODEL OBJECT
                param_grid=param_grid,
                cv=3,
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

            # store trained model back
            models[model_name] = best_model

        return report

    except Exception as e:
        raise CustomException(e, sys)
