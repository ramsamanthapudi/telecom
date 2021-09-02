from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def pipeline_estimate_cv(x_train,y_train):
    pipeline_transform_estimate=Pipeline([('scaling',StandardScaler()),('estimator',XGBClassifier())])
    parameters={'estimator__n_estimators':[5,10,15]}

    #for model,parameters in params.items():
    hyperparams=GridSearchCV(pipeline_transform_estimate,parameters,cv=10,return_train_score=False,verbose=False)
    hyperparams.fit(x_train,y_train)
    return hyperparams.best_estimator_


def pipeline_estimate(x_train,y_train,best_estimator):
    pipeline_transform_estimate = best_estimator
    pipeline_transform_estimate.fit(x_train,y_train)
    score=pipeline_transform_estimate.score(x_train, y_train)
    return pipeline_transform_estimate,score

