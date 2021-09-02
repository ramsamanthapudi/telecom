# Import Required Libraries.
import pandas as pd
from utils import drop_negative_values, outlier_treatment, clean, corr
from sklearn.model_selection import train_test_split
from Estimator import pipeline_estimate_cv,pipeline_estimate
import pickle


def train_tel():
    # Read the file.
    churn = pd.read_csv('train_churn_kg/train_churn_kg.csv')

    """
    CETEL_NUMBER : phonenumber customer
    DAYS_LIFE :  Days of life customer
    DEVICE_TECNOLOGY : 1 = AWS , 2 = No AWS, 3 = 3G Only, 4 = 4g Only, 5 sin clasificar phone device tecnology
    MIN_PLAN :  minutes include contract
    PRICE_PLAN :  money plan contracted
    TOT_MIN_CALL_OUT : count of minutes call out
    AVG_MIN_CALL_OUT_3: avg minutes call out last 3 months
    ROA_LASTMONTH : Data used in networks of other companies
    DEVICE: ( 1 Sony, 2 Motorola, 3 Huawei, 4 Apple, 5 LG 6 Nokia, 7 Samsung, 8 Alcatel, 9 Verykool, 10 Lenovo, 11 Mobiwire, 12 ZTE, 13 Sony Ericsson, 14 BlackBerry, 15 VSN Mobil, 16 VSN, 17 T610, 18 Other)
    TEC_ANT_DATA: Tecnology data antenna
    TEC_ANT_VOICE:  Tecnology voice antenna
    STATE/CITY: Tecnology antenna in state and city (State 1,2...100), (comuna 1,2,3,....320)
    CHURN:  0 = Churn Yes, 1 Churn No.
    """

    churn_updated = clean(churn)


    c_updated = drop_negative_values(churn_updated)


    c_updtd = corr(c_updated)

    c_co_updt=outlier_treatment(c_updtd)

    x=c_co_updt.drop(['CHURN'],axis=1)
    y=c_co_updt['CHURN']

    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.3,stratify=y)

    best_estimator=pipeline_estimate_cv(x_train,y_train)

    estimate,score=pipeline_estimate(x_train,y_train,best_estimator)

    with open('../churn.pkl','wb') as churn_pickle:
        pickle.dump(estimate,churn_pickle)

    print(score)


train_tel()


