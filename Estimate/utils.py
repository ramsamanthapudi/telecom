import pandas as pd


def clean(ch):
    # Removing first two columns.
    churn_updated = ch.iloc[:, 2:]

    # Removing the null columns.
    churn_updated.dropna(axis=0, inplace=True)

    return churn_updated

""""
Author: Mohith and Raju
Initial Version: 31/08/2021
Description: To remove the negative values present in the columns.
"""
def drop_negative_values(churn_updated):
    ch_updated = churn_updated.drop \
                (churn_updated[(churn_updated['DAYS_LIFE'] < 0) | (churn_updated['STATE_DATA'] < 0) | \
                (churn_updated['CITY_DATA'] < 0) | (churn_updated['CITY_VOICE'] < 0)].index)
    return ch_updated

"""
Description :In Days_life we have can observe that the maximum value Is '2567' And '75' percent of data lie under 523
IQR = Q3 – Q1
where Q3: '75' percentile of data
Q1: '25' percentile of data
IQR = '523' - '179'
IQR = '344'
The data points which fall below Q1 – '1.5' IQR OR above Q3 + '1.5' IQR are outliers.
The Values above '695' are consider as outliers but Your model is already a imbalance dataset so we have assume that the values above than '900' is consider as outlier .
"""
def outlier_treatment(churn_data):
    churn_data.drop(['TOT_MIN_IN_ULT_MES', 'AVG_MIN_IN_3'], axis=1, inplace=True)
    c_o_updated = churn_data[(churn_data['DAYS_LIFE'] < 900) & (churn_data['DEVICE_TECNOLOGY'] < 5)]
    c_co_updt = c_o_updated[c_o_updated['AVG_MIN_CALL_OUT_3'] < 1000]
    return c_co_updt


def corr(c):
    cols_to_drop = ['TOT_MIN_CALL_OUT', 'STATE_DATA', 'CITY_DATA', 'TEC_ANT_DATA']

    c_updtd = c.drop(cols_to_drop, axis=1)

    return c_updtd