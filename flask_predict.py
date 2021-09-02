from flask import Flask, request, jsonify
import pandas as pd
import pickle as pkl

# creating app for Flask class with __name__ as parameter.
app = Flask(__name__)

#load the model.
with open('churn.pkl','rb') as churn:
    algorithm=pkl.load(churn)

# predict the response.
def predict_request(req):
    cols_to_drop=['CETEL_NUMBER','CNI_CUSTOMER','TOT_MIN_CALL_OUT', 'STATE_DATA', 'CITY_DATA', 'TEC_ANT_DATA','TOT_MIN_IN_ULT_MES', 'AVG_MIN_IN_3']
    req.drop(cols_to_drop,axis=1,inplace=True)
    response=algorithm.predict(req)
    response=int(response[0])
    print('prediction is :',response)
    return response

# create a function to predict the output using the pickle file.
@app.route('/predict',methods=['GET'])
def predict_churn():
     print('started')
     try:
        # Initially check if request has some parameters.
        # Then validate them.
        # Take only required features and test using function.
        # return the response predicted by the model along with customer id, phone number.
         req_json=request.get_json()
         req_predict = pd.DataFrame(req_json,index=[0])
         customer_id = str(req_predict['CNI_CUSTOMER'][0])
         #print(type(customer_id))
         phone_number = str(req_predict['CETEL_NUMBER'][0])
         #print(type(phone_number))
         predict=predict_request(req_predict)
         response = {'Customer':customer_id,'phone':phone_number,'Churn_Prediction':predict,'response':'OK'}
         print('return is :',response)
         #print(req_predict)

         return jsonify(response)
     except Exception as e:
         print(str(e))
         return jsonify({'response':'Some Exception Occured, Contact support team.'})

if __name__=='__main__':
    app.run()