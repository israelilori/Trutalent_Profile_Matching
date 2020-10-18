import pandas as pd
from flask import Flask, jsonify, request
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import datetime
import threading
from time import sleep
from preprocess import process_resume
import os

#load model
model = pickle.load(open('tfidf_vectorizer.pkl','rb'))


# app
app = Flask(__name__)

#Use a service account to connect to the database

cred = credentials.Certificate({
  "type": os.environ['type'],
  "project_id": os.environ['project_id'],
  "private_key_id": os.environ['private_key_id'],
  "private_key": os.environ['private_key'].replace('\\n', '\n'),
  "client_id": os.environ['client_id'],
  "auth_uri": os.environ['auth_uri'],
  "client_email": os.environ['client_email'],
  "token_uri": os.environ['token_uri'],
  "auth_provider_x509_cert_url": os.environ['auth_provider_x509_cert_url'],
  "client_x509_cert_url": os.environ['client_x509_cert_url']
})
firebase_admin.initialize_app(cred)
db = firestore.client()


#routes
@app.route('/')
#define function to connect to the database
def home():
    return 'Hello'


#define function to predict and return the final page after prediction occurs
@app.route('/predict/<user_id>')
def predict(user_id):

    # get data
    #data = request.get_json(force=True)

    #Get one document
    doc_ref = db.collection(u'idealCandidateProfiles').document(user_id)

    doc = doc_ref.get()
    # convert data into dataframe
    #data.update((x, [y]) for x, y in data.items())
    #data_df = pd.DataFrame.from_dict(data)
    
    #Extract values from each key
    accessValues = doc.to_dict()

    a = (accessValues['personalStatementKeywords']['keyword1'], accessValues['personalStatementKeywords']['keyword2'], 
         accessValues['personalStatementKeywords']['keyword3'], accessValues['personalStatementKeywords']['keyword4'],
         accessValues['personalStatementKeywords']['keyword5'], accessValues['name'], 
         accessValues['skillsKeywords']['keyword1'], accessValues['skillsKeywords']['keyword2'],
         accessValues['skillsKeywords']['keyword3'], accessValues['skillsKeywords']['keyword4'], 
         accessValues['skillsKeywords']['keyword5'], accessValues['experience']['experience1'], 
         accessValues['experience']['experience2'], accessValues['experience']['experience3'], 
         accessValues['experience']['experience4'], accessValues['experience']['experience5'])

    ba = [a, a]
    
    data = pre_process(ba)

    # predictions
    result = model.predict(data)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

# start the flask app, allow remote connections
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(port = port, debug=True)