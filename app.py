from flask import Flask, request, jsonify
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import json
from json import JSONEncoder
import pandas as pd
import numpy as np
import pickle
import io

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(CustomJSONEncoder, self).default(obj)
    
app = Flask(__name__)
app.json_encoder = CustomJSONEncoder

# Load model from Azure Blob Storage
def load_model_from_blob():
  # connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
  # print(connect_str)
  connect_str = "DefaultEndpointsProtocol=https;AccountName=creditproject;AccountKey=ki155kFi7Q5RgnaOCui+rRKqKFW9D/8n9SL90GnCa9ZTNg8sVBdZC35wg0Y1CxC392oCLXkoBpRB+AStebLk7w==;EndpointSuffix=core.windows.net"
  blob_service_client = BlobServiceClient.from_connection_string(connect_str)
  blob_client = blob_service_client.get_blob_client(container="models", blob="model.xgb")
  download_stream = io.BytesIO()
  download_stream.write(blob_client.download_blob().readall())
  download_stream.seek(0)  # Rewind the buffer to the beginning
  model = pickle.load(download_stream)

  return model

def load_label_encoder():
    connect_str = "DefaultEndpointsProtocol=https;AccountName=creditproject;AccountKey=ki155kFi7Q5RgnaOCui+rRKqKFW9D/8n9SL90GnCa9ZTNg8sVBdZC35wg0Y1CxC392oCLXkoBpRB+AStebLk7w==;EndpointSuffix=core.windows.net"
    # Create a BlobServiceClient using the connection string
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a BlobClient to handle the file
    blob_client = blob_service_client.get_blob_client(container="models", blob="label_encoder.pkl")
    # Download the blob content
    download_stream = io.BytesIO()
    download_stream.write(blob_client.download_blob().readall())
    download_stream.seek(0)  # Rewind the buffer to the beginning
    # Load the object from the stream
    loaded_object = pickle.load(download_stream)

    return loaded_object

label_encoder = load_label_encoder()
model = load_model_from_blob()

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)
        df = pd.DataFrame([data])

        # use loaded label_encoder that is pre-fitted and saved during model training
        df['purpose'] = label_encoder.transform(df['purpose'])
        print("Preprocessed DataFrame:", df)

        predictions = model.predict(df)[0] # comes as list for some reason 
        print('PRINTED PREDICTIONS:', predictions)
        
        return jsonify({'prediction': predictions.tolist()})  # Convert to list for JSON serialization
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
