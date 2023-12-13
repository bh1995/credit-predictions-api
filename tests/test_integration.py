import pytest
import requests

def test_predict_endpoint():
    url = 'http://localhost:5000/predict'
    valid_payload = {"credit.policy": 1, "purpose": "home_improvement", "int.rate": 0.1128, "installment": 164.36, "log.annual.inc": 11.2664408, "dti": 13.53, "fico": 687, "days.with.cr.line": 8189.958333, "revol.bal": 16389, "revol.util": 57.7, "inq.last.6mths": 3, "delinq.2yrs": 0, "pub.rec": 1}  # Update with valid test data
    response = requests.post(url, json=valid_payload)
    assert response.status_code == 200
    assert 'prediction' in response.json()