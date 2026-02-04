from django.shortcuts import render
from django.http import JsonResponse
import json
import pickle
import numpy as np
import os

# --- Load Production Model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'elm_production_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    artifact = pickle.load(f)

predictor = artifact['predictor']
feature_order = artifact['feature_order']
class_labels = artifact['class_labels']
threshold = artifact['threshold']


def index_view(request):
    """
    Renders the main single-page application interface.
    """
    return render(request, 'index.html')


def predict_view(request):
    """
    Handles the API request for diabetes predictions.
    Expects a POST request with JSON data containing the 8 diabetes features.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Extract the 8 features
            pregnancies = float(data.get('pregnancies', 0))
            glucose = float(data.get('glucose', 0))
            blood_pressure = float(data.get('bloodPressure', 0))
            skin_thickness = float(data.get('skinThickness', 0))
            insulin = float(data.get('insulin', 0))
            bmi = float(data.get('bmi', 0))
            dpf = float(data.get('dpf', 0))
            age = float(data.get('age', 0))

            # Create input array in the correct feature order
            patient_data = np.array([[
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age
            ]])

            # Get prediction and probability
            prediction = predictor.predict(patient_data)[0]
            probability = predictor.predict_proba(patient_data)[0]

            # Prepare response
            response_data = {
                'prediction': int(prediction),
                'classification': class_labels[prediction],
                'probability_diabetes': round(float(probability), 4),
                'probability_no_diabetes': round(1 - float(probability), 4),
            }
            return JsonResponse(response_data)

        except (ValueError, TypeError, KeyError) as e:
            return JsonResponse({'error': f'Invalid input data: {str(e)}'}, status=400)

    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
