import base64
import io
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from django.conf import settings
from django.shortcuts import render

model = tf.keras.models.load_model(settings.MODEL_PATH)

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the input size of your model
    image = np.array(image) / 255.0   # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        image_data = data.get('image')
        if image_data:
            image_data = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            preprocessed_image = preprocess_image(image)
            predictions = model.predict(preprocessed_image)
            class_idx = np.argmax(predictions[0])

            if class_idx == 0:
                message = "The wall has a hole which can cause weaknesses. Recommended paint: XYZ."
            elif class_idx == 1:
                message = "The wall has peeled paint and can be taken care of by repainting."
            else:
                message = "The wall is healthy."
            return JsonResponse({'message': message})
        return JsonResponse({'error': 'No image data found'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def home(request):
    return render(request, 'base.html')
