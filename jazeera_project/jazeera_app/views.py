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
import cv2

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
            if ',' in image_data:
                image_data = base64.b64decode(image_data.split(',')[1])
            else:
                image_data = base64.b64decode(image_data)

            image = Image.open(io.BytesIO(image_data))
            preprocessed_image = preprocess_image(image)
            predictions = model.predict(preprocessed_image)
            class_idx = np.argmax(predictions[0])

            if class_idx == 0:
                message = "0"
            elif class_idx == 1:
                message = "1"
            else:
                message = "2"
            return JsonResponse({'message': message})
        return JsonResponse({'error': 'No image data found'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def visualize_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        image_data = data.get('image')
        if image_data:
            # Remove the 'data:image/jpeg;base64,' part if it's present
            if ',' in image_data:
                image_data = image_data.split(',')[1]

            try:
                image_data = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_data))
                image = np.array(image)

                # Identify the defect and solve it
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Calculate average color of the image
                avg_color = cv2.mean(image)[:3]
                fill_color = [int(avg_color[0]), int(avg_color[1]), int(avg_color[2])]

                for contour in contours:
                    cv2.drawContours(image, [contour], -1, fill_color, thickness=cv2.FILLED)

                # Convert image back to base64
                image_pil = Image.fromarray(image)
                buffered = io.BytesIO()
                image_pil.save(buffered, format="JPEG")
                visualized_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                return JsonResponse({'visualized_image': visualized_image_base64})

            except Exception as e:
                return JsonResponse({'error': str(e)}, status=400)

        return JsonResponse({'error': 'No image data found'}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def home(request):
    return render(request, 'base.html')