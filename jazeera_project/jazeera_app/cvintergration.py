import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('wall_inspection_model.keras')

cap = cv2.VideoCapture(0)

IMG_SIZE = (224,224)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    resized_frame = cv2.resize(frame, IMG_SIZE)

    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    prediction = model.predict(input_frame)

    class_idx = np.argmax(prediction[0])

    if class_idx == 0:
        prediction_label = 'Hole'
    
    elif class_idx == 1:
        prediction_label = 'Peeling paint'

    else:
        prediction_label = 'Normal wall'

    cv2.putText(frame, prediction_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Wall inspection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
