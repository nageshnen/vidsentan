import tensorflow as tf
import mlflow
from deepface import DeepFace
from fer import Video
from fer import FER
import cv2




video_filename = "test_video.mp4"
video = Video(video_filename)

# Analyze video, displaying the output
detector = FER(mtcnn=True)
raw_data = video.analyze(detector, display=True)
df = video.to_pandas(raw_data)
print(df)

model = DeepFace.build_model("OpenFace")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a video capture object
cap = cv2.VideoCapture('test_video.mp4')# Replace 'video_path.mp4' with your video file

# Start an MLflow run
with mlflow.start_run():

        total_faces = 0
        correct_predictions = 0

        while True:
                ret, frame = cap.read()

                if not ret:
                        break
        # Break the loop if no more frames are available

                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                new_height, new_width = 96, 96
                resized_image = cv2.resize(frame, (new_width, new_height))

                # Add a batch dimension to match the (None,) shape
                resized_image = resized_image.reshape((1, new_height, new_width, 3))

                # Now, resized_image has the shape (1, 55, 47, 3), where 1 is the batch size.

                #resized_frame = cv2.resize(frame, (47, 55), interpolation = cv2.INTER_AREA)
                # Detect faces
                faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

                if len(faces) == 0:
                        print("No Face Detected")
                else:
                        print("Face Detected")
                        for face in faces:
                                total_faces += 1

                                # Predict the facial expression for the face
                                result = model.predict(resized_image)

                                # Compare the predicted facial expression to the ground truth facial expression
                                try:
                                    ground_truth_expression = df.loc[total_faces - 1, "expression"]
                                    predicted_expression = result["emotion"]

                                    # If the predicted facial expression matches the ground truth facial expression, increment the correct predictions counter
                                    if predicted_expression == ground_truth_expression:
                                        correct_predictions += 1
                                except KeyError:
                                    print("The column 'expression' does not exist in the DataFrame.")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Finish the MLflow run
mlflow.end_run()

# Calculate the accuracy
accuracy = correct_predictions / total_faces

# Log the accuracy to MLflow
mlflow.log_metric("accuracy", accuracy)
