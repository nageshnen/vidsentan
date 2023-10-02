#
# import cv2
# from deepface import DeepFace
#
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# # Open a video capture object
# cap = cv2.VideoCapture('test_video.mp4')  # Replace 'video_path.mp4' with your video file
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break  # Break the loop if no more frames are available
#
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces
#     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
#
#     if len(faces) == 0:
#         test = 0
#         print("No Face Detected")
#     else:
#         print("Face Detected")
#         result = DeepFace.analyze(frame,actions=['emotion'],enforce_detection=False)
#         print(result)
#
#     # Display the frame
#     #cv2.imshow('Video', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()
#
# import cv2
# import frame
# from deepface import DeepFace
# from deepface.basemodels import DeepID, ArcFace, OpenFace
# from deepface.extendedmodels import Emotion
# import json
# import numpy as np
#
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# model = DeepFace.build_model('Emotion')
#
# # Define emotion labels
# emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
#
# # Start capturing video
# cap = cv2.VideoCapture('test_video.mp4')
#
# # Create a VideoWriter object to write the output video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     # Resize frame
#     resized_frame = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)
#
#     # Convert frame to grayscale
#     gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
#
#     if len(faces) == 0:
#         print("No Face Detected")
#     else:
#         # Preprocess the image for DEEPFACE
#         img = gray_frame.astype('float32') / 255.0
#         img = np.expand_dims(img, axis=-1)
#         img = np.expand_dims(img, axis=0)
#
#         # Predict emotions using DEEPFACE
#         preds = model.predict(img)
#         emotion_idx = np.argmax(preds)
#         emotion = emotion_labels[emotion_idx]
#         print(emotion)
#         for face in faces:
#             cv2.putText(frame, emotion,(face.left(), face.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#     # Write the frame to the output video
#     out.write(frame)
#
# # Release the capture and close all windows
# cap.release()
# out.release()
# cv2.destroyAllWindows()

#
# from fer import Video
# from fer import FER
#
# video_filename = "video.mp4"
# video = Video(video_filename)
#
# # Analyze video, displaying the output
# detector = FER(mtcnn=True)
# raw_data = video.analyze(detector, display=True)
# df = video.to_pandas(raw_data)
# print(df)
#
# model = DeepFace.build_model("OpenFace")
#
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# # Open a video capture object
# cap = cv2.VideoCapture('video.mp4')  # Replace 'video_path.mp4' with your video file
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break  # Break the loop if no more frames are available
#
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     new_height, new_width = 55, 47
#     resized_image = cv2.resize(frame, (new_width, new_height))
#
#     # Add a batch dimension to match the (None,) shape
#     resized_image = resized_image.reshape((1, new_height, new_width, 3))
#
#     # Now, resized_image has the shape (1, 55, 47, 3), where 1 is the batch size.
#
#     #resized_frame = cv2.resize(frame, (47, 55), interpolation = cv2.INTER_AREA)
#     # Detect faces
#     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
#
#     if len(faces) == 0:
#         print("No Face Detected")
#     else:
#         print("Face Detected")
#         result = model.predict(resized_image)
#         print(result)
#     # Display the frame
#     #cv2.imshow('Video', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()
