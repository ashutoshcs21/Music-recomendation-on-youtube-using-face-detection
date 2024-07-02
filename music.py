import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
# Load model and labels
try:
    model = load_model("model.h5")  # Update with your model file name and correct format
    label = np.load("labels.npy")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize Mediapipe components
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils
try:
    emotion=np.load("emotion.npy")
except:
    emotion=""
# Define EmotionProcessor class
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")  # Corrected typo: fromat -> format
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]
            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            np.save("emotion.npy",np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Streamlit app interface
lang = st.text_input("Language")
singer = st.text_input("Singer")
if lang and singer:
    webrtc_streamer(key="example", desired_playing_state=True, video_processor_factory=EmotionProcessor)

btn = st.button("Recommend me a song")
if btn:
    if not(emotion):
        st.warning("Please let me capture emotion first")
    else:
      webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}") if len(emotion)==0 else webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion[0]}+song+{singer}")



# import streamlit as st
# from streamlit_webrtc import webrtc_streamer
# import av
# import cv2
# import numpy as np
# import mediapipe as mp
# from keras.models import load_model
# import webbrowser

# # Load model and labels
# try:
#     model = load_model("model.h5")
#     label = np.load("labels.npy")
# except Exception as e:
#     st.error(f"Error loading model: {e}")
#     st.stop()

# # Initialize Mediapipe components
# holistic = mp.solutions.holistic
# hands = mp.solutions.hands
# holis = holistic.Holistic()
# drawing = mp.solutions.drawing_utils

# # Define EmotionProcessor class
# class EmotionProcessor:
#     def recv(self, frame):
#         frm = frame.to_ndarray(format="bgr24")
#         frm = cv2.flip(frm, 1)

#         res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
#         lst = []

#         if res.face_landmarks:
#             for i in res.face_landmarks.landmark:
#                 lst.append(i.x - res.face_landmarks.landmark[1].x)
#                 lst.append(i.y - res.face_landmarks.landmark[1].y)

#             if res.left_hand_landmarks:
#                 for i in res.left_hand_landmarks.landmark:
#                     lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
#                     lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
#             else:
#                 lst.extend([0.0] * 42)

#             if res.right_hand_landmarks:
#                 for i in res.right_hand_landmarks.landmark:
#                     lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
#                     lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
#             else:
#                 lst.extend([0.0] * 42)

#             lst = np.array(lst).reshape(1, -1)

#             pred = label[np.argmax(model.predict(lst))]
#             st.write(f"Detected emotion: {pred}")
#             np.save("emotion.npy", np.array([pred]))

#         drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
#         drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
#         drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

#         return av.VideoFrame.from_ndarray(frm, format="bgr24")

# # Streamlit app interface
# lang = st.text_input("Language")
# singer = st.text_input("Singer")

# if lang and singer:
#     webrtc_streamer(
#         key="example",
#         desired_playing_state=True,
#         video_processor_factory=EmotionProcessor
#     )

# btn = st.button("Recommend me a song")
# if btn:
#     emotion = np.load("emotion.npy", allow_pickle=True)
#     if not emotion:
#         st.warning("Please let me capture emotion first")
#     else:
#         emotion = emotion.item()  # Convert numpy array back to string
#         webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
