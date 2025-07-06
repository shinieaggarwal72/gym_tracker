import gradio as gr
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

count_left = 0
state_left = None

count_right = 0
state_right = None

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_frame(frame):
    global count_left, state_left, count_right, state_right

    rgb_img = frame.copy()
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    results = pose.process(rgb_img)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        angle_l = calc_angle(shoulder_l, elbow_l, wrist_l)

        cv2.putText(bgr_img, f"L: {round(angle_l, 1)}",
                    tuple(np.multiply(elbow_l, [bgr_img.shape[1], bgr_img.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if angle_l > 160:
            state_left = 'down'
        if angle_l < 30 and state_left == 'down':
            state_left = 'up'
            count_left += 1

        shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        angle_r = calc_angle(shoulder_r, elbow_r, wrist_r)

        cv2.putText(bgr_img, f"R: {round(angle_r, 1)}",
                    tuple(np.multiply(elbow_r, [bgr_img.shape[1], bgr_img.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        if angle_r > 160:
            state_right = 'down'
        if angle_r < 30 and state_right == 'down':
            state_right = 'up'
            count_right += 1

        txt = f"Curls Left: {count_right}  Right: {count_left}"
        cv2.putText(bgr_img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # mp_drawing.draw_landmarks(
        #     bgr_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        #     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        # )

    output_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return output_img

def reset_counter():
    global count_left, count_right, state_left, state_right
    count_left, count_right = 0, 0
    state_left, state_right = None, None
    return gr.update(value=None)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(sources=["webcam"], type="numpy", height=480, width=640)
        with gr.Column():
            output_img = gr.Image(label="Curl Counter", height=480, width=640)
            reset_btn = gr.Button("Reset Counter")

    input_img.stream(process_frame, inputs=[input_img], outputs=[output_img],
                     time_limit=30, stream_every=0.1, concurrency_limit=30)
    reset_btn.click(reset_counter, outputs=[output_img])

demo.launch()
