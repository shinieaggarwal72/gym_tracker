import mediapipe as mp
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

def calc_angle(a, b, c):
    a = np.array(a) #shouler joint
    b = np.array(b) #elbow joint
    c = np.array(c)  #wrist joint
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle>180.0:
        angle = 360-angle
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
count = 0
state = None
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence= 0.5) as pose:
    while True:
        _, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        res = pose.process(img) # makes detection
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        try:
            landmarks = res.pose_landmarks.landmark
            #print(landmarks)
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle_l = calc_angle(shoulder_l, elbow_l, wrist_l)
            #print(angle)
            # shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            # elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            # wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            # angle_r = calc_angle(shoulder_r, elbow_r, wrist_r)
            cv2.putText(img, str(round(angle_l,1)), tuple(np.multiply(elbow_l, [640,480]).astype(int)), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
            #cv2.putText(img, str(round(angle_r,1)), tuple(np.multiply(elbow_r, [640,480]).astype(int)), font, 1.1, (0,255,0), 2, cv2.LINE_AA)
            if angle_l>160:
                state = 'down'
            if angle_l < 30 and state=='down':
                state = 'up' 
                count += 1
                print(count)
            txt = "CURLS : "+str(count)+"    STATE: "+state
            cv2.putText(img, txt, (10,30), font, 1.1, (0,255,0), 2, cv2.LINE_AA)

                
        except:
            pass
        mp_drawing.draw_landmarks(img,res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color = (245,66, 230), thickness=2, circle_radius=2))
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
cap.release()
cv2.destroyAllWindows()