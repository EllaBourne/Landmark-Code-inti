import mediapipe as mp
import math
import cv2
import numpy as np
import json

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 

def to_px(lm, w , h):
    return np.array([lm.x * w, lm.y * h], dtype=float)

def angle_between(v1, v2):
    v1 = np.asarray(v1, dtype = float)
    v2 = np.asarray(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    c = np.clip(np.dot(v1,v2)/(n1 * n2), -1.0,1.0)
    return math.degrees(math.acos(c))

def angle_3pt(a,b,c):
    return angle_between(a-b,c-b)

cap = cv2.VideoCapture(0)


with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.1) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            h, w, _ = image.shape
            lms = results.pose_landmarks.landmark

            # landmark indices
            L_SH, L_EL, L_WR = 11, 13, 15   # left shoulder, elbow, wrist
            R_SH, R_EL, R_WR = 12, 14, 16   # right shoulder, elbow, wrist

            # left arm
            L_sh = to_px(lms[L_SH], w, h)
            L_el = to_px(lms[L_EL], w, h)
            L_wr = to_px(lms[L_WR], w, h)

            # right arm
            R_sh = to_px(lms[R_SH], w, h)
            R_el = to_px(lms[R_EL], w, h)
            R_wr = to_px(lms[R_WR], w, h)

            # compute angles
            L_angle = angle_3pt(L_sh, L_el, L_wr)
            R_angle = angle_3pt(R_sh, R_el, R_wr)

            if L_angle:
                cv2.putText(image, f"L: {L_angle:.1f}°", (int(L_el[0]+20), int(L_el[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            if R_angle:
                cv2.putText(image, f"R: {R_angle:.1f}°", (int(R_el[0]+20), int(R_el[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            
            # Write to JSON
            # Make sure to change the relative path to the correct position data.
            path = "./position_data.json"
            json_obj = {
                "rightShoulderPosition":{
                    "x":R_sh[0],
                    "y":R_sh[1],
                    "z":float(0)
                },
                "leftShoulderPosition":{
                    "x":L_sh[0],
                    "y":L_sh[1],
                    "z":float(0)
                },
                "rightElbowPosition":{
                    "x":R_el[0],
                    "y":R_el[1],
                    "z":float(0)
                },
                "leftElbowPosition":{
                    "x":L_el[0],
                    "y":L_el[1],
                    "z":float(0)
                },
                "rightWristPosition":{
                    "x":R_wr[0],
                    "y":R_wr[1],
                    "z":float(0)
                },
                "leftWristPosition":{
                    "x":L_wr[0],
                    "y":L_wr[1],
                    "z":float(0)
                }
            }
            json_str = json.dumps(json_obj)

            try:
                with open(path, "w") as f:
                    f.write(json_str)
            except:
                pass     

        cv2.imshow('Elbow Angle Display', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
