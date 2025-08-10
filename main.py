import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

finger_tips_ids = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks, handedness_label):
    fingers = []

    # Thumb: depends on hand side
    if handedness_label == "Right":
        # Thumb tip x < IP joint x means open
        if hand_landmarks.landmark[finger_tips_ids[0]].x < hand_landmarks.landmark[finger_tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        # Left hand
        if hand_landmarks.landmark[finger_tips_ids[0]].x > hand_landmarks.landmark[finger_tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other fingers: tip y < pip y means finger open
    for tip_id in finger_tips_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            label = handedness.classification[0].label  # "Right" or "Left"
            fingers_count = count_fingers(hand_landmarks, label)

            cv2.putText(frame, f"Fingers: {fingers_count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            action = {
                0: "Volume Down",
                1: "Volume Up",
                2: "Play/Pause"
            }.get(fingers_count, "No Action")

            cv2.putText(frame, action, (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    else:
        cv2.putText(frame, "Show your hand to camera", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("MediaPipe Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
