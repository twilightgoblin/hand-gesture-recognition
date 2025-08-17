import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Improved hand detection with better confidence thresholds
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support for 2 hands
            min_detection_confidence=0.8,  # Increased from 0.7
            min_tracking_confidence=0.8,   # Increased from 0.7
            model_complexity=1  # Use more complex model for better accuracy
        )
        
        # Finger tip IDs for more accurate detection
        self.finger_tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.finger_pip_ids = [3, 6, 10, 14, 18]   # PIP joints for comparison
        
        # Gesture history for smoothing
        self.gesture_history = deque(maxlen=10)
        self.finger_count_history = deque(maxlen=15)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Gesture definitions
        self.gestures = {
            'fist': [0, 0, 0, 0, 0],
            'point': [1, 1, 0, 0, 0],
            'peace': [1, 1, 0, 0, 0],
            'three': [1, 1, 1, 0, 0],
            'four': [1, 1, 1, 1, 0],
            'five': [1, 1, 1, 1, 1],
            'thumbs_up': [1, 0, 0, 0, 0],
            'okay': [1, 1, 0, 0, 1]
        }
        
        # Action mappings
        self.actions = {
            'fist': "Stop",
            'point': "Select",
            'peace': "Next",
            'three': "Volume Up",
            'four': "Volume Down", 
            'five': "Play/Pause",
            'thumbs_up': "Like",
            'okay': "OK"
        }

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two 3D points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

    def is_finger_extended(self, hand_landmarks, tip_id, pip_id, handedness_label):
        """Improved finger extension detection using multiple criteria"""
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        
        # Get wrist position for better reference
        wrist = hand_landmarks.landmark[0]
        
        # Calculate distances
        tip_to_pip = self.calculate_distance(tip, pip)
        tip_to_wrist = self.calculate_distance(tip, wrist)
        pip_to_wrist = self.calculate_distance(pip, wrist)
        
        # For thumb, use different logic based on hand side
        if tip_id == 4:  # Thumb
            if handedness_label == "Right":
                # Right hand: thumb extended if tip is to the left of PIP
                return tip.x < pip.x
            else:
                # Left hand: thumb extended if tip is to the right of PIP
                return tip.x > pip.x
        else:
            # For other fingers, use multiple criteria
            # 1. Tip should be above PIP (y coordinate)
            # 2. Tip should be further from wrist than PIP
            # 3. Distance from tip to PIP should be reasonable
            
            y_condition = tip.y < pip.y
            distance_condition = tip_to_wrist > pip_to_wrist
            reasonable_extension = tip_to_pip > 0.02  # Minimum extension threshold
            
            return y_condition and distance_condition and reasonable_extension

    def detect_fingers(self, hand_landmarks, handedness_label):
        """Detect which fingers are extended with improved accuracy"""
        fingers = []
        
        for i, tip_id in enumerate(self.finger_tips_ids):
            pip_id = self.finger_pip_ids[i]
            is_extended = self.is_finger_extended(hand_landmarks, tip_id, pip_id, handedness_label)
            fingers.append(1 if is_extended else 0)
        
        return fingers

    def recognize_gesture(self, fingers):
        """Recognize hand gesture based on finger pattern"""
        # Convert fingers list to tuple for dictionary lookup
        finger_pattern = tuple(fingers)
        
        # Check for exact matches first
        for gesture_name, pattern in self.gestures.items():
            if tuple(pattern) == finger_pattern:
                return gesture_name
        
        # If no exact match, find closest match
        best_match = None
        best_score = 0
        
        for gesture_name, pattern in self.gestures.items():
            # Calculate similarity score
            score = sum(1 for a, b in zip(fingers, pattern) if a == b)
            if score > best_score:
                best_score = score
                best_match = gesture_name
        
        # Only return match if similarity is high enough
        if best_score >= 4:  # At least 4 out of 5 fingers match
            return best_match
        
        return "unknown"

    def smooth_gesture(self, current_gesture):
        """Smooth gesture recognition using history"""
        self.gesture_history.append(current_gesture)
        
        # Get most common gesture in recent history
        if len(self.gesture_history) >= 5:
            recent_gestures = list(self.gesture_history)[-5:]
            gesture_counts = {}
            for gesture in recent_gestures:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            
            # Return most common gesture
            return max(gesture_counts.items(), key=lambda x: x[1])[0]
        
        return current_gesture

    def smooth_finger_count(self, current_count):
        """Smooth finger count using moving average"""
        self.finger_count_history.append(current_count)
        
        if len(self.finger_count_history) >= 5:
            # Use median for more stable counting
            recent_counts = list(self.finger_count_history)[-5:]
            return int(np.median(recent_counts))
        
        return current_count

    def draw_hand_info(self, frame, hand_landmarks, handedness_label, fingers, gesture, action):
        """Draw comprehensive hand information on frame"""
        # Draw hand landmarks
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Draw finger status
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        colors = [(0, 255, 0) if f else (0, 0, 255) for f in fingers]
        
        for i, (name, color, is_extended) in enumerate(zip(finger_names, colors, fingers)):
            tip = hand_landmarks.landmark[self.finger_tips_ids[i]]
            x, y = int(tip.x * frame.shape[1]), int(tip.y * frame.shape[0])
            
            # Draw finger status
            status = "ON" if is_extended else "OFF"
            cv2.putText(frame, f"{name}: {status}", (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw circle at finger tip
            cv2.circle(frame, (x, y), 8, color, -1)
        
        # Draw hand information
        info_y = 30
        cv2.putText(frame, f"Hand: {handedness_label}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        info_y += 30
        cv2.putText(frame, f"Fingers: {sum(fingers)}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        info_y += 30
        cv2.putText(frame, f"Gesture: {gesture.upper()}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        info_y += 30
        cv2.putText(frame, f"Action: {action}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def draw_performance_info(self, frame):
        """Draw performance information on frame"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
        else:
            fps = 0
        
        # Draw FPS and performance info
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw instructions
        instructions = [
            "Press 'q' to quit",
            "Press 'r' to reset",
            "Press 'h' to toggle help"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = frame.shape[0] - 80 + i * 25
            cv2.putText(frame, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    def process_frame(self, frame):
        """Process a single frame and return processed frame"""
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb)
        
        # Reset frame info
        cv2.putText(frame, "Show your hand to camera", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                
                # Detect fingers with improved accuracy
                fingers = self.detect_fingers(hand_landmarks, label)
                
                # Smooth finger count
                smoothed_count = self.smooth_finger_count(sum(fingers))
                
                # Recognize gesture
                gesture = self.recognize_gesture(fingers)
                smoothed_gesture = self.smooth_gesture(gesture)
                
                # Get action
                action = self.actions.get(smoothed_gesture, "No Action")
                
                # Draw comprehensive hand information
                self.draw_hand_info(frame, hand_landmarks, label, fingers, smoothed_gesture, action)
        
        # Draw performance information
        self.draw_performance_info(frame)
        
        return frame

    def run(self):
        """Main loop for hand gesture recognition"""
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Hand Gesture Recognition System Started!")
        print("Press 'q' to quit, 'r' to reset, 'h' for help")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display frame
            cv2.imshow("Advanced Hand Gesture Recognition", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset gesture history
                self.gesture_history.clear()
                self.finger_count_history.clear()
                print("Gesture history reset!")
            elif key == ord('h'):
                # Toggle help display
                print("\n=== HELP ===")
                print("Supported gestures:")
                for gesture, action in self.actions.items():
                    print(f"  {gesture}: {action}")
                print("=============")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        recognizer = HandGestureRecognizer()
        recognizer.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera is working and MediaPipe is properly installed")
