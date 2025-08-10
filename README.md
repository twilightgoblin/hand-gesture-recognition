# Hand Gesture Recognition with MediaPipe Hands

## Overview

This project implements real-time hand gesture recognition using **MediaPipe Hands**, a powerful and efficient hand tracking solution by Google. The system detects a single hand, counts the number of extended fingers, and maps finger counts to simple actions like volume control or play/pause commands.

---

## Features

* Real-time hand detection and tracking using webcam.  
* Accurate finger counting using MediaPipe hand landmarks.  
* Distinguishes between right and left hands for improved thumb detection.  
* Simple action mapping based on finger count.  
* Visual feedback with hand landmarks and detected finger count.  
* Cross-platform Python application using OpenCV and MediaPipe.

---

## Requirements

* Python 3.7 or higher  
* OpenCV  
* MediaPipe  

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script:

```bash
python main.py
```

---

## How to Use

* Ensure your webcam is connected and accessible.  
* Run the script.  
* Place your hand clearly in front of the camera.  
* The system will detect your hand and count the number of fingers you hold up.  
* Corresponding actions will be displayed on the screen.

---

## Action Mapping (Default)

| Fingers Extended | Action      |
| ---------------- | ----------- |
| 0                | Volume Down |
| 1                | Volume Up   |
| 2                | Play/Pause  |
| Other            | No Action   |

---

## Notes

* The system is designed for a single hand at a time.  
* Good lighting and a clear background improve detection accuracy.  
* Supports both right and left hand detection.  
* Press **`q`** to quit the application.

---

## Troubleshooting

* **`ModuleNotFoundError: No module named 'mediapipe'`**  
  Run `pip install mediapipe`.

* **Performance issues**  
  Close other heavy applications, and ensure your Python version is compatible.

* **Finger counting is inaccurate**  
  Adjust your hand position or lighting conditions; finger detection depends on landmark positions.

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Acknowledgments

* [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) for hand tracking technology.  
* OpenCV for video capture and rendering.  
* Inspired by open source gesture recognition projects.
