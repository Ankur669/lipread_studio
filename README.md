# LipRead Studio

**AI Dataset Recording System • Video + Audio + Lip Extraction**

LipRead Studio is a desktop application for collecting an English visual-speech / lip-reading dataset. It records video and synchronized audio for a predefined list of English words, detects the speaker's face and mouth using MediaPipe FaceMesh, extracts a lip region of interest (ROI), stores metadata in CSV format, and performs dataset quality control with selective re-recording.

The application is designed for speaker-independent dataset collection and is currently structured around a target of **100 speakers × 100 English words**.

---

## Features

- 🎥 Webcam-based video recording
- 🎙️ Microphone / Bluetooth audio-device selection
- 👄 Automatic face and mouth detection with MediaPipe FaceMesh
- 🧠 Automatic lip ROI extraction
- 📹 Per-word video files
- 🔊 Per-word WAV audio files
- 🖼️ 112 × 112 lip frames
- 📄 CSV metadata
- ✅ Automatic quality-control scoring
- 🔁 Failed-word detection and selective re-recording
- ⏸️ Pause / resume session controls
- 🛑 Safe session stopping
- 🔎 Camera and microphone testing
- 📊 Live quality, FPS, word-progress and microphone telemetry
- 🌙 Dark / light display options
- 🔊 Optional Windows voice instructions
- 📝 Crash and diagnostic logging
- 🧵 Thread-safe camera, audio and MediaPipe resource handling

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.12 |
| GUI | Tkinter / ttk |
| Computer vision | OpenCV |
| Face & lip detection | MediaPipe FaceMesh |
| Numerical processing | NumPy |
| Audio recording/playback | sounddevice |
| WAV file handling | SoundFile |
| Image handling | Pillow |
| Logging | Python standard library |

### Required package versions

The project currently uses the following pinned versions:

```text
numpy==1.26.4
opencv-python==4.10.0.84
mediapipe==0.10.21
sounddevice==0.5.1
soundfile==0.12.1
Pillow==10.4.0
```

> **Important:** Use **Python 3.12** for this project. Do not change the MediaPipe version casually. The application uses `mp.solutions.face_mesh`, so the development environment is intentionally pinned to `mediapipe==0.10.21`.

---

# 1. System Requirements

## Recommended operating system

- Windows 10 or Windows 11
- Python 3.12
- Working webcam
- Working microphone or Bluetooth microphone/headset
- Sufficient free disk space for video, audio and extracted lip frames

The application contains Windows-specific functionality for voice instructions through PowerShell/System.Speech. On non-Windows systems, the voice-instruction feature is skipped.

---

# 2. Project Structure

A recommended project directory is:

```text
LipReadStudio/
│
├── lipread_studio.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── videos/
│   ├── audio/
│   ├── lips/
│   └── metadata/
│
└── logs/
    ├── lipread_app.log
    ├── crash.log
    └── fatal_native.log
```

The application automatically creates the dataset directories when it starts.

The exact Python filename can be different. Replace `lipread_studio.py` in the commands below with the actual filename of your main Python file.

---

# 3. Installation

## Step 1 — Install Python 3.12

Install Python **3.12.x**.

After installation, open Command Prompt and check:

```bash
py -3.12 --version
```

You should see something similar to:

```text
Python 3.12.x
```

If this command does not work, make sure Python 3.12 is installed and available through the Windows Python launcher.

---

## Step 2 — Open the project folder

Open Command Prompt in the project directory.

For example:

```bash
cd path\to\LipReadStudio
```

You should be able to see:

```text
requirements.txt
README.md
your_python_file.py
```

---

# 4. Recommended: Create a Virtual Environment

Using a virtual environment prevents this project from interfering with other Python projects.

Create it:

```bash
py -3.12 -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

After activation, the command prompt should show something similar to:

```text
(.venv)
```

---

# 5. Upgrade pip

Run:

```bash
python -m pip install --upgrade pip
```

---

# 6. Install Project Dependencies

Run:

```bash
python -m pip install -r requirements.txt
```

Or, without activating the virtual environment:

```bash
py -3.12 -m pip install -r requirements.txt
```

This installs:

```text
NumPy
OpenCV
MediaPipe
sounddevice
SoundFile
Pillow
```

---

# 7. Verify the Installation

Before starting the complete application, it is useful to verify the important packages.

Run:

```bash
py -3.12 -c "import cv2, numpy, mediapipe, sounddevice, soundfile; from PIL import Image; print('All dependencies imported successfully')"
```

Expected result:

```text
All dependencies imported successfully
```

You can also check the MediaPipe version:

```bash
py -3.12 -c "import mediapipe as mp; print(mp.__version__)"
```

It should report:

```text
0.10.21
```

---

# 8. Run the Application

If your main file is named:

```text
lipread_studio.py
```

run:

```bash
py -3.12 lipread_studio.py
```

If your virtual environment is activated:

```bash
python lipread_studio.py
```

---

# 9. Camera Setup

The application automatically scans for available cameras.

The default camera index is:

```text
CAMERA_INDEX = 0
```

The application probes multiple camera indexes and displays detected cameras in the **Camera** dropdown.

## Recommended setup

Before recording:

1. Connect the webcam.
2. Close applications that may be using the webcam.
3. Start LipRead Studio.
4. Allow Windows camera access if prompted.
5. Click **Refresh** under Camera.
6. Select the required camera.
7. Click **Test**.
8. Confirm that a video frame is received.

The application attempts to use:

```text
DirectShow
```

on Windows because it is generally suitable for USB webcam capture.

---

# 10. Microphone / Bluetooth Audio Setup

The application detects available input audio devices using `sounddevice`.

You can select:

- Laptop microphone
- USB microphone
- Bluetooth microphone/headset
- Other available Windows input devices

## Before recording

1. Connect the microphone or Bluetooth headset.
2. Make sure Windows recognizes it.
3. Start the application.
4. Click **Refresh** under Microphone / Bluetooth.
5. Select the desired input.
6. Click **Test**.
7. Confirm that the microphone test succeeds.

The application records:

```text
Sample rate: 44100 Hz
Channels: 1
```

---

# 11. Windows Permissions

If the camera or microphone does not work, check Windows privacy permissions.

Open:

```text
Windows Settings
→ Privacy & security
→ Camera
```

Make sure camera access is enabled.

Also check:

```text
Windows Settings
→ Privacy & security
→ Microphone
```

Make sure microphone access is enabled.

If you are using a Bluetooth device, also confirm that Windows has selected the intended input device.

---

# 12. Recording Workflow

The normal recording workflow is:

```text
Enter Speaker ID
        ↓
Select Camera
        ↓
Select Microphone
        ↓
Test Camera
        ↓
Test Microphone
        ↓
Start Session
        ↓
Face Detection
        ↓
Word Prompt
        ↓
Countdown
        ↓
Speak Word
        ↓
Record Video + Audio
        ↓
Extract Lip ROI
        ↓
Quality Control
        ↓
Next Word
        ↓
Repeat
```

---

# 13. Speaker ID

Enter a unique speaker identifier.

The default value is:

```text
S001
```

Examples:

```text
S001
S002
S003
...
S100
```

Use a consistent naming convention for all participants.

Do not use duplicate speaker IDs unless you intentionally want to overwrite/replace an existing speaker's data according to the application's recording/re-recording workflow.

---

# 14. Timing Controls

The application provides three timing controls:

### READY

Time before recording begins.

Default:

```text
3 seconds
```

### RECORD

Recording duration.

Default:

```text
3 seconds
```

### PAUSE

Pause between recordings.

Default:

```text
1 second
```

The available ranges are:

```text
READY  : 1–10 seconds
RECORD : 1–10 seconds
PAUSE  : 0–5 seconds
```

Adjust these according to your data-collection protocol.

---

# 15. Word Prompt

The speaker-facing interface displays the current word above the camera.

The application contains a predefined list of **100 English words**, including words such as:

```text
Analyse
Acquire
Adult
At least
Average
Area
Exam
Engineer
East
Earn
...
January
February
...
Today
Tomorrow
Yesterday
```

The speaker should say the word exactly as displayed.

For dataset consistency:

- Speak clearly.
- Face the camera.
- Keep your face within the tracking area.
- Avoid unnecessary head movement.
- Avoid covering your mouth.
- Maintain approximately the same distance from the camera.
- Follow the same pronunciation protocol for every recording.

---

# 16. Face and Lip Detection

The application uses:

```python
MediaPipe FaceMesh
```

with:

```text
max_num_faces = 1
refine_landmarks = True
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
```

The application detects the face and identifies a mouth region using selected MediaPipe facial landmarks.

The lip ROI is automatically expanded using configurable margins.

---

# 17. Lip ROI

The extracted lip images are resized to:

```text
112 × 112 pixels
```

The application uses the mouth landmarks:

```text
61
291
0
17
78
308
13
14
```

These landmarks are used to determine the mouth/lip bounding region.

---

# 18. Dataset Output

The application creates:

```text
dataset/
├── videos/
├── audio/
├── lips/
└── metadata/
```

### Videos

Per-word video recordings are stored under:

```text
dataset/videos/
```

### Audio

Per-word WAV recordings are stored under:

```text
dataset/audio/
```

### Lips

Extracted lip ROI frames are stored under:

```text
dataset/lips/
```

The target lip frame size is:

```text
112 × 112
```

### Metadata

CSV metadata is stored under:

```text
dataset/metadata/
```

The metadata records information associated with the collected samples.

---

# 19. Quality Control

The application contains automatic dataset quality control.

The configured quality thresholds are:

```text
85% and above       → PASS / GOOD
70% to below 85%    → REVIEW
Below 70%           → FAILED
```

The primary detection metric is lip detection.

The quality dashboard also considers:

- Face detection
- Lip detection
- Microphone activity

The live quality indicator is displayed in the application.

---

# 20. Failed-Word Re-recording

The application supports selective re-recording.

The maximum number of re-recording attempts is:

```text
2
```

Instead of recording the entire dataset again, failed words can be identified and re-recorded.

This is useful for maintaining dataset quality while reducing unnecessary recording time.

---

# 21. Pause and Stop

## Pause

Press:

```text
Ⅱ PAUSE
```

The application safely pauses the session.

If a word is currently being recorded, the application allows the current recording operation to finish safely before pausing.

Press:

```text
▶ RESUME
```

to continue.

## Stop

Press:

```text
■ STOP
```

to stop the session.

The application includes synchronization around camera and audio resources to reduce the risk of native-resource conflicts during stopping.

---

# 22. Voice Instructions

Voice instructions are optional.

The checkbox:

```text
Voice instructions
```

can be enabled or disabled.

On Windows, the application uses PowerShell/System.Speech for spoken instructions.

On non-Windows systems, the voice feature is skipped.

---

# 23. Live Dashboard

The application provides live telemetry including:

```text
QUALITY
FPS
WORDS
MIC LEVEL
SESSION PROGRESS
```

The camera preview can also display:

```text
FACE TRACK
LIP ROI
FACE LOCK / FACE SEARCH
LIPS LOCK / LIPS SEARCH
RECORDING STATUS
```

This allows the operator and speaker to monitor the recording conditions in real time.

---

# 24. Logs and Crash Diagnostics

The application contains extensive diagnostic logging.

A `logs` directory is created next to the application.

Typical files include:

```text
logs/lipread_app.log
logs/crash.log
logs/fatal_native.log
```

There may also be:

```text
lipread_crash.log
```

depending on the logging configuration in the current application version.

## If the application crashes

Check these files first.

### Python exception

Look in:

```text
logs/crash.log
```

### General application information

Look in:

```text
logs/lipread_app.log
```

### Native crash / access violation

Look in:

```text
logs/fatal_native.log
```

These logs are especially useful when troubleshooting OpenCV, MediaPipe, PortAudio, or device-related problems.

---

# 25. Common Problems

## Problem 1 — `No module named ...`

Example:

```text
ModuleNotFoundError: No module named 'cv2'
```

Run:

```bash
py -3.12 -m pip install -r requirements.txt
```

If using a virtual environment:

```bash
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

---

## Problem 2 — MediaPipe installation error

Make sure you are actually using Python 3.12:

```bash
py -3.12 --version
```

Then:

```bash
py -3.12 -m pip install --force-reinstall mediapipe==0.10.21
```

Then verify:

```bash
py -3.12 -c "import mediapipe as mp; print(mp.__version__)"
```

Expected:

```text
0.10.21
```

### Important

Do not install a random newer MediaPipe version simply because pip lists it.

This project is currently configured around:

```text
mediapipe==0.10.21
```

because the application uses:

```python
mp.solutions.face_mesh
```

---

## Problem 3 — `mp.solutions` is missing

If you see an error similar to:

```text
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

first check the installed version:

```bash
py -3.12 -c "import mediapipe as mp; print(mp.__version__)"
```

The project's expected version is:

```text
0.10.21
```

If necessary:

```bash
py -3.12 -m pip uninstall mediapipe -y
py -3.12 -m pip install mediapipe==0.10.21
```

Also make sure your project folder does not contain a file or folder named:

```text
mediapipe.py
```

or:

```text
mediapipe/
```

because that can interfere with Python imports.

---

## Problem 4 — Camera not detected

Try the following:

1. Disconnect and reconnect the webcam.
2. Close Zoom/Teams/Meet/OBS or other applications using the camera.
3. Restart the application.
4. Click **Refresh**.
5. Select another camera from the dropdown.
6. Click **Test**.
7. Check Windows camera permissions.

If necessary, check the camera index configuration:

```python
CAMERA_INDEX = 0
```

---

## Problem 5 — Microphone not detected

Try:

1. Reconnect the microphone.
2. Reconnect the Bluetooth headset.
3. Check Windows sound settings.
4. Make sure the device has an input microphone.
5. Click **Refresh**.
6. Select the device.
7. Click **Test**.

The application uses `sounddevice` to query available input devices.

---

## Problem 6 — Bluetooth microphone has poor quality

Bluetooth headsets may expose separate playback and microphone profiles in Windows.

Check the selected input device carefully.

For high-quality dataset collection, a dedicated USB microphone or wired microphone may provide more consistent audio than a Bluetooth headset.

---

## Problem 7 — Face is not detected

Try:

- Increase lighting.
- Face the camera directly.
- Move closer if the face is too small.
- Avoid extreme side angles.
- Remove objects covering the face.
- Make sure the camera preview is working.
- Check that only one primary speaker is in the camera frame.

The application is configured for:

```text
max_num_faces = 1
```

---

## Problem 8 — Lips are not detected

Make sure:

- The mouth is visible.
- The speaker is facing the camera.
- Lighting is adequate.
- The face is large enough in the frame.
- The speaker is not covering the mouth.
- The camera is focused.

The quality score depends heavily on successful lip detection.

---

## Problem 9 — Application closes unexpectedly

Do not immediately restart and ignore the problem.

First inspect:

```text
logs/
```

especially:

```text
lipread_app.log
crash.log
fatal_native.log
```

If the crash is native, also check whether it occurs consistently when:

- Starting the camera
- Stopping the camera
- Starting audio recording
- Playing tones
- Switching audio devices
- Running MediaPipe processing

The application contains locks and shutdown handling specifically to reduce unsafe concurrent access to native camera/audio/MediaPipe resources.

---

# 26. Clean Reinstallation

If the Python environment becomes corrupted, recreate it.

Delete the virtual environment:

```text
.venv/
```

Then create it again:

```bash
py -3.12 -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

# 27. Recommended Setup for Dataset Collection

For consistent research data, use the same setup for every speaker whenever possible.

Recommended:

- Same camera
- Same camera position
- Same camera resolution
- Same lighting
- Same background
- Same microphone
- Same microphone position
- Same recording distance
- Same word list
- Same timing
- Same pronunciation instructions

Avoid changing hardware or recording conditions halfway through the dataset unless the change is intentional and documented.

---

# 28. Suggested Recording Environment

For best consistency:

```text
Camera
  ↓
Speaker positioned centrally
  ↓
Face clearly visible
  ↓
Mouth unobstructed
  ↓
Even lighting
  ↓
Quiet environment
  ↓
Stable microphone
```

The speaker should remain approximately stationary and look toward the camera while speaking.

---

# 29. Dataset Naming and Organization

Use consistent speaker IDs.

Example:

```text
S001
S002
S003
...
S100
```

Keep a separate record of:

- Speaker ID
- Recording date
- Camera used
- Microphone used
- Any unusual recording conditions
- Re-recorded samples
- Quality-control issues

This information can be useful later when training and evaluating the lip-reading model.

---

# 30. Research Pipeline

The intended high-level pipeline is:

```text
Data Collection
      ↓
Video + Audio
      ↓
Face Detection
      ↓
Mouth/Lip Detection
      ↓
Lip ROI Extraction
      ↓
Quality Control
      ↓
Failed Sample Re-recording
      ↓
Clean Dataset
      ↓
Model Training
      ↓
Visual Speech Recognition
```

The project is intended to provide the data-collection foundation for a future visual speech recognition system.

The creator information currently describes the planned AI pipeline as:

```text
Video + Audio + MediaPipe + CNN/LSTM
```

---

# 31. Important Files

## `requirements.txt`

Contains the exact Python package versions required by the project.

Install with:

```bash
python -m pip install -r requirements.txt
```

## Main Python application

Contains:

- GUI
- Camera management
- Audio management
- MediaPipe processing
- Recording
- Lip extraction
- Quality control
- Dataset management
- Logging

## `README.md`

This documentation.

---

# 32. Quick Start

For someone who just received the project:

### 1. Install Python 3.12

Check:

```bash
py -3.12 --version
```

### 2. Open the project directory

```bash
cd path\to\LipReadStudio
```

### 3. Create virtual environment

```bash
py -3.12 -m venv .venv
```

### 4. Activate it

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Verify MediaPipe

```bash
python -c "import mediapipe as mp; print(mp.__version__)"
```

Expected:

```text
0.10.21
```

### 7. Run the application

```bash
python lipread_studio.py
```

### 8. Test devices

```text
Refresh Camera
→ Select Camera
→ Test Camera

Refresh Microphone
→ Select Microphone
→ Test Microphone
```

### 9. Start recording

```text
Enter Speaker ID
→ Start Session
→ Follow word prompts
→ Speak each word
→ Allow automatic lip extraction
→ Review QC results
→ Re-record failed samples if required
```

---

# 33. One-Command Installation

After Python 3.12 is installed, a friend can set up the project with:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then:

```bash
python lipread_studio.py
```

---

# 34. Development Notes

This project uses several native libraries:

- OpenCV
- MediaPipe
- PortAudio through sounddevice

For this reason, version consistency and safe resource management are important.

The application uses synchronization mechanisms around:

```text
Camera access
Audio access
MediaPipe FaceMesh processing
Session state
Detection state
Dataset state
```

Do not remove these synchronization mechanisms without testing carefully for native crashes and resource conflicts.

---

# 35. Version Information

Current application information shown in the application:

```text
LipRead Studio
Version: 4.0 • Futuristic HUD
```

Project target:

```text
100 speakers × 100 English words
```

Lip ROI:

```text
112 × 112
```

Audio:

```text
44,100 Hz
Mono
```

Default video target:

```text
1280 × 720
30 FPS
```

Quality thresholds:

```text
GOOD     ≥ 85%
REVIEW   70–84%
FAILED   < 70%
```

Maximum re-recording attempts:

```text
2
```

---

# 36. Important Reminder for Contributors

When sharing or running this project on another computer:

1. Use **Python 3.12**.
2. Install dependencies using `requirements.txt`.
3. Keep `mediapipe==0.10.21`.
4. Test the camera before recording.
5. Test the microphone before recording.
6. Do not delete the `dataset` directory while a session is running.
7. Check the `logs` directory if something goes wrong.
8. Keep the recording environment consistent for research-quality data.

---


## Project Goal

**LipRead Studio aims to provide a structured, quality-controlled system for collecting synchronized speech video, audio and lip-region data for research in visual speech recognition and lip-reading.**

