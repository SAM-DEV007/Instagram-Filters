# Hand Gesture
  It is an attempt to replicate the hand gesture filter of Instagram.

# Installation
  The required libraries are mentioned in `requirements.txt` file. It can be installed via `pip install -r \path\to\requirements.txt`.

# About
  The project relies heavily on webcam or at least any camera connected to the computer. The program will try to capture the hand gesture
  and identify it as one of the sign displayed on the screen. It uses a [custom trained model](Model/Model_Data). The [training
  files](Model) are mentioned, as well as the [information](Model/README.md) on the model. After detecting
  the gesture, it will take a snapshot and will show it. The video is recorded and saved under `Captured Video` folder (created after first launch).
  
  The window can be closed via Esc, Q or the Close button.
