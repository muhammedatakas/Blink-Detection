# Eye Blink Detection for DeepFake Detection

This repository contains a blink detection system developed for a competition that identifies key frames during eye blinks in video footage. The goal is to accurately detect three critical moments in the blinking process.

![Blink Detection Demo](visualization.gif)
## Competition Task

This project was developed for a competition where the task is to identify:
- **Start frame**: When the eyes begin to close
- **Middle frame**: When the eyes are fully closed
- **End frame**: When the eyes are fully reopened after a blink

The detection accuracy is crucial for deepfake detection applications, as unnatural blinking patterns can indicate synthetic media.

## Features

- Dual backend support: MediaPipe (recommended) or dlib
- Advanced pattern recognition for precise frame identification
- Real-time visualization of Eye Aspect Ratio (EAR)
- CSV output format compatible with competition requirements

## Requirements

- See `requirements.txt` for Python dependencies

For dlib backend, download the shape predictor file (MediaPipe recommended):
- http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

## Usage

```python
from blink_detector import BlinkDetector

# Initialize detector (MediaPipe recommended for best results)
detector = BlinkDetector(detector_type='mediapipe')

# Process a video file
results = detector.process_video(
    input_path="path/to/video.mp4",
    output_csv_path="blinks.csv",
    output_video_path="visualization.mp4"  # Optional but useful for verification
)
```

## Output Format

The detector produces two CSV files that match the competition requirements:

- `blinks.csv`: Contains `file_name`, `start_blink`, and `end_blink` columns
- `blinks_middle_frames.csv`: Contains `file_name` and `middle_frame` columns

## Methodology

The detector uses a hybrid approach combining:

1. Eye Aspect Ratio (EAR) calculation using facial landmarks
2. Pattern-based detection analyzing EAR velocity changes
3. State machine tracking for blink lifecycle management
4. Post-processing to refine blink boundary accuracy

This approach achieves high precision in identifying the exact frames where blinks begin and end.

