import cv2
import dlib
import mediapipe as mp
import numpy as np
import csv
import os
from scipy.spatial import distance
from collections import deque

class BlinkDetector:
    def __init__(self, detector_type='dlib'):
        self.detector_type = detector_type
        
        # MediaPipe initialization
        if detector_type == 'mediapipe':
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
            self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        else:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # Parameters adjusted for simple code behavior
        self.EAR_THRESHOLD = 0.21
        self.MIN_BLINK_FRAMES = 1  # Minimum 1 frame for very fast blinks
        self.MAX_BLINK_FRAMES = 30  # Reduced from 60 for natural blinks
        self.SMOOTH_WINDOW = 3
        
        # State management
        self.ear_history = deque(maxlen=20)  # Increased to look back further
        self.blink_state = "open"
        self.current_blink = {"start_frame": None, "end_frame": None}
        self.pending_blinks = []  # Track blinks awaiting end frame confirmation
        
        # Add slope tracking
        self.ear_buffer = deque(maxlen=6)
        self.slope_threshold = 0.01
        self.stability_threshold = 0.002
        self.pattern_window = 20
        self.potential_blink = {
            'start_frame': None,
            'min_ear': float('inf'),
            'min_frame': None
        }
        # Add pattern detection parameters
        self.ear_window = deque(maxlen=30)  # Larger window for pattern detection
        self.velocity_threshold = 0.002
        self.frame_adjustment = 2  # Frames to look back/forward
        self.local_maxima_window = 5
        # Visualization data
        self.visualization_data = {
            'frame_ears': [],
            'blinks': [],
            'video_size': (0, 0),
            'fps': 30
        }

    def _calculate_ear(self, eye_points):
        points = np.array(eye_points)
        A = np.linalg.norm(points[1] - points[5])
        B = np.linalg.norm(points[2] - points[4])
        C = np.linalg.norm(points[0] - points[3])
        return (A + B) / (2.0 * C)

    def _find_precise_frames(self, start_frame, end_frame):
        """Improved local maxima detection with adaptive windows"""
        search_window = 10  # Reduced window for quicker search
        start_window = [(d['frame_number'], d['ear']) for d in self.visualization_data['frame_ears']
                    if max(0, start_frame - search_window) <= d['frame_number'] <= start_frame + search_window]
        
        end_window = [(d['frame_number'], d['ear']) for d in self.visualization_data['frame_ears']
                    if max(0, end_frame - search_window) <= d['frame_number'] <= end_frame + search_window]
        
        # Find most recent high EAR before blink
        refined_start = start_frame
        if start_window:
            start_peaks = [frame for frame, ear in start_window 
                        if ear == max(ear for (_, ear) in start_window)]
            if start_peaks:
                refined_start = min(start_peaks)
        
        # Find highest EAR after blink
        refined_end = end_frame
        if end_window:
            end_peaks = [frame for frame, ear in end_window 
                    if ear == max(ear for (_, ear) in end_window)]
            if end_peaks:
                refined_end = max(end_peaks)  # Take latest peak
        
        return refined_start, refined_end

    def _detect_blink_frames(self, current_frame, ear):
        """Enhanced blink detection with pattern validation"""
        self.ear_buffer.append((current_frame, ear))
        self.ear_window.append((current_frame, ear))
        
        if len(self.ear_window) < self.pattern_window:
            return
        
        velocities = [self.ear_window[i+1][1] - self.ear_window[i][1] 
                    for i in range(len(self.ear_window)-1)]
        
        if not self.potential_blink['start_frame']:
            if all(v < -self.velocity_threshold for v in velocities[-3:]):
                self.potential_blink['start_frame'] = current_frame - 3
                self.potential_blink['min_ear'] = float('inf')
        
        if self.potential_blink['start_frame']:
            if ear < self.potential_blink['min_ear']:
                self.potential_blink['min_ear'] = ear
                self.potential_blink['min_frame'] = current_frame
            
            if (len(velocities) >= 3 and
                all(v > self.velocity_threshold for v in velocities[-3:]) and
                abs(velocities[-1]) < self.stability_threshold):
                
                start_frame, end_frame = self._find_precise_frames(
                    self.potential_blink['start_frame'],
                    current_frame
                )
                
                duration = end_frame - start_frame
                if self.MIN_BLINK_FRAMES <= duration <= self.MAX_BLINK_FRAMES:
                    self.visualization_data['blinks'].append((start_frame, end_frame))
                
                self.potential_blink['start_frame'] = None

    def _update_blink_state(self, current_frame, ear):
        """Improved state management with deeper historical analysis"""
        if ear is None:
            return

        self.ear_history.append((current_frame, ear))
        
        # Check pending blinks and update their max_ear
        to_remove = []
        for idx, pending in enumerate(self.pending_blinks):
            if ear > pending['max_ear']:
                pending['max_ear'] = ear
                pending['max_ear_frame'] = current_frame
            pending['frames_since_end'] += 1

            # Finalize after 5 frames
            if pending['frames_since_end'] >= 5:
                start = pending['start_frame']
                end = pending['max_ear_frame']
                duration = end - start
                if self.MIN_BLINK_FRAMES <= duration <= self.MAX_BLINK_FRAMES:
                    # Check for overlaps
                    overlap = False
                    for s, e in self.visualization_data['blinks']:
                        if not (end < s or start > e):
                            overlap = True
                            break
                    if not overlap:
                        self.visualization_data['blinks'].append((start, end))
                to_remove.append(idx)
        
        # Remove processed pending blinks
        for idx in reversed(to_remove):
            del self.pending_blinks[idx]

        # State transitions
        if ear < self.EAR_THRESHOLD and self.blink_state == "open":
            # Find last peak before drop
            max_ear = -float('inf')
            start_candidate = current_frame
            for hist_frame, hist_ear in reversed(self.ear_history):
                if hist_ear > max_ear:
                    max_ear = hist_ear
                    start_candidate = hist_frame + 1  # Start after peak
                else:
                    break  # EAR started decreasing
            self.current_blink["start_frame"] = start_candidate
            self.blink_state = "closed"
            
        elif ear >= self.EAR_THRESHOLD and self.blink_state == "closed":
            # Find first frame above threshold in buffer
            end_candidate = current_frame
            for (future_frame, future_ear) in self.ear_buffer:
                if future_ear >= self.EAR_THRESHOLD:
                    end_candidate = future_frame
                    break
            # Add to pending blinks for end refinement
            self.pending_blinks.append({
                'start_frame': self.current_blink["start_frame"],
                'tentative_end': end_candidate,
                'max_ear': ear,
                'max_ear_frame': end_candidate,
                'frames_since_end': 0
            })
            self.blink_state = "open"
            self.current_blink = {"start_frame": None, "end_frame": None}

    def _process_frame_mediapipe(self, frame, frame_number):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        ear = None
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            landmarks = np.array([[int(lm.x * w), int(lm.y * h)] 
                                for lm in face_landmarks.landmark])
            
            left_ear = self._calculate_ear(landmarks[self.LEFT_EYE])
            right_ear = self._calculate_ear(landmarks[self.RIGHT_EYE])
            ear = (left_ear + right_ear) / 2.0
            
            # Add pattern-based detection
            self._detect_blink_frames(frame_number, ear)
            # Keep state-based detection as fallback
            self._update_blink_state(frame_number, ear)
            
            self.visualization_data['frame_ears'].append({
                'frame_number': frame_number,
                'ear': ear,
                'smoothed_ear': ear,
                'left_ear': left_ear,
                'right_ear': right_ear
            })
        
        return frame, ear

    def _create_visualization(self, frame, frame_data):
        """Create visualization overlay for frame"""
        if frame_data['ear'] is not None:
            y_pos = 30
            # Draw EAR values
            for label, value in [('Left EAR', frame_data['left_ear']),
                                ('Right EAR', frame_data['right_ear']),
                                ('Avg EAR', frame_data['ear'])]:
                cv2.putText(frame, f"{label}: {value:.2f}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                y_pos += 30
            
            # Draw blink state and count
            cv2.putText(frame, f"State: {self.blink_state}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            y_pos += 30
            
            cv2.putText(frame, f"Blinks: {len(self.visualization_data['blinks'])}", 
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            y_pos += 30
            
            cv2.putText(frame, f"Frame: {frame_data['frame_number']}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # Draw visualization elements
        self._draw_ear_graph(frame, frame_data['frame_number'])
        self._draw_blink_intervals(frame, frame_data['frame_number'])
        return frame

    def _draw_ear_graph(self, frame, current_frame):
        """Draw scrolling EAR history graph"""
        graph_height = 150
        graph_width = frame.shape[1] - 20
        margin = 10
        
        # Create graph canvas
        graph = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
        
        # Plot settings
        max_ear = 0.4
        frames_to_show = 100
        start_idx = max(0, len(self.visualization_data['frame_ears']) - frames_to_show)
        
        # Plot EAR values
        for i in range(start_idx, len(self.visualization_data['frame_ears'])):
            data = self.visualization_data['frame_ears'][i]
            if data['ear'] is None:
                continue
                
            x = int((i - start_idx) * (graph_width / frames_to_show))
            y = int(graph_height * (1 - (data['ear'] / max_ear)))
            color = (0,255,0) if data['ear'] > self.EAR_THRESHOLD else (0,0,255)
            cv2.circle(graph, (x, y), 2, color, -1)
        
        # Draw threshold line
        thresh_y = int(graph_height * (1 - (self.EAR_THRESHOLD / max_ear)))
        cv2.line(graph, (0, thresh_y), (graph_width, thresh_y), (255,255,255), 1)
        
        # Overlay graph on frame
        frame[-graph_height-margin:-margin, margin:margin+graph_width] = graph

    def _draw_blink_intervals(self, frame, current_frame):
        """Draw detected blink intervals on timeline"""
        timeline_height = 20
        timeline_y = frame.shape[0] - timeline_height - 10
        cv2.rectangle(frame, (0, timeline_y), (frame.shape[1], timeline_y+timeline_height), 
                    (50,50,50), -1)
        
        total_frames = len(self.visualization_data['frame_ears'])
        if total_frames == 0:
            return
        
        # Draw blink intervals
        for start, end in self.visualization_data['blinks']:
            x_start = int((start / total_frames) * frame.shape[1])
            x_end = int((end / total_frames) * frame.shape[1])
            cv2.rectangle(frame, (x_start, timeline_y), (x_end, timeline_y+timeline_height), 
                        (0,255,0), -1)
        
        # Current frame indicator
        x_current = int((current_frame / total_frames) * frame.shape[1])
        cv2.line(frame, (x_current, timeline_y), (x_current, timeline_y+timeline_height), 
                (255,0,0), 2)

    def _save_middle_frames(self, output_path, input_path):
        """Save middle frames of blinks to CSV"""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'middle_frame'])
            
            for start, end in self.visualization_data['blinks']:
                middle_frame = start + (end - start) // 2
                writer.writerow([
                    os.path.basename(input_path),
                    middle_frame
                ])

    def process_video(self, input_path, output_csv_path, output_video_path=None):
        cap = cv2.VideoCapture(input_path)
        frame_number = 0
        
        self.visualization_data['video_size'] = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        self.visualization_data['fps'] = cap.get(cv2.CAP_PROP_FPS)
        
        writer = None
        if output_video_path:
            # Use H.264 codec for WhatsApp compatibility
            if os.name == 'nt':  # Windows
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            else:
                fourcc = cv2.VideoWriter_fourcc(*'X264')  # H.264 codec
                
            # Ensure dimensions are even (required for H.264)
            width = self.visualization_data['video_size'][0]
            height = self.visualization_data['video_size'][1]
            if width % 2:
                width -= 1
            if height % 2:
                height -= 1
                
            writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                min(30.0, self.visualization_data['fps']),  # Cap at 30fps for WhatsApp
                (width, height),
                isColor=True
            )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, ear = self._process_frame_mediapipe(frame, frame_number)
            
            if writer and ear is not None:
                frame_data = self.visualization_data['frame_ears'][-1]
                viz_frame = self._create_visualization(processed_frame, frame_data)
                writer.write(viz_frame)
            
            frame_number += 1
        
        cap.release()
        if writer:
            writer.release()
        
        # Split output path to create middle frames path
        path_without_ext = os.path.splitext(output_csv_path)[0]
        middle_frames_path = f"{path_without_ext}_middle_frames.csv"
        
        self._save_results(output_csv_path, input_path)
        self._save_middle_frames(middle_frames_path, input_path)
        return self.visualization_data

    def _save_results(self, output_path, input_path):
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'start_blink', 'end_blink'])
            for start, end in self.visualization_data['blinks']:
                writer.writerow([
                    os.path.basename(input_path),
                    start,
                    end
                ])


if __name__ == "__main__":
    detector = BlinkDetector(detector_type='mediapipe')  # Changed to mediapipe as it's more robust
    results = detector.process_video(
        input_path="path/to/your/video.mp4",  # Update with your video path
        output_csv_path="blinks.csv",
        output_video_path="blinks_visualization.mp4"
    )