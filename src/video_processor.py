import cv2
import numpy as np
import time
import os
import argparse
from typing import Dict, List, Any

# Import system components
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sensor_fusion import SensorFusionSystem
from collision_avoidance import CollisionAvoidanceSystem
from utils.visualization import Visualizer

class VideoProcessor:
    """Video processor for ADAS using webcam or video file"""
    
    def __init__(self, output_path: str = None, use_gpu: bool = False):
        self.output_path = output_path
        self.use_gpu = use_gpu
        
        # Create output directory
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            
        # Initialize components
        self.sensor_fusion = SensorFusionSystem()
        self.collision_avoidance = CollisionAvoidanceSystem()
        self.visualizer = Visualizer(output_path)
        
        # Initialize YOLO model
        self._initialize_model()
            
    def _initialize_model(self):
        """Initialize object detection model"""
        try:
            from ultralytics import YOLO
            
            print("Loading YOLO model...")
            self.yolo_model = YOLO('yolov8n.pt')
            
            if self.use_gpu:
                # Set device to GPU if available
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if device == 'cuda':
                    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                    print(f"Using GPU acceleration")
                else:
                    print("No GPU detected, using CPU")
                
                self.yolo_model.to(device)
                
        except ImportError:
            print("YOLO not available, using dummy detection")
            self.yolo_model = None
            
    def _dummy_detection(self, frame):
        """Create dummy detection for testing when YOLO is not available"""
        # Create dummy detection (car-like rectangle)
        height, width = frame.shape[:2]
        center_x = width // 2
        box = np.array([[center_x - 100, height - 200, center_x + 100, height - 100]])
        
        # Create object with same structure as YOLO results
        return [type('obj', (), {
            'boxes': type('boxes', (), {
                'xyxy': box,
                'cls': np.array([0]),  # class 0 = car
                'conf': np.array([0.9])
            })
        })]
            
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Dict:
        """Process a single frame"""
        # Run object detection
        if self.yolo_model:
            yolo_results = self.yolo_model(frame)
        else:
            yolo_results = self._dummy_detection(frame)
            
        # Create dummy radar and lidar data
        # In a real system, this would come from actual sensors
        radar_data = {
            'objects': [1, 2],
            'distances': [10.0, 20.0],
            'velocities': [2.0, 1.0],
            'angles': [0.1, -0.2]
        }
        
        # Create simple lidar data (could be more sophisticated in real system)
        lidar_data = np.random.rand(100, 4)  # 100 points with x,y,z,intensity
            
        # Process sensor data
        fused_data = self.sensor_fusion.process_sensor_data(yolo_results, radar_data, lidar_data)
        
        # Calculate collision risks
        risks = self.sensor_fusion.get_collision_risks(fused_data)
        
        # Determine appropriate response
        response = self.collision_avoidance.process_risks(risks)
        
        # Generate visualizations
        objects_vis = self.visualizer.draw_objects(frame, fused_data.get('objects', []))
        tracks_vis = self.visualizer.draw_tracks(objects_vis, fused_data.get('tracks', []))
        radar_vis = self.visualizer.visualize_radar(frame, radar_data)
        lidar_vis = self.visualizer.visualize_lidar_points(radar_vis, lidar_data)
        
        # Add warning overlay if necessary
        if response['warning_level'] in ['warning', 'critical']:
            tracks_vis = self.collision_avoidance.generate_visual_warning(tracks_vis, response)
            
        # Create dashboard
        dashboard = self.visualizer.create_dashboard(frame, tracks_vis, lidar_vis, response)
        
        # Save visualization if output path specified
        if self.output_path:
            self.visualizer.save_visualization(dashboard, f"frame_{frame_idx:04d}.jpg")
            
        return {
            'frame_id': frame_idx,
            'fused_data': fused_data,
            'risks': risks,
            'response': response,
            'visualizations': {
                'objects': objects_vis,
                'tracks': tracks_vis,
                'radar': radar_vis,
                'lidar': lidar_vis,
                'dashboard': dashboard
            }
        }
        
    def process_video(self, video_path: str = None, start_frame: int = 0, 
                    max_frames: int = None, display: bool = True, save_output: bool = False):
        """Process video file or webcam feed"""
        # Open video file or webcam
        if video_path:
            cap = cv2.VideoCapture(video_path)
            print(f"Processing video: {video_path}")
        else:
            cap = cv2.VideoCapture(0)  # Use default camera (webcam)
            print("Processing webcam feed")
            
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video source")
            return None
            
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if save_output and self.output_path:
            # Create video writer
            output_path = os.path.join(self.output_path, 'output_video.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width*2, frame_height*2))
        else:
            out = None
                
        # Skip to start frame
        for _ in range(start_frame):
            ret, _ = cap.read()
            if not ret:
                print(f"Error: Could not skip to frame {start_frame}")
                return None
                
        # Process frames
        frame_idx = start_frame
        results = []
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
                
            # Check if max frames reached
            if max_frames is not None and frame_idx - start_frame >= max_frames:
                print(f"Reached maximum number of frames ({max_frames})")
                break
                
            print(f"Processing frame {frame_idx}...")
            
            # Process frame
            result = self.process_frame(frame, frame_idx)
            results.append(result)
            
            # Display dashboard
            if display:
                dashboard = result['visualizations']['dashboard']
                cv2.imshow('ADAS Dashboard', dashboard)
                
                # Write frame to output video
                if out:
                    out.write(dashboard)
                    
                # Wait for key press (1ms)
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    print("User interrupted")
                    break
                    
            frame_idx += 1
            
        # Release resources
        cap.release()
        if out:
            out.write(dashboard)
            out.release()
            print(f"Output video saved to {output_path}")
            
        if display:
            cv2.destroyAllWindows()
            
        return results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ADAS Video Processor')
    parser.add_argument('--video', type=str, default=None, help='Path to video file (default: use webcam)')
    parser.add_argument('--output-path', type=str, default='../output', help='Path to output directory')
    parser.add_argument('--start-frame', type=int, default=0, help='Starting frame ID')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames to process')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--save-output', action='store_true', help='Save output video')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Create video processor
    processor = VideoProcessor(args.output_path, args.gpu)
    
    # Process video
    processor.process_video(
        args.video, 
        args.start_frame, 
        args.max_frames, 
        not args.no_display, 
        args.save_output
    )
    
if __name__ == "__main__":
    main() 