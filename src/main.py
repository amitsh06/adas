import os
import cv2
import numpy as np
import time
import argparse
from typing import Dict, List, Any

# Import system components
from data_loader import DatasetHandler
from sensor_fusion import SensorFusionSystem
from collision_avoidance import CollisionAvoidanceSystem
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utilities
from utils.visualization import Visualizer

class ADASSystem:
    """Main ADAS system integrating all components"""
    
    def __init__(self, data_path: str, output_path: str = None, use_gpu: bool = False):
        self.data_path = data_path
        self.output_path = output_path
        self.use_gpu = use_gpu
        
        # Initialize components
        self.dataset_handler = DatasetHandler(data_path)
        self.sensor_fusion = SensorFusionSystem()
        self.collision_avoidance = CollisionAvoidanceSystem()
        self.visualizer = Visualizer(output_path)
        
        # Create output directory
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            
        # Flag to indicate if the system has been initialized
        self.initialized = False
            
    def initialize(self, dataset_name: str = 'kitti'):
        """Initialize the system and download dataset"""
        print(f"Initializing ADAS system with {dataset_name} dataset...")
        self.dataset_handler.download_dataset(dataset_name)
        
        # Initialize YOLO model with GPU if available
        if self.use_gpu:
            try:
                import torch
                from ultralytics import YOLO
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if device == 'cuda':
                    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                    print(f"CUDA Version: {torch.version.cuda}")
                    print(f"Using GPU acceleration")
                else:
                    print("No GPU detected, using CPU")
                
                self.yolo_model = YOLO('yolov8n.pt')
                self.yolo_model.to(device)
            except ImportError:
                print("PyTorch or YOLO not available, GPU acceleration disabled")
        
        self.initialized = True
        
    def process_frame(self, frame_id: int) -> Dict:
        """Process a single frame"""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        # Get sensor data
        frame_data = self.dataset_handler.get_dataset_frame(frame_id)
        if frame_data is None:
            return None
            
        # Get individual sensor data
        camera_frame = frame_data['camera']
        radar_data = frame_data['radar']
        lidar_data = frame_data['lidar']
        
        # YOLO object detection on camera frame
        try:
            from ultralytics import YOLO
            
            if not hasattr(self, 'yolo_model'):
                print("Loading YOLO model...")
                self.yolo_model = YOLO('yolov8n.pt')
                if self.use_gpu:
                    # Set device to GPU if available
                    import torch
                    self.yolo_model.to('cuda' if torch.cuda.is_available() else 'cpu')
                    
            # Run object detection
            yolo_results = self.yolo_model(camera_frame)
            
        except ImportError:
            print("YOLO not available, using dummy detection")
            # Create dummy detection for testing
            yolo_results = [type('obj', (), {
                'boxes': type('boxes', (), {
                    'xyxy': np.array([[100, 300, 300, 400], [400, 350, 430, 450]]),
                    'cls': np.array([0, 1]),
                    'conf': np.array([0.9, 0.85])
                })
            })]
        
        # Process sensor data
        fused_data = self.sensor_fusion.process_sensor_data(yolo_results, radar_data, lidar_data)
        
        # Calculate collision risks
        risks = self.sensor_fusion.get_collision_risks(fused_data)
        
        # Determine appropriate response
        response = self.collision_avoidance.process_risks(risks)
        
        # Generate visualizations
        objects_vis = self.visualizer.draw_objects(camera_frame, fused_data['objects'])
        tracks_vis = self.visualizer.draw_tracks(objects_vis, fused_data['tracks'])
        radar_vis = self.visualizer.visualize_radar(camera_frame, radar_data)
        lidar_vis = self.visualizer.visualize_lidar_points(radar_vis, lidar_data)
        
        # Add warning overlay if necessary
        if response['warning_level'] in ['warning', 'critical']:
            tracks_vis = self.collision_avoidance.generate_visual_warning(tracks_vis, response)
            
        # Create dashboard
        dashboard = self.visualizer.create_dashboard(camera_frame, tracks_vis, lidar_vis, response)
        
        # Save visualizations if output path specified
        if self.output_path:
            self.visualizer.save_visualization(dashboard, f"frame_{frame_id:04d}.jpg")
            
        return {
            'frame_id': frame_id,
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
        
    def run_simulation(self, start_frame: int = 0, num_frames: int = 10, display: bool = True):
        """Run the ADAS simulation on multiple frames"""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        results = []
        
        for frame_id in range(start_frame, start_frame + num_frames):
            print(f"Processing frame {frame_id}...")
            frame_result = self.process_frame(frame_id)
            
            if frame_result is None:
                print(f"Error processing frame {frame_id}, skipping.")
                continue
                
            results.append(frame_result)
            
            # Display dashboard
            if display:
                dashboard = frame_result['visualizations']['dashboard']
                cv2.imshow('ADAS Dashboard', dashboard)
                
                # Wait for key press
                key = cv2.waitKey(100)
                if key == 27:  # ESC key
                    break
                    
        if display:
            cv2.destroyAllWindows()
            
        return results
        
    def generate_report(self, results: List[Dict], report_path: str = None):
        """Generate a report from the simulation results"""
        if not report_path and self.output_path:
            report_path = os.path.join(self.output_path, 'report.html')
        elif not report_path:
            report_path = 'report.html'
            
        print(f"Generating report to {report_path}...")
        
        # Count warning and critical events
        warning_count = 0
        critical_count = 0
        
        for result in results:
            response = result['response']
            if response['warning_level'] == 'warning':
                warning_count += 1
            elif response['warning_level'] == 'critical':
                critical_count += 1
                
        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ADAS Simulation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .frames {{ display: flex; flex-wrap: wrap; }}
                .frame {{ margin: 10px; border: 1px solid #ddd; padding: 10px; }}
                .frame img {{ max-width: 500px; }}
                .warning {{ background-color: #fff3cd; }}
                .critical {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>ADAS Simulation Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total frames processed: {len(results)}</p>
                <p>Warning events: {warning_count}</p>
                <p>Critical events: {critical_count}</p>
            </div>
            
            <h2>Frame Details</h2>
            <div class="frames">
        """
        
        # Add frame details
        for result in results:
            frame_id = result['frame_id']
            response = result['response']
            warning_level = response['warning_level']
            
            # Determine frame CSS class based on warning level
            frame_class = warning_level if warning_level in ['warning', 'critical'] else ''
            
            # Add frame to report
            html += f"""
                <div class="frame {frame_class}">
                    <h3>Frame {frame_id}</h3>
                    <p>Warning Level: {warning_level.upper()}</p>
                    <p>Message: {response['message']}</p>
                    <img src="frame_{frame_id:04d}.jpg" alt="Frame {frame_id}">
                </div>
            """
            
        # Close HTML
        html += """
            </div>
        </body>
        </html>
        """
        
        # Write report to file
        with open(report_path, 'w') as f:
            f.write(html)
            
        print(f"Report generated: {report_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ADAS Simulation')
    parser.add_argument('--data-path', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--output-path', type=str, default='../output', help='Path to output directory')
    parser.add_argument('--dataset', type=str, default='kitti', help='Dataset to use (kitti, nuscenes)')
    parser.add_argument('--start-frame', type=int, default=0, help='Starting frame ID')
    parser.add_argument('--num-frames', type=int, default=10, help='Number of frames to process')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Create ADAS system
    adas = ADASSystem(args.data_path, args.output_path, args.gpu)
    
    # Initialize system
    adas.initialize(args.dataset)
    
    # Run simulation
    results = adas.run_simulation(args.start_frame, args.num_frames, not args.no_display)
    
    # Generate report
    adas.generate_report(results)
    
if __name__ == "__main__":
    main() 