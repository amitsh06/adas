import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os

class Visualizer:
    """Visualization utilities for ADAS system"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Colors for different object classes (BGR format)
        self.colors = {
            0: (0, 0, 255),    # Car (red)
            1: (0, 255, 0),    # Pedestrian (green)
            2: (255, 0, 0),    # Bicycle (blue)
            3: (255, 255, 0),  # Motorcycle (cyan)
            4: (0, 255, 255),  # Truck (yellow)
            5: (255, 0, 255),  # Bus (magenta)
            -1: (200, 200, 200)  # Unknown (gray)
        }
        
        # Class names
        self.class_names = {
            0: "Car",
            1: "Pedestrian",
            2: "Bicycle",
            3: "Motorcycle",
            4: "Truck",
            5: "Bus",
            -1: "Unknown"
        }
        
    def draw_objects(self, frame: np.ndarray, objects: List[Dict]) -> np.ndarray:
        """
        Draw detected objects on the camera frame
        """
        output_frame = frame.copy()
        
        for obj in objects:
            # Skip if no bounding box
            if 'box' not in obj or obj['box'] is None:
                continue
                
            # Get object properties
            box = obj['box']
            class_id = obj.get('class', -1)
            confidence = obj.get('confidence', 0)
            
            # Get color for this class
            color = self.colors.get(class_id, (200, 200, 200))
            
            # Draw bounding box
            cv2.rectangle(
                output_frame,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color,
                2
            )
            
            # Get class name
            class_name = self.class_names.get(class_id, "Unknown")
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            
            # Get size of text to draw background rectangle
            (text_width, text_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                output_frame,
                (int(box[0]), int(box[1] - text_height - 5)),
                (int(box[0] + text_width + 5), int(box[1])),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                output_frame,
                label,
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # If distance is available, display it
            if 'distance' in obj and obj['distance'] is not None:
                distance_text = f"{obj['distance']:.1f}m"
                cv2.putText(
                    output_frame,
                    distance_text,
                    (int(box[0]), int(box[1] - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return output_frame
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[tuple]) -> np.ndarray:
        """
        Draw tracked objects with their IDs and trajectories
        """
        output_frame = frame.copy()
        
        for obj_id, track in tracks:
            obj_data = track['data']
            
            # Skip if no center position
            if 'center' not in obj_data or obj_data['center'] is None:
                continue
                
            # Get center position
            center = obj_data['center']
            
            # Draw circle at center
            cv2.circle(
                output_frame,
                (int(center[0]), int(center[1])),
                5,
                (0, 255, 255),  # Yellow
                -1
            )
            
            # Draw object ID
            cv2.putText(
                output_frame,
                f"ID: {obj_id}",
                (int(center[0]) + 10, int(center[1]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
            
            # If Kalman filter prediction is available, draw it
            if hasattr(track['filter'], 'x'):
                predicted_pos = track['filter'].x[:2].flatten()
                
                # Draw line from current to predicted position
                cv2.line(
                    output_frame,
                    (int(center[0]), int(center[1])),
                    (int(predicted_pos[0]), int(predicted_pos[1])),
                    (0, 0, 255),  # Red
                    2
                )
                
                # Draw predicted position
                cv2.circle(
                    output_frame,
                    (int(predicted_pos[0]), int(predicted_pos[1])),
                    3,
                    (0, 0, 255),  # Red
                    -1
                )
        
        return output_frame
    
    def visualize_radar(self, frame: np.ndarray, radar_data: Dict) -> np.ndarray:
        """
        Visualize radar data as overlay on camera frame
        """
        output_frame = frame.copy()
        
        # Get image center (as reference for radar visualization)
        center_x = output_frame.shape[1] // 2
        center_y = output_frame.shape[0] // 2
        
        # Draw radar field of view (cone)
        radius = min(center_x, center_y) - 20
        start_angle = -30
        end_angle = 30
        
        # Draw radar cone
        cv2.ellipse(
            output_frame,
            (center_x, center_y),
            (radius, radius),
            0,
            start_angle,
            end_angle,
            (0, 255, 0),  # Green
            1
        )
        
        # Draw distance circles
        for distance in [5, 10, 20]:
            # Scale distance to pixels
            scaled_radius = int(radius * distance / 20)
            
            cv2.circle(
                output_frame,
                (center_x, center_y),
                scaled_radius,
                (0, 255, 0),  # Green
                1
            )
            
            # Add distance label
            cv2.putText(
                output_frame,
                f"{distance}m",
                (center_x + 5, center_y - scaled_radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        # Plot radar detections
        if radar_data and 'objects' in radar_data:
            for i in range(len(radar_data['objects'])):
                if 'distances' in radar_data and 'angles' in radar_data:
                    distance = radar_data['distances'][i]
                    angle = radar_data['angles'][i]
                    
                    # Scale distance to pixels
                    scaled_distance = int(radius * distance / 20)
                    
                    # Convert to cartesian coordinates
                    x = int(center_x + scaled_distance * np.sin(angle))
                    y = int(center_y - scaled_distance * np.cos(angle))
                    
                    # Draw detection point
                    cv2.circle(
                        output_frame,
                        (x, y),
                        5,
                        (0, 0, 255),  # Red
                        -1
                    )
                    
                    # Add distance label
                    cv2.putText(
                        output_frame,
                        f"{distance:.1f}m",
                        (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )
        
        return output_frame
    
    def visualize_lidar_points(self, frame: np.ndarray, lidar_data: np.ndarray) -> np.ndarray:
        """
        Visualize LIDAR point cloud as overlay on camera frame
        """
        output_frame = frame.copy()
        
        # Skip if no LIDAR data
        if lidar_data is None or len(lidar_data) == 0:
            return output_frame
            
        # Get image center
        center_x = output_frame.shape[1] // 2
        center_y = output_frame.shape[0] // 2
        
        # Scale for visualization
        scale_factor = 20  # Pixels per meter
        
        # Draw LIDAR points
        for point in lidar_data:
            # Get x, y coordinates (first two columns)
            x, y = point[:2]
            
            # Skip points too far away
            if np.sqrt(x**2 + y**2) > 20:
                continue
                
            # Convert to image coordinates
            img_x = int(center_x + x * scale_factor)
            img_y = int(center_y - y * scale_factor)
            
            # Skip if outside image bounds
            if (img_x < 0 or img_x >= output_frame.shape[1] or
                img_y < 0 or img_y >= output_frame.shape[0]):
                continue
                
            # Get intensity if available (typically 4th column)
            intensity = point[3] if len(point) > 3 else 1.0
            
            # Map intensity to color (brighter = higher intensity)
            color_value = int(255 * intensity)
            
            # Draw point
            cv2.circle(
                output_frame,
                (img_x, img_y),
                1,
                (color_value, color_value, 0),  # Yellow-ish with intensity
                -1
            )
        
        return output_frame
    
    def create_dashboard(self, camera_frame: np.ndarray, 
                         objects_frame: np.ndarray, 
                         radar_frame: np.ndarray,
                         risk_data: Dict) -> np.ndarray:
        """
        Create a dashboard with all visualization elements
        """
        # Resize frames to be the same size
        height, width = camera_frame.shape[:2]
        objects_frame = cv2.resize(objects_frame, (width, height))
        radar_frame = cv2.resize(radar_frame, (width, height))
        
        # Create blank canvas for dashboard
        dashboard = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
        
        # Place frames in dashboard
        dashboard[:height, :width] = camera_frame  # Top-left: Original camera
        dashboard[:height, width:] = objects_frame  # Top-right: Objects detection
        dashboard[height:, :width] = radar_frame  # Bottom-left: Radar visualization
        
        # Create info panel for bottom-right
        info_panel = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray
        
        # Add system status information
        cv2.putText(
            info_panel,
            "ADAS Status: Active",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),  # Green
            2
        )
        
        # Add risk information if available
        if risk_data:
            # Risk level
            warning_level = risk_data.get('warning_level', 'none')
            
            if warning_level == 'critical':
                level_color = (0, 0, 255)  # Red
                level_text = "CRITICAL"
            elif warning_level == 'warning':
                level_color = (0, 255, 255)  # Yellow
                level_text = "WARNING"
            else:
                level_color = (0, 255, 0)  # Green
                level_text = "NORMAL"
                
            cv2.putText(
                info_panel,
                f"Risk Level: {level_text}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                level_color,
                2
            )
            
            # Add message
            cv2.putText(
                info_panel,
                risk_data.get('message', ''),
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                level_color,
                2
            )
            
            # Add details if available
            if 'details' in risk_data and risk_data['details']:
                details = risk_data['details']
                
                y_offset = 150
                
                if 'distance' in details and details['distance'] is not None:
                    cv2.putText(
                        info_panel,
                        f"Closest Object: {details['distance']:.2f}m",
                        (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 200, 200),  # Light gray
                        1
                    )
                    y_offset += 30
                    
                if 'ttc' in details and details['ttc'] is not None:
                    cv2.putText(
                        info_panel,
                        f"Time to Collision: {details['ttc']:.2f}s",
                        (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 200, 200),  # Light gray
                        1
                    )
                    y_offset += 30
                    
                if 'risk_score' in details:
                    cv2.putText(
                        info_panel,
                        f"Risk Score: {details['risk_score']:.2f}",
                        (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 200, 200),  # Light gray
                        1
                    )
        
        # Place info panel in dashboard
        dashboard[height:, width:] = info_panel
        
        return dashboard
    
    def save_visualization(self, frame: np.ndarray, filename: str) -> None:
        """
        Save visualization frame to file
        """
        if self.output_dir:
            file_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(file_path, frame) 