import numpy as np
from typing import Dict, List, Any
import cv2

class CollisionAvoidanceSystem:
    """Collision avoidance system for ADAS"""
    def __init__(self):
        self.warning_threshold = 0.5  # Risk score above which to issue warning
        self.emergency_threshold = 0.8  # Risk score above which to take emergency action
        self.prev_risk_scores = {}  # Store previous risk scores for smoothing
        
    def process_risks(self, risk_data: List[Dict]) -> Dict:
        """
        Process collision risks and determine appropriate response
        """
        if not risk_data:
            # No risks detected
            return {
                'action': 'none',
                'warning_level': 'none',
                'message': 'No obstacles detected',
                'details': {}
            }
            
        # Find highest risk score
        max_risk = max(risk_data, key=lambda x: x['risk_score'])
        risk_score = max_risk['risk_score']
        
        # Apply temporal smoothing if we have previous data for this object
        obj_id = max_risk['object_id']
        if obj_id in self.prev_risk_scores:
            # Weighted average with previous score (70% current, 30% previous)
            risk_score = 0.7 * risk_score + 0.3 * self.prev_risk_scores[obj_id]
            
        # Update previous score
        self.prev_risk_scores[obj_id] = risk_score
        
        # Determine response based on risk score
        response = {
            'action': 'none',
            'warning_level': 'none',
            'message': 'Normal operation',
            'details': {
                'object_id': obj_id,
                'risk_score': risk_score,
                'ttc': max_risk.get('ttc'),
                'distance': max_risk.get('distance'),
                'velocity': max_risk.get('velocity')
            }
        }
        
        if risk_score >= self.emergency_threshold:
            # Emergency situation
            response.update({
                'action': 'emergency_brake',
                'warning_level': 'critical',
                'message': 'EMERGENCY BRAKING!'
            })
        elif risk_score >= self.warning_threshold:
            # Warning situation
            response.update({
                'action': 'warning',
                'warning_level': 'warning',
                'message': 'Collision warning!'
            })
            
        return response
        
    def generate_visual_warning(self, frame: np.ndarray, response: Dict) -> np.ndarray:
        """
        Generate a visual warning overlay on the camera frame
        """
        warning_frame = frame.copy()
        
        if response['warning_level'] == 'critical':
            # Red border for critical warning
            border_color = (0, 0, 255)  # BGR format
            border_thickness = 30
            
            # Add red border
            cv2.rectangle(
                warning_frame,
                (border_thickness, border_thickness),
                (warning_frame.shape[1] - border_thickness, warning_frame.shape[0] - border_thickness),
                border_color,
                border_thickness
            )
            
            # Add warning text
            cv2.putText(
                warning_frame,
                response['message'],
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )
            
        elif response['warning_level'] == 'warning':
            # Yellow border for warning
            border_color = (0, 255, 255)  # BGR format
            border_thickness = 20
            
            # Add yellow border
            cv2.rectangle(
                warning_frame,
                (border_thickness, border_thickness),
                (warning_frame.shape[1] - border_thickness, warning_frame.shape[0] - border_thickness),
                border_color,
                border_thickness
            )
            
            # Add warning text
            cv2.putText(
                warning_frame,
                response['message'],
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                2
            )
            
        # Add detailed information
        if 'details' in response and response['details']:
            details = response['details']
            info_text = []
            
            if 'distance' in details and details['distance'] is not None:
                info_text.append(f"Distance: {details['distance']:.2f}m")
                
            if 'ttc' in details and details['ttc'] is not None:
                info_text.append(f"TTC: {details['ttc']:.2f}s")
                
            if 'risk_score' in details:
                info_text.append(f"Risk: {details['risk_score']:.2f}")
                
            # Display information
            for i, text in enumerate(info_text):
                cv2.putText(
                    warning_frame,
                    text,
                    (50, 150 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                
        return warning_frame
        
    def execute_response(self, response: Dict) -> bool:
        """
        Execute the determined response action
        Returns True if emergency action was taken
        """
        if response['action'] == 'emergency_brake':
            # In a real system, this would activate the braking system
            print("EXECUTING EMERGENCY BRAKING!")
            return True
            
        elif response['action'] == 'warning':
            # In a real system, this would activate visual/audio warnings
            print("WARNING: Possible collision detected!")
            return False
            
        return False 