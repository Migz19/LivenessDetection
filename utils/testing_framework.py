"""Simple testing utility for liveness detection"""

import requests
import json
from pathlib import Path
from typing import Dict, List


class LivenessTester:
    """Simple tester for liveness detection API"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []
    
    def test_video(self, video_path: str, expected: str = None) -> Dict:
        """Test single video"""
        try:
            with open(video_path, 'rb') as f:
                resp = requests.post(
                    f"{self.api_url}/api/v1/liveness/detect-detailed",
                    files={'file': f},
                    timeout=60
                )
            
            if resp.status_code != 200:
                return {'error': f"API error: {resp.status_code}"}
            
            data = resp.json()
            result = {
                'video': video_path,
                'predicted': data['status'],
                'confidence': data['confidence']['final_confidence'],
                'warnings': data['decision_factors'].get('warning_flags', [])
            }
            
            if expected:
                result['expected'] = expected
                result['correct'] = (expected.lower() == data['status'])
            
            self.results.append(result)
            return result
        except Exception as e:
            return {'error': str(e), 'video': video_path}
    
    def test_batch(self, video_dir: str, labels: Dict = None) -> List[Dict]:
        """Test all videos in directory"""
        labels = labels or {}
        results = []
        
        for video_file in Path(video_dir).glob("*"):
            if video_file.suffix.lower() not in {'.mp4', '.avi', '.mov', '.mkv'}:
                continue
            
            expected = labels.get(video_file.name)
            result = self.test_video(str(video_file), expected)
            results.append(result)
            print(f"{video_file.name}: {result.get('predicted', 'error')}, conf: {result.get('confidence', 'N/A')}")
        
        return results
    
    def accuracy(self) -> float:
        """Calculate accuracy from labeled results"""
        labeled = [r for r in self.results if r.get('correct') is not None]
        if not labeled:
            return None
        return sum(r['correct'] for r in labeled) / len(labeled)
