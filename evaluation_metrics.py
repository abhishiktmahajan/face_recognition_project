import time
from collections import defaultdict

class FaceRecognitionEvaluator:
    def __init__(self):
        self.reset()
        self.results = []  # For storing batch results

    def reset(self):
        """Reset all metrics."""
        self.total_faces = 0
        self.detected_faces = 0
        self.recognized_faces = 0
        self.predictions = []  # List of (predicted, actual, distance) tuples
        self.processing_times = []
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
        self.results = []  # Reset stored batch results

    def add_detection_result(self, detected_count, actual_count=1):
        """Record face detection result."""
        self.total_faces += actual_count
        self.detected_faces += detected_count

    def add_recognition_result(self, predicted_name, actual_name, distance):
        """Record face recognition result."""
        self.predictions.append((predicted_name, actual_name, distance))
        self.confusion_matrix[actual_name][predicted_name] += 1

    def add_processing_time(self, start_time, end_time):
        """Record processing time (in seconds)."""
        self.processing_times.append(end_time - start_time)

    def add_results(self, results):
        """Add batch processing results."""
        self.results.extend(results)

    def get_detection_rate(self):
    # If batch results are available, compute detection rate from them.
        if self.results:
            detected = sum(1 for r in self.results if r.get('detected', False))
            return detected / len(self.results)
        # Otherwise, fall back on the traditional method.
        if self.total_faces == 0:
            return 0
        return self.detected_faces / self.total_faces

    def get_recognition_accuracy(self):
        if self.results:
            # Assume recognition is successful if recognition is not 'Unknown'
            detected = sum(1 for r in self.results if r.get('detected', False))
            recognized = sum(1 for r in self.results if r.get('recognition', 'Unknown') != "Unknown")
            if detected == 0:
                return 0
            return recognized / detected
        if not self.predictions:
            return 0
        correct = sum(1 for pred, actual, _ in self.predictions if pred == actual)
        return correct / len(self.predictions)


    def get_average_processing_time(self):
        """Get average processing time in milliseconds."""
        if not self.processing_times:
            return 0
        return (sum(self.processing_times) / len(self.processing_times)) * 1000

    def get_fps(self):
        """Calculate frames per second."""
        if not self.processing_times or sum(self.processing_times) == 0:
            return 0
        return len(self.processing_times) / sum(self.processing_times)
    
    def generate_report(self, include_confusion=True):
        total_images = len(self.results)
        detection_rate = self.get_detection_rate() * 100  # as percentage
        recognition_accuracy = self.get_recognition_accuracy() * 100  # as percentage
        report = {
            "total_images": total_images,
            "detection": {
                "total_faces": self.total_faces,
                "detected_faces": self.detected_faces,
                "detection_rate": detection_rate  # This key must be present
            },
            "recognition": {
                "total_predictions": len(self.predictions),
                "accuracy": recognition_accuracy
            },
            "performance": {
                "avg_processing_time_ms": self.get_average_processing_time(),
                "fps": self.get_fps()
            }
        }
        if include_confusion and self.confusion_matrix:
            report["confusion_matrix"] = dict(self.confusion_matrix)
        return report



    # Add this method to satisfy your GUI call:
    def calculate_metrics(self):
        """Calculate and return evaluation metrics."""
        return self.generate_report(include_confusion=True)

    def print_report(self):
        """Print evaluation report to console."""
        report = self.generate_report(include_confusion=False)
        print("\n📊 FACE RECOGNITION EVALUATION REPORT")
        print("=" * 50)
        print(f"\n🔍 DETECTION:")
        print(f"  Total faces: {report['detection']['total_faces']}")
        print(f"  Detected faces: {report['detection']['detected_faces']}")
        print(f"  Detection rate: {report['detection']['detection_rate']*100:.1f}%")
        print(f"\n🏷️ RECOGNITION:")
        print(f"  Total predictions: {report['recognition']['total_predictions']}")
        print(f"  Accuracy: {report['recognition']['accuracy']*100:.1f}%")
        print(f"\n⚡ PERFORMANCE:")
        print(f"  Avg. processing time: {report['performance']['avg_processing_time_ms']:.1f} ms/image")
        print(f"  Processing speed: {report['performance']['fps']:.1f} FPS")
        print("=" * 50)
