import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor
import time
from collections import defaultdict

# -----------------------------
# FaceRecognitionEvaluator Class (Your provided code)
# -----------------------------
class FaceRecognitionEvaluator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_faces = 0
        self.detected_faces = 0
        self.recognized_faces = 0
        self.predictions = []  # List of (predicted, actual, distance) tuples
        self.processing_times = []
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    def add_detection_result(self, detected_count, actual_count=1):
        """Record face detection result"""
        self.total_faces += actual_count
        self.detected_faces += detected_count
    
    def add_recognition_result(self, predicted_name, actual_name, distance):
        """Record face recognition result"""
        self.predictions.append((predicted_name, actual_name, distance))
        self.confusion_matrix[actual_name][predicted_name] += 1
    
    def add_processing_time(self, start_time, end_time):
        """Record processing time (in seconds)"""
        self.processing_times.append(end_time - start_time)
    
    def get_detection_rate(self):
        """Calculate face detection rate"""
        if self.total_faces == 0:
            return 0
        return self.detected_faces / self.total_faces
    
    def get_recognition_accuracy(self):
        """Calculate recognition accuracy"""
        if not self.predictions:
            return 0
        correct = sum(1 for pred, actual, _ in self.predictions if pred == actual)
        return correct / len(self.predictions)
    
    def get_average_processing_time(self):
        """Get average processing time in milliseconds"""
        if not self.processing_times:
            return 0
        return (sum(self.processing_times) / len(self.processing_times)) * 1000
    
    def get_fps(self):
        """Calculate frames per second"""
        if not self.processing_times or sum(self.processing_times) == 0:
            return 0
        return len(self.processing_times) / sum(self.processing_times)
    
    def generate_report(self, include_confusion=True):
        """Generate evaluation report"""
        report = {
            "detection": {
                "total_faces": self.total_faces,
                "detected_faces": self.detected_faces,
                "detection_rate": self.get_detection_rate()
            },
            "recognition": {
                "total_predictions": len(self.predictions),
                "accuracy": self.get_recognition_accuracy()
            },
            "performance": {
                "avg_processing_time_ms": self.get_average_processing_time(),
                "fps": self.get_fps()
            }
        }
        
        if include_confusion and self.confusion_matrix:
            report["confusion_matrix"] = dict(self.confusion_matrix)
        
        return report
    
    def print_report(self):
        """Print evaluation report to console"""
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

# -----------------------------
# Main Face Recognition Code
# -----------------------------
# Set device and initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

mtcnn = MTCNN(keep_all=False, device=device)  # For known faces (one face per image)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("FaceNet model loaded.")

# Load known faces and compute embeddings
known_faces_dir = "known_faces"
known_faces = {}

print("\n📥 Loading known faces...")
for person_name in os.listdir(known_faces_dir):
    person_path = os.path.join(known_faces_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    embeddings = []
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        try:
            img = Image.open(image_path).convert("RGB")
            face = mtcnn(img, return_prob=False)
            if face is None:
                print(f"❌ No face detected in {image_name}, skipping...")
                continue

            face = face.unsqueeze(0).to(device)
            embedding = resnet(face).detach().cpu().numpy().flatten()
            embeddings.append(embedding)
            print(f"✅ Processed face in {image_name}")
        except Exception as e:
            print(f"❌ Error processing {image_name}: {str(e)}")
            continue

    if embeddings:
        known_faces[person_name] = np.mean(np.stack(embeddings), axis=0)
        print(f"📌 {person_name} - {len(embeddings)} images processed, embedding shape: {known_faces[person_name].shape}")

print(f"\n✅ Loaded {len(known_faces)} known people.")

# Initialize evaluator instance
evaluator = FaceRecognitionEvaluator()

# For demonstration purposes, assume each known face is a "ground truth" for itself.
# In a realistic test, you would have ground truth labels for your test images.
# Here, we'll simulate that for each test image processed, we know the actual person.
# For now, let's assume the test image's ground truth is "Tom Cruise" (adjust as needed).

# Process a test image and update evaluator metrics
def process_and_display_image(image_path, confidence_threshold=0.7, distance_threshold=1.0, ground_truth="David"):
    start_time = time.time()
    
    # Load test image using OpenCV and convert to RGB
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load test image {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Try to load a font for drawing text
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Use MTCNN to detect faces in the test image
    boxes, probs = mtcnn.detect(img_pil)
    if boxes is None or len(boxes) == 0:
        print(f"❌ No faces detected in {image_path}")
        evaluator.add_detection_result(0, actual_count=1)
        return
    
    evaluator.add_detection_result(len(boxes), actual_count=1)
    print(f"\n✅ Detected {len(boxes)} faces")
    
    draw = ImageDraw.Draw(img_pil)
    for i, (box, prob) in enumerate(zip(boxes, probs)):
        if prob < confidence_threshold:
            continue
        box = [int(b) for b in box]
        x1, y1, x2, y2 = box
        
        try:
            face_img = img_pil.crop((x1, y1, x2, y2)).resize((160, 160))
            aligned_face = mtcnn(face_img)
            if aligned_face is None:
                print(f"⚠️ Face {i+1}: Could not align face")
                continue
            
            aligned_face = aligned_face.unsqueeze(0).to(device)
            embedding = resnet(aligned_face).detach().cpu().numpy().flatten()
            
            # Update evaluator with recognition result
            predicted_name = "Unknown"
            min_dist = float("inf")
            for name, known_embedding in known_faces.items():
                dist = np.linalg.norm(embedding - known_embedding)
                if dist < min_dist:
                    min_dist = dist
                    predicted_name = name
            
            # If the best distance is above threshold, mark as Unknown
            if min_dist > distance_threshold:
                predicted_name = "Unknown"
            
            evaluator.add_recognition_result(predicted_name, ground_truth, min_dist)
            label = f"{predicted_name} ({min_dist:.2f})"
            print(f"🔍 Face {i+1}: {label}")
            
            # Draw rectangle and label
            color = (0, 255, 0) if predicted_name != "Unknown" else (255, 0, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - 25), label, fill="white", font=font)
        except Exception as e:
            print(f"⚠️ Error processing face {i+1}: {str(e)}")
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1, y1 - 25), "Error", fill="red", font=font)
    
    end_time = time.time()
    evaluator.add_processing_time(start_time, end_time)
    
    result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("Face Recognition", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result_img

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    test_image_path = os.path.join("test_images", "test3.jpg")  # Change as needed
    print(f"\nProcessing test image: {test_image_path}")
    process_and_display_image(test_image_path, confidence_threshold=0.7, distance_threshold=1.0, ground_truth="David")
    
    # After processing, print the evaluation report.
    evaluator.print_report()
