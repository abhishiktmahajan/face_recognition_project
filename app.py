import torch
import cv2
import numpy as np
import os
import time
import argparse
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1

# Import our custom modules
# Assuming the previous code is saved in separate files
from config_system import load_config, save_config
from face_database import FaceDatabase
from evaluation_metrics import FaceRecognitionEvaluator

def get_font():
    """Try to load a suitable font"""
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        try:
            if os.name == 'nt':  # Windows
                font = ImageFont.truetype("calibri.ttf", 20)
            else:  # Unix/Linux/Mac
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
    return font

def process_image(img, face_db, mtcnn, config, evaluator=None):
    """Process a single image and return the result image with detections"""
    start_time = time.time()
    
    # Convert to PIL if it's a numpy array
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = img
    
    img_np = np.array(img_pil)
    draw_img = img_pil.copy()
    draw = ImageDraw.Draw(draw_img)
    font = get_font()
    
    # Get detection parameters from config
    confidence_threshold = config["detection"]["confidence_threshold"]
    distance_threshold = config["recognition"]["distance_threshold"]
    
    # Detect faces
    boxes, probs = mtcnn.detect(img_np)
    
    if evaluator:
        evaluator.add_detection_result(boxes, 1)  # Assume 1 face in ground truth for simplicity
    
    # Draw faces and recognize
    if boxes is not None and len(boxes) > 0:
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < confidence_threshold:
                continue
                
            # Convert box coordinates to integers
            box = [int(b) for b in box]
            x1, y1, x2, y2 = box
            
            try:
                # Extract and align the face
                face_img = img_pil.crop((x1, y1, x2, y2))
                # Use the single-face MTCNN for alignment
                aligned_face = face_db.mtcnn(face_img)
                
                if aligned_face is None:
                    continue
                
                # Get embedding
                aligned_face = aligned_face.unsqueeze(0).to(face_db.device)
                embedding = face_db.resnet(aligned_face).detach().cpu().numpy().flatten()
                
                # Find the best match
                best_match, min_dist = face_db.recognize_face(
                    embedding, distance_threshold)
                
                # Prepare label
                parts = []
                if config["display"]["show_confidence"]:
                    parts.append(f"Conf: {prob:.2f}")
                if config["display"]["show_distance"]:
                    parts.append(f"Dist: {min_dist:.2f}")
                
                label = f"{best_match}"
                detail = ", ".join(parts) if parts else ""
                
                # Draw rectangle
                color = (0, 255, 0) if best_match != "Unknown" else (255, 0, 0)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=config["display"]["box_thickness"])
                
                # Add text background for main label
                if hasattr(draw, 'textbbox'):
                    text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                    draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill=color)
                else:
                    # Fallback for older PIL versions
                    text_width, text_height = draw.textsize(label, font=font)
                    draw.rectangle([x1, y1 - text_height - 5, x1 + text_width, y1], fill=color)
                
                # Draw name label
                draw.text((x1, y1 - 25), label, font=font, fill="white")
                
                # Add detail text if needed
                if detail:
                    y_offset = -45  # Further up from the name
                    if hasattr(draw, 'textbbox'):
                        text_bbox = draw.textbbox((x1, y1 + y_offset), detail, font=font)
                        draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill=(0, 0, 0))
                    else:
                        text_width, text_height = draw.textsize(detail, font=font)
                        draw.rectangle([x1, y1 + y_offset - text_height, x1 + text_width, y1 + y_offset], fill=(0, 0, 0))
                    
                    draw.text((x1, y1 + y_offset), detail, font=font, fill=(255, 255, 255))
                
                # Record for evaluation if evaluator is provided
                if evaluator:
                    # In a real system, you would get the actual name from ground truth
                    # For now, we'll just use recognition result to simulate
                    evaluator.add_recognition_result(best_match, best_match, min_dist)
                
            except Exception as e:
                print(f"⚠️ Error processing face {i+1}: {str(e)}")
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    
    end_time = time.time()
    if evaluator:
        evaluator.add_processing_time(start_time, end_time)
    
    # Convert back for display
    result_image = cv2.cvtColor(np.array(draw_img), cv2.COLOR_RGB2BGR)
    
    # Add FPS info
    fps = 1.0 / (end_time - start_time)
    cv2.putText(result_image, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return result_image

def process_image_batch(image_dir, face_db, mtcnn, config, output_dir=None, evaluator=None):
    """Process all images in a directory"""
    if not os.path.exists(image_dir):
        print(f"❌ Directory not found: {image_dir}")
        return
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_images = 0
    processed_images = 0
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image_path = os.path.join(image_dir, filename)
                img = Image.open(image_path).convert("RGB")
                
                print(f"Processing {filename}...")
                result_img = process_image(img, face_db, mtcnn, config, evaluator)
                
                if output_dir:
                    output_path = os.path.join(output_dir, f"result_{filename}")
                    cv2.imwrite(output_path, result_img)
                    print(f"✅ Saved result to {output_path}")
                
                # Display the result
                cv2.imshow("Face Recognition", result_img)
                key = cv2.waitKey(0)
                if key == 27:  # ESC key
                    break
                
                processed_images += 1
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
            
            total_images += 1
    
    cv2.destroyAllWindows()
    print(f"✅ Processed {processed_images}/{total_images} images")

def process_webcam(face_db, mtcnn, config, evaluator=None):
    """Process webcam feed for face recognition"""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("✅ Webcam started. Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Process every other frame to improve performance
        if frame_count % 2 == 0:
            result_frame = process_image(frame, face_db, mtcnn, config, evaluator)
            cv2.imshow("Face Recognition (Webcam)", result_frame)
        else:
            # Just show the original frame
            cv2.imshow("Face Recognition (Webcam)", frame)
        
        frame_count += 1
        
        # Calculate FPS every second
        if frame_count % 30 == 0:
            end_time = time.time()
            fps = 30 / (end_time - start_time)
            start_time = time.time()
            print(f"FPS: {fps:.1f}")
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam closed")

def main():
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument("--mode", choices=["test", "batch", "webcam"], default="test",
                        help="Operation mode: test (single image), batch (directory), webcam")
    parser.add_argument("--image", default="test_images/test3.jpg",
                        help="Path to test image (for test mode)")
    parser.add_argument("--dir", default="test_images",
                        help="Directory with images (for batch mode)")
    parser.add_argument("--output", default="results",
                        help="Output directory for results")
    parser.add_argument("--report", action="store_true",
                        help="Generate evaluation report")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Initialize face database
    face_db = FaceDatabase(
        config["paths"]["known_faces_dir"],
        device=config["device"],
        model_name=config["recognition"]["model"]
    )
    
    # Initialize MTCNN for detection
    mtcnn = MTCNN(
        keep_all=config["detection"]["keep_all"],
        device=face_db.device
    )
    
    # Initialize evaluator if reporting is enabled
    evaluator = FaceRecognitionEvaluator() if args.report else None
    
    # Process based on mode
    if args.mode == "test":
        if os.path.exists(args.image):
            img = Image.open(args.image).convert("RGB")
            result = process_image(img, face_db, mtcnn, config, evaluator)
            
            # Display the result
            cv2.imshow("Face Recognition", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save the result
            if not os.path.exists(args.output):
                os.makedirs(args.output)
            output_path = os.path.join(args.output, f"result_{os.path.basename(args.image)}")
            cv2.imwrite(output_path, result)
            print(f"✅ Saved result to {output_path}")
        else:
            print(f"❌ Image not found: {args.image}")
    
    elif args.mode == "batch":
        process_image_batch(args.dir, face_db, mtcnn, config, args.output, evaluator)
    
    elif args.mode == "webcam":
        process_webcam(face_db, mtcnn, config, evaluator)
    
    # Print evaluation report if enabled
    if evaluator:
        evaluator.print_report()

if __name__ == "__main__":
    main()