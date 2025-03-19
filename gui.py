import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import threading
import os
import time
import json
from facenet_pytorch import MTCNN, InceptionResnetV1 # type: ignore
import torch
import glob

# Import our custom modules
from config import load_config, save_config
from database import FaceDatabase
from evaluation_metrics import FaceRecognitionEvaluator
import pillow_heif  # type: ignore
# ---------------------------
# Helper Function for Image Loading with HEIC Support
# ---------------------------
def load_image(file_path):
    """
    Loads an image from file_path. Supports HEIC format using pillow-heif.
    Returns a PIL.Image object or None if loading fails.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.heic':
            # Use pillow-heif for HEIC images
           
            heif_file = pillow_heif.read_heif(file_path)
            img_pil = PIL.Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw"
            )
        else:
            # Use OpenCV for other formats
            img = cv2.imread(file_path)
            if img is None:
                print(f"Could not load image: {file_path}")
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = PIL.Image.fromarray(img)
        return img_pil
    except Exception as e:
        print(f"Error loading image {file_path}: {str(e)}")
        return None

# ---------------------------
# Main GUI Application
# ---------------------------
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x700")
        
        # Load configuration
        self.config = load_config()
        
        # Initialize face database
        self.face_db = FaceDatabase(
            self.config["paths"]["known_faces_dir"],
            device=self.config["device"],
            model_name=self.config["recognition"]["model"]
        )
        
        # Initialize MTCNN for detection
        self.mtcnn = MTCNN(
            keep_all=self.config["detection"]["keep_all"],
            device=self.face_db.device
        )
        
        # Initialize evaluator
        self.evaluator = FaceRecognitionEvaluator()
        
        # Setup camera variables
        self.cap = None
        self.camera_active = False
        self.camera_thread = None
        self.stop_camera = False
        
        # Initialize status variable BEFORE creating the UI
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Select an image or start webcam.")
        
        # Setup UI
        self.create_ui()
        
        # Image display variables
        self.current_image = None
        self.photo = None
        
        # Update status bar and DB info
        self.update_db_info()

    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top panel with buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Start Webcam", command=self.toggle_webcam).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Batch Process", command=self.batch_process).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Add Person", command=self.add_person).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Settings", command=self.open_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="View Report", command=self.show_report).pack(side=tk.LEFT, padx=5)
        
        # Content panel for image display
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.image_panel = ttk.Label(content_frame)
        self.image_panel.pack(fill=tk.BOTH, expand=True)
        
        # Right panel for DB info and settings
        right_panel = ttk.Frame(main_frame, width=300)
        right_panel.pack(fill=tk.Y, side=tk.RIGHT, padx=5)
        
        # Database info panel
        db_frame = ttk.LabelFrame(right_panel, text="Database Info")
        db_frame.pack(fill=tk.X, pady=5)
        self.db_info_text = tk.Text(db_frame, height=10, width=35)
        self.db_info_text.pack(fill=tk.X)
        self.db_info_text.config(state=tk.DISABLED)
        
        # Status bar panel
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
        # Settings panel (hidden by default)
        self.settings_frame = ttk.LabelFrame(right_panel, text="Settings")
        ttk.Label(self.settings_frame, text="Detection Confidence:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.detection_scale = ttk.Scale(self.settings_frame, from_=0.1, to=1.0, 
                                         value=self.config["detection"]["confidence_threshold"],
                                         length=200)
        self.detection_scale.grid(row=0, column=1, pady=2)
        ttk.Label(self.settings_frame, text="Recognition Distance:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.recognition_scale = ttk.Scale(self.settings_frame, from_=0.1, to=2.0, 
                                           value=self.config["recognition"]["distance_threshold"],
                                           length=200)
        self.recognition_scale.grid(row=1, column=1, pady=2)
        ttk.Button(self.settings_frame, text="Save Settings", command=self.save_settings).grid(row=2, column=0, columnspan=2, pady=10)
    
    def update_db_info(self):
        """Update the database info text widget with current statistics."""
        self.db_info_text.config(state=tk.NORMAL)
        self.db_info_text.delete(1.0, tk.END)
        people_count = len(self.face_db.known_faces)
        total_images = 0
        for person in self.face_db.known_faces:
            person_path = os.path.join(self.config["paths"]["known_faces_dir"], person)
            total_images += len([f for f in os.listdir(person_path) if f.lower().endswith(('.jpg','.jpeg','.png', '.heic'))])
        
        info_text = f"People in database: {people_count}\n"
        info_text += f"Total face images: {total_images}\n"
        info_text += f"Recognition model: {self.config['recognition']['model']}\n"
        info_text += f"Device: {self.config['device']}\n\nPeople:\n"
        for person in self.face_db.known_faces:
            info_text += f"- {person}\n"
        
        self.db_info_text.insert(tk.END, info_text)
        self.db_info_text.config(state=tk.DISABLED)
    
    def open_image(self):
        """Open an image file and process it."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.heic"), ("All files", "*.*"))
        )
        if not file_path:
            return
        try:
            img_pil = load_image(file_path)
            if img_pil is None:
                raise Exception("Image could not be loaded.")
            # Convert PIL image to numpy array (RGB)
            img = np.array(img_pil)
            self.process_image(img)
            self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Could not open image: {str(e)}")
    
    def toggle_webcam(self):
        """Start or stop the webcam feed."""
        if self.camera_active:
            self.stop_camera = True
            self.status_var.set("Webcam stopped")
            self.camera_active = False
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            self.stop_camera = False
            self.camera_active = True
            self.status_var.set("Webcam active")
            self.camera_thread = threading.Thread(target=self.update_webcam)
            self.camera_thread.daemon = True
            self.camera_thread.start()
    
    def update_webcam(self):
        """Continuously update the webcam feed in a separate thread."""
        fps_counter = 0
        fps_start = time.time()
        while not self.stop_camera:
            ret, frame = self.cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fps_counter += 1
            # Process every 3rd frame to balance performance
            if fps_counter % 3 == 0:
                self.process_image(rgb_frame.copy())
            if time.time() - fps_start > 1:
                fps = fps_counter / (time.time() - fps_start)
                self.status_var.set(f"Webcam active - {fps:.1f} FPS")
                fps_counter = 0
                fps_start = time.time()
            # Small delay
            time.sleep(0.01)
        if self.cap:
            self.cap.release()
    
    def process_image(self, img):
        """Detect faces in the image, perform recognition, and display the result."""
        try:
            confidence_threshold = self.detection_scale.get()
            distance_threshold = self.recognition_scale.get()
            
            # Use the face database's mtcnn for alignment
            img_pil = PIL.Image.fromarray(img)
            boxes, probs = self.mtcnn.detect(np.array(img_pil))
            display_img = img.copy()
            
            if boxes is not None and len(boxes) > 0:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob < confidence_threshold:
                        continue
                    box = [int(b) for b in box]
                    x1, y1, x2, y2 = box
                    try:
                        face_img = img_pil.crop((x1, y1, x2, y2))
                        aligned_face = self.mtcnn(face_img)
                        if aligned_face is None:
                            continue

                        # Fix incorrect dimensions
                        aligned_face = aligned_face.squeeze(0)
                        if aligned_face.dim() == 5:
                            aligned_face = aligned_face.squeeze(1)
                        if aligned_face.dim() == 3:
                            aligned_face = aligned_face.unsqueeze(0)
                        if aligned_face.dim() != 4:
                            print(f"⚠️ Unexpected tensor shape: {aligned_face.shape}")
                        
                        aligned_face = aligned_face.to(self.face_db.device)
                        embedding = self.face_db.resnet(aligned_face).detach().cpu().numpy().flatten()
                        best_match, min_dist = self.face_db.recognize_face(embedding, distance_threshold)
                        
                        # Choose rectangle color based on recognition result
                        color = (0, 255, 0) if best_match != "Unknown" else (255, 0, 0)
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, self.config["display"]["box_thickness"])
                        label = f"{best_match}"
                        cv2.putText(display_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 2)
                        
                        # Record evaluation data if needed
                        self.evaluator.add_recognition_result(best_match, best_match, min_dist)
                    
                    except Exception as e:
                        print(f"Error processing face {i}: {str(e)}")
            self.display_image(display_img)
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Error processing image: {str(e)}")
    
    def display_image(self, img):
        """Resize and display the image on the GUI."""
        h, w = img.shape[:2]
        max_h = self.root.winfo_height() - 100
        max_w = self.root.winfo_width() - 350
        scale = min(max_w/w, max_h/h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.current_image = PIL.Image.fromarray(img)
        self.photo = PIL.ImageTk.PhotoImage(self.current_image)
        self.image_panel.configure(image=self.photo)
    
    def batch_process(self):
        """Allow the user to select a folder and process all images within it."""
        folder_path = filedialog.askdirectory(title="Select Test Images Folder")
        if not folder_path:
            return
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.heic']:
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        if not image_files:
            messagebox.showinfo("Info", "No image files found in the selected folder")
            return
        threading.Thread(target=self._process_batch, args=(image_files,)).start()
    
    def _process_batch(self, image_files):
        """Process images in a separate thread for evaluation."""
        self.status_var.set(f"Processing {len(image_files)} images...")
        results = []
        for i, img_path in enumerate(image_files):
            start_time = time.time()  # Record start time for this image
            try:
                # Use load_image to support HEIC files
                img_pil = load_image(img_path)
                if img_pil is None:
                    results.append({'image': os.path.basename(img_path), 'error': 'Image could not be loaded'})
                    continue
                
                # Convert PIL image to numpy array
                img = np.array(img_pil)
                boxes, probs = self.mtcnn.detect(np.array(img_pil))
                if boxes is not None and len(boxes) > 0:
                    idx = np.argmax(probs)
                    box = [int(b) for b in boxes[idx]]
                    x1, y1, x2, y2 = box
                    face_img = img_pil.crop((x1, y1, x2, y2))
                    aligned_face = self.mtcnn(face_img)
                    if aligned_face is not None:
                        print("Original aligned_face shape:", aligned_face.shape)
                        
                        # Remove extra dimension if the tensor is 5D ([1, 1, 3, 160, 160])
                        if aligned_face.dim() == 5 and aligned_face.size(1) == 1:
                            aligned_face = aligned_face.squeeze(1)
                        
                        # If unbatched (3D: [3, 160, 160]), add batch dimension
                        if aligned_face.dim() == 3:
                            aligned_face = aligned_face.unsqueeze(0)
                        
                        if aligned_face.dim() != 4:
                            print(f"⚠️ Unexpected tensor shape: {aligned_face.shape}")
                        
                        aligned_face = aligned_face.to(self.face_db.device)
                        embedding = self.face_db.resnet(aligned_face).detach().cpu().numpy().flatten()
                        best_match, min_dist = self.face_db.recognize_face(embedding, distance_threshold=self.recognition_scale.get())
                        
                        # Update evaluator metrics manually:
                        self.evaluator.add_detection_result(1, actual_count=1)
                        self.evaluator.add_recognition_result(best_match, best_match, min_dist)
                        
                        results.append({
                            'image': os.path.basename(img_path),
                            'detected': True,
                            'recognition': best_match,
                            'confidence': float(probs[idx]),
                            'distance': float(min_dist)
                        })
                    else:
                        results.append({'image': os.path.basename(img_path), 'detected': False})
                else:
                    results.append({'image': os.path.basename(img_path), 'detected': False})
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                results.append({'image': os.path.basename(img_path), 'error': str(e)})
            end_time = time.time()  # Record end time
            self.evaluator.add_processing_time(start_time, end_time)
        self.evaluator.add_results(results)
        self.status_var.set(f"Processed {len(image_files)} images. View report for details.")
        self.root.after(100, self.show_report)
    
    def add_person(self):
        """Open a popup to add a new person with selected images."""
        popup = tk.Toplevel(self.root)
        popup.title("Add New Person")
        popup.geometry("400x200")
        popup.transient(self.root)
        popup.grab_set()
        ttk.Label(popup, text="Person Name:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        name_var = tk.StringVar()
        name_entry = ttk.Entry(popup, textvariable=name_var, width=30)
        name_entry.grid(row=0, column=1, padx=10, pady=10)
        name_entry.focus()
        ttk.Label(popup, text="Face Images:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        image_var = tk.StringVar()
        image_var.set("No images selected")
        ttk.Label(popup, textvariable=image_var).grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)
        selected_images = []
        
        def browse_images():
            files = filedialog.askopenfilenames(
                title="Select Face Images",
                filetypes=(("Image files", "*.jpg *.jpeg *.png *.heic"), ("All files", "*.*"))
            )
            nonlocal selected_images
            if files:
                selected_images = files
                image_var.set(f"{len(files)} images selected")
        
        ttk.Button(popup, text="Browse...", command=browse_images).grid(row=1, column=2, padx=10, pady=10)
        
        def save_person():
            person_name = name_var.get().strip()
            if not person_name:
                messagebox.showerror("Error", "Person name is required")
                return
            if not selected_images:
                messagebox.showerror("Error", "At least one face image is required")
                return
            popup.destroy()
            threading.Thread(target=self._add_person_thread, args=(person_name, selected_images)).start()
        
        ttk.Button(popup, text="Add Person", command=save_person).grid(row=2, column=0, columnspan=3, pady=20)
    
    def _add_person_thread(self, person_name, image_paths):
        """Background thread to add a person to the database."""
        self.status_var.set(f"Adding {person_name} to database...")
        person_dir = os.path.join(self.config["paths"]["known_faces_dir"], person_name)
        os.makedirs(person_dir, exist_ok=True)
        processed = 0
        for i, img_path in enumerate(image_paths):
            try:
                # Use load_image to support HEIC files
                img_pil = load_image(img_path)
                if img_pil is None:
                    print(f"Could not load image: {img_path}")
                    continue
                
                # Convert the image to RGB (if not already in RGB)
                # (load_image returns a PIL.Image in RGB mode if successful)
                
                # Detect and align face
                boxes, probs = self.mtcnn.detect(np.array(img_pil))
                if boxes is None or len(boxes) == 0:
                    print(f"No face detected in image: {img_path}")
                    continue  # Skip this image if no face is detected

                idx = np.argmax(probs)
                box = [int(b) for b in boxes[idx]]
                x1, y1, x2, y2 = box
                face_img = img_pil.crop((x1, y1, x2, y2))
                aligned_face = self.mtcnn(face_img)
                
                if aligned_face is not None:
                    print("Aligned face shape before adjustment:", aligned_face.shape)
                    
                    # If the tensor is 5D (e.g. [1, 1, 3, 160, 160]) and the extra dimension is 1, remove it.
                    if aligned_face.dim() == 5 and aligned_face.size(1) == 1:
                        aligned_face = aligned_face.squeeze(1)
                    
                    # If the tensor is 4D (e.g. [1, 3, 160, 160]), remove the batch dimension.
                    if aligned_face.dim() == 4:
                        aligned_face = aligned_face.squeeze(0)
                    
                    print("Aligned face shape after adjustment:", aligned_face.shape)
                    
                    # Rearrange dimensions from [C, H, W] to [H, W, C] so Pillow can use it.
                    face_np = (aligned_face.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    face_pil = PIL.Image.fromarray(face_np)
                    
                    # Build filename and define save_path
                    img_name = f"{person_name}_{i+1:03d}.jpg"
                    save_path = os.path.join(person_dir, img_name)
                    
                    face_pil.save(save_path)
                    processed += 1
                else:
                    print(f"Face alignment failed for image: {img_path}")
                    continue
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
        # Reload the face database and update UI
        self.face_db.load_known_faces()
        self.update_db_info()
        self.status_var.set(f"Added {processed} face images for {person_name}")
        messagebox.showinfo("Success", f"Added {processed} face images for {person_name}")
    
    def open_settings(self):
        """Toggle the visibility of the settings panel."""
        if self.settings_frame.winfo_ismapped():
            self.settings_frame.pack_forget()
        else:
            self.settings_frame.pack(fill=tk.X, pady=5)
    
    def save_settings(self):
        """Save the current settings to the configuration file."""
        self.config["detection"]["confidence_threshold"] = self.detection_scale.get()
        self.config["recognition"]["distance_threshold"] = self.recognition_scale.get()
        save_config(self.config)
        self.status_var.set("Settings saved")
        messagebox.showinfo("Success", "Settings saved successfully")
    
    def show_report(self):
        metrics = self.evaluator.calculate_metrics()  # Now returns full dictionary
        report_text = "# Face Recognition Performance Report\n\n"
        report_text += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report_text += f"Total images processed: {metrics['total_images']}\n"
        report_text += f"Face detection rate: {metrics['detection']['detection_rate']:.2f}%\n"
        report_text += f"Recognition accuracy: {metrics['recognition']['accuracy']:.2f}%\n"
        report_text += f"Avg. processing time: {metrics['performance']['avg_processing_time_ms']:.1f} ms/image\n"
        report_text += f"Processing speed: {metrics['performance']['fps']:.1f} FPS\n\n"
        
        if "confusion_matrix" in metrics:
            report_text += "Confusion Matrix:\n"
            for actual, preds in metrics["confusion_matrix"].items():
                report_text += f"{actual}: {preds}\n"
        
        report_window = tk.Toplevel(self.root)
        report_window.title("Recognition Performance Report")
        report_window.geometry("600x500")
        report_text_widget = tk.Text(report_window, wrap=tk.WORD, padx=10, pady=10)
        report_text_widget.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(report_text_widget, command=report_text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        report_text_widget.config(yscrollcommand=scrollbar.set)
        report_text_widget.insert(tk.END, report_text)
        report_text_widget.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
