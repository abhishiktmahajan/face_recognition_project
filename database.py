import os
import shutil
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceDatabase:
    def __init__(self, known_faces_dir, device="cuda", model_name="vggface2"):
        self.known_faces_dir = known_faces_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained=model_name).eval().to(self.device)
        
        # Load known faces
        self.known_faces = {}
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load all known faces from the database directory"""
        print(f"\n📥 Loading known faces from {self.known_faces_dir}...")
        self.known_faces = {}
        
        for person_name in os.listdir(self.known_faces_dir):
            person_path = os.path.join(self.known_faces_dir, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            embeddings = []
            valid_images = 0
            skipped_images = 0
            
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                
                try:
                    img = Image.open(image_path).convert("RGB")
                    face = self.mtcnn(img, return_prob=False)
                    
                    if face is None:
                        skipped_images += 1
                        continue
                    
                    face = face.unsqueeze(0).to(self.device)
                    embedding = self.resnet(face).detach().cpu().numpy().flatten()
                    embeddings.append(embedding)
                    valid_images += 1
                    
                except Exception as e:
                    print(f"❌ Error processing {image_name}: {str(e)}")
                    skipped_images += 1
            
            if embeddings:
                embeddings_array = np.array(embeddings)
                self.known_faces[person_name] = np.mean(embeddings_array, axis=0)
                print(f"📌 {person_name}: {valid_images} images processed, {skipped_images} skipped")
        
        print(f"✅ Loaded {len(self.known_faces)} people")
        return len(self.known_faces)
    
    def add_person(self, name, image_paths=None):
        """Add a new person to the database"""
        person_dir = os.path.join(self.known_faces_dir, name)
        
        # Create directory if it doesn't exist
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            print(f"✅ Created directory for {name}")
        
        # Add images if provided
        if image_paths:
            valid_images = 0
            for i, img_path in enumerate(image_paths):
                try:
                    # Check if face can be detected
                    img = Image.open(img_path).convert("RGB")
                    face = self.mtcnn(img, return_prob=False)
                    
                    if face is None:
                        print(f"❌ No face detected in {img_path}, skipping...")
                        continue
                    
                    # Save the image to person directory
                    new_filename = f"{name}_{i+1:03d}.jpg"
                    new_path = os.path.join(person_dir, new_filename)
                    shutil.copy(img_path, new_path)
                    valid_images += 1
                    
                except Exception as e:
                    print(f"❌ Error processing {img_path}: {str(e)}")
            
            print(f"✅ Added {valid_images} images for {name}")
        
        # Reload database to include new person
        self.load_known_faces()
        return True
    
    def remove_person(self, name):
        """Remove a person from the database"""
        person_dir = os.path.join(self.known_faces_dir, name)
        
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
            print(f"✅ Removed {name} from database")
            
            # Reload database
            self.load_known_faces()
            return True
        else:
            print(f"❌ Person {name} not found in database")
            return False
    
    def get_all_people(self):
        """Get list of all people in the database"""
        return list(self.known_faces.keys())
    
    def recognize_face(self, embedding, distance_threshold=1.0):
        """Find the best match for a face embedding"""
        min_dist = float("inf")
        best_match = "Unknown"
        
        for name, known_embedding in self.known_faces.items():
            dist = np.linalg.norm(embedding - known_embedding)
            if dist < min_dist:
                min_dist = dist
                best_match = name
        
        # Return unknown if distance is above threshold
        if min_dist > distance_threshold:
            return "Unknown", min_dist
        
        return best_match, min_dist