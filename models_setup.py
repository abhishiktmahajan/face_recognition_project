import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
print(torch.cuda.is_available())

# Set device: use GPU if available, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize MTCNN for face detection.
# keep_all=True returns all detected faces.
mtcnn = MTCNN(keep_all=True, device=device)
print("MTCNN initialized.")

# Initialize FaceNet (InceptionResnetV1) for face recognition.
# The model is pretrained on the vggface2 dataset.
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("FaceNet (InceptionResnetV1) model loaded.")
