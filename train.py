import os
import cv2
import numpy as np

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_ids = {}
    current_id = 0
    
    # Path folder dataset train
    train_dir = "dataset/train"
    
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "_")
                
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                
                id_ = label_ids[label]
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    labels.append(id_)
                    
    # Latih model dan simpan (trainer.yml)
    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")
    print("Pelatihan selesai! Label terdaftar:", label_ids)

if __name__ == "__main__":
    train_recognizer()