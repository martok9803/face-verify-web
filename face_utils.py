# face_utils.py
import os
import cv2
import numpy as np
import dlib
from deepface import DeepFace

OUTPUT_DIR_MATCHED = "output_images/matched"
OUTPUT_DIR_ROTATED = "output_images/rotated"

def ensure_dirs():
    os.makedirs(OUTPUT_DIR_MATCHED, exist_ok=True)
    os.makedirs(OUTPUT_DIR_ROTATED, exist_ok=True)

def load_image(image_path: str):
    print(f"[load_image] {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img

def rotate_image(image, angle: float):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0]); sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, M, (new_w, new_h))

def detect_face(detector, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) > 0:
        print(f"[detect_face] found {len(faces)} face(s)")
        return faces[0], image
    print("[detect_face] no faces")
    return None, image

def rotate_and_detect_face(detector, image, tag: str):
    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(image, angle)
        cv2.imwrite(f"{OUTPUT_DIR_ROTATED}/rotated_{tag}_{angle}.jpg", rotated)
        face, rotated_img = detect_face(detector, rotated)
        if face is not None:
            print(f"[rotate_and_detect_face] success at {angle}° for {tag}")
            return face, rotated_img
    print(f"[rotate_and_detect_face] no face found for {tag}")
    return None, None

def extract_face(face, image):
    h, w = image.shape[:2]
    top = max(0, face.top())
    left = max(0, face.left())
    bottom = min(h, face.bottom())
    right = min(w, face.right())
    if bottom <= top or right <= left:
        print("[extract_face] invalid crop bounds")
        return np.array([])
    return image[top:bottom, left:right]

def save_face(face_image, filename: str):
    if face_image.size == 0:
        print(f"[save_face] empty face image: {filename}")
        return False
    path = f"{OUTPUT_DIR_MATCHED}/{filename}"
    ok = cv2.imwrite(path, face_image)
    print(f"[save_face] saved: {path} -> {ok}")
    return ok

def concatenate_and_save_faces(face_image1, face_image2, filename: str):
    if face_image1.size == 0 or face_image2.size == 0:
        print("[concatenate] one or both faces empty")
        return False
    a = cv2.resize(face_image1, (500, 500))
    b = cv2.resize(face_image2, (500, 500))
    if a.dtype != b.dtype:
        print("[concatenate] dtype mismatch")
        return False
    combined = cv2.hconcat([a, b])
    ok = cv2.imwrite(filename, combined)
    print(f"[concatenate] saved: {filename} -> {ok}")
    return ok

def compare_faces():
    id_path = f"{OUTPUT_DIR_MATCHED}/id_face.jpg"
    photo_path = f"{OUTPUT_DIR_MATCHED}/photo_face.jpg"
    print("[compare_faces] verifying…")
    result = DeepFace.verify(id_path, photo_path, enforce_detection=False)
    print(f"[compare_faces] verified={result.get('verified')} distance={result.get('distance')} threshold={result.get('threshold')}")
    return result
