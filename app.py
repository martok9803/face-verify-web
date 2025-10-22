# app.py
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename

import dlib
from face_utils import (
    ensure_dirs, load_image, rotate_and_detect_face, extract_face,
    save_face, concatenate_and_save_faces, compare_faces
)

#Configs
UPLOAD_FOLDER = "uploads"
STATIC_OUTPUTS = os.path.join("static", "outputs")
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

#Please make sure this folder exist (might automate it later)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_OUTPUTS, exist_ok=True)
ensure_dirs()

# aaaaa dlib detector
DETECTOR = dlib.get_frontal_face_detector()

@app.route('/output_images/<path:filename>')
def output_images(filename):
    return send_from_directory('output_images', filename)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify():
    id_file = request.files.get("id_image")
    photo_file = request.files.get("photo_image")

    if not id_file or not photo_file:
        return render_template("result.html", error="Please select both images.")

    if id_file.filename == "" or photo_file.filename == "":
        return render_template("result.html", error="One of the images has no filename.")

    ext1 = id_file.filename.rsplit(".", 1)[-1].lower()
    ext2 = photo_file.filename.rsplit(".", 1)[-1].lower()
    if ext1 not in ALLOWED_EXT or ext2 not in ALLOWED_EXT:
        return render_template("result.html", error="Only JPG, JPEG, PNG, WEBP are supported.")

    # Here we are saving the files
    id_name = secure_filename(id_file.filename)
    photo_name = secure_filename(photo_file.filename)
    id_path = os.path.join(app.config["UPLOAD_FOLDER"], id_name)
    photo_path = os.path.join(app.config["UPLOAD_FOLDER"], photo_name)
    id_file.save(id_path)
    photo_file.save(photo_path)

    try:
        id_img = load_image(id_path)
        id_face, id_img_rot = rotate_and_detect_face(DETECTOR, id_img, "id")
        if id_face is None:
            return render_template("result.html", error="No face detected in ID image.")
        id_face_img = extract_face(id_face, id_img_rot)
        save_face(id_face_img, "id_face.jpg")

        ph_img = load_image(photo_path)
        ph_face, ph_img_rot = rotate_and_detect_face(DETECTOR, ph_img, "photo")
        if ph_face is None:
            return render_template("result.html", error="No face detected in Photo image.")
        ph_face_img = extract_face(ph_face, ph_img_rot)
        save_face(ph_face_img, "photo_face.jpg")

        res = compare_faces()
        out_name = f"{uuid.uuid4().hex}.jpg"
        out_path = os.path.join(STATIC_OUTPUTS, out_name)
        concatenate_and_save_faces(id_face_img, ph_face_img, out_path)

        message = "✅ The faces MATCH." if res.get("verified") else "❌ Faces DO NOT match."
        details = {
            "distance": round(float(res.get("distance", 0)), 4),
            "threshold": round(float(res.get("threshold", 0)), 4),
            "model": res.get("model") or "VGG-Face"
        }

        out_img_url = url_for("static", filename=f"outputs/{out_name}")
        return render_template("result.html", message=message, details=details, out_img_url=out_img_url)

    except Exception as e:
        return render_template("result.html", error=f"Processing error: {e}")

if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    app.run(host="0.0.0.0", port=5000, debug=True)
