from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import base64


app = Flask(__name__)






# --- MoveNetモデルロード ---
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

KEYPOINT_EDGES = [
    (0, 1),(0, 2),(1, 3),(2, 4),(0, 5),(0, 6),
    (5, 7),(7, 9),(6, 8),(8, 10),(5, 6),
    (5, 11),(6, 12),(11, 12),(11, 13),(13, 15),
    (12, 14),(14, 16)
]

def preprocess_image(image):
    image = cv2.resize(image, (192,192))
    image = image.astype(np.int32)
    return np.expand_dims(image, axis=0)

def draw_keypoints_and_edges(image, keypoints, edges=None):
    image = image.copy()
    height, width, _ = image.shape
    for i in range(keypoints.shape[0]):
        y, x = keypoints[i]
        cv2.circle(image, (int(x*width), int(y*height)), 5, (0,255,0), -1)
    if edges:
        for (i,j) in edges:
            y1, x1 = keypoints[i]
            y2, x2 = keypoints[j]
            cv2.line(image, (int(x1*width), int(y1*height)), (int(x2*width), int(y2*height)), (255,0,0), 2)
    return image

def draw_golden_line_auto(image, keypoints):
    height, width, _ = image.shape
    ear_x = int(keypoints[0][0]*width)
    cv2.line(image, (ear_x, 0), (ear_x, height), (0, 255, 255), 2)

    target_indices = [5, 11, 13, 15]
    distances = []
    for idx in target_indices:
        dx = abs(keypoints[idx][0]*width - ear_x)
        distances.append(dx)
        cv2.putText(image, f"{dx:.1f}px", 
                    (int(keypoints[idx][0]*width)+5, int(keypoints[idx][1]*height)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    score = max(0, 100 * (1 - np.mean(distances)/(width/10)))
    cv2.putText(image, f"Score: {score:.1f}%", (20,40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return image, score


@app.route("/")
def home():
    return "Hello, Flask!"



@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    img_data = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_tensor = preprocess_image(img)
    outputs = movenet(tf.convert_to_tensor(input_tensor))
    keypoints_with_scores = outputs['output_0'].numpy()[0,0,:,:]

    image_with_kp = draw_keypoints_and_edges(img, keypoints_with_scores[:,:2], KEYPOINT_EDGES)
    result_img, score = draw_golden_line_auto(image_with_kp, keypoints_with_scores)

    _, buffer = cv2.imencode(".png", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    img_str = "data:image/png;base64," + base64.b64encode(buffer).decode()
    return jsonify({"result": img_str, "score": score})

if __name__ == "__main__":
    app.run(debug=True)
