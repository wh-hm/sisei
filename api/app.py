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

PART_NAMES = {
    0: "鼻", 1: "左目", 2: "右目", 3: "左耳", 4: "右耳",
    5: "左肩", 6: "右肩", 7: "左肘", 8: "右肘",
    9: "左手首", 10: "右手首", 11: "左腰", 12: "右腰",
    13: "左膝", 14: "右膝", 15: "左足首", 16: "右足首"
}

COMMENTS = {
    "腰": "腰の位置がゴールデンラインから外れています。骨盤の傾きに注意しましょう。",
    "肩": "肩のラインが崩れています。左右の高さを意識して整えましょう。",
    "足首": "足首の位置が不安定です。立ち姿勢を見直してみましょう。",
}

REQUIRED_PARTS = {
    "肩": [5, 6],
    "腰": [11, 12],
    "足首": [15, 16],
    "耳": [3, 4]
}

def preprocess_image(image):
    image = cv2.resize(image, (192,192))
    image = image.astype(np.int32)
    return np.expand_dims(image, axis=0)

def draw_keypoints_and_edges(image, keypoints, edges=None):
    image = image.copy()
    height, width, _ = image.shape
    return image  # 関節描画は省略中

def draw_golden_line_auto(image, keypoints):
    height, width, _ = image.shape
    nose_x = keypoints[0][1]
    left_shoulder_x = keypoints[5][1]
    right_shoulder_x = keypoints[6][1]

    if nose_x < min(left_shoulder_x, right_shoulder_x):
        base_idx = 4
        indices = [4, 6, 12, 16]
    elif nose_x > max(left_shoulder_x, right_shoulder_x):
        base_idx = 3
        indices = [3, 5, 11, 15]
    else:
        base_idx = 0
        indices = [3, 5, 11, 15]

    ear_x = int(keypoints[base_idx][1] * width)
    cv2.line(image, (ear_x, 0), (ear_x, height), (255, 255, 0), 2)

    x_coords = [keypoints[i][1] * width for i in indices]
    y_coords = [keypoints[i][0] * height for i in indices]

    for i in range(len(indices) - 1):
        pt1 = (int(x_coords[i]), int(y_coords[i]))
        pt2 = (int(x_coords[i + 1]), int(y_coords[i + 1]))
        cv2.line(image, pt1, pt2, (255, 0, 0), 2)

    avg_x = np.mean(x_coords)
    dx_list = [abs(x - avg_x) for x in x_coords]
    score = max(0, 100 * (1 - np.mean(dx_list) / (width / 10)))

    max_dx = max(dx_list)
    max_idx = indices[dx_list.index(max_dx)]

    if max_idx in [11, 12]:
        part = "腰"
    elif max_idx in [5, 6]:
        part = "肩"
    elif max_idx in [15, 16]:
        part = "足首"
    else:
        part = "不明"

    worst_part = part


    if score >= 95:
        hanntei = "ゴールデンラインです。"
        comment = "ゴールデンラインと言っていいでしょう。全体的に良好な姿勢です。"
    else:
        hanntei = "ゴールデンラインではありません。"
        comment = COMMENTS.get(part, "姿勢に改善の余地があります。")

    #cv2.putText(image, f"姿勢Score: {float(score):.1f}%", (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv2.putText(image, comment, (20, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return image, float(score), comment, worst_part,hanntei

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        print("⚠️ JSONデータが不正です:", data)
        return jsonify({"error": "No image received"}), 400

    # base64デコード処理
    try:
        img_data = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("⚠️ 画像デコード失敗:", e)
        return jsonify({"error": "Image decode failed"}), 500

    # 姿勢推定
    input_tensor = preprocess_image(img)
    outputs = movenet(tf.convert_to_tensor(input_tensor))
    keypoints_with_scores = outputs['output_0'].numpy()[0, 0, :, :]

    # 必須部位の検出チェック
    missing_parts = []
    for part_name, indices in REQUIRED_PARTS.items():
        if not any(keypoints_with_scores[i][2] > 0.3 for i in indices):
            missing_parts.append(part_name)

    # キーポイント描画
    image_with_kp = draw_keypoints_and_edges(img, keypoints_with_scores[:, :2], KEYPOINT_EDGES)

    # golden line 描画 or 通常描画
    if missing_parts:
        result_img = image_with_kp
        score = "不明"
        comment = "肩、腰、足首、耳。画像にしっかり映るようにしてください。"
        worst_part = "不明"
        hanntei = "不明"
        missing_msg = "以下の部位が検出できませんでした: " + "、".join(missing_parts) + "。画像にしっかり映るようにしてください。"
    else:
        result_img, score, comment, worst_part, hanntei = draw_golden_line_auto(image_with_kp, keypoints_with_scores)
        missing_msg = ""

    # 画像をbase64に変換
    _, buffer = cv2.imencode(".png", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    img_str = "data:image/png;base64," + base64.b64encode(buffer).decode()

    # 結果を返す
    return jsonify({
        "result": img_str,
        "score": score,
        "comment": comment,
        "worst_part": worst_part,
        "hanntei": hanntei,
        "missing": missing_msg
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
