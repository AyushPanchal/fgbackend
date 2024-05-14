from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

cassava_model = load_model("models/sweet-potato-effnet.h5")
cashew_model = load_model("models/cashew-effnet.h5")
maize_model = load_model("models/maize-effnet.h5")
tomato_model = load_model("models/tomato-effnet.h5")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/predict", methods=["POST"])
def dummy_response():
    labels = ['green mite', 'bacterial blight', 'mosaic', 'healthy', 'brown spot']
    img = cv2.imread("dataset/Cassava brown spot/brown spot3_.jpg")

    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencv_image, (224, 224))
    img = img.reshape((1, 224, 224, 3))
    prediction = cassava_model.predict(img)
    prediction = np.argmax(prediction, axis=1)[0]
    print(labels[prediction])

    result = {'prediction': prediction, "info": "info"}

    return jsonify({"prediction": "Cashew Gummosis",
                    "info": "Gummosis infection in cashew plants typically presents as dark cankers spread throughout the trunk or woody branches that occasionally crack and ooze a transparent resin-like gum."})


@app.route("/predict_cashew", methods=["POST"])
def predict_cashew():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    print(file)
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save uploaded image to server
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    labels = ['anthracnose', 'gumosis', 'healthy', 'leaf miner', "red rust"]
    # Preprocess the image
    img = cv2.imread(filename)
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencv_image, (224, 224))
    img = img.reshape((1, 224, 224, 3))
    prediction = cashew_model.predict(img)
    prediction = np.argmax(prediction, axis=1)[0]
    print(labels[prediction])
    class_info = [
        "Cashew anthracnose is a fungal disease that affects cashew trees and their nuts. It is caused by several species of fungi in the genus Colletotrichum, particularly Colletotrichum gloeosporioides and Colletotrichum acutatum.",
        "Cashew gummosis is a plant disease that affects cashew trees, characterized by the formation of gummy exudates or resinous substances on the bark, branches, or wounds of the tree.",
        "Healthy cashew",
        "The cashew leaf miner is a pest that can cause significant damage to cashew trees by tunneling and feeding on the leaves",
        'Cashew red rust, also known as cashew rust or cashew leaf rust, is a fungal disease that affects cashew trees. It is caused by the fungus Puccinia kuehnii. This disease primarily targets the leaves of cashew trees and can lead to significant economic losses if not managed effectively']
    print(f"Prediction : {labels[prediction]}")
    print(prediction)
    result = {'prediction': labels[prediction], "info": class_info[prediction]}  # Thresholding at 0.5 probability
    print(result)
    return jsonify(result), 200


@app.route("/predict_cassava", methods=["POST"])
def predict_cassava():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    print(file)
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save uploaded image to server
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    labels = ['green mite', 'bacterial blight', 'mosaic', 'healthy', 'brown spot']
    # Preprocess the image
    img = cv2.imread(filename)
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencv_image, (224, 224))
    img = img.reshape((1, 224, 224, 3))
    prediction = cassava_model.predict(img)
    prediction = np.argmax(prediction, axis=1)[0]
    print(labels[prediction])
    class_info = [
        "Green mite, also known as cassava green mite, infests cassava plants, causing leaf discoloration and drop, impacting photosynthesis and plant health.",
        "Cassava bacterial blight is caused by bacteria like Xanthomonas axonopodis pv. manihotis, leading to leaf spots, stem lesions, and blight, requiring management to prevent rapid spread.",
        "Cassava mosaic disease is a viral disease causing mosaic-like patterns on leaves, spread by infected plant material and whiteflies, potentially stunting plant growth and reducing yield.",
        "Represents a disease-free, vigorous cassava plant with green and intact leaves, essential for optimal growth and production.",
        "Cassava brown spot, caused by fungi like Pseudocercospora spp., results in small, brown to black spots on leaves, requiring sanitation and fungicidal treatment."
    ]

    print(f"Prediction : {labels[prediction]}")
    print(prediction)
    result = {'prediction': labels[prediction], "info": class_info[prediction]}  # Thresholding at 0.5 probability
    print(result)
    return jsonify(result), 200


@app.route("/predict_maize", methods=["POST"])
def predict_maize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    print(file)
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save uploaded image to server
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    labels = ['leaf beetle',
              'healthy',
              'leaf blight',
              'grasshoper',
              'fall armyworm',
              'streak virus',
              'leaf spot']
    # Preprocess the image
    img = cv2.imread(filename)
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencv_image, (224, 224))
    img = img.reshape((1, 224, 224, 3))
    prediction = maize_model.predict(img)
    prediction = np.argmax(prediction, axis=1)[0]
    print(labels[prediction])
    class_info = [
        "Maize leaf beetle is a common pest that feeds on maize leaves, causing damage such as holes and skeletonization, which can weaken the plant and reduce yield if infestations are severe.",
        "Represents a healthy maize plant with vigorous growth, characterized by green and intact leaves, essential for optimal development and yield.",
        "Maize leaf blight is a fungal disease caused by pathogens like Exserohilum turcicum, resulting in irregular lesions and blighting of maize leaves, which can impact overall plant health and grain production.",
        "Maize grasshopper is a significant pest that consumes maize foliage, leading to defoliation and potentially affecting the plant's ability to photosynthesize and produce grain.",
        "Maize fall armyworm is a destructive pest that can cause extensive damage by feeding on maize leaves and tunneling into maize ears, leading to yield losses if not controlled.",
        "Maize streak virus is a viral disease transmitted by leafhoppers, resulting in streaks or stripes on maize leaves and stunted growth, ultimately reducing maize yield and quality.",
        "Maize leaf spot, caused by fungi like Bipolaris spp. or Cercospora spp., leads to the development of small, dark spots on maize leaves, affecting photosynthesis and potentially reducing grain yield."
    ]

    print(f"Prediction : {labels[prediction]}")
    print(prediction)
    result = {'prediction': labels[prediction], "info": class_info[prediction]}  # Thresholding at 0.5 probability
    print(result)
    return jsonify(result), 200


@app.route("/predict_tomato", methods=["POST"])
def predict_tomato():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    print(file)
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save uploaded image to server
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    labels = ['verticulium vilt', 'septoria leaf spot', 'healthy', 'leaf blight', 'leaf curl']
    # Preprocess the image
    img = cv2.imread(filename)
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencv_image, (224, 224))
    img = img.reshape((1, 224, 224, 3))
    prediction = tomato_model.predict(img)
    prediction = np.argmax(prediction, axis=1)[0]
    print(labels[prediction])
    class_info = [
        "Tomato Verticillium wilt is a fungal disease caused by Verticillium species, which infects tomato plants through the roots and causes wilting of leaves and yellowing of lower leaves. It can lead to reduced yield and plant death in severe cases.",
        "Tomato Septoria leaf spot is a common fungal disease caused by Septoria lycopersici, characterized by small dark spots with a lighter center on tomato leaves. It can defoliate the plant and reduce fruit production if left unchecked.",
        "Represents a healthy tomato plant with vigorous growth, characterized by green and disease-free leaves, essential for optimal fruit development and yield.",
        "Tomato leaf blight, often caused by fungal pathogens like Alternaria solani or Phytophthora infestans, results in large irregular lesions on tomato leaves, leading to defoliation and reduced fruit quality.",
        "Tomato leaf curl is a viral disease transmitted by whiteflies, causing tomato leaves to curl upward and become distorted. Infected plants may have reduced fruit set and yield."
    ]

    print(f"Prediction : {labels[prediction]}")
    print(prediction)
    result = {'prediction': labels[prediction], "info": class_info[prediction]}  # Thresholding at 0.5 probability
    print(result)
    return jsonify(result), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
