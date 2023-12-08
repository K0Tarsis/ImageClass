from PIL import Image
from flask import Flask, request, jsonify
from clip_vit import predict_classes_clip

app = Flask(__name__)
app.secret_key = "secret key12333"


@app.route('/', methods=['POST'])
def main():

    classes = [
        "a photo of a electronics", "a photo of a clothes", "a photo of a food", "a photo of a shoes"
    ]

    img = Image.open(request.files.get('image').stream)
    # classes = predict_classes(img)
    prediction = predict_classes_clip(classes, img)
    # return classify(classes)
    return prediction


if __name__ == '__main__':
    app.run()
