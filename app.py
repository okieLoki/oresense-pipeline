from flask import Flask, request, jsonify
import requests
import joblib
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

app = Flask(__name__)

loaded_model = joblib.load("LR_final.joblib")


class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 64 * 64, 64)
        self.fc2 = nn.Linear(64 + 1, 4)

    def forward(self, x, silica_impurity):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        combined_features = torch.cat((x, silica_impurity.unsqueeze(1)), dim=1)
        x = self.fc2(combined_features)
        return x


image_cnn = ImageCNN()

model_path = "Final_model.pth"
image_cnn.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
image_cnn.eval()

class_names = ["Grade A", "Grade B", "Grade C", "Grade D"]

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)


@app.route("/", methods=["GET"])
def index():
    return "This is the flask pipeline!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        image_url = data["image_url"]
        silica_impurity = float(data["silica_impurity"])
        other_params = data["other_params"]

        response = requests.get(image_url)
        with open("./photo.jpg", "wb") as file:
            file.write(response.content)
        img_path = "./photo.jpg"
        image = Image.open(img_path)
        input_image = transform(image).unsqueeze(0)

        with torch.no_grad():
            predicted_labels = image_cnn(input_image, torch.Tensor([silica_impurity]))

        probabilities = F.softmax(predicted_labels, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()

        feature_names = [
            "% Iron Feed",
            "% Silica Feed",
            "Starch Flow",
            "Amina Flow",
            "Ore Pulp Flow",
            "Ore Pulp pH",
            "Ore Pulp Density",
        ]
        input_data = [other_params[feature] for feature in feature_names]
        data = pd.DataFrame([input_data], columns=feature_names)
        linear_regression_prediction = loaded_model.predict(data)[0]

        response = {
            "Status": "success",
            "Predicted Grade": class_names[predicted_class_index],
            "Confidence": 100 * probabilities.squeeze()[predicted_class_index].item(),
            "Linear Regression Prediction": linear_regression_prediction,
            "Other Parameters": other_params,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
