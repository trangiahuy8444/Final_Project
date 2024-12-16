
import json
from config import *
from db import *
from RelTR.inference import *
# reset_database()
"""
with open('image_data.json') as f:
    image_data = json.load(f)
list_img_path = []
for image_info in image_data[:50]:
    list_img_path.append(image_info['url'])

model = load_model("./RelTR/ckpt/checkpoint0149.pth")
all_predictions = []

# Loop over all images in the dataset
for index_img in list_img_path[:50]:
    predictions = predict(index_img, model)

    # Add the predictions for this image to the overall list
    all_predictions.append({
        "image_path": index_img,
        "predictions": predictions
    })

# Save all predictions to a JSON file
with open("all_predictions.json", "w") as f:
    json.dump(all_predictions, f, indent=4)
"""

with open("all_predictions.json", "r") as f:
    data = json.load(f)

save_to_neo4j_with_img_path(data)