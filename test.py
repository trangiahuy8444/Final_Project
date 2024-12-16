import gradio as gr
from PIL import Image
import json
from config import *
from db import *
from RelTR.inference import *
from collections import Counter

# all_predicts = []
# with open("all_predictions.json") as f:
#     all_predicts = json.load(f)

# images_path = []
# subjects = []
# relationships = []
# objects = []

# for i in range(len(all_predicts)):
#     images_path.append(all_predicts[i]['image_path'])
#     for j in all_predicts[i]['predictions']:
#         subjects.append(j['subject']['class'])
#         relationships.append(j['relation']['class'])
#         objects.append(j['object']['class'])
    
# # Thống kê số lượng object
# subject_counter = Counter()
# relationship_counter = Counter()
# object_counter = Counter()

# # Lọc ra các object có tần suất nhiều nhất
# subject_counter.update(subjects)
# relationship_counter.update(relationships)
# object_counter.update(objects)

# # Lọc ra các subject có tần suất nhiều nhất
# most_common_subjects = subject_counter.most_common()
# print("Most common subjects:", most_common_subjects, '\n')

# # Lọc ra các relationship có tần suất nhiều nhất
# most_common_relationships = relationship_counter.most_common()
# print("Most common relationships:", most_common_relationships, '\n')

# # Lọc ra các object có tần suất nhiều nhất
# most_common_objects = object_counter.most_common()
# print("Most common objects:", most_common_objects, '\n')

# num_subjects = len(most_common_subjects)
# num_relations = len(most_common_relationships)
# num_objects = len(most_common_objects)

# print(num_subjects)
# print(num_relations)
# print(num_objects)

model = load_model("./RelTR/ckpt/checkpoint0149.pth")
img_path = './img_test/1.jpg'
predictions = predict(img_path,model)

print(predictions)