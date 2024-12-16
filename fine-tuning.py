import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch.nn as nn
import json

class RelationshipDataset(Dataset):
    def __init__(self, data, subject_to_idx, relationship_to_idx):
        self.data = []
        for image_data in data:
            for pred in image_data['predictions']:
                subject = pred['subject']['class']
                relation = pred['relation']['class']
                
                # Only include if subject and relation are in our vocabulary
                if (subject in subject_to_idx and 
                    relation in relationship_to_idx):
                    self.data.append({
                        'subject': subject_to_idx[subject],
                        'relation': relationship_to_idx[relation]
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'subject': torch.tensor(self.data[idx]['subject']),
            'relation': torch.tensor(self.data[idx]['relation'])
        }
    
class RelationshipModel(nn.Module):
    def __init__(self, num_subjects, num_relations):
        super().__init__()
        self.embedding_dim = 256
        
        # Embedding layers
        self.subject_embedding = nn.Embedding(num_subjects, self.embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, self.embedding_dim)
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8),
            num_layers=6
        )
        
        # Output layers
        self.fc = nn.Linear(self.embedding_dim, num_relations)

    def forward(self, subject, relation):
        # Embed inputs
        subj_emb = self.subject_embedding(subject)
        rel_emb = self.relation_embedding(relation)
        
        # Combine embeddings bằng cách lấy trung bình
        x = (subj_emb + rel_emb) / 2
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Get predictions
        out = self.fc(x)
        return out
    
def train_model(model, train_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            subject = batch['subject'].to(device)
            relation = batch['relation'].to(device)
            
            # Forward pass
            outputs = model(subject, relation)
            loss = criterion(outputs, relation)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Load and analyze data
all_predicts = []
with open("Neo4j/all_predictions.json") as f:
    all_predicts = json.load(f)

# Dictionary to store subject frequencies per image
image_subjects = {}
# Counter for overall statistics
subject_counter = Counter()
relationship_counter = Counter()
object_counter = Counter()  # Thêm counter cho objects

# Analyze data
for image_data in all_predicts:
    image_path = image_data['image_path']
    image_subjects[image_path] = Counter()
    
    for pred in image_data['predictions']:
        # Count subjects per image
        subject = pred['subject']['class']
        object_class = pred['object']['class']  # Lấy object class
        image_subjects[image_path][subject] += 1
        
        # Overall statistics
        subject_counter[subject] += 1
        relationship_counter[pred['relation']['class']] += 1
        object_counter[object_class] += 1  # Đếm objects

# Get most common items
TOP_K = 20
most_common_subjects = subject_counter.most_common(TOP_K)
most_common_relationships = relationship_counter.most_common(TOP_K)
most_common_objects = object_counter.most_common(TOP_K)  # Lấy top objects

# Print statistics
print("\nMost common subjects (main focus of images):")
for subject, count in most_common_subjects:
    print(f"{subject}: {count}")

print("\nMost common relationships:")
for rel, count in most_common_relationships:
    print(f"{rel}: {count}")

print("\nMost common objects:")  # In thống kê về objects
for obj, count in most_common_objects:
    print(f"{obj}: {count}")

# Print subjects that appear frequently in individual images
print("\nSubjects that appear multiple times in single images:")
threshold = 2  # Minimum occurrences in a single image
frequent_subjects_in_images = set()

for image_path, subj_counter in image_subjects.items():
    for subj, count in subj_counter.items():
        if count >= threshold:
            frequent_subjects_in_images.add(subj)
            print(f"Image {image_path}: {subj} appears {count} times")

# Create mapping dictionaries for training (focused on subjects)
subject_to_idx = {subject: idx for idx, (subject, _) in enumerate(most_common_subjects)}
relationship_to_idx = {rel: idx for idx, (rel, _) in enumerate(most_common_relationships)}

# Reverse mappings
idx_to_subject = {idx: subject for subject, idx in subject_to_idx.items()}
idx_to_relation = {idx: rel for rel, idx in relationship_to_idx.items()}

# Initialize dataset with mappings
dataset = RelationshipDataset(all_predicts, subject_to_idx, relationship_to_idx)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model with updated vocabulary sizes
model = RelationshipModel(
    num_subjects=len(subject_to_idx),
    num_relations=len(relationship_to_idx)
)

# Training
train_model(model, train_loader)

def predict_relationship(model, subject):
    model.eval()
    with torch.no_grad():
        # Convert input to tensor
        subject_tensor = torch.tensor([subject_to_idx[subject]])
        
        # Predict
        output = model(subject_tensor, None)
        _, predicted = torch.max(output.data, 1)
        
        return idx_to_relation[predicted.item()]