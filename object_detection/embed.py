import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import argparse

class BoundingBoxDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.data = []

        # Prepare the dataset by loading all bounding boxes
        for image_name in os.listdir(image_dir):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(image_dir, image_name)
                json_path = os.path.join(json_dir, image_name.replace('.jpg', '.json'))

                with open(json_path, 'r') as f:
                    objects = json.load(f)
                    for obj in objects:
                        box = obj['box']
                        box_tuple = (box['x1'], box['y1'], box['x2'], box['y2'])
                        self.data.append((image_path, box_tuple))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, box = self.data[idx]
        image = Image.open(image_path).convert('RGB')

        x1, y1, x2, y2 = box
        crop = image.crop((x1, y1, x2, y2))
        if self.transform:
            crop = self.transform(crop)

        return crop, image_path, box


def main(image_dir, json_dir, batch_size, output_file):
    # Preprocessing transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and dataloader
    dataset = BoundingBoxDataset(image_dir, json_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # Load pre-trained ResNet model and modify it to output embeddings
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # Process images and extract embeddings
    all_embeddings = []
    for batch in dataloader:
        crops, image_paths, boxes = batch
        with torch.no_grad():
            embeddings = model(crops).squeeze()
            if len(embeddings.shape) == 1:
                embeddings = embeddings.unsqueeze(0)  # Handle case when there's only one embedding
                
            for i, embedding in enumerate(embeddings):
                
                all_embeddings.append({
                    "image_path": image_paths[i],
                    "box": boxes[i].tolist(),
                    "embedding": embedding.tolist() 
                })

    # Save embeddings to the output file
    with open(output_file, 'w') as f:
        json.dump(all_embeddings, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings from bounding boxes using ResNet50.")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--json_dir', type=str, required=True, help='Directory containing JSON files with bounding boxes.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing images.')
    parser.add_argument('--output_file', type=str, required=True, help='Output file to save embeddings.')

    args = parser.parse_args()
    main(args.image_dir, args.json_dir, args.batch_size, args.output_file)
