import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import argparse
import pickle
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm
import uuid

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
                        box = np.array([box['x1'], box['y1'], box['x2'], box['y2']])  # Directly create numpy array
                        self.data.append((image_path, box))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, box = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = box
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        
        coordinates = (x1, y1, x2, y2)
        crop = image.crop(coordinates)
        #print(crop)

        if self.transform:
            crop = self.transform(crop)
        return crop, image_path, box

def main(image_dir, json_dir, batch_size, n_workers, device):
    mp.set_sharing_strategy('file_system')
    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')

    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join('embeddings', unique_id)
    os.makedirs(temp_dir)

    output_file = 'embeddings/' + image_dir.replace('datasets/', '') + '_local_ResNet50_avg.pkl'

    # Preprocessing transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset and dataloader
    dataset = BoundingBoxDataset(image_dir, json_dir, transform=transform)
    print('Total bounding boxes: ', len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=False)

    # Load pre-trained ResNet model and modify it to output embeddings
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    model.to(device)

    temp_files = []
    for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        crops, image_paths, boxes = batch
        crops = crops.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                embeddings = model(crops).squeeze()
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)  # Handle case when there's only one embeddings
        
        # Move embeddings to CPU before appending
        embeddings = embeddings.cpu()

        batch_embeddings = []
        for j, embedding in enumerate(embeddings):
            #print(embedding)
            batch_embeddings.append({
                "image_path": image_paths[j],
                "box": boxes[j],
                "embedding": embedding
            })

        # Save the current batch embeddings to a temporary file
        temp_file = os.path.join(temp_dir, f'temp_batch_{i}.pkl')
        with open(temp_file, 'wb') as f:
            pickle.dump(batch_embeddings, f)
        temp_files.append(temp_file)

    # Combine all temporary files into the final output file
    all_embeddings = []
    for temp_file in temp_files:
        with open(temp_file, 'rb') as f:
            batch_embeddings = pickle.load(f)
            all_embeddings.extend(batch_embeddings)
        os.remove(temp_file)  # Clean up the temporary file

    with open(output_file, 'wb') as f:
        pickle.dump(all_embeddings, f)

    os.rmdir(temp_dir)  # Clean up the temporary directory
    print('Embeddings written in', output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings from bounding boxes using ResNet50.")
    parser.add_argument('--image_dir', type=str, default='datasets/brueg_small', help='Directory containing images.')
    parser.add_argument('--json_dir', type=str, default='object_detection/brueg_small_detections', help='Directory containing JSON files with bounding boxes.')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size for processing images.')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of worker processes to use for data loading.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu or cuda).')

    args = parser.parse_args()
    main(args.image_dir, args.json_dir, args.batch_size, args.n_workers, args.device)
