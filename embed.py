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
import logging
from tqdm import tqdm

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

        coordinates = tuple([c for c in box])

        crop = image.crop(coordinates)
        if self.transform:
            crop = self.transform(crop)
        return crop, image_path, box

def main(image_dir, json_dir, batch_size, n_workers, device):
    mp.set_sharing_strategy('file_system')
    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')

    output_file = 'embeddings/' + image_dir.replace('datasets/', '') + '.pkl'

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

    # Process images and extract embeddings
    all_embeddings = []
    for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        crops, image_paths, boxes = batch
        crops = crops.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                embeddings = model(crops).squeeze()
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)  # Handle case when there's only one embedding

        # Move embeddings to CPU before appending
        embeddings = embeddings.cpu()

        for j, embedding in enumerate(embeddings):
            all_embeddings.append({
                "image_path": image_paths[j],
                "box": boxes[j].numpy().tolist(),
                "embedding": embedding.numpy().tolist()
            })

        # Save intermediate results to the output file to avoid OOM
        # I think the memory issue here was the storing of all the embeddings at once.
        #to try to avoid tranferring data from GPU to CPU too often, I empirically choes to write 100 embeddings at a time
        # embeddings have dim=2048, as the last ResNet layer is 2048 neurons.
        #however, the dimentsionality is very high and might be impractical for later processung steps
        if (i + 1) % 100 == 0:  # Adjust the frequency as needed
            with open(output_file, 'ab') as f:
                pickle.dump(all_embeddings, f)
            all_embeddings = []  # Clear the list to free memory

    # Save any remaining embeddings to the output file
    if all_embeddings:
        with open(output_file, 'ab') as f:
            pickle.dump(all_embeddings, f)
        print('Embeddings written!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings from bounding boxes using ResNet50.")
    parser.add_argument('--image_dir', type=str, default='datasets/brueg_small', help='Directory containing images.')
    parser.add_argument('--json_dir', type=str, default='object_detection/brueg_small_detections', help='Directory containing JSON files with bounding boxes.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for processing images.')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of worker processes to use for data loading.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu or cuda).')

    args = parser.parse_args()
    main(args.image_dir, args.json_dir, args.batch_size, args.n_workers, args.device)
