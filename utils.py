# embedding_utils.py

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pickle
import torch.multiprocessing as mp
from tqdm import tqdm
import uuid

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('jpg', 'jpeg', 'png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return img_path, image

def embed_global(image_dir, batch_size=1, n_workers=0, device='cpu'):
    mp.set_sharing_strategy('file_system')
    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')

    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join('embeddings', unique_id)
    os.makedirs(temp_dir)

    output_file = 'embeddings/' + image_dir.replace('datasets/', '') + '_global.pkl'
    output_dir = os.path.dirname(output_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=True)

    # Load pre-trained ResNet model and modify it to output embeddings
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)

    temp_files = []
    for i, (image_paths, imgs) in enumerate(tqdm(dataloader, desc="Processing batches")):
        imgs = imgs.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                embeddings = model(imgs).squeeze()
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)  # Handle case when there's only one embedding

        embeddings = embeddings.cpu()

        batch_embeddings = []
        for j, embedding in enumerate(embeddings):
            batch_embeddings.append({
                "image_path": image_paths[j],
                "embedding": embedding
            })

        temp_file = os.path.join(temp_dir, f'temp_batch_{i}.pkl')
        with open(temp_file, 'wb') as f:
            pickle.dump(batch_embeddings, f)
        temp_files.append(temp_file)

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

    def embed_one(img_path_or_pil, device='cuda'):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load pre-trained ResNet model and modify it to output embeddings
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(device)

        if isinstance(img_path_or_pil, str):
            # If input is a file path, open the image
            image = Image.open(img_path_or_pil).convert("RGB")
        elif isinstance(img_path_or_pil, Image.Image):
            # If input is already a PIL Image
            image = img_path_or_pil
        else:
            raise TypeError("Input should be either a file path or a PIL Image object.")

        # Process the image
        dataset = ImageDataset(transform=transform)
        image = dataset.process_image(image)
        image = image.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                embedding = model(image).squeeze()

        embedding = embedding.cpu().numpy()  # Convert to numpy array for easier handling
        return embedding

            
        
        

