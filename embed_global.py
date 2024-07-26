import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import argparse
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

def get_params(params):
    parser = argparse.ArgumentParser(description="Extract embeddings from bounding boxes using ResNet50.")
    parser.add_argument('--image_dir', type=str, default='datasets/Brueghel', help='Directory containing images.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing images.')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of worker processes to use for data loading.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu or cuda).')
    return parser.parse_args(params)

def embed_global(opts):
    mp.set_sharing_strategy('file_system')
    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')

    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join('embeddings', unique_id)
    os.makedirs(temp_dir)

    output_file = 'embeddings/' + opts.image_dir.replace('datasets/', '') + '_global.pkl'
    output_dir = os.path.dirname(output_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(opts.image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, num_workers=opts.n_workers, shuffle=False, pin_memory=True)

    # Load pre-trained ResNet model and modify it to output embeddings
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(opts.device)

    temp_files = []
    for i, (image_paths, imgs) in enumerate(tqdm(dataloader, desc="Processing batches")):
        imgs = imgs.to(opts.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(opts.device == 'cuda')):
                embeddings = model(imgs).squeeze()
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)  # Handle case when there's only one embedding
    

        # Move embeddings to CPU before appending
        embeddings = embeddings.cpu()

        batch_embeddings = []
        for j, embedding in enumerate(embeddings):
            batch_embeddings.append({
                "image_path": image_paths[j],
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

def main(params):
    opts = get_params(params)
    embed_global(opts)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
