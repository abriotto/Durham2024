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

    all_embeddings = []
    for i, (image_paths, imgs) in enumerate(tqdm(dataloader, desc="Processing batches")):
        imgs = imgs.to(opts.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(opts.device == 'cuda')):
                embeddings = model(imgs).squeeze()
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)  # Handle case when there's only one embedding

        # Move embeddings to CPU before appending
        embeddings = embeddings.cpu()

        for j, embedding in enumerate(embeddings):
            all_embeddings.append({
                "image_path": image_paths[j],
                "embedding": embedding.numpy().tolist()
            })

        # Save intermediate results to the output file to avoid OOM
        if (i + 1) % 100 == 0:  # Adjust the frequency as needed
            with open(output_file, 'ab') as f:
                pickle.dump(all_embeddings, f)
            all_embeddings = []  # Clear the list to free memory

    # Save any remaining embeddings to the output file
    if all_embeddings:
        with open(output_file, 'ab') as f:
            pickle.dump(all_embeddings, f)
    print('Embeddings written!')

def main(params):
    opts = get_params(params)
    embed_global(opts)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
