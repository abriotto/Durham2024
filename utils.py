# embedding_utils.py

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, vgg19, VGG19_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pickle
import torch.multiprocessing as mp
from tqdm import tqdm
import uuid
import json
import numpy as np
from sklearn.decomposition import PCA

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

                if os.path.exists(json_path):  # Check if the JSON file exists
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

def create_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def initialize_model(model_name, device):
    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'vgg19':
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-2])

    else: raise(NotImplementedError)
        
    model.eval()
    model.to(device)
    return model

def prepare_image(img_path_or_pil, transform):
    if isinstance(img_path_or_pil, str):
        image = Image.open(img_path_or_pil).convert("RGB")
    elif isinstance(img_path_or_pil, Image.Image):
        image = img_path_or_pil
    else:
        raise TypeError("Input should be either a file path or a PIL Image object.")
    
    if transform:
        image = transform(image)
    
    return image

def embed_global(model, image_dir, batch_size=32, n_workers=0, device='cpu'):
    mp.set_sharing_strategy('file_system')

    dataset_name = os.path.basename(image_dir)
    dataset_embeddings_folder = os.path.join('embeddings', dataset_name, model)

    if not os.path.exists(dataset_embeddings_folder):
        os.makedirs(dataset_embeddings_folder)
    
    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join(dataset_embeddings_folder, unique_id)
    os.makedirs(temp_dir)


    transform = create_transform()

    dataset = ImageDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=True)

    model_ready = initialize_model(model, device)

    temp_files = []
    for i, (image_paths, imgs) in enumerate(tqdm(dataloader, desc="Processing batches")):
        imgs = imgs.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                embeddings = model_ready(imgs).squeeze()
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

    output_file = os.path.join(dataset_embeddings_folder, 'global.pkl')

    with open(output_file, 'wb') as f:
        pickle.dump(all_embeddings, f)
        
    os.rmdir(temp_dir)  # Clean up the temporary directory
    print('Embeddings written in', output_file)

    return(output_file)


def embed_patches(model, image_dir, json_dir, batch_size, n_workers, device, query = False):
    dataset_name = os.path.basename(image_dir)
    if query :
        dataset_embeddings_folder = os.path.join('embeddings', dataset_name + '_queries', model)
        
    else: dataset_embeddings_folder = os.path.join('embeddings', dataset_name, model)

    mp.set_sharing_strategy('file_system')

    if not os.path.exists(dataset_embeddings_folder):
        os.makedirs(dataset_embeddings_folder)
    

    unique_id = str(uuid.uuid4())
    temp_dir = os.path.join(dataset_embeddings_folder, unique_id)
    os.makedirs(temp_dir)

    # Create dataset and dataloader
    dataset = BoundingBoxDataset(image_dir, json_dir, transform=create_transform())
    print('Total bounding boxes: ', len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=False)

    # Load pre-trained ResNet model and modify it to output embeddings
    model_ready = initialize_model(model, device)
    temp_files = []
    for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        crops, image_paths, boxes = batch
        crops = crops.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                embeddings = model_ready(crops).squeeze()
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
    
    output_file = os.path.join(dataset_embeddings_folder, 'local.pkl')

    with open(output_file, 'wb') as f:
        pickle.dump(all_embeddings, f)

    os.rmdir(temp_dir)  # Clean up the temporary directory
    print('Embeddings written in', output_file)

    return(output_file)

def concatenate_embeddings(local_path, global_path):

    # Load local embeddings
    with open(local_path, 'rb') as f:
        local_embeddings = pickle.load(f)

    # Load global embeddings
    with open(global_path, 'rb') as f:
        global_embeddings = pickle.load(f)

    # Create a dictionary for global embeddings for quick lookup by image_path
    global_dict = {item['image_path']: item['embedding'] for item in global_embeddings}

    # List to store concatenated embeddings
    concatenated_embeddings = []

    # Iterate through local embeddings
    for local_item in local_embeddings:
        image_path = local_item['image_path']
        local_embedding = local_item['embedding']
        box = local_item['box']
        
        if image_path in global_dict:
            global_embedding = global_dict[image_path]
            # Concatenate global and local embeddings
            concatenated_embedding = np.concatenate((np.array(global_embedding), np.array(local_embedding)), axis=None)
            # Append the result to the list
            concatenated_embeddings.append({
                'image_path': image_path,
                'box': box,
                'embedding': concatenated_embedding
            })

    # Save the concatenated embeddings into a new pickle file
    output_path = local_path.replace('local', 'conc')

    with open(output_path, 'wb') as f:
        pickle.dump(concatenated_embeddings, f)

    print(len(concatenated_embedding), 'concatenated embeddings created')

    print(f"Concatenated embeddings have been saved to '{output_path}'.")

    return(output_path)



def apply_pca(input_file, n_components):

    transform_file = input_file.replace('.pkl', '')+'_transformMat_' + str(n_components) +'.pkl'
    output_file = input_file.replace('.pkl', '') + '_pca_' + str(n_components) + '.pkl'
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = np.array([np.array(item['embedding']) for item in data])
    print(embeddings.shape)
    print(embeddings[1].shape)

    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)  # Fit and transform in one step

    # Save the PCA components matrix (transformation matrix)
    with open(transform_file, 'wb') as f:
        pickle.dump(pca.components_, f)
    
    # Collect PCA results
    pca_results = []
    for i, item in enumerate(data):
        if 'box' in item:
            pca_results.append({
                "image_path": item["image_path"],
                "box" : item["box"],
                "embedding": pca_embeddings[i]
            })
        else:
            pca_results.append({
                "image_path": item["image_path"],
                "embedding": pca_embeddings[i]
             })

             
    print(len(pca_results))

    # Save PCA embeddings to the output file
    with open(output_file, 'wb') as f_out:
            pickle.dump(pca_results, f_out)

    print(f'PCA applied and results saved to {output_file}')

    return(output_file)

def apply_transform_mat(input_file, transform_mat): 
    n_components = transform_mat.shape[0]
    output_file = input_file.replace('.pkl', '') + '_pca_' + str(n_components) + '.pkl'
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = np.array([np.array(item['embedding']) for item in data])

    pca_embeddings = np.dot(embeddings,transform_mat.T) 
    
    # Collect PCA results
    pca_results = []
    for i, item in enumerate(data):
        if 'box' in item:
            pca_results.append({
                "image_path": item["image_path"],
                "box" : item["box"],
                "embedding": pca_embeddings[i]
            })
        else:
            pca_results.append({
                "image_path": item["image_path"],
                "embedding": pca_embeddings[i]
             })

             
    print(len(pca_results))

    # Save PCA embeddings to the output file
    with open(output_file, 'wb') as f_out:
            pickle.dump(pca_results, f_out)

    print(f'PCA applied and results saved to {output_file}')

    return(output_file)

def box_pixels_to_ratio(box, img_path):
    image = Image.open(img_path).convert('RGB')
    width, height = image.size

    # Convert normalized coordinates to pixel coordinates
    x1, y1, x2, y2 = box
    x1_r = round(x1 / width, 3)
    y1_r = round(y1 / height, 3)
    x2_r = round(x2 / width, 3)
    y2_r = round(y2 / height, 3)

    box_r = [x1_r, y1_r, x2_r, y2_r]
    return(box_r)


        



