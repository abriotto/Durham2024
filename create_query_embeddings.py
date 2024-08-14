import utils
import argparse
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--detections_dir', type=str)
    parser.add_argument('--dataset_path', type=str, default='datasets/Brueghel', help='Directory containing images.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing images.')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of worker processes to use for data loading.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cpu or cuda).')
    parser.add_argument('--patch_t_matrix', type=str, help='Path to the patch transformation matrix.')
    parser.add_argument('--global_embeddings', type=str, help='Path to the global embeddings.')

    return parser.parse_args()


def embed_query(model, dataset_path, detections_dir, batch_size, n_workers, device, patch_t_matrix, global_embeddings):
    # Create global and patch embeddings for all images in the dataset
    local_embeddings_dict_path= utils.embed_patches(model, dataset_path, detections_dir, batch_size, n_workers, device, query=True)
    # Apply PCA on them, save the matrix!
    with open(patch_t_matrix, 'rb') as f:
        t_mat = pickle.load(f)

    local_embeddings_pca = utils.apply_transform_mat(local_embeddings_dict_path, t_mat)

    conc_embeddings = utils.concatenate_embeddings(local_embeddings_pca, global_embeddings)
    
    return(conc_embeddings)


def main():
    args = parse_args()
    embed_query(args.model, args.dataset_path, args.detections_dir, args.batch_size, args.n_workers, args.device, args.patch_t_matrix, args.global_embeddings)

if __name__ == "__main__":
    main()  # Call the main function with parentheses
