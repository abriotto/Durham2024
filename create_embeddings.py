import utils
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg19')
    parser.add_argument('--dataset_detections_dir', type=str)
    parser.add_argument('--pca_components', type=int, default=50)  # Note: default should be an integer
    parser.add_argument('--dataset_path', type=str, default='datasets/Brueghel', help='Directory containing images.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing images.')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of worker processes to use for data loading.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu or cuda).')
    return parser.parse_args()

def embed(model, dataset_path, detections_dir, pca_components, batch_size, n_workers, device):
    # Create global and patch embeddings for all images in the dataset
    global_embeddings_path = utils.embed_global(model, dataset_path, batch_size, n_workers, device)
    local_embeddings_path = utils.embed_patches(model, dataset_path, detections_dir, batch_size, n_workers, device)

    # Apply PCA on them, save the matrix!
    global_embeddings_pca = utils.apply_pca(global_embeddings_path, pca_components)
    local_embeddings_pca = utils.apply_pca(local_embeddings_path, pca_components)

    utils.concatenate_embeddings(local_embeddings_pca, global_embeddings_pca)

def main():
    args = parse_args()
    embed(args.model, args.dataset_path, args.dataset_detections_dir, args.pca_components, args.batch_size, args.n_workers, args.device)

if __name__ == "__main__":
    main()  # Call the main function with parentheses
