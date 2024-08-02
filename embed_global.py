

import argparse
from utils import embed_global

def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from bounding boxes using ResNet50.")
    parser.add_argument('--image_dir', type=str, default='datasets/Brueghel', help='Directory containing images.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing images.')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of worker processes to use for data loading.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu or cuda).')
    return parser.parse_args()

def main():
    args = parse_args()
    embed_global(args.image_dir, args.batch_size, args.n_workers, args.device)

if __name__ == "__main__":
    main()
