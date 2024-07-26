import pickle
import numpy as np
import argparse

def concatenate_embeddings(local_path, global_path, output_path):

    
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
    with open(output_path, 'wb') as f:
        pickle.dump(concatenated_embeddings, f)

    print(f"Concatenated embeddings have been saved to '{output_path}'.")

def main(local_path, global_path, output_path):
    concatenate_embeddings(local_path, global_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate local and global embeddings.')
    parser.add_argument('--local_path', type=str, default = 'embeddings/Brueghel_local_pca_50.pkl', help='Path to the local embeddings pickle file.')
    parser.add_argument('--global_path', type=str, default = 'embeddings/Brueghel_global_pca_50.pkl', help='Path to the global embeddings pickle file.')
    parser.add_argument('--output_path', type=str, default = 'embeddings/Brueghel_conc.pkl', help='Path to save the concatenated embeddings pickle file.')

    args = parser.parse_args()
    main(args.local_path, args.global_path, args.output_path)
