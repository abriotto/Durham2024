import pickle
import argparse
import numpy as np
from sklearn.decomposition import PCA

def get_params(params):
    parser = argparse.ArgumentParser(description="Apply PCA to embeddings.")
    parser.add_argument('--input_file', type=str, default='embeddings/brueg_small_local.pkl', help='Path to the input pickle file containing embeddings.')
    parser.add_argument('--n_components', type=int, default=2, help='Number of PCA components.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing embeddings.')
    return parser.parse_args(params)

def load_embeddings(input_file):
    all_embeddings = []
    try:
        with open(input_file, 'rb') as f:
            while True:
                try:
                    item = pickle.load(f)
                    all_embeddings.extend(item)
                except EOFError:
                    break
    except FileNotFoundError:
        print(f"File {input_file} not found.")
        return []
    return all_embeddings

def apply_pca(input_file, n_components, batch_size):
    output_file = input_file.replace('.pkl', '') + '_pca_'+ str(n_components) + '.pkl'
    all_embeddings = load_embeddings(input_file)
    if not all_embeddings:
        return

    embeddings = np.array([item["embedding"] for item in all_embeddings])

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)

    # Save PCA embeddings in batches
    for i in range(0, len(all_embeddings), batch_size):
        batch = all_embeddings[i:i+batch_size]
        pca_batch_embeddings = pca_embeddings[i:i+batch_size]

        pca_results = []
        for j, item in enumerate(batch):
            pca_results.append({
                "image_path": item["image_path"],
                "embedding": pca_batch_embeddings[j]

            })


        # Write the current batch of PCA embeddings to the output file
        with open(output_file, 'ab') as f_out:
            pickle.dump(pca_results, f_out)

    print(f'PCA applied and results saved to {output_file}')

def main(params):
    opts = get_params(params)
    apply_pca(opts.input_file, opts.n_components, opts.batch_size)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
