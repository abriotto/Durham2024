import pickle
import argparse
import numpy as np
from sklearn.decomposition import PCA

def get_params(params):
    parser = argparse.ArgumentParser(description="Apply PCA to embeddings.")
    parser.add_argument('--input_file', type=str, default='embeddings/Brueghel_local_ResNet50_avg.pkl', help='Path to the input pickle file containing embeddings.')
    parser.add_argument('--n_components', type=int, default=50, help='Number of PCA components.')
    #parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing embeddings.')
    return parser.parse_args(params)


def apply_pca(opts):
    output_file = opts.input_file.replace('.pkl', '') + '_pca_' + str(opts.n_components) + '.pkl'

    with open(opts.input_file, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = np.array([np.array(item['embedding']) for item in data])
    print('before PCA', type(embeddings))


    # Apply PCA
    pca = PCA(n_components=opts.n_components)
    pca_embeddings = pca.fit_transform(embeddings)
    print('after PCA', len(pca_embeddings))

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

def main(params):
    opts = get_params(params)
    apply_pca(opts)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
