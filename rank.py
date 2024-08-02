import numpy as np
import pickle
import sklearn.metrics
from PIL import Image
import pca
import utils
from sklearn.decomposition import PCA

def embed_local_query(query, box, pca_components):

    image = Image.open(query).convert('RGB')
    width, height = image.size

    # Convert normalized coordinates to pixel coordinates
    x1, y1, x2, y2 = box
    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)
    
    coordinates = (x1, y1, x2, y2)
    patch = image.crop(coordinates)

    patch_embedding= utils.embed_one(patch)
    global_embedding= utils.embed_one(image)

    pca = PCA(n_components=pca_components)
    pca_patch = pca.fit_transform(patch_embedding)
    pca_global = pca.fit_transform(global_embedding)

    query_embedding = np.concatenate((np.array(global_embedding), np.array(patch_embedding)), axis=None)
    query_dict = {'image_path': query, 'box': box, 'embedding': query_embedding}

    return query_dict


def cos_sim(query_path, embeddings_path, patch_box = None):
    
    query_embeddings_dicts = []
    
    # Load all embeddings from the file
    with open(embeddings_path, 'rb') as f:
        all_embeddings_dicts = pickle.load(f)
    
    # Collect embeddings for the query image
    for e in all_embeddings_dicts:
        if e['image_path'] == query_path:
            query_embeddings_dicts.append(e)

    # Remove query image embeddings from the list
    for e in query_embeddings_dicts:
        all_embeddings_dicts.remove(e)
    
    
    # Prepare the embeddings for similarity calculation
    if patch_box:
        query_embeddings = embed_local_query(query_path, patch_box, 50)
        query_embeddings_dicts = {}
        query_embeddings_dicts['image_path'] = query_path
        query_embeddings_dicts['box'] = patch_box
        query_embeddings_dicts['embedding'] = query_embeddings


    else: 
        print(len(query_embeddings_dicts), 'detections found for query')
        query_embeddings = [e['embedding'] for e in query_embeddings_dicts]

    all_embeddings = [e['embedding'] for e in all_embeddings_dicts]

    # Ensure embeddings are consistent
    for i, e in enumerate(all_embeddings):
        assert np.array_equal(e, all_embeddings_dicts[i]['embedding'])
    
    for i, q in enumerate(query_embeddings):
        assert np.array_equal(q, query_embeddings_dicts[i]['embedding'])
        
    # Calculate cosine similarities
    similarity_mat = sklearn.metrics.pairwise.cosine_similarity(np.array(query_embeddings), np.array(all_embeddings)) 
    print(similarity_mat.shape)

    # Get the top 5 matches for each query embedding with similarity values
    top_matches_list = []
    for i in range(similarity_mat.shape[0]):
        top_10_indices = np.argsort(-similarity_mat[i])[:10]
        top_10_matches = [(all_embeddings_dicts[j], similarity_mat[i][j]) for j in top_10_indices]
        top_matches_list.append(top_10_matches)

    return [(query_embeddings_dicts[i], top_matches_list[i]) for i in range(len(query_embeddings_dicts))]

# Example usage
# results = cos_sim('path/to/query/image.jpg', 'path/to/embeddings.pkl')
# for query, matches in results:
#     print(f"Query: {query['image_path']}")
#     for match, similarity in matches:
#         print(f"Match: {match['image_path']} with similarity {similarity}")