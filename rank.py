import numpy as np
import pickle
import sklearn.metrics
from PIL import Image
import utils
from sklearn.decomposition import PCA


###NOT USED NOW
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


def find_top_matches_for_patch(query_path, sim_mat_row, all_embeddings_dicts, n):
    """
    Find the top `n` unique matches based on `sim_mat_row` and `image_path` from `all_embeddings_dicts`.
    """
    # Get the indices that would sort the array in descending order
    top_n_indices = np.argsort(-sim_mat_row)
    
    # Initialize a set to track seen image_paths and a list to hold the top matches
    seen_image_paths = set()
    top_n_matches = []
    
    # Iterate over the indices to find unique top matches
    for idx in top_n_indices:
        if len(top_n_matches) >= n:
            break
        
        entry = all_embeddings_dicts[idx]
        image_path = entry['image_path']
        similarity = sim_mat_row[idx]
        
        # Check if the image_path has already been seen
        if image_path not in seen_image_paths and image_path!= query_path:
            top_n_matches.append((entry, similarity))
            seen_image_paths.add(image_path)
    
    return top_n_matches

def rank_local_to_global(query_embeddings_path, embeddings_path, top_n):
    '''
    Returns top matches for each query.

    query is a tuple(img_path, [box])
    '''
    with open(query_embeddings_path, 'rb') as f:
        query_embeddings_dicts = pickle.load(f)

    query_paths = [e['image_path'] for e in query_embeddings_dicts]
    #print(query_paths)
    

    with open(embeddings_path, 'rb') as f:
        all_embeddings_dicts = pickle.load(f)

    
    all_embeddings= [e['embedding'] for e in all_embeddings_dicts]
    embeddings_queries = [e['embedding'] for e in query_embeddings_dicts]

    #find sim mat for all queries at once
    similarity_mat = sklearn.metrics.pairwise.cosine_similarity(np.array(embeddings_queries), np.array(all_embeddings)) 
    print(similarity_mat.shape)

    top_matches_list = []
    for i in range(similarity_mat.shape[0]):
        top_matches = find_top_matches_for_patch(query_paths[i], similarity_mat[i], all_embeddings_dicts, top_n)
        top_matches_list.append((query_embeddings_dicts[i],top_matches))
    
    
    return top_matches_list





    




# Example usage
# results = cos_sim('path/to/query/image.jpg', 'path/to/embeddings.pkl')
# for query, matches in results:
#     print(f"Query: {query['image_path']}")
#     for match, similarity in matches:
#         print(f"Match: {match['image_path']} with similarity {similarity}")