import numpy as np
import pickle
import sklearn.metrics

def cos_sim(query_path, embeddings_path):
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
    
    print(len(query_embeddings_dicts), 'detections found for query')

    # Prepare the embeddings for similarity calculation
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
        top_5_indices = np.argsort(-similarity_mat[i])[:5]
        top_5_matches = [(all_embeddings_dicts[j], similarity_mat[i][j]) for j in top_5_indices]
        top_matches_list.append(top_5_matches)

    return [(query_embeddings_dicts[i], top_matches_list[i]) for i in range(len(query_embeddings_dicts))]

# Example usage
# results = cos_sim('path/to/query/image.jpg', 'path/to/embeddings.pkl')
# for query, matches in results:
#     print(f"Query: {query['image_path']}")
#     for match, similarity in matches:
#         print(f"Match: {match['image_path']} with similarity {similarity}")
