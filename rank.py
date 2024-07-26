import numpy as np
import pickle
import sklearn.metrics.pairwise
import argparse
from collections import Counter

def cos_sim(opt):

    query_embeddings_dicts = []
    
    with open(opts.embeddings_path, 'rb') as f:
        all_embeddings_dicts = pickle.load(f)
    
    for e in all_embeddings_dicts:
        if e['image_path'] == opts.query_path:
            query_embeddings_dicts.append(e)

    ## remove all embeddings of query image from list on embeddings so we do not get query img in top matches
    for e in query_embeddings_dicts:
        all_embeddings_dicts.remove(e)
    
    print(len(query_embeddings_dicts), ' detections found for query')
    #print(query_embeddings[0])

    ### calculate cosine similarities for each embedding of the query image
    query_embeddings= []
    all_embeddings = []

    for e in query_embeddings_dicts:
        query_embeddings.append(e['embedding'])
    
    for e in all_embeddings_dicts:
        all_embeddings.append(e['embedding'])
    
    #### just checking the shape
    for i, e in enumerate(all_embeddings):
        assert np.array_equal(e, all_embeddings_dicts[i]['embedding'])
    
    for i, q in enumerate(query_embeddings):
        assert np.array_equal(q, query_embeddings_dicts[i]['embedding'])
        

    similarity_mat = sklearn.metrics.pairwise.cosine_similarity(np.array(query_embeddings), np.array(all_embeddings)) 
    print(similarity_mat.shape)

      # Get the top 5 matches for each query embedding
    top_matches_list = []
    for i in range(similarity_mat.shape[0]):
        top_5_indices = np.argsort(-similarity_mat[i])[:5]
        top_5_matches = [all_embeddings_dicts[j]['image_path'] for j in top_5_indices]
        top_matches_list.append(top_5_matches)

    for i, matches in enumerate(top_matches_list):
        print(f"Top 5 matches for query embedding {i+1}:")
        for match in matches:
            print(f"\t{match}")

    # Count occurrences of each image in the top matches
    all_top_matches = [match for sublist in top_matches_list for match in sublist]
    counter = Counter(all_top_matches)
    
    # Get the 5 most common images
    most_common_images = counter.most_common(5)
    
    print("Top 5 most frequent images in top matches:")
    for image, count in most_common_images:
        print(f"{image}: {count} times")


def main(opts):

    cos_sim(opts)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate local and global embeddings.')
    parser.add_argument('--query_path', type=str, required = True)
    parser.add_argument('--embeddings_path', type=str, required =True)

    opts = parser.parse_args()

    main(opts)
