import pickle

input_file = '/home/annab/Durham2024/embeddings/Brueghel_global_pca_50.pkl'

try:
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    print(data)
except FileNotFoundError:
    print(f"File {input_file} not found.")
except pickle.PickleError as e:
    print(f"Error loading pickle file: {e}")