import pickle

input_file = '/home/annab/Durham2024/embeddings/Brueghel_global_pca_50.pkl'


with open(input_file, 'rb') as f:
        data = pickle.load(f)
print(data[0])
print(len(data))

print(type(data))
   
