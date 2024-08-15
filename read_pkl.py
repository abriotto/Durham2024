import pickle
import numpy as np
input_file = 'embeddings/Brueghel/resnet50/conc_pca_50.pkl'


with open(input_file, 'rb') as f:
        data = pickle.load(f)

print(len(data))


 
print(data[0])


   
