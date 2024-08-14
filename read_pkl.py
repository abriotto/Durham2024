import pickle
import numpy as np
input_file = 'embeddings/Brueghel/local_pca_50.pkl'


with open(input_file, 'rb') as f:
        data = pickle.load(f)

print(len(data))


for i in data:
 if len(i['embedding'])!= 50:
      raise(ValueError)
 
print(data[0])


   
