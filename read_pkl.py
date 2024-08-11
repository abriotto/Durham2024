import pickle
import numpy as np
input_file = '/home/annab/Durham2024/embeddings/Brueghel_query/resnet50_conc_pca_50.pkl'


with open(input_file, 'rb') as f:
        data = pickle.load(f)

print(len(data))


for i in data:
 if len(i['embedding'])!= 100:
      raise(ValueError)
 
print(data[0])


   
