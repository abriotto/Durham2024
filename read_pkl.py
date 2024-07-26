import pickle
import numpy as np
input_file = '/home/annab/Durham2024/embeddings/Brueghel_conc.pkl'


with open(input_file, 'rb') as f:
        data = pickle.load(f)

print(len(data))


for i in data:
 if len(i['embedding'])!= 100:
      raise(ValueError)
 
assert np.array_equal(data[1]['embedding'][:50], data[0]['embedding'][:50])


   
