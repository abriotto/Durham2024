import pickle

input_file = '/home/annab/Durham2024/embeddings/Brueghel_global.pkl'


with open(input_file, 'rb') as f:
        data = pickle.load(f)

print(len(data))
print(data[999])
for i in data:
 if len(i['embedding'])!= 2048:
      print('STOP')
 
print('OK')

print(type(data))

   
