import json

input_file = 'datasets/brueghelTest.json'


with open(input_file, 'r') as f:
        data = json.load(f)

print(len(data))


print(data.keys())

