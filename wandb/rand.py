import random
import json
# Get 100 random numbers

random_set = set()

while len(random_set) < 100:
    x = random.randint(0, 1033)
    random_set.add(x)

random_list = list(random_set)
random_list.sort()
with open("indices.json", "w") as f:
    json.dump(random_list, f)
