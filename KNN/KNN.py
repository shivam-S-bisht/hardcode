from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import warnings
from collections import Counter

dataset = {'k' : [[1,2], [2,3], [3,1]], 'r' : [[6,5], [7,7], [8,6]]}
new_features = [5,7]

# [[plt.scatter(j[0], j[1], s = 50, color = i) for j in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s = 50, color = 'm')
# plt.show()        

import warnings

def knn(data, predict, k = 3):
    if len(data) >= k:
        warnings.warn('probably a wrong number....')

    distances = []



    for group in data:
        for features in data[group]:
            # ed_dist = np.sqrt(np.sum(np.array(features) - np.array(predict))**2)
            ed_dist = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([ed_dist, group])


    print(distances)
    votes  = []
    for i in sorted(distances):
        votes.append(i[1])

    print(votes)    

    print(Counter(votes).most_common(1))

    vote_result = Counter(votes).most_common(1)[0][0]
       



    return vote_result



result = knn(dataset, new_features, 4)
print(result)