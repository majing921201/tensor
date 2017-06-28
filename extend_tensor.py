import numpy as np
A = np.load("yelp_onemonth.npy")
new_tensor = []
for i in range(300):
    new_tensor.append(A[i%100])
new_tensor = np.array(new_tensor)
print new_tensor.shape
np.save("50_t300",new_tensor)
