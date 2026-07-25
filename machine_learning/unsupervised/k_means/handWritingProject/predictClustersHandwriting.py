import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# ============================================================
# K-Means On Handwritten Digits
# ============================================================
#
# What this method does:
# - clusters digit images without using their labels during training
# - learns 10 cluster centers that act like prototype digit shapes
# - predicts which learned cluster a new digit image is closest to
#
# Why we use it:
# - it shows unsupervised learning on a more realistic dataset
# - it demonstrates that clusters can capture visual structure
# - it helps explain the difference between true labels and cluster IDs

digits = datasets.load_digits()
#print(digits.DESCR)
print(digits.data)
print(digits.target)

#plt.gray() 
#plt.matshow(digits.images[100])
#print(digits.target[100])
#plt.show()


k = 10
model = KMeans(n_clusters=k, random_state=42)

model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

new_samples = np.array([
[0,0.83,4.16,7.99,7.66,4.49,0,0,0,6.16,8.32,8.32,8.32,7.91,0,0,0,0.83,7.07,8.32,5.83,1.5,0,0,0,3.08,8.32,7.07,4.16,3.91,0.91,0,0,2.58,7.41,7.49,7.49,8.32,6.41,0,0,0,0,0,0.17,5.82,8.24,0,0,0,1.91,6.74,7.99,8.32,7.07,0,0,0,1.24,4.99,4.66,2.75,0.17,0],
[0,0,0,0.91,3.16,4.16,2.41,0,0,0,2.91,8.07,8.32,8.32,8.16,0,0,0,4.33,8.32,3.82,7.49,7.74,0,0,0,0.58,7.82,8.32,8.32,5.65,0,0,0,0.25,5.41,5.83,6.82,8.32,1.66,0,0,0,0,0.66,5.65,8.32,2.25,0,0,2.16,6.49,7.82,8.32,4.91,0.08,0,0,2.16,5.74,4.74,2.16,0,0],
[0,0,0,2.41,5.49,5.83,2.74,0,0,0,2.66,8.32,7.74,7.82,5.83,0,0,0,1.25,5.58,2.07,7.66,5.74,0,0,0,0,2.41,7.57,8.16,2.16,0,0,0,3.58,8.24,8.32,8.32,1.83,0,0,0,3.91,7.99,7.16,8.32,2.41,0,0,0,7.07,8.32,7.66,4.74,0.17,0,0,0,2.25,2.25,0.17,0,0,0],
[0,0,0,2.41,4.41,4.99,3.74,0,0,0,0.5,8.16,7.99,7.57,8.32,0.75,0,0,0,0.58,1.83,6.91,8.24,0.42,0,0,0.25,5.91,8.32,8.32,5.41,0,0,0,0.25,5.41,5.68,7.66,6.41,0,0,2.33,2.5,2.5,3.33,8.07,5.99,0,1.16,8.32,8.32,8.32,8.32,7.4,1.08,0,0,0.67,0.83,0.83,0.83,0.25,0,0]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
