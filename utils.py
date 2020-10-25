import matplotlib.pyplot as plt

def plot(dataset, index):
    image, label = dataset[index]
    print(dataset.mappings[label])
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
