import matplotlib.pyplot as plt
import torch
import torchvision

# image-label tuples.
training_data = torchvision.datasets.CIFAR10("./data", download=True)
hw_inches = 8  # width, height in inches.
figure = plt.figure(figsize=(hw_inches, hw_inches))
cols, rows = 3, 3  # grid shape.
for i in range(1, cols * rows + 1):  # 1 to 9 inclusive.
    sample_idx = int(torch.randint(len(training_data), size=(1,)).item())
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(str(label))
    plt.axis("off")
    plt.imshow(img)
plt.show()
