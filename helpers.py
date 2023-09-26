import matplotlib.pyplot as plt


def display_images(columns, rows, images, figsize=(27, 10)):
    fig=plt.figure(figsize=(20, 10))
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])
        plt.axis('off')
    plt.show()