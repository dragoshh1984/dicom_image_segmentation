import matplotlib.pyplot as plt

def plot_image(image, title='DICOM Image'):
    image_height, image_width = image.shape
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    plt.show()
    plt.imsave("{}.png".format(title), image, cmap='gray')
    