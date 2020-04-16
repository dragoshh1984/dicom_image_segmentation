from skimage import measure
import matplotlib.pyplot as plt

def plot_image(image, title='Solution'):
    image_height, image_width = image.shape
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image, cmap='gray')
    ax.set_title(title)

    ax.axis('off')
    plt.show()
    
def plot_comparison(dicom_image, solution_iamge, doctor_image, title='Result'):
    solution_contours = measure.find_contours(solution_iamge, 
                                             0, 
                                             fully_connected='high',
                                             positive_orientation='low')

    doctor_contours = measure.find_contours(doctor_image, 
                                            0, 
                                            fully_connected='high',
                                            positive_orientation='low')

    plt.figure(figsize=(10, 10))
    
    plt.imshow(dicom_image, cmap='gray')
    
    for _, contour in enumerate(solution_contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='r')
    for _, contour in enumerate(doctor_contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='g')

    plt.title(title)
    plt.axis('off')

    plt.show()
