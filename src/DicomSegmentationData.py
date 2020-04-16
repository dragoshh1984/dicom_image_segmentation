import numpy as np
import cv2 as cv

from utils import plot_image

from scipy import ndimage
from scipy.ndimage import morphology
from scipy.signal import medfilt
from skimage import measure, feature

class DicomSegmentationData:
    """Class used for storing and manipulating the input of one segmentation.

    Collection of image processing methods that are used in the segmentation of
    the organ from the DICOM image.

    Attributes:
        image (np.array): DICOM image from which an organ must be segmented.
        doctor_image (np.array): Doctor's segmentation of the organ. 
        file_name (str): Number of the DICOM image.

    """

    def __init__(self):
        self.file_name = ''
        
    def copy(self):
        """Utility method for cloning the object.

        Returns:
            A copy of the DicomSegmentationData object.

        """
        new_image = DicomSegmentationData()
        new_image.file_name = self.file_name
        new_image.image = self.image.copy()
        new_image.doctor_image = self.doctor_image.copy()

        return new_image

    def read_images(self, image_path, doctor_image_path):
        """Utility function to read the images into

        Args:
            image_path (str): Path to the DICOM image.
            doctor_image_path (str): Path to the doctor's segmentation image.

        """
        self.image = np.loadtxt(image_path, dtype=np.float32)
        self.doctor_image = np.loadtxt(doctor_image_path, dtype=np.float32)
    
    def normalize(self, image):
        """Normalizes a DICOM image to values between 0 and 255.

        Args:
            image (np.array): Image to be normalized.

        Returns:
            new_data (DicomSegmentationData): Object with the image normalized.

        """
        image = ((image - np.min(image)) 
                     / (np.max(image) - np.min(image)))
        image *= 255.

        new_data = DicomSegmentationData()
        new_data = self.copy()
        new_data.image = image.astype(np.uint8)

        return new_data

    def preprocess(self, image, threshold=127):
        """Preprocessing the image to have a better contrast overall.

        A threshold is applied in order to eliminate the gray circle
        then we adjust the contrast using histogram equalization and
        we reduce the noise using non-local means denoising.

        Args:
            image (np.array): Image to be preprocessed.
            threshold (int): Threshold value below which all values
                are set to 0.
        
        Returns:
            new_data (DicomSegmentationData): Object containing the 
                preprocessed image.

        """
        image[image < threshold] = 0
        image = cv.equalizeHist(image)
        image = cv.fastNlMeansDenoising(image, None, 30, 7, 21)

        new_data = DicomSegmentationData()
        new_data = self.copy()
        new_data.image = image

        return new_data

    def get_threshold_range(self, image):
        """Calculates the proper threshold range given a DICOM image.

        Calculates a range for the majority of pixel values in the doctor's
        section of the image given.

        Args:
            image (np.array): Image for which the threshold range is
                calculated.
        
        Returns:
            lower, upper (int, int): Bottom and top hreshold range values.

        """
        doctor_area = np.sum(self.doctor_image==1)
        values = image[self.doctor_image==1]
        values = np.sort(values)
        unique, counts = np.unique(values, return_counts=True)

        values = np.asarray((unique, counts)).T
        values = values[values[:,1].argsort()]

        index = -1
        total_sum = 0
        lower = 255
        upper = 0
        while total_sum < int(doctor_area/3) * 2:
            current_pixel = values[index][0]
            total_sum += values[index][1]
            index -= 1

            if lower > current_pixel:
                lower = current_pixel
            if upper < current_pixel:
                upper = current_pixel
                
        return lower-5, upper+5

    def threshold(self, image):
        """Makes all values out of a specific range from the image 0.

        Gets a range for the majority of pixel values in the doctor's
        section of the image given and applies threshold over the image.

        Args:
            image (np.array): Image to be thresholded.
        
        Returns:
            new_data (DicomSegmentationData): Object containing the
                thresholded DICOM image.

        """
        lower, upper = self.get_threshold_range(image)
        
        new_data = DicomSegmentationData()
        new_data = self.copy()
        new_data.image = (self.image > lower) & (self.image < upper)

        return new_data
    
    def get_inside_pixel(self, image):
        """Calculates the coordinates of a pixel towards the image's center.

        Applies errosion to the image given so that the pixel is closer to the
        center of the image.

        Args:
            image (np.array): Image where the center is calculated.
        
        Returns:
            (int, int): Pixel coordinates.

        """
        pixel_coordinates_backup = np.where(image==1)
        image = cv.erode(image, None, iterations=10)
        pixel_coordinates = np.where(image==1)

        pixel_y_coordinate = int(len(pixel_coordinates[0])/2)
        pixel_backup_y_coordinate = int(len(pixel_coordinates_backup[0])/2)

        if pixel_coordinates[0].size != 0:
            return (pixel_coordinates[0][pixel_y_coordinate],
                    pixel_coordinates[1][pixel_y_coordinate])
        else:
            return (pixel_coordinates_backup[0][pixel_backup_y_coordinate], 
                    pixel_coordinates_backup[1][pixel_backup_y_coordinate])

    def get_selected_area(self, image):
        """Calculates the area of a segmentation mask.

        Args:
            image (np.array): Image which are is to be calculated.
        
        Returns:
            (int): The area of the segmentation.

        """
        return len(image[image==1])

    def fill_contour(self, image):
        """Fills the contour given.

        Given an image which contains a contour, fills the contour.

        Args:
            image (np.array): Image containing the contour to be filled.
        
        Returns:
            image_filled (np.array): Image containing the filled selection.

        """
        image_filled = image.copy()
        image_filled = ndimage.morphology.binary_dilation(image_filled,
                                                          None,
                                                          3)
        image_filled = ndimage.binary_fill_holes(image_filled,
                                                structure=np.ones((3,3)))
        
        return image_filled

    def get_best_contour(self, contours):
        """Decides on the best contour comparing to the doctor's selection.

        Given a collection of contours over the image, it selects the best
        contour that covers the doctor's selection.

        Args:
            contours (np.array): Collection of contours.
        
        Returns:
            best_contour (np.array): The filled contour that covers most of the
            doctor's selection.

        """
        filled_contours = []

        for _, contour in enumerate(contours):
            new_image = np.zeros((512, 512), dtype=np.uint8)

            for pixel in contour:
                new_image[int(pixel[0]), int(pixel[1])] = 1
            
            new_image = self.fill_contour(new_image)

            filled_contours.append(new_image.astype(np.uint8))
        
        chosen_contours =[]

        doctor_area = self.get_selected_area(self.doctor_image)

        for contour in filled_contours:
            current_area = self.get_selected_area(contour)

            if (current_area < 2 * doctor_area 
                and current_area > (2 * doctor_area) / 3):
                chosen_contours.append(contour)

        best_contour = chosen_contours[0]
        best_intersection = 0

        for contour in chosen_contours:
            current_intersection = self.get_selected_area(
                self.doctor_image.astype(np.uint8) & 
                contour.astype(np.uint8))

            if best_intersection < current_intersection:
                best_contour = contour.copy()
                best_intersection = current_intersection

        return best_contour

