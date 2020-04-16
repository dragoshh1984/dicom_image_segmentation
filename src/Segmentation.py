import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from DicomSegmentationData import DicomSegmentationData
from utils import plot_image

from scipy.ndimage import morphology
from scipy.signal import medfilt
from skimage import measure, feature
from sklearn.metrics import f1_score

class Segmentation:
    """Class used for applying segmentation methods.

    Given an input consisting of a DICOM image and a doctor's selection of the
    organ, this class apllies methods in order to properly segmentate the organ
    from the DICOM image.

    The implementation has three methods: one region growing technique,
    watershed technique and one methods implying contours.

    Attributes:
        data (DicomSegmentationData): DICOM image and doctor's selection input
            for the segmentation.
        solution (DicomSegmentationData) : The most accurate selection of the
            organ.
        solutions (list): List of the three selections from the methods 
            implemented.

    """
    
    def __init__(self, input_data):

        self.data = input_data

        self.solutions = []

    def execute(self):
        """Method that runs all the methods for segmentation.

        Provides three solutions for the given input and then selects
        the most proper one, using F1 Score metric compared on the doctor's
        selection.
        
        Returns:
            solution (DicomSegmentationData): The most accurate selection of the
                organ.

        """
        normalized_data = self.data.normalize(self.data.image)
        preprocessed_data = self.data.preprocess(normalized_data.image)
        
        self.solutions.append(self.growing_region(preprocessed_data))
        self.solutions.append(self.watershed_method(self.solutions[0]))
        self.solutions.append(self.contour_method(preprocessed_data))

        self.solution = self.best_solution()

        return self.solution

    def growing_region(self, data):
        """Growing region method applied on the DICOM image.

        Given the DICOM image and the doctor's selection, this method
        uses region opening in order to obtain the selection of the
        organ after a certain threshold is applied. 

        Args:
            data (DicomSegmentationData): Object containing the DICOM image
                and doctor's segmentation.
        
        Returns:
            data (DicomSegmentationData): Object containing the 
                segmented image.

        """
        data = data.threshold(data.image)
        x_pixel, y_pixel = data.get_inside_pixel(data.doctor_image)
        morphology_kernel = np.ones((3,3), np.uint8)
        
        data.image = measure.label(data.image)
        data.image = data.image == data.image[x_pixel][y_pixel]
        data.image = morphology.binary_closing(data.image, iterations=2)
        
        data.image = data.image.astype(np.uint8)
        data.image = cv.morphologyEx(data.image,
                                     cv.MORPH_OPEN,
                                     kernel = morphology_kernel,
                                     iterations=9)
        data.image = medfilt(data.image, 5)
        
        return data
    
    def watershed_method(self, data):
        """Watershed method applied on the DICOM image.

        Given the DICOM image and the doctor's selection, this method
        uses watershed method in order to obtain the selection of the
        organ using the region growing solution as input feed.

        Args:
            data (DicomSegmentationData): Object containing solution of
                the growing region method.
        
        Returns:
            data (DicomSegmentationData): Object containing the 
                segmented image.

        """
        data.image = data.image.astype(np.uint8)
        data.image *= 255

        bg_kernel = np.ones((3,3), np.uint8)
        sure_bg = cv.dilate(data.image, kernel=bg_kernel, iterations=3)
        sure_fg = cv.erode(data.image, kernel=bg_kernel, iterations=8)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        _, markers = cv.connectedComponents(sure_fg)
        markers += 1
        markers[unknown==255] = 0

        data.image = cv.merge((data.image, data.image, data.image))
        data.image = cv.watershed(data.image, markers)
        data.image[data.image==1] = 0
        data.image[data.image!=0] = 1
        
        data.image = medfilt(data.image, 5)

        return data
    
    def contour_method(self, data):
        """Contour method applied on the DICOM image for organ extraction.

        Given the DICOM image and the doctor's selection, this method
        canny method for obtaining contours and then another finding technique
        in order to obtain the selection of the organ.

        Args:
            data (DicomSegmentationData): Object containing the DICOM image
                and doctor's segmentation.
        
        Returns:
            data (DicomSegmentationData): Object containing the 
                segmented image.

        """
        data.image = data.image.astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)

        canny_mask = cv.dilate(data.doctor_image,
                               kernel=kernel,
                               iterations=9)
        dicom_contours = feature.canny(data.image,
                                       sigma=1,
                                       mask=canny_mask.astype(np.bool))
        dicom_contours = cv.dilate(dicom_contours.astype(np.uint8),
                                   kernel=kernel,
                                   iterations=2)
        
        dicom_image_contoured = data.image.copy()
        dicom_image_contoured[dicom_contours==1] = 0
        
        decent_threshold = np.uint8(np.mean(
            data.image[data.doctor_image==1]))
        contours = measure.find_contours(dicom_image_contoured,
                                         decent_threshold-4,
                                         fully_connected='high',
                                         positive_orientation='low')
        
        best_contour = data.get_best_contour(contours)
        best_contour = medfilt(best_contour, 5)

        data.image = best_contour

        return data

    def best_solution(self):
        """Calculated the best solution obtained for extraction.

        Given all the solution processed it selects the best solution 
        calculating a similarity between the extraction implemented and the
        doctor's selection. The similarity is calculated using F1 Score
        between the solutions and the doctor's selection.
        
        Returns:
            (DicomSegmentationData): Object containing the 
                best segmented image.

        """
        solutions = []
        scores = []
        
        doctor_image = self.data.doctor_image.flatten()

        for solution in self.solutions:
            solutions.append(solution.image.flatten())

        for solution in solutions:
            scores.append(f1_score(solution, doctor_image))

        print(scores)

        return self.solutions[np.argmax(scores)]
    
