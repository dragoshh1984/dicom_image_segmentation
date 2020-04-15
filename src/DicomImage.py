import numpy as np
import cv2 as cv

from utils import plot_image

from scipy import ndimage
from scipy.ndimage import morphology
from scipy.signal import medfilt
from skimage import measure, feature

class DicomImage:
    def __init__(self):
        self.file_name = ''
        
    def copy(self):
        new_image = DicomImage()
        new_image.file_name = self.file_name
        new_image.image = self.image.copy()
        new_image.doctor_image = self.doctor_image.copy()

        return new_image

    def read_image(self, image_path):
        self.image = np.loadtxt(image_path, dtype=np.float32)

    def read_doctor_image(self, image_path):
        self.doctor_image = np.loadtxt(image_path, dtype=np.float32)

    def read_images(self, image_path, doctor_image_path):
        self.read_image(image_path)
        self.read_doctor_image(doctor_image_path)
    
    def preprocess_image(self):
        self.image = ((self.image - np.min(self.image)) 
                     / (np.max(self.image) - np.min(self.image)))
        self.image *= 255.
        self.image = self.image.astype(np.uint8)

    def apply_trivial_threshold(self, threshold=127):
        self.image[self.image < threshold] = 0
    
    def apply_histogram_equalization(self):
        self.image = cv.equalizeHist(self.image)
    
    def apply_denoise(self):
        # magic numbers, should be change into variables
        self.image = cv.fastNlMeansDenoising(self.image, None, 30, 7, 21)
    
    def apply_threshold(self, lower, upper):
        self.image = (self.image > lower) & (self.image < upper)
    
    def get_inside_pixel(self, image):
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
        
    def apply_region_growing_method(self):
        x_pixel, y_pixel = self.get_inside_pixel(self.doctor_image)
        
        morphology_kernel = np.ones((3,3), np.uint8)

        self.image = measure.label(self.image)
        self.image = self.image == self.image[x_pixel][y_pixel]

        # magic numbers, should be change into variables
        self.image = morphology.binary_closing(self.image, iterations=2)
        
        self.image = self.image.astype(np.uint8)
        # magic numbers, should be change into variables
        self.image = cv.morphologyEx(self.image,
                                     cv.MORPH_OPEN,
                                     kernel = morphology_kernel,
                                     iterations=9)

    def apply_smooth_filter(self):
        # magic numbers, should be change into variables
        self.image = medfilt(self.image, 5)
    
    def apply_watershed_method(self):
        self.image = self.image.copy().astype(np.uint8)
        self.image *= 255
        bg_kernel = np.ones((3,3), np.uint8)

        # magic numbers, should be change into variables
        sure_bg = cv.dilate(self.image, kernel=bg_kernel, iterations=3)
        
        # magic numbers, should be change into variables
        sure_fg = cv.erode(self.image, kernel=bg_kernel, iterations=8)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv.subtract(sure_bg, sure_fg)

        _, markers = cv.connectedComponents(sure_fg)
        markers += 1
        markers[unknown==255] = 0

        self.image = cv.merge((self.image, self.image, self.image))
        self.image = cv.watershed(self.image, markers)
        self.image[self.image==1] = 0
        self.image[self.image!=0] = 1

    def get_selected_area(self, image):
        return len(image[image==1])

    def fill_contour(self, image):
        image_filled = image.copy()
        image_filled = ndimage.morphology.binary_dilation(image_filled,
                                                          None,
                                                          3)
        image_filled = ndimage.binary_fill_holes(image_filled,
                                                structure=np.ones((3,3)))
        
        return image_filled

    def get_best_contour(self, contours):
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

    def apply_contour_method(self):
        self.image = self.image.copy().astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)

        canny_mask = cv.dilate(self.doctor_image,
                               kernel=kernel,
                               iterations=9)
        dicom_contours = feature.canny(self.image,
                                       sigma=1,
                                       mask=canny_mask.astype(np.bool))
        dicom_contours = cv.dilate(dicom_contours.astype(np.uint8),
                                   kernel=kernel,
                                   iterations=2)
        
        dicom_image_contoured = self.image.copy()
        dicom_image_contoured[dicom_contours==1] = 0
        
        decent_threshold = np.uint8(np.mean(
            self.image[self.doctor_image==1]))
        contours = measure.find_contours(dicom_image_contoured,
                                         decent_threshold-4,
                                         fully_connected='high',
                                         positive_orientation='low')
        
        best_contour = self.get_best_contour(contours)

        self.image = best_contour
