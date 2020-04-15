import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from DicomImage import DicomImage
from utils import plot_image

from scipy.ndimage import morphology
from scipy.signal import medfilt
from skimage import measure
from sklearn.metrics import f1_score

class Segmentation:
    def __init__(self, input_data):
        self.data = input_data

        self.solutions = []

    def execute(self):
        self.preprocess()
        
        self.solutions.append(self.growing_region())
        self.solutions.append(self.watershed_method())
        self.solutions.append(self.contour_method())

        self.solution = self.best_solution()

        return self.solution
    
    def preprocess(self):
        self.data.preprocess_image()
        self.data.apply_trivial_threshold()
        self.data.apply_histogram_equalization()
        self.data.apply_denoise()
    
    def growing_region(self):
        image = self.data.copy()

        lower, upper = self.get_lower_upper(image)
        image.apply_threshold(lower, upper)
        image.apply_region_growing_method()
        image.apply_smooth_filter()
        
        return image
    
    def watershed_method(self):
        image = self.solutions[0].copy()
        image.apply_watershed_method()
        image.apply_smooth_filter()

        return image
    
    def contour_method(self):
        image = self.data.copy()
        image.apply_contour_method()
        image.apply_smooth_filter()

        return image

    def best_solution(self):
        solutions = []
        scores = []
        
        doctor_image = self.data.doctor_image.flatten()

        for solution in self.solutions:
            solutions.append(solution.image.flatten())

        for solution in solutions:
            scores.append(f1_score(solution, doctor_image))

        print(scores)

        return self.solutions[np.argmax(scores)]
    
    def get_lower_upper(self, image):
        doctor_area = np.sum(self.data.doctor_image==1)
        values = image.image[self.data.doctor_image==1]
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
