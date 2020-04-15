import numpy as np
import os
import time
from utils import plot_image

from config import get_parser
from DicomImage import DicomImage
from Segmentation import Segmentation

from scipy import ndimage
from skimage import measure

class Solution:
    def __init__(self, args):
        self.args = args
        self.input_data = []

        self.get_input_data()
    
    def get_input_data(self):
        if self.args.parse_folder == True:
            files = os.listdir(self.args.input_dir)
            
            input_image_names = []
            doctor_image_names = []
            for file in files:
                if file.endswith('-HU.in'):
                    input_image_names.append(file)
                if file.endswith('-seg.in'):
                    doctor_image_names.append(file)
            
            for image in input_image_names:
                image_name = image.split('-')[0]

                for doctor_image in doctor_image_names:
                    doctor_image_name = doctor_image.split('-')[0]

                    if image_name == doctor_image_name:
                        image_path = os.path.join(self.args.input_dir, image)
                        doctor_image_path = os.path.join(self.args.input_dir,
                                                         doctor_image)

                        new_input_data = DicomImage()
                        new_input_data.file_name = image_name
                        new_input_data.read_images(image_path, 
                                                   doctor_image_path)

                        self.input_data.append(new_input_data)

                        break
        else:
            try:
                image_path = os.path.join(self.args.input_dir,
                                          self.args.input_HU)
                doctor_image_path = os.path.join(self.args.input_dir,
                                                 self.args.input_seg)
                file_name = self.args.input_HU.split('/')[-1]
                file_name = file_name.split('-')[0]
                
                new_input_data = DicomImage()
                new_input_data.file_name = file_name
                new_input_data.read_images(image_path, doctor_image_path)

                self.input_data.append(new_input_data)
                pass
            except NameError:
                print("Invalid input file.")
    
    def fill_contour(self, image):
        image_filled = image.copy()
        image_filled = ndimage.morphology.binary_dilation(image_filled,
                                                          None,
                                                          3)
        image_filled = ndimage.binary_fill_holes(image_filled,
                                                 structure=np.ones((3,3)))
        
        return image_filled

    def get_segments(self, data):
        segments = []
        sections = measure.find_contours(data.doctor_image,
                                         0,
                                         fully_connected='high',
                                         positive_orientation='low')
        
        if len(sections) == 1:
            segments.append(data)
        else:
            organ_sections = []

            for _, section in enumerate(sections):
                new_image = np.zeros((512, 512), dtype=np.uint8)
                
                for pixel in section:
                    new_image[int(pixel[0]), int(pixel[1])] = 1
                
                new_image = self.fill_contour(new_image)
                
                organ_sections.append(new_image.astype(np.uint8))
            
            for section in organ_sections:
                new_data = DicomImage()
                new_data.file_name = data.file_name
                new_data.image = data.image.copy()
                new_data.doctor_image = section.copy()

                segments.append(new_data)
        
        return segments

    def save_solution(self, sections_segmented):
        solution = np.zeros((512, 512), dtype=np.uint8)

        for section in sections_segmented:
            solution += section.image.astype(np.uint8)
        
        solution[solution > 1] = 1
        
        solution_name = "{}/{}-opt.out".format(self.args.output_dir,
                                               sections_segmented[0].file_name)

        np.savetxt(solution_name, solution)
        # plot_image(solution)

    def run(self):
        print("Started running...")
        for input in self.input_data:
            start_time = time.time()
            
            segments = self.get_segments(input)

            print("\nExtracting from {} file.".format(segments[0].file_name))
            print("Organs found: {}".format(len(segments)))
            
            sections_segmented = []
            for segment in segments:
                segmentation = Segmentation(segment)
                section_segmented = segmentation.execute()
                
                sections_segmented.append(section_segmented)
            
            self.save_solution(sections_segmented)
            elapsed_time = int(time.time() - start_time)

            minutes = elapsed_time % 3600 // 60
            seconds = elapsed_time % 60
            print("Finished extracting in {:02d}:{:02d}".format(minutes, 
                                                                seconds))
        print("Finished extracting.")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    solution = Solution(args=args)
    solution.run()
