import numpy as np
import os
import sys
import time

from config import get_parser
from DicomSegmentationData import DicomSegmentationData
from Segmentation import Segmentation
from utils import plot_image, plot_comparison

from scipy import ndimage
from skimage import measure

class Solution:
    """Class used for obtaining the extraction.

    Class used for reading the input files and running the segmentation
    algorithms for each input.

    If a doctor's image has multiple organs, for each organ it will provide
    a solution and at the end it will unify those solution in a single image.

    Examples:
        $ python Solution.py 'path/to/DICOM/image' 'path/to/doctors/image'

        An error message will be displayer here, due to the fact that argparse
        will not recognise the arguments, but the solution will contiue to run.

        $ python Solution.py --parse_dir=True --input_dir='path/to/directory'

        Will run through all the files in the directory and selects matching
        input files (same_number-HU.in with same_number-seg.in) and then will
        run the segmentation for all the input data found.    

    Attributes:
        args (DicomSegmentationData): argparse arguments or simply the path to
            the two input files.
        no_parser (bool) : Detects if the solution was run not using the
            argparse parameters; True if so, False otherwise.
            
        input_data (list): List of all the input data collection of files (two
            per each run). 

    """

    def __init__(self, args, no_parser=False):
        self.args = args
        self.input_data = []
        self.no_parser = no_parser

        self.get_input_data()

    def get_images_names(self, files):
        """Utility function that extracts two lists with the input images.

        Given a collection of strings (file names) it will select those that
        end in '-HU.in' and '-seg.in' and group them in two lists.

        Args:
            lists (list): Files names from the directory selected.
        
        Returns:
            input_image_names (list): All the image input file names.
            doctor_image_names (list): All the doctor input file names.

        """
        input_image_names = []
        doctor_image_names = []
        for file in files:
            if file.endswith('-HU.in'):
                input_image_names.append(file)
            if file.endswith('-seg.in'):
                doctor_image_names.append(file)
        
        return input_image_names, doctor_image_names

    def select_input_data(self, input_image_names, doctor_image_names):
        """Utility function that matches each input image with its doctor image.

        Given a collection of input file names it will find the matching doctor
        image and append each two in a DicomSegmentationData object.

        Args:
            input_image_names (list): File names of the input DICOM images.
            doctor_image_names (list): File names of the input doctor images.

        """
        for image in input_image_names:
            image_name = image.split('-')[0]
            doctor_image_name = image_name + '-seg.in'
            
            if doctor_image_name in doctor_image_names:
                image_path = os.path.join(self.args.input_dir, image)
                doctor_image_path = os.path.join(self.args.input_dir,
                                                    doctor_image_name)

                new_input_data = DicomSegmentationData()
                new_input_data.file_name = image_name
                new_input_data.read_images(image_path, 
                                            doctor_image_path)

                self.input_data.append(new_input_data)

    def read_one_input(self, image_path, doctor_image_path):
        """Appends only one input file.

        Appends the input data consisting of the DICOM image and doctor
        image specified in the arguments.

        Args:
            image_path (str): DICOM image path.
            doctor_image_path (str): Doctor image path.

        """
        file_name = image_path.split('/')[-1]
        file_name = file_name.split('-')[0]
        
        new_input_data = DicomSegmentationData()
        new_input_data.file_name = file_name
        new_input_data.read_images(image_path, doctor_image_path)

        self.input_data.append(new_input_data)

    def get_input_data(self):
        """Reads all the input data specified.

        Creates an DicomSegmentationData object for each data specified,
        appends them in a list which will be run afterwards.

        """
        if self.no_parser:
            self.read_one_input(image_path=self.args[0],
                                doctor_image_path=self.args[1])
            return

        if self.args.parse_dir == True:
            files = os.listdir(self.args.input_dir)
            
            input_image_names, doctor_image_names = self.get_images_names(files)
            self.select_input_data(input_image_names, doctor_image_names)

        else:
            try:
                self.read_one_input(image_path=self.args.input_HU,
                                    doctor_image_path=self.args.input_seg)
                pass
            except NameError:
                print("Invalid input file.")
    
    def fill_contour(self, image):
        """Utility function that fills the contour given.

        After identifying that there are multiple organs in the doctor
        image, each contour indentified must be filled using this method.

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

    def get_segments(self, data):
        """Creates a DicomSegmentationData object for each contour found.

        Searches in the doctor segmentation image for contours. If they
        are found, an input data will be created  for segmentation for each
        contour found.

        Args:
            data (DicomSegmentationData): Object containing a DICOM image and
                the doctor image.
        
        Returns:
            segments (list): List of DicomSegmentationData objects consisting of
                the same DICOM image and each contour, filled, separated.

        """
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
                new_data = DicomSegmentationData()
                new_data.file_name = data.file_name
                new_data.image = data.image.copy()
                new_data.doctor_image = section.copy()

                segments.append(new_data)
        
        return segments

    def save_solution(self, sections_segmented, original_image):
        """Saves the solution provided by Segmentation.

        Each organ found will be grouped in one image and that image will be
        saved as the solution.

        If the argparse arguments were not used, then the solution will be saved
        in the current folder under the name 'optim.out'.

        Otherwise, the solutions will be saved in the self.args.output_dir
        directory with names 'same_numer-opt.out'.

        For more visualisation of the solutions: uncomment the last two lines
        of this method. (elapsed time for each extraction will increase the time
        the plots are opened)

        Args:
            sections_segmented (list): All the DicomSegmentationData objects
                for the current input data (there is one for each organ,
                if there is more than one).
            original_image (np.array) Image of the original DICOM image,
                used for visualisation purposes only.

        """
        solution = np.zeros((512, 512), dtype=np.uint8)
        doctor_image = np.zeros((512, 512), dtype=np.uint8)

        for section in sections_segmented:
            solution += section.image.astype(np.uint8)
            doctor_image += section.doctor_image.astype(np.uint8)
        
        solution[solution > 1] = 1
        solution = solution.astype(np.uint8)
        doctor_image[doctor_image > 1] = 1

        
        if self.no_parser:
            solution_name = 'optim.out'
        else:
            solution_name = "{}/{}-opt.out".format(self.args.output_dir,
                sections_segmented[0].file_name)

        np.savetxt(solution_name, solution, fmt="%d")
        
        plot_image(solution)
        plot_comparison(original_image, solution, doctor_image)

    def run(self):
        """Executes segmentation for each input provided.

        For each pair of input created, runs the segmentation algorithm. If
        there are multiple organs in the same input, it will produce only one
        solution.

        """
        print("Started running...")
        for input in self.input_data:
            start_time = time.time()
            
            segments = self.get_segments(input)

            print("\nExtracting from {} file.".format(segments[0].file_name))
            print("Organs found: {}".format(len(segments)))
            
            original_image = segments[0].normalize(segments[0].image)
            original_image = segments[0].preprocess(original_image.image)

            sections_segmented = []
            for segment in segments:
                segmentation = Segmentation(segment)
                section_segmented = segmentation.execute()
                
                sections_segmented.append(section_segmented)
            
            self.save_solution(sections_segmented, original_image.image)
            elapsed_time = int(time.time() - start_time)

            minutes = elapsed_time % 3600 // 60
            seconds = elapsed_time % 60
            print("Finished extracting in {:02d}:{:02d}".format(minutes, 
                                                                seconds))
        print("Finished extracting.")

if __name__ == "__main__":
    parser = get_parser()
    no_parser = False
    
    try:
        args = parser.parse_args()
        pass
    except:
        args = sys.argv[1:]
        no_parser = True

    if no_parser:
        if args[0] != '-h':
            solution = Solution(args=args, no_parser=no_parser)
            solution.run()
    else:
        solution = Solution(args=args, no_parser=no_parser)
        solution.run()
        
