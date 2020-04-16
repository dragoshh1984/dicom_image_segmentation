import argparse

def get_parser():
    parser = argparse.ArgumentParser(description=(
        "DICOM Image Segmentation from DICOM image and doctor selection."
        " If parse_folder is True then each image should respect"
        "the following format:"
        "\"number-HU.in\" for the DICOM image."
        "\"number-seg.in\" for the doctor's segmentation image."
        " You can also run $python 'path_to_image' 'path_to_doctor_image'"
        "and ignore the argparse arguments."
    ))

    parser.add_argument('--input_dir', type=str, default='input_folder',
        help="The root directory location for the input data.")
    
    parser.add_argument('--input_HU', type=str, default='',
        help="The full path of the DICOM input image for segmentation.")
    
    parser.add_argument('--input_seg', type=str, default='',
        help="The full path of the doctor's segmentation.")
    
    parser.add_argument('--parse_dir', type=bool, default=False,
        help="If True is passed, each \"number-HU.in\" and \"number-seg.in\""
        "will be taken as input, one by one.")
    
    parser.add_argument('--output_dir', type=str, default='output_folder',
        help="The location of the directory"
        "where the solution are to be saved.")
    
    return parser

