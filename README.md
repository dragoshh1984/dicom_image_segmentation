# DICOM Image Segmentation

Solution for organ segmentation from DICOM images.

# Repository Structure

1. ```src``` -> Code for the repo and folder for input and output.
    * [Solution.py](https://github.com/dragoshh1984/dicom_image_segmentation/blob/master/src/Solution.py) : File to run the segmentation.
    * [Segmentation.py](https://github.com/dragoshh1984/dicom_image_segmentation/blob/master/src/Segmentation.py) : File containing the segmentation methods.
    * [DicomSegmentationData.py](https://github.com/dragoshh1984/dicom_image_segmentation/blob/master/src/DicomSegmentationData.py) : File containing image processing methods.
    * [create_input_file.py](https://github.com/dragoshh1984/dicom_image_segmentation/blob/master/src/create_input_file.py) : Script used to create .in file from .png images.
    * [utils.py](https://github.com/dragoshh1984/dicom_image_segmentation/blob/master/src/utils.py) : File containing visualisation methods.
    * [config.py](https://github.com/dragoshh1984/dicom_image_segmentation/blob/master/src/config.py) : File containing the argparse setup for [Solution.py](https://github.com/dragoshh1984/dicom_image_segmentation/blob/master/src/Solution.py).
2. [workspace.ipynb](https://github.com/dragoshh1984/dicom_image_segmentation/blob/master/workspace.ipynb) : Notebook where all the experimenting was done.

# How to use

Via the argparse arguments:

```
usage: Solution.py [-h] [--input_dir INPUT_DIR] [--input_HU INPUT_HU]
                   [--input_seg INPUT_SEG] [--parse_dir PARSE_DIR]
                   [--output_dir OUTPUT_DIR]

DICOM Image Segmentation from DICOM image and doctor selection. If
parse_folder is True then each image should respectthe following
format:"number-HU.in" for the DICOM image."number-seg.in" for the doctor's
segmentation image. You can also run $python 'path_to_image'
'path_to_doctor_image'and ignore the argparse arguments.

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        The root directory location for the input data.
  --input_HU INPUT_HU   The full path of the DICOM input image for
                        segmentation.
  --input_seg INPUT_SEG
                        The full path of the doctor's segmentation.
  --parse_dir PARSE_DIR
                        If True is passed, each "number-HU.in" and "number-
                        seg.in"will be taken as input, one by one.
  --output_dir OUTPUT_DIR
                        The location of the directorywhere the solution are to
                        be saved.
```
Via terminal direct arguments:

```
$ python Solution.py 'path/to/dicom/image/999-HU.in' 'path/to/doctor/image/999-seg.in'
```
The images name must respect the format:
```
same_number-HU.in
same_number-seg.in
```

Chosing the later version, the Solution.py acts as it is specified in the doc file
saving the solution under the name optim.out, instead of 999-opt.out as in the other
cases.

# Implementation details

Before the image would go into processing techniques to extract the desired
organ, the image is first normalized scaling its values from 0 to 255.

After normalization, a threshold is applied in order to get rid of the unnecessary 
information (the gray circle surrounding the image, for example).

In order to create a higher contrast, a histogram equalization is done, followed
by a denoising method; have tried here denoising using median blur, gaussian blur
and non-local means denoising. The later proven beter results so we implemented
that one.

From this point on I've tried 4 different methods:

## 1. Region Growing

In order to apply this technique I needed to have the desired region separated
from the rest, via thresholding. I've experimented with few methods like:
using the image mean, bimodal thresholding, maximixe interclass variance
thresholding and local thresholding out of which the last two provided decent results
but not necessarily good.

I've chosen to use a manual thresholding, keeping the values from a range of pixel
values that are present in the doctor's section of the DICOM image.

Using morphology opening I was able to get rid of the white noise around the 
section and using binary closing, I've filled the holes from that section. Antoher
morphology transformation and some blur using a median filter and that is the 
first solution.

## 2. Watershed Method

I've taken the solution from the previous method as input for this solution.
I had to separate the sure foreground, sure background and unkown region, and then
based on these regions and looking at the DICOM image, the algorithm will fill
the unknown region up to where it finds it suitable (less than the sure background).

I applied again a median filter to blur the edges and that's it for this solution.

## 3. Contour Method

Using the denoised version of the DICOM image, detailed above, I've decided to
plot the contours found in the image, first by using the Canny algorithm. Canny was
able to detect the contours nicely, but without closing them. 

I've tried also plotting the contours using the marching squares technique, that
was also able to close them, but didn't do this quite well. As I was analysing both
plots I saw that canny nicely closes the contours that were not closed well with the
marching sqares method.

Next step was to draw the contours around the section of interest provided by Canny
algorithm over the DICOM image, afterward I've applied the marching squares and it
provided much better results.

Having a collection of contours from the DICOM image I've selected the one that
interested me the most by calculating the intersection area between each contour
and the doctor's section contour.

I applied a median filter to blur the edges for this method as well and I've
obtained the 3rd and most probably the best solution.

## 4. Superpixel Method

The last method I've experimented with is the SLIC (Simple Linear Iterative Clustering)
algorithm for Superpixel generation. Unfortunately, the clusters that were found
didn't quite demarcate well the organs from the DICOM image. I've tried many 
values for the parameters, but no decent solution was found.

There should be a nice solution done by this method, as it is widely used in 
computer vision, maybe using other clustering method or tweaking a bit more the
parameters.

## 5. Other interesting methods available (have not tried them)

Random Walker, Graph Cuts.

# Multiple organ selections

I've used the marching squares method on the doctor's image
in order to determine how many contours are found. Given that the image is
binary, the contours are very easily found. If there is more than one organ 
then more contours would be found.

After all the contours are found and stored, next I've filled them using binary 
dilations and save each filled contour on a different binary image.

For each binary image with filled contour, I've grouped it with the same DICOM
image and run the segmentation.

At the end I've added altogether the solutions and obtained the solution for multiple organ segmentation.

# Visualisation and Evaluation

While experimenting I've used precision, recall and f1-score between my solution
and yours; in order to determine how close the solution was.

Because I thought that one solution might be better in certain cases than other,
I've kept all the solutions in a list and calculate the f1-score between the solution
and the doctor's image (because, the doctor is the closest thing we have to our
solution) and selected the solution with the highest score.

For visualisation I've decided to plot my solution's contour and the doctor's
over the DICOM image in order to be able to appreciate if the solution is good.
