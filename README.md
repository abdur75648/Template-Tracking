# Template-Tracking #
- - - - 
*Tracking a moving object over time using OpenCV*

Video tracking is the process of locating a moving object (or multiple objects) over time using a moving camera. This repo uses traditional non-learning-based object tracking techniques to track specific objects in the given test sequences. Such a system needs no prior training and even may not have seen an object prior to tracking.

This repo uses OpenCV to do template tracking using 3 different methods:

1. ***Block Matching Technique:*** This is an appearance based method. We've implemented this using 2 metrices: sum of squared distance (SSD) as well as normalised cross correlation (NCC)
2. ***Lucas Kanade Algorithm:*** This is one of the methods based on the optical flow measurement
3. ***Pyramid based Lucas Kanade:*** This an improvement over the LKA using a multiscale Gaussian pyramid

### Dataset Structure ###
The dataset consists of video sequences and their ground truth bounding boxes.

Each datatset can be put in a folder (say dataset_name) inside this folder.

The video sequence resides inside a subfolder 'img' of the 'dataset_name' folder.

The ground truth file that resides inside the 'dataset_name' folder. It consists of bounding boxes per frame of the format (x, y, h, w) where (x,y) are the coordinates of the leftmost corner of the object and (h,w) are the height and width of the object.

We've considered the ground truth for the first frame as the object template for tracking the object in future frames.

### Using This Repo ###
1. Download & put the data folder inside the folder. Ensure the same format as described above
2. Perform the template tracking using any of the 4 methods using command:

` python script_name -i datasetfolder `

3. Evaluate the performance using:

` python eval.py -i datasetfolder `

where i is the input dataset folder, and script_name can be any one of these: [block_matching_ncc.py](https://github.com/abdur75648/Template-Tracking/blob/main/block_matching_ncc.py), [block_matching_ssd.py](https://github.com/abdur75648/Template-Tracking/blob/main/block_matching_ssd.py), [lka.py](https://github.com/abdur75648/Template-Tracking/blob/main/lka.py) or [pylka.py](https://github.com/abdur75648/Template-Tracking/blob/main/pylka.py)
