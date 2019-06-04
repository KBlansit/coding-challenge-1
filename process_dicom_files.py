"""Script for parsing dicom image data/contour data"""

import os
import ntpath
import logging

import pandas as pd

from src.parsing import parse_contour_file, parse_dicom_file, poly_to_mask

CONTOUR_DATA_PATH = "data/contourfiles"
DICOM_DATA_PATH = "data/dicoms"
LINK_DATA_PATH = "data/link.csv"

logging.basicConfig(
    filename = "logs/dicom_processing_dicom.log",
    format = '%(asctime)-15s %(pathname)s %(message)-15s',
    level = logging.DEBUG,
)

# user defined function

# read in link csv
link_df = pd.read_csv(LINK_DATA_PATH)

for i, row in link_df.iterrows():

    # initialize
    curr_img_

    # construct directories for input
    dcm_dir = os.path.join(DICOM_DATA_PATH, row["patient_id"])

    # get list of dicom paths
    for curr_file in os.listdir(dcm_dir):
        curr_dicom_path = os.path.join(dcm_dir, curr_file)

        parse_dicom_file(curr_dicom_path)['pixel_data']






    contour_dir = os.path.join(DICOM_DATA_PATH, row["original_id"])

    # construct subdirectories for inner contours
    inner_contour_dir = os.path.join(contour_dir, "i-contours")
