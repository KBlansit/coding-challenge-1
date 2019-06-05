"""Script for parsing dicom image data/contour data"""

import re
import os
import h5py
import ntpath
import logging

import pandas as pd

from tqdm import tqdm
from datetime import datetime
from matplotlib import pylab as plt

from src.parsing import parse_contour_file, parse_dicom_file, poly_to_mask

CONTOUR_DATA_PATH = "data/contourfiles"
DICOM_DATA_PATH = "data/dicoms"
LINK_DATA_PATH = "data/link.csv"

contour_RGX = re.compile("\d{4}")

logging.basicConfig(
    filename = "logs/dicom_processing_dicom.log",
    format = '%(asctime)-15s %(pathname)s %(message)-15s',
    level = logging.INFO,
)

# user defined functions
def write_image(path, img_matrix, overlay):
    """Writes image with contour to the given path

    param: path: the path to write to
    param: img_matrix: the dicom image
    param: overlay: the overlay image
    efect: writes image
    """
    fig, ax = plt.subplots(1, dpi = 125)

    ax.imshow(img_matrix, "gray")
    ax.imshow(overlay, "inferno", alpha = 0.3)

    fig.savefig(path)
    plt.close()

def process_row(curr_row):
    """Processes a current row and results in a dictionary
    key are format of either:
        /patient_id/instance_number/image_matrix
        /patient_id/instance_number/contour
    value:
        the numpy image or contour

    param: curr_row: a pandas series to process
    return: dictionary of /patient_id/instance_number/
    """
    # construct directories for input
    dicom_dir = os.path.join(DICOM_DATA_PATH, curr_row["patient_id"])

    curr_dicom_dict = {}
    for curr_dicom_file in os.listdir(dicom_dir):
        # get dicom path
        curr_dicom_path = os.path.join(dicom_dir, curr_dicom_file)

        # will return either none or list of contours
        dicom_dict = parse_dicom_file(curr_dicom_path)
        if dicom_dict:
            dicom_id = ntpath.basename(curr_dicom_path).split(".")[0]
            curr_dicom_dict[dicom_id] = dicom_dict["pixel_data"]


    # construct directories for contour label; original_id is label id
    i_contour_dir = os.path.join(
        CONTOUR_DATA_PATH, curr_row["original_id"], "i-contours"
    )

    curr_contour_dict = {}
    for curr_contour_file in os.listdir(i_contour_dir):
        # get contour path
        curr_contour_path = os.path.join(i_contour_dir, curr_contour_file)

        # will return either list of countirs
        contour_rslt = parse_contour_file(curr_contour_path)
        if contour_rslt:
            # get the contour id to match to instance number
            # we use index [1] since it's the 2nd part of the string name
            contour_id = ntpath.basename(curr_contour_path).split(".")[0]
            contour_id = str(int(contour_RGX.findall(contour_id)[1]))

            # if we don't have contour, skip pass case and log
            if contour_id in curr_dicom_dict:
                dims = curr_dicom_dict[contour_id].shape
            else:
                logging.warning("Missing dicom file dicom: patient {},\
                                 contour {}".format(curr_row["patient_id"], curr_missing))
                continue

            # save to dict
            curr_contour_dict[contour_id] = poly_to_mask(contour_rslt, *dims)


    # determine which contours we have missing and log
    missing_contours = list(set(curr_dicom_dict.keys()) -\
                            set(curr_contour_dict.keys()))
    for curr_missing in missing_contours:
        logging.warning("Missing i-contour file: patient {}, contour {}".\
                        format(curr_row["patient_id"], curr_missing))

                        
    # get intersection of files so we know what to interate over
    intersection_lst = list(set(curr_contour_dict.keys()) &\
                            set(curr_dicom_dict.keys()))

    # create rslt_dict to store results
    # rslt_dict has paths organized as:
    # /patient_id/instance_number/contour
    # /patient_id/instance_number/image
    rslt_dict = {}
    for curr_k in intersection_lst:
        base_rslt_key = "{}/{}".format(curr_row["patient_id"], curr_k)

        rslt_dict[base_rslt_key+"/image"] = curr_dicom_dict[curr_k]
        rslt_dict[base_rslt_key+"/i_contour"] = curr_contour_dict[curr_k]

        # write contour mask overlaid on image as a file
        img_path = "{}_{}.png".format(curr_row["patient_id"], curr_k)
        img_path = os.path.join("output", img_path)

        # write image
        write_image(img_path, curr_dicom_dict[curr_k], curr_contour_dict[curr_k])

    return rslt_dict


# read in link csv
link_df = pd.read_csv(LINK_DATA_PATH)

print("Processing Files:")

rslt_dict_lst = []
for i, row in tqdm(link_df.iterrows()):
    rslt_dict_lst.append(process_row(row))

# combine into a single dict
combined_dict = {k: v for d in rslt_dict_lst for k, v in d.items()}

# create data connection, write files, and clean up
print("Writing Files:")
f_conn = h5py.File("output/contour_annotations.h5py", "w")
for k, v in combined_dict.items():
    f_conn[k] = v

f_conn.close()
