"""Script for parsing dicom image data/contour data"""

import re
import os
import h5py
import ntpath
import logging

import pandas as pd

from tqdm import tqdm
from functools import reduce
from datetime import datetime
from matplotlib import pylab as plt

from src.parsing import parse_contour_file, parse_dicom_file, poly_to_mask

# system wide parameters
CONTOUR_DATA_PATH = "data/contourfiles"
DICOM_DATA_PATH = "data/dicoms"
LINK_DATA_PATH = "data/link.csv"
OUTPUT_NAME = "output/contour_annotations.hdf5"

contour_RGX = re.compile("\d{4}")

logging.basicConfig(
    filename = "logs/dicom_processing_dicom.log",
    format = '%(asctime)-15s %(pathname)s %(message)-15s',
    level = logging.INFO,
)


# user defined functions
def write_image(path, img_matrix, i_contour, o_contour):
    """Writes image with contour to the given path

    :param: path: the path to write to
    :param: img_matrix: the dicom image
    :param: i_contour: the i-contour image
    :param: o_contour: the o-contour image
    :efect: writes image
    """

    # make subplots
    fig, ax = plt.subplots(2, dpi = 125)

    ax[0].imshow(img_matrix, "gray")
    ax[0].imshow(i_contour, "inferno", alpha = 0.3)

    ax[1].imshow(img_matrix, "gray")
    ax[1].imshow(o_contour, "inferno", alpha = 0.3)

    fig.savefig(path)
    plt.close()

def process_contour(org_id, contour_name, curr_dicom_dict):
    """processes a current patient id (named 'original_id') contour

    :param: org_id: the original_id (from curr_row)
    :param: contour_name: the name of the contour
    :param: curr_dicom_dict: dictionary of dicom files
        keys: dicom id
        value: dicom image np matrix
    :return: dict with
        keys: contour id (mapped to dicom id)
        value: of contour mask
    """
    # construct directories for contour label; original_id is label id
    contour_dir = os.path.join(
        CONTOUR_DATA_PATH, org_id, contour_name
    )

    
    curr_contour_dict = {}
    for curr_contour_file in os.listdir(contour_dir):
        # get contour path
        curr_contour_path = os.path.join(contour_dir, curr_contour_file)

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
                logging.warning("Missing dicom file dicom: orginal_id {},\
                                 contour {}".format(org_id, curr_missing))
                continue

            # save to dict
            curr_contour_dict[contour_id] = poly_to_mask(contour_rslt, *dims)

    return curr_contour_dict

def process_row(curr_row):
    """Processes a current row and results in a dictionary
    key are format of either:
        /patient_id/instance_number/image_matrix
        /patient_id/instance_number/contour
    value:
        the numpy image or contour

    :param: curr_row: a pandas series to process
    :return: dictionary of /patient_id/instance_number/
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


    # process contours into dictionaries
    org_id = curr_row["original_id"]
    i_contour_dict = process_contour(org_id, "i-contours", curr_dicom_dict)
    o_contour_dict = process_contour(org_id, "o-contours", curr_dicom_dict)


    # determine which contours we have missing and log
    missing_contours_i = list(set(curr_dicom_dict.keys()) -\
                            set(i_contour_dict.keys()))
    for curr_missing in missing_contours_i:
        logging.warning("Missing i-contour file: patient {}, contour {}".\
                        format(curr_row["patient_id"], curr_missing))

    missing_contours_o = list(set(curr_dicom_dict.keys()) -\
                            set(o_contour_dict.keys()))
    for curr_missing in missing_contours_o:
        logging.warning("Missing o-contour file: patient {}, contour {}".\
                        format(curr_row["patient_id"], curr_missing))


    # make list of dicts, get keys, and find interesection
    dict_lst = [
        curr_dicom_dict,
        i_contour_dict,
        o_contour_dict
    ]
    k_lst = [list(x.keys()) for x in dict_lst]
    complete_lst = list(reduce(set.intersection, [set(item) for item in k_lst]))


    # using interesection of keys(complete_lst), we now know what to iterate over
    # create rslt_dict to store results
    # rslt_dict has paths organized as:
    # /patient_id/instance_number/image_matrix
    # /patient_id/instance_number/contour
    rslt_dict = {}
    for curr_k in complete_lst:
        base_rslt_key = "{}/{}".format(curr_row["patient_id"], curr_k)

        rslt_dict[base_rslt_key+"/image_matrix"] = curr_dicom_dict[curr_k]
        rslt_dict[base_rslt_key+"/i_contour"] = i_contour_dict[curr_k]
        rslt_dict[base_rslt_key+"/o_contour"] = o_contour_dict[curr_k]

        # write contour mask overlaid on image as a file
        img_path = "{}_{}.png".format(curr_row["patient_id"], curr_k)
        img_path = os.path.join("output", img_path)


        # img_lst is ordered according to dict_lst
        img_lst = [x[curr_k] for x in dict_lst]


        # write image
        write_image(img_path, *img_lst)


    return rslt_dict


# read in link csv
link_df = pd.read_csv(LINK_DATA_PATH)


print("Processing Files:")
rslt_dict_lst = []
for _, row in tqdm(link_df.iterrows()):
    rslt_dict_lst.append(process_row(row))


# create data connection, write files, and clean up
print("Writing Files:")
f_conn = h5py.File(OUTPUT_NAME, "w")


# combine into a single dict
combined_dict = {k: v for d in rslt_dict_lst for k, v in d.items()}
for k, v in combined_dict.items():
    f_conn[k] = v


f_conn.close()
