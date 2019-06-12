# import libraries
import h5py
import logging
import numpy as np

from matplotlib import pylab as plt

# system wide variables
DATA_PATH = "output/contour_annotations.hdf5"

# user defined functions
def make_subtraction_contour(curr_k, f_conn):
    """makes contours between outer and inner contours
    :params: curr_k: the current key to process
    :params: f_conn: hdf5 file connection
    :returns: contour between outer and inner contours
    """
    # get contours
    i_contr = f_conn[curr_k]["i_contour"][()]
    o_contr = f_conn[curr_k]["o_contour"][()]

    # convert to int for easier substraction
    i_contr = i_contr.astype('int')
    o_contr = o_contr.astype('int')

    # subtract, convert to bool, and return
    return (o_contr - i_contr).astype('bool')


# read in hdf5 connection
f_conn = h5py.File(DATA_PATH, "r")


# get keys for cases
f_keys = list(f_conn.keys())
f_keys = [["{}/{}".format(x,y) for y in f_conn[x]] for x in f_keys]
f_keys = [x for y in f_keys for x in y]


# get inner contour and subtraction contours (outer - inner) into matrix
i_contr_mtx = np.stack([f_conn[x]["i_contour"] for x in f_keys])
sub_contr_mtx = np.stack([make_subtraction_contour(x, f_conn) for x in f_keys])
