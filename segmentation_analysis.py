# import libraries
import h5py
import logging
import numpy as np
import seaborn as sns

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

# get images
img_mtx = np.stack([f_conn[x]["image_matrix"] for x in f_keys])

# get indicies of masks
# indexing: 0: img index, 1: x, 2: y
i_cont_loc = np.where(i_contr_mtx)
sub_cont_loc = np.where(sub_contr_mtx)


# show plot of inner and subtraction (between i- and o- contours) contour
# make subplots
img_indx = 0
fig, ax = plt.subplots(2, dpi = 150)

# inner contour
ax[0].imshow(img_mtx[img_indx], "gray")
ax[0].imshow(i_contr_mtx[img_indx], "inferno", alpha = 0.3)

# contour between inner and outer contours
ax[1].imshow(img_mtx[img_indx], "gray")
ax[1].imshow(sub_contr_mtx[img_indx], "inferno", alpha = 0.3)
ax[1].arrow(50, 100, 80, 27, color="red", head_width=5, head_length=7)

# figure titple
fig.suptitle("Inner and in bwetween inner/outer contours.", fontsize=10)


# assessment of vairance of i-contours between cases
for i, curr_k in enumerate(f_keys):
    # get contour indicies for the current case, and plot to seaborn
    curr_i_indx = np.where(i_contr_mtx[i])
    sns.kdeplot(img_mtx[i][curr_inner_indx])
plt.xlim(0, 750) # ~ min max of contour values
plt.suptitle("Distribution of per slice inner contours.\n\
             Each line represents a unqiue slice's distribution", fontsize=10)


# assessment of vairance of subtraction contours (outer - inner) between cases
for i, curr_k in enumerate(f_keys):
    # get subtraction contour indicies for the current case, and plot to seaborn
    curr_sub_indx = np.where(sub_contr_mtx[i])
    sns.kdeplot(img_mtx[i][curr_sub_indx])
plt.xlim(0, 750) # ~ min max of contour values
plt.suptitle("Distribution of per slice in between inner/outer contours.\n\
             Each line represents a unqiue slice's distribution", fontsize=10)


# assessment of overall
sns.kdeplot(img_mtx[i_cont_loc], shade=True, label="Inner Contour")
sns.kdeplot(img_mtx[sub_cont_loc], shade=True, label="Between Inner and Outer Contour")
plt.suptitle("Distributions of intensities.\n", fontsize=10)


# lets say the threshold is ~125
thresh = 125
