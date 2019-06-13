# import libraries
import cv2
import h5py
import logging
import numpy as np
import seaborn as sns

from matplotlib import pylab as plt

# system wide variables
DATA_PATH = "output/contour_annotations.hdf5"

# user defined functions
def make_myocardium_contour(curr_k, f_conn):
    """makes contours between outer and inner contours to define myocardium
    :params: curr_k: the current key to process
    :params: f_conn: hdf5 file connection
    :returns: contour myocardium contour
    """
    # get contours
    i_contr = f_conn[curr_k]["i_contour"][()]
    o_contr = f_conn[curr_k]["o_contour"][()]

    # convert to int for easier substraction
    i_contr = i_contr.astype('int')
    o_contr = o_contr.astype('int')

    # subtract, convert to bool, and return
    return (o_contr - i_contr).astype('bool')

def make_threshold_based_segmentation(curr_k, f_conn):
    """makes threshold based contours of myocardium
    :params: curr_k: the current key to process
    :params: f_conn: hdf5 file connection
    :returns: segmentation mask of myocardium
    """

    # get outer contour and image
    o_contr = f_conn[curr_k]["o_contour"][()]
    img = f_conn[curr_k]["image_matrix"][()]

    # otsu binary threshold
    threshold = cv2.threshold(
        img.astype(np.uint8),
        img.min(),
        img.max(),
        type=cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )[0]

    # find pixels that are about threshold within outer contour
    # these pixels are defined then as the
    pred_i_contour = o_contr.copy()
    o_indx = np.where(pred_i_contour)
    pred_i_contour[o_indx] = img[o_indx] > threshold

    # return
    return pred_i_contour

def dice(im1, im2):
    """Computes the Dice coefficient, a measure of set similarity.
    adapted from: https://gist.github.com/JDWarner/6730747

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.

    Maximum similarity = 1
    No similarity = 0

    :params: im1: array-like, bool
    :params: im2: array-like, bool
    :returns: dice similarity coefficient (float) [0-1]
    """

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

# read in hdf5 connection
f_conn = h5py.File(DATA_PATH, "r")


# get keys for cases
f_keys = list(f_conn.keys())
f_keys = [["{}/{}".format(x,y) for y in f_conn[x]] for x in f_keys]
f_keys = [x for y in f_keys for x in y]


# get all contours
# here, m_contour_mtx is the GT true contour
i_contr_mtx = np.stack([f_conn[x]["i_contour"] for x in f_keys])
o_contr_mtx = np.stack([f_conn[x]["o_contour"] for x in f_keys])
m_contr_mtx = np.stack([make_myocardium_contour(x, f_conn) for x in f_keys])

# get images
img_mtx = np.stack([f_conn[x]["image_matrix"] for x in f_keys])

# get indicies of masks
# indexing: 0: img index, 1: x, 2: y
i_cont_loc = np.where(i_contr_mtx)
m_cont_loc = np.where(m_contr_mtx)


# show plot of inner and subtraction (between i- and o- contours) contour
# make subplots
img_indx = 0
fig, ax = plt.subplots(2, dpi = 150)

# inner contour
ax[0].imshow(img_mtx[img_indx], "gray")
ax[0].imshow(i_contr_mtx[img_indx], "inferno", alpha = 0.3)

# contour between inner and outer contours
ax[1].imshow(img_mtx[img_indx], "gray")
ax[1].imshow(m_contr_mtx[img_indx], "inferno", alpha = 0.3)
ax[1].arrow(50, 100, 80, 27, color="red", head_width=5, head_length=7)

# figure titple
fig.suptitle("Blood pool (inner) and myocardial (outer - inner) contours.",
             fontsize=10)


# assessment of vairance of i-contours between cases
for i, curr_k in enumerate(f_keys):
    # get contour indicies for the current case, and plot to seaborn
    curr_i_indx = np.where(i_contr_mtx[i])
    sns.kdeplot(img_mtx[i][curr_i_indx])
plt.xlim(0, 750) # ~ min max of contour values
plt.suptitle("Distribution of per slice blood pool (inner) contours.\n\
             Each line represents a unqiue slice's distribution", fontsize=10)


# assessment of vairance of subtraction contours (outer - inner) between cases
for i, curr_k in enumerate(f_keys):
    # get subtraction contour indicies for the current case, and plot to seaborn
    curr_m_indx = np.where(m_contr_mtx[i])
    sns.kdeplot(img_mtx[i][curr_m_indx])
plt.xlim(0, 750) # ~ min max of contour values
plt.suptitle("Distribution of per slice myocardial (outer - inner) contours.\n\
             Each line represents a unqiue slice's distribution", fontsize=10)


# assessment of overall
sns.kdeplot(img_mtx[i_cont_loc], shade=True,
            label="Blood pool (inner) contour")
sns.kdeplot(img_mtx[m_cont_loc], shade=True,
            label="Myocardial (outer - inner) contour")
plt.suptitle("Distributions of intensities.", fontsize=10)


# get predicted contours from this
pred_i_contour = [make_threshold_based_segmentation(x, f_conn) for x in f_keys]

# get all dice scores
dice_lst = []
for i, curr_k in enumerate(f_keys):
    dice_lst.append(dice(pred_i_contour[i], i_contr_mtx[i]))

dice_txt = "Mean DICE: {}±{}".format(
    round(np.mean(dice_lst), 2),
    round(np.std(dice_lst), 2),
)
sns.boxplot(dice_lst)
plt.suptitle("Box and whisker plot of DICE coefficients\n\
             " + dice_txt, fontsize=10)
plt.xlabel("DICE Coefficient")
