"""Defines the data generator functions"""

import h5py
import numpy as np

class DataGenerator():
    def __init__(self, f_conn_path, batch_size, debug = False):
        """Constructor
        :param: f_conn_path: path to hdf5 data
        :param: batch_size: number of cases to show to NN at a time
        :debug: debug: debugging flag used for unit tests
        """
        self.f_conn = h5py.File(f_conn_path, "r")
        self.batch_size = batch_size
        self.train_keys = self._get_all_f_keys()
        self.debug = debug

    def _get_all_f_keys(self):
        """Given stored data connection, returns paths that define each instance

        Note:
        key are format of either:
            /patient_id/instance_number/image_matrix
            /patient_id/instance_number/contour

        :return: numpy array of keys/paths
        """
        # list all key paths
        f_keys = list(self.f_conn.keys())
        f_keys = [["{}/{}".format(x,y) for y in self.f_conn[x]] for x in f_keys]
        f_keys = [x for y in f_keys for x in y]

        return np.array(f_keys)

    def _construct_data(self, path_lst):
        """Given a list of paths (for keys), return the image and contours

        :param: path_lst: list of path keys
        "param: f_conn: the hdf5 data connection
        :return: tuple of: x_input, y_input
        """
        # construct paths
        x_paths = ["{}/image_matrix".format(x) for x in path_lst]
        y_paths = ["{}/i_contour".format(x) for x in path_lst]

        # make numpy arrays
        x_input = np.stack([self.f_conn[x][()] for x in x_paths])
        y_input = np.stack([self.f_conn[x][()] for x in y_paths])

        # expand dims
        x_input = np.expand_dims(x_input, axis=-1)
        y_input = np.expand_dims(y_input, axis=-1)

        return x_input, y_input

    def _make_indicies(self):
        """Makes list of lists for slices

        :return: returns list of lists
        """
        # determine min number of cuts so we can make that many indicies lists
        min_cuts = np.floor(len(self.train_keys)/self.batch_size).astype('int')

        slices_ary = np.arange(0, min_cuts*self.batch_size)
        slices_ary = slices_ary.reshape(min_cuts, self.batch_size)

        slice_lst = slices_ary.tolist()

        # add additional slice if necessary
        if len(self.train_keys) % self.batch_size:
            additional_slc = np.arange(min_cuts*self.batch_size, len(self.train_keys)).tolist()
            slice_lst.append(additional_slc)

        # convert into arrays for proper indexing
        slice_lst = [np.array(x) for x in slice_lst]

        return slice_lst

    def get_num_steps(self):
        """Getter method for determining number of steps per epoch

        :return: number of steps per epoch
        """
        return np.ceil(len(self.train_keys)/self.batch_size).astype('int')

    def data_generator(self, debug = False):
        """The data generator for fit_generator

        :debug: debug flag
        :yield: next batch of self.batch_size inputs/labels
        """
        slc_lst = self._make_indicies()

        # do forever
        while True:
            # randomize at start
            rnd_k_ary = np.random.choice(self.train_keys, len(self.train_keys),\
                                         replace=False)

            for curr_slc in slc_lst:
                # get slice of keys
                curr_keys = rnd_k_ary[curr_slc]

                # return prematurely with debug flag
                if self.debug:
                    yield curr_keys
                else:
                    yield self._construct_data(curr_keys)
