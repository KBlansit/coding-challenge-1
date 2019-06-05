"""Unit tests for DataGenerator"""

import unittest

import numpy as np

from src.data_generator import DataGenerator

np.random.seed(11928127)

DATA_PATH = "output/contour_annotations.hdf5"

class TestDataGenerator(unittest.TestCase):

    @classmethod
    def setUp(cls):
        """Set up class method
        """
        cls.dg = DataGenerator(DATA_PATH, 8, True)

    def test_data_generator(self):
        """Tests that the data generator works properly
        """
        # get generator object
        curr_gen = self.dg.data_generator()

        # calculate number of steps
        num_steps = self.dg.get_num_steps()

        # do once
        rslt_lst_1 = []
        for i in range(num_steps):
            curr_ids = next(curr_gen)
            rslt_lst_1.append(curr_ids)

        rslt_lst_1 = [list(x) for x in rslt_lst_1]
        rslt_lst_1 = [x for y in rslt_lst_1 for x in y]

        # do twice
        rslt_lst_2 = []
        for i in range(num_steps):
            curr_ids = next(curr_gen)
            rslt_lst_2.append(curr_ids)

        rslt_lst_2 = [list(x) for x in rslt_lst_2]
        rslt_lst_2 = [x for y in rslt_lst_2 for x in y]

        # asssert both lists are not the same
        self.assertNotEqual(rslt_lst_1, rslt_lst_2)

        # get all keys
        all_keys = self.dg.train_keys

        # find differences between rslt lists and actual keys
        diff_rslt_1 = len(set(self.dg.train_keys) - set(rslt_lst_1))
        diff_rslt_2 = len(set(self.dg.train_keys) - set(rslt_lst_2))

        self.assertIs(diff_rslt_1, 0)
        self.assertIs(diff_rslt_2, 0)


if __name__ == "__main__":
    unittest.main()
