#!/usr/bin/env python

# quick and dirty fix to solve h5py file locks
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import MiST

MiST.init()
