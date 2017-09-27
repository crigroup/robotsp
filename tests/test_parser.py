#! /usr/bin/env python
from __future__ import print_function
import os
import re
import unittest
import numpy as np
import resource_retriever
# Tested module
from robotsp import parser


class Test_tsp_Module(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    folder = 'package://lkh_solver/tsplib'
    path = resource_retriever.get_filename(folder, use_protocol=False)
    tmp = []
    for name in os.listdir(path):
      fullpath = os.path.join(path, name)
      extension = os.path.splitext(name)
      if os.path.isfile(fullpath) and extension in ['.gtsp', '.tsp', '.tour']:
        tmp.append(fullpath)
    cls.files = tmp
    print('---')
    print('Parsing all the files here:', folder)

  def test_read_tsplib(self):
    for filename in self.files:
      num_nodes = np.max([int(s) for s in re.findall(r'\d+', filename)])
      # Construct the graph only for instances smaller than 100 nodes
      if num_nodes < 100:
        g = parser.read_tsplib(filename)

  def test_tsplib_to_dict(self):
    # Parse all the available instances
    [parser.tsplib_to_dict(f) for f in self.files]
