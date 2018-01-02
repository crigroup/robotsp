#! /usr/bin/env python
from __future__ import print_function
import time
import unittest
import numpy as np
import resource_retriever
# Tested modules
import robotsp as rtsp

found_pyscipopt = True
try:
  from pyscipopt import Model, quicksum
except ImportError:
  found_pyscipopt = False


class Test_tsp_Module(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Load the test instance only once
    np.set_printoptions(precision=6, suppress=True)
    uri = 'package://lkh_solver/tsplib/berlin52.tsp'
    filename = resource_retriever.get_filename(uri, use_protocol=False)
    cls.graph = rtsp.parser.read_tsplib(filename)
    print('---')
    print('Processing TSPLIB instance:', filename)

  def _run_tsp_test(self, method):
    starttime = time.time()
    tour = method(self.graph)
    duration = time.time() - starttime
    # Assert the tour
    self.assertEqual(len(tour), len(set(tour)))
    self.assertEqual(len(tour), self.graph.number_of_nodes())
    # Report
    cost = rtsp.tsp.compute_tour_cost(self.graph, tour, weight='weight')
    print('') # Dummy line
    print('Cost: {:.3f}'.format(cost))
    if duration is not None:
      print('Duration: {:.4f} s.'.format(duration))
    print(' -> '.join([str(i) for i in tour]))

  def test_metrics(self):
    if found_pyscipopt:
      names = []
      names.append(('dantzig42', 699))  # EXPLICIT - LOWER_DIAG_ROW
      names.append(('eil51', 426))      # EUC_2D
      names.append(('burma14', 3323))   # GEO
      for name, bounds in names:
        uri = 'package://lkh_solver/tsplib/{}.tsp'.format(name)
        filename = resource_retriever.get_filename(uri, use_protocol=False)
        graph = rtsp.parser.read_tsplib(filename)
        tour = rtsp.tsp.scip_solver(graph)
        cost = rtsp.tsp.compute_tour_cost(graph, tour)
        self.assertEqual(cost, bounds)

  def test_nearest_neighbor(self):
    self._run_tsp_test(rtsp.tsp.nearest_neighbor)

  def test_scip_solver(self):
    if found_pyscipopt:
      self._run_tsp_test(rtsp.tsp.scip_solver)

  def test_two_opt(self):
    self._run_tsp_test(rtsp.tsp.two_opt)
