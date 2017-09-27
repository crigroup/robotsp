#!/usr/bin/env python
import itertools
import numpy as np
import networkx as nx
import progressbar as pbar
# Own modules
from . import metric


def from_coordinate_list(coordinates, distfn=None, args=(), weight='weight'):
  """
  Construct a ``nx.Graph`` using the input coordinates

  Parameters
  ----------
  coordinates: list
    The input coordinates
  distfn: Callable, optional
    The function to be used to compute the distance/weight between nodes. If
    `None`, :meth:`robotsp.metric.euclidean_fn` will be used.
  args: tuple, optional
    Additional arguments to be passed to the ``distfn``
  weight: str, optional
    Use this edge attribute as the edge weight

  Returns
  -------
  rotated: list
    The rotated tour
  """
  if distfn is None:
    distfn = metric.euclidean_fn
  num_nodes = len(coordinates)
  graph = nx.Graph()
  for i in xrange(num_nodes):
    for j in xrange(i+1, num_nodes):
      graph.add_node(i, value=coordinates[i])
      graph.add_node(j, value=coordinates[j])
      dist = distfn(coordinates[i], coordinates[j], *args)
      graph.add_edge(i, j, weight=dist)
  return graph

def from_setslist(setslist, distfn, args=(), weight='weight'):
  if distfn is None:
    distfn = metric.euclidean_fn
  set_sizes = [len(s) for s in setslist]
  num_sets = len(setslist)
  graph = nx.Graph()
  # Generate the list of nodes ids
  start = 0
  sets = []
  for size in set_sizes:
    stop = start+size
    sets.append(range(start, stop))
    start = stop
  # Add nodes and edges
  for i in xrange(num_sets-1):
    set_i_indices = range(set_sizes[i])
    for j in xrange(i+1, num_sets):
      set_j_indices = range(set_sizes[j])
      for k,l in itertools.product(set_i_indices, set_j_indices):
        x = setslist[i][k]
        y = setslist[j][l]
        u = sets[i][k]
        v = sets[j][l]
        graph.add_node(u, value=x)
        graph.add_node(v, value=y)
        graph.add_edge(u, v, weight=distfn(x, y, *args))
  return graph, sets

def from_sorted_setslist(setslist, distfn, args=(), weight='weight',
                                                                verbose=False):
  if distfn is None:
    distfn = metric.euclidean_fn
  set_sizes = [len(s) for s in setslist]
  num_sets = len(setslist)
  graph = nx.Graph()
  # Generate the list of nodes ids
  num_edges = 0
  start = 0
  sets = []
  for i,size in enumerate(set_sizes):
    stop = start+size
    sets.append(range(start, stop))
    start = stop
    if i < len(set_sizes)-1:
      num_edges += set_sizes[i]*set_sizes[i+1]
  # Configure the status bar
  if verbose:
    widgets = ['Populating graph edges: ', pbar.SimpleProgress()]
    widgets += [' ', pbar.Bar(), ' ', pbar.Timer()]
    bar = pbar.ProgressBar(widgets=widgets, maxval=num_edges).start()
    count = 0
  # Add nodes and edges
  for i in xrange(num_sets-1):
    j = i+1
    set_i_indices = range(set_sizes[i])
    set_j_indices = range(set_sizes[j])
    for k,l in itertools.product(set_i_indices, set_j_indices):
      if verbose:
        bar.update(count)
        count += 1
      x = setslist[i][k]
      y = setslist[j][l]
      u = sets[i][k]
      v = sets[j][l]
      graph.add_node(u, value=x)
      graph.add_node(v, value=y)
      graph.add_edge(u, v, weight=distfn(x, y, *args))
  if verbose:
    bar.finish()
  return graph, sets
