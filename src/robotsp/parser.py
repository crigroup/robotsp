#! /usr/bin/env python
import os
import enum
import math
import numpy as np
import networkx as nx
# Local modules
import robotsp as rtsp


class SUPPORTED_PROBLEM_TYPES(enum.Enum):
  TSP   = 1
  GTSP  = 2
  TOUR  = 3

class SUPPORTED_WEIGHT_TYPES(enum.Enum):
  MAN_2D    = 1
  EUC_2D    = 2
  GEO       = 3
  EXPLICIT  = 4

class SUPPORTED_WEIGHT_FORMATS(enum.Enum):
  LOWER_DIAG_ROW  = 1
  FULL_MATRIX     = 2

def fn_euc_2d(x, y):
  distance = rtsp.metric.euclidean_fn(x, y)
  return int(distance + 0.5)

def fn_man_2d(x, y):
  distance = rtsp.metric.manhattan_fn(x, y)
  return int(distance + 0.5)

def get_keyword_index(lines, keyword):
  for i,line in enumerate(lines):
    if keyword in line:
      return i
  return None

def read_tsplib(filename):
  tsp_dict = tsplib_to_dict(filename)
  # Problem type
  problem_type = SUPPORTED_PROBLEM_TYPES[tsp_dict['TYPE']]
  if problem_type == SUPPORTED_PROBLEM_TYPES.GTSP:
    sets = tsp_dict['GTSP_SET_SECTION']
  if problem_type == SUPPORTED_PROBLEM_TYPES.TOUR:
    return tsp_dict['TOUR_SECTION']
  # Select the distance function
  edge_weight_type = SUPPORTED_WEIGHT_TYPES[tsp_dict['EDGE_WEIGHT_TYPE']]
  if edge_weight_type == SUPPORTED_WEIGHT_TYPES.MAN_2D:
    distfn = fn_man_2d
  elif edge_weight_type == SUPPORTED_WEIGHT_TYPES.EUC_2D:
    distfn = fn_euc_2d
  elif edge_weight_type == SUPPORTED_WEIGHT_TYPES.GEO:
    distfn = rtsp.metric.geo_fn
  # Populate the graph
  graph = nx.Graph()
  dimension = tsp_dict['DIMENSION']
  if edge_weight_type != SUPPORTED_WEIGHT_TYPES.EXPLICIT:
    # First the nodes
    for n,x,y in tsp_dict['NODE_COORD_SECTION']:
      graph.add_node(n, coordinate=np.array([x,y]))
    # Add the graph edges
    for i in graph.nodes_iter():
      for j in graph.nodes_iter():
        if i == j:
          continue
        # Skip node pairs from the same set
        if problem_type == SUPPORTED_PROBLEM_TYPES.GTSP:
          for members in sets:
            found_i = i in members
            found_j = j in members
            if found_i and found_j:
              same_set = True
              break
            elif found_i or found_j:
              same_set = False
              break
          if same_set:
            continue
        # Add the edge
        x = graph.node[i]['coordinate']
        y = graph.node[j]['coordinate']
        graph.add_edge(i, j, weight=distfn(x, y))
  else:
    weight_format = SUPPORTED_WEIGHT_FORMATS[tsp_dict['EDGE_WEIGHT_FORMAT']]
    if weight_format == SUPPORTED_WEIGHT_FORMATS.LOWER_DIAG_ROW:
      i,j = (0,0)
      for row in tsp_dict['EDGE_WEIGHT_SECTION']:
        for w in row:
          if i != j:
            graph.add_edge(i, j, weight=w)
          j += 1
          if j > i:
            i += 1
            j = 0
    elif weight_format == SUPPORTED_WEIGHT_FORMATS.FULL_MATRIX:
      i,j = (0,0)
      for row in tsp_dict['EDGE_WEIGHT_SECTION']:
        for w in row:
          if i != j:
            graph.add_edge(i, j, weight=w)
          j += 1
          if j == dimension:
            i += 1
            j = 0
  # Check soundness
  num_nodes = graph.number_of_nodes()
  if num_nodes != dimension:
    raise IOError('DIMENSION mismatch: {0}/{1}'.format(num_nodes, dimension))
  # Return
  if problem_type == SUPPORTED_PROBLEM_TYPES.TSP:
    return graph
  elif problem_type == SUPPORTED_PROBLEM_TYPES.GTSP:
    return (graph, sets)

def tsplib_to_dict(filename):
  lines = []
  # Read the raw file and dump it into a list
  with open(filename, 'rb') as f:
    while True:
      line = f.readline()
      if line.find('EOF') != -1 or not line:
        break
      lines.append(line)
  # Parse the named fields
  remaining_lines = []
  tsp_dict = dict()
  for line in lines:
    if ':' in line:
      key, val_str = (item.strip() for item in line.split(':', 1))
      # Skip the special GTSP_SET_SECTION case
      if key == 'GTSP_SET_SECTION':
        remaining_lines.append(line)
        continue
      try:
        value = int(val_str)
      except:
        value = val_str
      tsp_dict[key] = value
    else:
      remaining_lines.append(line)
  lines = list(remaining_lines)
  # Check that we have a valid problem type
  try:
    problem_type = SUPPORTED_PROBLEM_TYPES[tsp_dict['TYPE']]
  except KeyError:
    raise NotImplementedError(
      'TYPE not supported: {}'.format(tsp_dict['TYPE']))
  if problem_type != SUPPORTED_PROBLEM_TYPES.TOUR:
    # Check that we have a valid edge weight type
    try:
      edge_weight_type = SUPPORTED_WEIGHT_TYPES[tsp_dict['EDGE_WEIGHT_TYPE']]
    except KeyError:
      raise NotImplementedError(
          'EDGE_WEIGHT_TYPE error: {}'.format(tsp_dict['EDGE_WEIGHT_TYPE']))
    if edge_weight_type == SUPPORTED_WEIGHT_TYPES.EXPLICIT:
      # Check that we have a valid edge weight format
      try:
        SUPPORTED_WEIGHT_FORMATS[tsp_dict['EDGE_WEIGHT_FORMAT']]
      except KeyError:
        raise NotImplementedError(
          'EDGE_WEIGHT_FORMAT error: {}'.format(tsp_dict['EDGE_WEIGHT_FORMAT']))
    if edge_weight_type != SUPPORTED_WEIGHT_TYPES.EXPLICIT:
      # Read the coordinates
      keyword = 'NODE_COORD_SECTION'
      first = get_keyword_index(lines, keyword) + 1
      last = first + tsp_dict['DIMENSION']
      tsp_dict[keyword] = []
      for line in lines[first:last]:
        (i,x,y) = line.split()
        tsp_dict[keyword].append([int(i), float(x), float(y)])
    else:
      # Read the matrix as a list of lists
      keyword = 'EDGE_WEIGHT_SECTION'
      i = get_keyword_index(lines, keyword) + 1
      tsp_dict[keyword] = []
      while True:
        try:
          line = lines[i]
          tsp_dict[keyword].append([int(s) for s in line.split()])
          i += 1
        except (ValueError, IndexError):
          break
  else:
    # Read the TOUR_SECTION
    keyword = 'TOUR_SECTION'
    first = get_keyword_index(lines, keyword) + 1
    last = first + tsp_dict['DIMENSION']
    tsp_dict[keyword] = []
    for line in lines[first:last]:
      tsp_dict[keyword].append(int(line))
  # Read the sets for the GTSP case
  if problem_type == SUPPORTED_PROBLEM_TYPES.GTSP:
    keyword = 'GTSP_SET_SECTION'
    first = get_keyword_index(lines, keyword) + 1
    last = first + tsp_dict['GTSP_SETS']
    tsp_dict[keyword] = []
    for line in lines[first:last]:
      tsp_dict[keyword].append([int(s) for s in line.split()[1:-1]])
  return tsp_dict

def write_gtsplib(filename, graph, sets, params):
  num_sets = len(sets)
  num_nodes = graph.number_of_nodes()
  basename = os.path.splitext(os.path.basename(filename))[0]
  ## GTSP File
  header =  'NAME: {}\n'.format(basename)
  header += 'TYPE: GTSP\n'
  header += 'COMMENT: Task with {} targets\n'.format(num_sets)
  header += 'DIMENSION: {}\n'.format(num_nodes)
  header += 'GTSP_SETS: {}\n'.format(num_sets)
  header += 'EDGE_WEIGHT_TYPE: EXPLICIT\n'
  header += 'EDGE_WEIGHT_FORMAT: FULL_MATRIX \n'
  header += 'EDGE_WEIGHT_SECTION\n'
  # Prepare the edge weight section
  nodelist = graph.nodes()
  matrix = np.asarray(nx.to_numpy_matrix(graph, nodelist, weight='weight'))
  matrix = np.int0(matrix * (10**params.sigfigs))
  matrix_str = '\n'.join(' '.join(str(i) for i in row) for row in matrix)
  # Prepare the sets section
  sets_str = '\nGTSP_SET_SECTION\n'
  for i, members in enumerate(sets):
    sets_str += '{} '.format(i+1)
    sets_str += ' '.join(map(str, np.int0(members)+1))
    sets_str += ' -1\n'
  # Write the file
  with open(filename, 'w') as f:
    f.write(header)
    f.write(matrix_str)
    f.write(sets_str)
    f.write('EOF')
  ## PARAMETER_FILE
  path = os.path.dirname(filename)
  fullbasename = os.path.join(path, basename)
  content =  'PROBLEM_FILE = {}\n'.format(filename)
  content += 'ASCENT_CANDIDATES = {:d}\n'.format(params.ascent_candidates)
  # content += 'INITIAL_PERIOD = {:d}\n'.format(params.initial_period)
  content += 'MAX_CANDIDATES = {:d}\n'.format(params.max_candidates)
  content += 'MAX_TRIALS = {:d}\n'.format(params.max_trials)
  content += 'MOVE_TYPE = {:d}\n'.format(params.move_type)
  content += 'OUTPUT_TOUR_FILE = {}\n'.format(fullbasename+'.tour')
  content += 'PI_FILE = {}\n'.format(fullbasename+'.pi')
  content += 'POPULATION_SIZE = {:d}\n'.format(params.population_size)
  content += 'PRECISION = {:d}\n'.format(params.precision)
  content += 'RUNS = {:d}\n'.format(params.runs)
  content += 'SEED = {:d}\n'.format(params.seed)
  content += 'TRACE_LEVEL = {:d}'.format(params.trace_level)
  # Write the file
  with open(fullbasename+'.par', 'w') as f:
    f.write(content)
  return fullbasename
