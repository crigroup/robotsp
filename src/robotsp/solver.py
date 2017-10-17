#!/usr/bin/env python
import os
import time
import itertools
import numpy as np
import networkx as nx
import openravepy as orpy
from subprocess import Popen, PIPE
# Utils
import raveutils as ru
# Local modules
import robotsp as rtsp


class SolverParameters(object):
  def __init__(self):
    # Task space parameters
    self.tsp_solver = rtsp.tsp.two_opt
    self.tspace_metric = rtsp.metric.euclidean_fn
    self.tspace_metric_args = ()
    # Configuration space parameters
    self.cspace_metric = rtsp.metric.max_joint_diff_fn
    self.cspace_metric_args = None
    # kinematics parameters
    self.iktype = orpy.IkParameterizationType.Transform6D
    self.qhome = np.deg2rad([0, -20, 130, 0, 70, 0.])
    self.standoff = None
    self.step_size = np.pi/4.
    self.translation_only = True
    self.penalize_jnt_limits = False
    self.jac_link_name = None
    # Planning parameters
    self.try_swap = True
    self.planner = 'BiRRT'
    self.max_iters = 100
    self.max_ppiters = 30
    # GLKH solver parameters
    self.sigfigs = 3
    self.ascent_candidates = 500
    self.initial_period = 1000
    self.max_candidates = 30
    self.max_trials = 1000
    self.move_type = 5
    self.population_size = 1
    self.precision = 10
    self.runs = 1
    self.seed = 1
    self.trace_level = 1

  def __repr__(self):
    output = ''
    for name in dir(self):
      attr = getattr(self, name)
      if not name.startswith('_') and not callable(attr):
        output += '{0}: {1}\n'.format(name, attr)
    output = output[:-1] # Remove last new line
    return output

  def __str__(self):
    return self.__repr__()

  def initialized(self):
    none_is_valid = ['jac_link_name']
    initialized = True
    for name in dir(self):
      attr = getattr(self, name)
      if name.startswith('_') or callable(attr) or (name in none_is_valid):
        continue
      if attr is None:
        initialized = False
        break
    return initialized

def compute_cspace_trajectories(robot, cgraph, ctour, params):
  cpu_times = []
  trajectories = []
  for idx in xrange(len(ctour)-1):
    u = ctour[idx]
    v = ctour[idx+1]
    qstart = cgraph.node[u]['value']
    qgoal = cgraph.node[v]['value']
    starttime = time.time()
    with robot:
      robot.SetActiveDOFValues(qstart)
      traj = ru.planning.plan_to_joint_configuration(robot, qgoal, params.planner,
                params.max_iters, params.max_ppiters, try_swap=params.try_swap)
    cputime = time.time() - starttime
    cpu_times.append(cputime)
    trajectories.append(traj)
  return trajectories, cpu_times

def compute_execution_time(trajectories):
  execution_time = 0
  for traj in trajectories:
    if traj is None:
      return float('inf')
    execution_time += traj.GetDuration()
  return execution_time

def compute_robot_configurations(robot, targets, params, qstart=None):
  manip = robot.GetActiveManipulator()
  # Compute the IK solutions
  starttime = time.time()
  configurations = []
  if qstart is not None:
    configurations.append([qstart])
  for i,ray in enumerate(targets):
    newpos = ray.pos() - params.standoff*ray.dir()
    newray = orpy.Ray(newpos, ray.dir())
    solutions = ru.kinematics.find_ik_solutions(robot, newray, params.iktype,
                                  collision_free=True, freeinc=params.step_size)
    if len(solutions) == 0:
      raise Exception('Failed to find IK solution for target %d' % i)
    configurations.append(solutions)
  cpu_time = time.time() - starttime
  return configurations, cpu_time

def generate_gtsp_graph(robot, targets, params):
  # Compute the IK solutions and populate the GTSP graph
  if params.qhome is None:
    qhome = robot.GetActiveDOFValues()
  else:
    qhome = np.array(params.qhome)
  configurations, ik_cpu_time = compute_robot_configurations(robot, targets,
                                              params, qstart=qhome, qgoal=qhome)
  # Generate the GTSP graph

  info = dict()
  info['ik_cpu_time'] = ik_cpu_time
  info['metric_cpu_time'] = metric_cpu_time
  return graph, sets, info

def glkh_solver(robot, targets, params, path='~/.ros', cache=None):
  solver_starttime = time.time()
  # Create the working directories where we will save the GTSP files
  def create_dir(dpath):
    if not os.path.isdir(dpath):
      try:
        os.mkdir(dpath)
      except OSError:
        raise OSError('Failed to create: {}'.format(dpath))
  working_path = os.path.join(os.path.expanduser(path), 'robotsp')
  tmp_path = os.path.join(working_path, 'TMP')
  create_dir(working_path)
  create_dir(tmp_path)
  # Check parameters have been initialized
  if not params.initialized():
    raise ValueError('SolverParameters has not been initialized')
  # Compute the IK solutions and populate the GTSP graph
  if params.qhome is None:
    qhome = robot.GetActiveDOFValues()
  else:
    qhome = np.array(params.qhome)
  configurations, ik_cpu_time = compute_robot_configurations(robot, targets,
                                                          params, qstart=qhome)
  # Generate/load the GTSP graph
  starttime = time.time()
  if cache is None:
    graph, sets = rtsp.construct.from_setslist(configurations,
                                params.cspace_metric, params.cspace_metric_args)
  else:
    graph, sets = cache
  metric_cpu_time = time.time() - starttime
  # Write GTSP graph to disc
  num_nodes = graph.number_of_nodes()
  num_sets = len(sets)
  gtsp_file = os.path.join(working_path, '{0}task{1}.gtsp'.format(num_sets,
                                                                  num_nodes))
  basename = rtsp.parser.write_gtsplib(gtsp_file, graph, sets, params)
  # Call the glkh solver
  if params.trace_level > 0:
    outpipe = None
  else:
    outpipe = PIPE
  starttime = time.time()
  process = Popen(['rosrun', 'glkh_solver', 'glkh_solver', basename+'.par'],
                              cwd=working_path, stdout=outpipe, stderr=outpipe)
  stdout, stderr = process.communicate()
  # Read the tour
  tour_filename = basename+'.tour'
  if not os.path.isfile(tour_filename):
    tour = [-1] * (num_sets+1)
    ccost = float('inf')
    traj_cpu_times = [float('inf')] * num_sets
    gtsp_cpu_time = float('inf')
    trajectories = [None] * num_sets
    solver_cpu_time = time.time() - solver_starttime
  else:
    tour = (np.int0(rtsp.parser.read_tsplib(tour_filename)) - 1).tolist()
    tour = rtsp.tsp.rotate_tour(tour, start=0)  # Start from robot home
    tour.append(0)                              # Go back to robot home
    ccost = rtsp.tsp.compute_tour_cost(graph, tour, is_cycle=False)
    gtsp_cpu_time = time.time() - starttime
    # Compute the c-space trajectories
    trajectories, traj_cpu_times = compute_cspace_trajectories(robot, graph,
                                                                  tour, params)
    solver_cpu_time = time.time() - solver_starttime
  # Extra info
  info = dict()
  info['cgraph'] = graph
  info['bins'] = sets
  info['ctour'] = tour
  info['ccost'] = ccost
  info['ik_cpu_time'] = ik_cpu_time
  info['metric_cpu_time'] = metric_cpu_time
  info['gtsp_cpu_time'] = gtsp_cpu_time
  info['traj_cpu_times'] = traj_cpu_times
  info['solver_cpu_time'] = solver_cpu_time
  info['task_execution_time'] = compute_execution_time(trajectories)
  info['stdout'] = stdout
  info['stderr'] = stderr
  # Clean up
  os.rename(basename+'.tour', basename+'-{}.tour'.format(params.max_trials))
  os.remove(basename+'.pi')
  os.remove(basename+'.par')
  return trajectories, info

def robotsp_solver(robot, targets, params):
  """
  Parameters
  ----------
  robot: orpy.robot
    OpenRAVE robot
  targets: list
    List of rays (`orpy.Ray`)
  """
  if not params.initialized():
    raise ValueError('SolverParameters has not been initialized')
  solver_starttime = time.time()
  # Compute the IK solutions and populate the coordinates for the T-Space TSP
  if params.qhome is None:
    qhome = robot.GetActiveDOFValues()
  else:
    qhome = np.array(params.qhome)
  configurations, ik_cpu_time = compute_robot_configurations(robot, targets,
                                                          params, qstart=qhome)
  manip = robot.GetActiveManipulator()
  coordinates = []
  for solutions in configurations:
    qrobot = solutions[0]
    with robot:
      robot.SetActiveDOFValues(qrobot)
      position = manip.GetEndEffectorTransform()[:3,3]
      coordinates.append(position)
  # Step 1: Find the task-space tour
  tgraph = rtsp.construct.from_coordinate_list(coordinates,
                    distfn=params.tspace_metric, args=params.tspace_metric_args)
  starttime = time.time()
  ttour = params.tsp_solver(tgraph)
  ttour = rtsp.tsp.rotate_tour(ttour, start=0)  # Start from robot home
  tsp_cpu_time = time.time() - starttime
  # Step 2: Find optimal robot configurations for the order obtained in step 1
  setslist = [configurations[n] for n in ttour]
  setslist += [[qhome]]
  starttime = time.time()
  cgraph, bins = rtsp.construct.from_sorted_setslist(setslist,
                    distfn=params.cspace_metric, args=params.cspace_metric_args)
  metric_cpu_time = time.time() - starttime
  starttime = time.time()
  ctour = nx.dijkstra_path(cgraph, source=0, target=cgraph.number_of_nodes()-1)
  dijkstra_cpu_time = time.time() - starttime
  # Step 3: Compute the c-space trajectories
  trajectories, traj_cpu_times = compute_cspace_trajectories(robot, cgraph,
                                                                  ctour, params)
  solver_cpu_time = time.time() - solver_starttime
  info = dict()
  # Return the graphs
  info['tgraph'] = tgraph
  info['cgraph'] = cgraph
  info['bins'] = bins
  # Return the tours
  info['ttour'] = ttour
  info['ctour'] = ctour
  # Return the costs
  info['tcost'] = rtsp.tsp.compute_tour_cost(tgraph, ttour, is_cycle=True)
  info['ccost'] = rtsp.tsp.compute_tour_cost(cgraph, ctour, is_cycle=False)
  # Return the cpu times
  info['ik_cpu_time'] = ik_cpu_time
  info['tsp_cpu_time'] = tsp_cpu_time
  info['metric_cpu_time'] = metric_cpu_time
  info['dijkstra_cpu_time'] = dijkstra_cpu_time
  info['traj_cpu_times'] = traj_cpu_times
  info['solver_cpu_time'] = solver_cpu_time
  # Task execution time
  info['task_execution_time'] = compute_execution_time(trajectories)
  return trajectories, info

def tsp_cspace_solver(robot, targets, params):
  if not params.initialized():
    raise ValueError('SolverParameters has not been initialized')
  solver_starttime = time.time()
  # Working entities
  if params.qhome is None:
    qhome = robot.GetActiveDOFValues()
  else:
    qhome = np.array(params.qhome)
  with robot:
    robot.SetActiveDOFValues(qhome)
    home_yoshi = ru.kinematics.compute_yoshikawa_index(robot, params.jac_link_name,
                            params.translation_only, params.penalize_jnt_limits)
  # Select the IK solution with the best manipulability for each ray
  cspace_nodes = [qhome]
  indices = [home_yoshi]
  for i,ray in enumerate(targets):
    newpos = ray.pos() - params.standoff*ray.dir()
    newray = orpy.Ray(newpos, ray.dir())
    solutions = ru.kinematics.find_ik_solutions(robot, newray, params.iktype,
                            collision_free=True, freeinc=params.step_size)
    if len(solutions) == 0:
      raise Exception('Failed to find IK solution for target %d' % i)
    max_yoshikawa = -float('inf')
    for j,q in enumerate(solutions):
      with robot:
        robot.SetActiveDOFValues(q)
        yoshikawa = ru.kinematics.compute_yoshikawa_index(robot, params.jac_link_name,
                            params.translation_only, params.penalize_jnt_limits)
        if yoshikawa > max_yoshikawa:
          max_yoshikawa = yoshikawa
          max_idx = j
    cspace_nodes.append(solutions[max_idx])
    indices.append(max_yoshikawa)
  # Solve the TSP on the configuration space
  cgraph = rtsp.construct.from_coordinate_list(cspace_nodes,
                                params.cspace_metric, params.cspace_metric_args)
  ctour = params.tsp_solver(cgraph)
  ctour = rtsp.tsp.rotate_tour(ctour, start=0) # Start from robot home
  ctour.append(0)                         # Go back to the robot home
  # Compute the c-space trajectories
  trajectories, traj_cpu_times = compute_cspace_trajectories(robot, cgraph,
                                                                  ctour, params)
  solver_cpu_time = time.time() - solver_starttime
  # Store the yoshikawa index in the C-Space graph
  for n in cgraph.nodes_iter():
    cgraph.node[n]['yoshikawa'] = indices[n]
  info = dict()
  # Return the graph
  info['cgraph'] = cgraph
  # Return the tour
  info['ctour'] = ctour
  # Return the costs
  info['ccost'] = rtsp.tsp.compute_tour_cost(cgraph, ctour, is_cycle=False)
  # Return the cpu times
  info['traj_cpu_times'] = traj_cpu_times
  info['solver_cpu_time'] = solver_cpu_time
  # Task execution time
  info['task_execution_time'] = compute_execution_time(trajectories)
  return trajectories, info
