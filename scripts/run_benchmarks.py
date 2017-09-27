#!/usr/bin/env python
import os
import h5py
import time
import rospy
import logging
import argparse
import numpy as np
import networkx as nx
import resource_retriever
import progressbar as pbar
import openravepy as orpy
from collections import defaultdict
# Utils
import criutils
import raveutils.conversions as orconv
import raveutils.kinematics as orkin
# RoboTSP
import robotsp as rtsp

logger = logging.getLogger('run_benchmarks')


def parse_args():
  # Remove extra IPython notebook args
  clean_argv = rospy.myargv()[1:]
  if '-f' in clean_argv:
    clean_argv = clean_argv[2:]
  # Parse
  format_class = argparse.RawDescriptionHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=format_class,
                  description='Run RoboTSP benchmarks')
  # Kinematics parameters
  parser.add_argument('--standoff', metavar='', type=float, default=0.01,
    help='Standoff distance from the panel (meters). Default=%(default).3f')
  parser.add_argument('--step-size', metavar='', type=float, default=45,
    help='Discretization step-size for the free-DoF (deg). Default=%(default).1f')
  # Planning PlannerParameters
  parser.add_argument('--planner', metavar='', type=str, default='birrt',
    help='OpenRAVE planner to be used. Default=%(default)s')
  parser.add_argument('--max-iters', metavar='', type=int, default=100,
    help='Max planning iterations. Default=%(default)d')
  parser.add_argument('--max-ppiters', metavar='', type=int, default=30,
    help='Max post-processing iterations. Default=%(default)d')
  # Benchmarking
  parser.add_argument('--all', action='store_true',
    help='If set, will run all the below benchmarks')
  parser.add_argument('--tsp-solvers', action='store_true',
    help='If set, will benchmark three TSP solvers')
  parser.add_argument('--metrics', action='store_true',
    help='If set, will benchmark three C-Space metrics')
  parser.add_argument('--discretization', action='store_true',
    help='If set, will benchmark several discretization step sizes')
  parser.add_argument('--others', action='store_true',
    help='If set, will benchmark RoboTSP against two alternative methods')
  args = parser.parse_args(clean_argv)
  return args

def tsp_solvers_benchmarking(f, targets, num_targets_list):
  # Create HDF5 group for the TSP solvers benchmarks
  group_name = 'tsp'
  if group_name in f.keys():
    del f[group_name]
  group = f.create_group(group_name)
  group['solvers'] = ['Exact', '2-Opt', 'RNN']
  group['number_of_targets'] = num_targets_list
  tsp_solvers = [ rtsp.tsp.scip_solver,
                  rtsp.tsp.two_opt,
                  rtsp.tsp.nearest_neighbor]
  # Initialize the datasets
  dset_shape = (len(tsp_solvers), len(num_targets_list))
  cpu_time_dset = group.create_dataset('cpu_time', dset_shape)
  cost_dset = group.create_dataset('cost', dset_shape)
  gap_dset = group.create_dataset('optimality_gap', dset_shape)
  tour_dsets = [[] for _ in num_targets_list]
  for j,num in enumerate(num_targets_list):
    tour_dsets[j] = group.create_dataset('tour{0:03d}'.format(num),
                                        (len(tsp_solvers), num), dtype='int64')
  # Report progress
  maxval = np.prod(dset_shape)
  widgets = ['Benchmarking TSP solvers: ', pbar.Bar(), ' ', pbar.Timer()]
  bar = pbar.ProgressBar(widgets=widgets, maxval=maxval).start()
  count = 0
  # Compute CPU time and cost for each solver
  coordinates = [t.pos() for t in targets]
  for j,num in enumerate(num_targets_list):
    coordinates_subset = coordinates[:num]
    graph = rtsp.construct.from_coordinate_list(coordinates_subset,
                                                distfn=rtsp.metric.euclidean_fn)
    for i, solver in enumerate(tsp_solvers):
      starttime = time.time()
      tour = solver(graph)
      cpu_time = time.time() - starttime
      cost = rtsp.tsp.compute_tour_cost(graph, tour)
      # Save the benchmarking results
      cost_dset[i,j] = cost
      cpu_time_dset[i,j] = cpu_time
      # Save the tour
      tour_dsets[j][i,:] = rtsp.tsp.rotate_tour(tour, start=0)
      # Update progress bar
      count += 1; bar.update(count)
  bar.finish()
  # Compute the optimality gap
  for j in xrange(len(num_targets_list)):
    ideal_cost = cost_dset[0,j]
    for i in xrange(1,len(tsp_solvers)):
        cost = cost_dset[i,j]
        optimality_gap = 100*(cost-ideal_cost) / ideal_cost
        gap_dset[i,j] = optimality_gap
  return

def metrics_benchmarking(f, robot, targets, targets_indices_list, args):
  # Create HDF5 group for the metrics benchmarks
  group_name = 'metric'
  if group_name in f.keys():
    del f[group_name]
  group = f.create_group(group_name)
  group['metrics'] = ['Euclidean', 'Max Joint Difference',
                                                        'Linear Interpolation']
  num_targets_list = [len(l) for l in targets_indices_list]
  group['number_of_targets'] = num_targets_list
  # Configuration-space metrics
  qd_max = robot.GetActiveDOFMaxVel()
  metrics = [ (rtsp.metric.euclidean_fn, (robot.GetActiveDOFWeights(),)),
              (rtsp.metric.max_joint_diff_fn, (1./qd_max,)),
              (rtsp.metric.linear_traj_duration, (robot,))]
  # Initialize the datasets
  dset_shape = (len(metrics), len(num_targets_list))
  cpu_time_dset = group.create_dataset('cpu_time', dset_shape)
  execution_time_dset = group.create_dataset('execution_time', dset_shape)
  # Report progress
  maxval = np.prod(dset_shape)
  widgets = ['Benchmarking C-Space metrics: ', pbar.Bar(), ' ', pbar.Timer()]
  bar = pbar.ProgressBar(widgets=widgets, maxval=maxval).start()
  count = 0
  # Configure the RoboTSP solver
  params = rtsp.solver.SolverParameters()
  params.standoff = args.standoff
  params.planner = args.planner
  params.max_iters = args.max_iters
  params.max_ppiters = args.max_ppiters
  params.step_size = np.deg2rad(args.step_size)
  for j,num in enumerate(num_targets_list):
    targets_subset = [targets[l] for l in targets_indices_list[j]]
    for i, (metric,margs) in enumerate(metrics):
      params.cspace_metric = metric
      params.cspace_metric_args = margs
      trajs, info = rtsp.solver.robotsp_solver(robot, targets_subset, params)
      # Populate the HDF5 file
      cpu_time_dset[i,j] = info['metric_cpu_time']
      execution_time_dset[i,j] = info['task_execution_time']
      # Update progress bar
      count += 1; bar.update(count)
  bar.finish()
  return

def discretization_benchmarking(f, robot, targets, targets_indices_list, args):
  # Create HDF5 group for the discretization benchmarks
  step_sizes = [np.pi, np.pi/2., np.pi/3., np.pi/4., np.pi/6., np.pi/12.]
  group_name = 'discretization'
  if group_name in f.keys():
    del f[group_name]
  group = f.create_group(group_name)
  group['step_sizes'] = step_sizes
  num_targets_list = [len(l) for l in targets_indices_list]
  group['number_of_targets'] = num_targets_list
  # Initialize the datasets
  dset_shape = (len(step_sizes), len(num_targets_list))
  cpu_time_dset = group.create_dataset('cpu_time', dset_shape)
  execution_time_dset = group.create_dataset('execution_time', dset_shape)
  configs_dset = group.create_dataset('number_of_configurations', dset_shape,
                                                                  dtype='int64')
  # Report progress
  maxval = np.prod(dset_shape)
  widgets = ['Benchmarking discretization step-size: ', pbar.Bar(), ' ',
                                                                  pbar.Timer()]
  bar = pbar.ProgressBar(widgets=widgets, maxval=maxval).start()
  count = 0
  # Configure the RoboTSP solver
  params = rtsp.solver.SolverParameters()
  params.standoff = args.standoff
  params.planner = args.planner
  params.max_iters = args.max_iters
  params.max_ppiters = args.max_ppiters
  params.cspace_metric = rtsp.metric.max_joint_diff_fn
  params.cspace_metric_args = (1./robot.GetActiveDOFMaxVel(), )
  for j,num in enumerate(num_targets_list):
    targets_subset = [targets[l] for l in targets_indices_list[j]]
    for i,step_size  in enumerate(step_sizes):
      params.step_size = step_size
      trajs, info = rtsp.solver.robotsp_solver(robot, targets_subset, params)
      # Populate the HDF5 file
      cpu_time_dset[i,j] = (info['ik_cpu_time'] + info['metric_cpu_time'] +
                                                    info['dijkstra_cpu_time'])
      execution_time_dset[i,j] = info['task_execution_time']
      configs_dset[i,j] = sum([len(b) for b in info['bins']])
      # Update progress bar
      count += 1; bar.update(count)
  bar.finish()
  return

def others_benchmarking(f, robot, targets, targets_indices_list, args):
  # Create HDF5 group for the discretization benchmarks
  group_name = 'others'
  if group_name in f.keys():
    del f[group_name]
  group = f.create_group(group_name)
  labels = ['GLKH1000', 'GLKH500', 'GLKH100', 'TSP C-Space', 'RoboTSP']
  group['methods'] = labels
  num_targets_list = [len(l) for l in targets_indices_list]
  group['number_of_targets'] = num_targets_list
  tour_dsets = [[] for _ in num_targets_list]
  for j,num in enumerate(num_targets_list):
    tour_dsets[j] = group.create_dataset('tour{0:03d}'.format(num),
                                            (len(labels), num+2), dtype='int64')
  # Initialize the datasets
  dset_shape = (len(labels), len(num_targets_list))
  cpu_time_dset = group.create_dataset('cpu_time', dset_shape)
  cost_dset = group.create_dataset('cspace_cost', dset_shape)
  execution_time_dset = group.create_dataset('execution_time', dset_shape)
  # Report progress
  maxval = np.prod(dset_shape)
  widgets = ['Benchmarking other methods vs RoboTSP: ', pbar.Bar(), ' ',
                                                                  pbar.Timer()]
  bar = pbar.ProgressBar(widgets=widgets, maxval=maxval).start()
  count = 0
  # Configure the solvers
  params = rtsp.solver.SolverParameters()
  # Shared parameters
  params.standoff = args.standoff
  params.planner = args.planner
  params.max_iters = args.max_iters
  params.max_ppiters = args.max_ppiters
  params.cspace_metric = rtsp.metric.max_joint_diff_fn
  params.cspace_metric_args = (1./robot.GetActiveDOFMaxVel(), )
  params.step_size = np.deg2rad(args.step_size)
  # GLKH parameters
  params.trace_level = 0
  for j,num in enumerate(num_targets_list):
    targets_subset = [targets[l] for l in targets_indices_list[j]]
    # GLKH 100 trials
    i = 0
    params.max_trials = 100
    trajectories, info = rtsp.solver.glkh_solver(robot, targets_subset, params)
    glkh_metric_cpu_time = info['metric_cpu_time']
    cpu_time_dset[i,j] = info['solver_cpu_time']
    cost_dset[i,j] = info['ccost']
    execution_time_dset[i,j] = info['task_execution_time']
    tour_dsets[j][i,:] = info['ctour']
    count += 1; bar.update(count)
    # Save time by re-using the generated GTSP graph
    cache = (info['cgraph'], info['bins'])
    # GLKH 500 trials
    i = 1
    params.max_trials = 500
    trajectories, info = rtsp.solver.glkh_solver(robot, targets_subset,
                                                            params, cache=cache)
    cpu_time_dset[i,j] = (info['solver_cpu_time'] + glkh_metric_cpu_time -
                                                        info['metric_cpu_time'])
    cost_dset[i,j] = info['ccost']
    execution_time_dset[i,j] = info['task_execution_time']
    tour_dsets[j][i,:] = info['ctour']
    count += 1; bar.update(count)
    # GLKH 1000 trials
    i = 2
    params.max_trials = 1000
    trajectories, info = rtsp.solver.glkh_solver(robot, targets_subset,
                                                            params, cache=cache)
    cpu_time_dset[i,j] = (info['solver_cpu_time'] + glkh_metric_cpu_time -
                                                        info['metric_cpu_time'])
    cost_dset[i,j] = info['ccost']
    execution_time_dset[i,j] = info['task_execution_time']
    tour_dsets[j][i,:] = info['ctour']
    count += 1; bar.update(count)
    # TSP C-Space
    i = 3
    trajectories, info = rtsp.solver.tsp_cspace_solver(robot, targets_subset,
                                                                        params)
    cpu_time_dset[i,j] = info['solver_cpu_time']
    cost_dset[i,j] = info['ccost']
    execution_time_dset[i,j] = info['task_execution_time']
    tour_dsets[j][i,:] = info['ctour']
    count += 1; bar.update(count)
    # RoboTSP
    i = 4
    trajectories, info = rtsp.solver.robotsp_solver(robot, targets_subset,
                                                                        params)
    cpu_time_dset[i,j] = info['solver_cpu_time']
    cost_dset[i,j] = info['ccost']
    execution_time_dset[i,j] = info['task_execution_time']
    tour_dsets[j][i,:] = info['ctour']
    count += 1; bar.update(count)
  bar.finish()
  return


if __name__ == '__main__':
  args = parse_args()
  np.set_printoptions(precision=6, suppress=True)
  criutils.logger.initialize_logging(format_level=logging.INFO)
  # Load the OpenRAVE environment
  uri = 'package://robotsp/data/worlds/airbus_challenge.env.xml'
  world_xml = resource_retriever.get_filename(uri, use_protocol=False)
  env = orpy.Environment()
  if not env.Load(world_xml):
    logger.error('Failed to load: {0}'.format(world_xml))
    raise IOError
  orpy.RaveSetDebugLevel(orpy.DebugLevel.Fatal)
  # Setup robot and manip
  robot = env.GetRobot('robot')
  manip = robot.SetActiveManipulator('drill')
  robot.SetActiveDOFs(manip.GetArmIndices())
  qhome = np.deg2rad([0, -20, 130, 0, 70, 0.])
  with env:
    robot.SetActiveDOFValues(qhome)
  # Load IKFast and links stats
  iktype = orpy.IkParameterizationType.Transform6D
  if not orkin.load_ikfast(robot, iktype):
    logger.error('Failed to load IKFast {0}'.format(iktype.name))
    exit()
  success = orkin.load_link_stats(robot, xyzdelta=0.01)
  # The robot velocity limits are quite high in the model
  robot.SetDOFVelocityLimits([1., 0.7, 0.7, 1., 0.7, 1.57])
  robot.SetDOFAccelerationLimits([5., 4.25, 4.25, 5.25, 6., 8.])
  # Get all the targets (rays)
  panel = env.GetKinBody('panel')
  targets = []
  for link in panel.GetLinks():
    lname = link.GetName()
    if lname.startswith('hole'):
      targets.append( orconv.to_ray(link.GetTransform()) )
  num_targets = len(targets)
  # Select the targets subsets randomly
  np.random.seed(1)
  indices = range(num_targets)
  num_targets_list = [25, 50, 100, 150, 200, 245]
  targets_indices_list = []
  for num in num_targets_list:
    subset = np.random.choice(num_targets, size=num, replace=False).tolist()
    targets_indices_list.append(subset)
  # Check if the benchmarks HDF5 file exists
  f = None
  generate = (args.all or args.tsp_solvers or args.metrics or
                                            args.discretization or args.others)
  if generate:
    uri = 'package://robotsp/data/benchmarks.hdf5'
    filename = resource_retriever.get_filename(uri, use_protocol=False)
    if os.path.isfile(filename):
      logger.warning('Benchmarks file already exists: {}'.format(filename))
      answer = raw_input('Overwrite selected benchmarks? (y/[n]): ')
      if answer.lower() == 'y':
        f = h5py.File(filename, 'r+')
    else:
      f = h5py.File(filename, 'w')
  if f is not None:
    try:
      # Update targets
      group_name = 'targets'
      if group_name in f.keys():
        del f[group_name]
      f[group_name] = [t.pos().tolist() + t.dir().tolist() for t in targets]
      # Run requested benchmarks
      if args.tsp_solvers or args.all:
        tsp_solvers_benchmarking(f, targets, num_targets_list)
      if args.metrics or args.all:
        metrics_benchmarking(f, robot, targets, targets_indices_list, args)
      if args.discretization or args.all:
        discretization_benchmarking(f, robot, targets, targets_indices_list,
                                                                          args)
      if args.others or args.all:
        others_benchmarking(f, robot, targets, targets_indices_list, args)
    finally:
      f.close()
  # Debug
  import IPython
  IPython.embed(banner1='')
  exit()
