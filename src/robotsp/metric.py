#!/usr/bin/env python
import math
import itertools
import numpy as np
import openravepy as orpy
import raveutils.planning as orplan


def manhattan_fn(x, y, weights=None):
  if weights is None:
    distance = np.sum(np.abs(x-y))
  else:
    distance = np.sum(weights*np.abs(x-y))
  return distance

def euclidean_fn(x, y, weights=None):
  if weights is None:
    distance = math.sqrt(np.sum((x-y)**2))
  else:
    distance = math.sqrt(np.sum(weights*(x-y)**2))
  return distance

def euclidean_fn_int(x, y, weights = None, mult_factor=1000):
  if weights is None:
    distance = math.sqrt(np.sum((x-y)**2))
  else:
    distance = math.sqrt(np.sum(weights*(x-y)**2))
  return int(math.floor(distance*mult_factor))

def geo_fn(x, y):
  def to_radians(v):
    degrees = np.int0(v)
    minutes = v - degrees
    return np.deg2rad(degrees + 5.0 * minutes / 3.0)
  latitude1, longitude1 = to_radians(x)
  latitude2, longitude2 = to_radians(y)
  radius = 6378.388
  q1 = math.cos(longitude1 - longitude2)
  q2 = math.cos(latitude1 - latitude2)
  q3 = math.cos(latitude1 + latitude2)
  distance = math.floor(radius *
                        math.acos(0.5 * ((1.0 + q1)*q2 - (1.0 - q1)*q3)) + 1.0)
  return distance

def max_joint_diff_fn(x, y, weights=None):
  if weights is None:
    distance = max(np.abs(x-y))
  else:
    distance = max(weights*np.abs(x-y))
  return distance

def birrt_shortcut_traj_duration(qstart, qgoal, robot, max_iters, max_ppiters,
                                                                try_swap=False):
  traj = orplan.plan_to_joint_configuration(robot, qgoal, 'BiRRT', max_iters,
                                                max_ppiters, try_swap=try_swap)
  duration = traj.GetDuration() if traj is not None else float('inf')
  return duration

def cubic_traj_duration(qa, qb, robot):
  duration = retimed_traj_duration(qa, qb, robot, 'CubicTrajectoryRetimer')
  return duration

def linear_traj_duration(qa, qb, robot):
  duration = retimed_traj_duration(qa, qb, robot, 'LinearTrajectoryRetimer')
  return duration

def parabolic_traj_duration(qa, qb, robot):
  duration = retimed_traj_duration(qa, qb, robot, 'ParabolicTrajectoryRetimer')
  return duration

def retimed_traj_duration(qa, qb, robot, method):
  traj = orplan.trajectory_from_waypoints(robot, [qa, qb])
  status = orplan.retime_trajectory(robot, traj, method)
  if status == orpy.PlannerStatus.HasSolution:
    duration = traj.GetDuration()
  else:
    duration = float('inf')
  return duration
