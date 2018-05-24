#!/usr/bin/env python
import time
import logging
import numpy as np
import openravepy as orpy
# Utils
import baldor as br
import criutils as cu
import raveutils as ru
# RoboTSP
import robotsp as rtsp


def play_trajectory(robot, traj, speed=1.0, rate=100.):
  indices = robot.GetActiveManipulator().GetArmIndices()
  spec = traj.GetConfigurationSpecification()
  starttime = time.time()
  while time.time()-starttime < traj.GetDuration()/speed:
    curtime = (time.time()-starttime)*speed
    with env: # have to lock environment since accessing robot
      trajdata=traj.Sample(curtime)
      qrobot = spec.ExtractJointValues(trajdata, robot, indices, 0)
      robot.SetActiveDOFValues(qrobot)
    time.sleep(1./rate)

def visualize_trajectories(robot, trajectories, speed=2., pausetime=0.0):
  handles = []
  Toffset = np.eye(4)
  Toffset[2,3] = 0.01
  for traj in trajectories:
    play_trajectory(robot, traj, speed)
    # Draw the ray
    Tdrill = manip.GetEndEffectorTransform()
    ray = ru.conversions.to_ray(np.dot(Tdrill, Toffset))
    handles.append( ru.visual.draw_ray(env, ray, linewidth=1.5) )
    if pausetime > 0:
      time.sleep(pausetime)
  return handles

if __name__ == '__main__':
  # Configure the logger
  logger = logging.getLogger('rtsp_challenge')
  cu.logger.initialize_logging(format_level=logging.INFO)
  # Load OpenRAVE environment
  world_xml = 'worlds/airbus_challenge.env.xml'
  env = orpy.Environment()
  logger.info('Loading OpenRAVE environment: {}'.format(world_xml))
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
  if not ru.kinematics.load_ikfast(robot, iktype):
    logger.error('Failed to load IKFast {0}'.format(iktype.name))
    exit()
  success = ru.kinematics.load_link_stats(robot, xyzdelta=0.01)
  velocity_limits = robot.GetDOFVelocityLimits()
  acceleration_limits = robot.GetDOFVelocityLimits()
  robot.SetDOFVelocityLimits([1., 0.7, 0.7, 1., 0.7, 1.57])
  robot.SetDOFAccelerationLimits([5., 4.25, 4.25, 5.25, 6., 8.])
  # Get all the targets (rays)
  panel = env.GetKinBody('panel')
  targets = []
  for link in panel.GetLinks():
    lname = link.GetName()
    if lname.startswith('hole'):
      targets.append( ru.conversions.to_ray(link.GetTransform()) )
  # RoboTSP parameters
  params = rtsp.solver.SolverParameters()
  # Task space parameters
  params.tsp_solver = rtsp.tsp.two_opt
  params.tspace_metric = rtsp.metric.euclidean_fn
  # Configuration space parameters
  params.cspace_metric = rtsp.metric.max_joint_diff_fn
  params.cspace_metric_args = (1./robot.GetActiveDOFMaxVel(),)
  # kinematics parameters
  params.iktype = orpy.IkParameterizationType.Transform6D
  params.qhome = qhome
  params.standoff = 0.01
  params.step_size = np.pi/6.
  # Planning parameters
  params.try_swap = False
  params.planner = 'BiRRT'
  params.max_iters = 100
  params.max_ppiters = 30
  # Run RobotTSP
  logger.info('Running RoboTSP on {0} targets'.format(len(targets)))
  logger.info('It takes ~20 secs for 245 targets...')
  trajs, info = rtsp.solver.robotsp_solver(robot, targets, params)
  # Report
  ptime = info['solver_cpu_time']
  etime = info['task_execution_time']
  logger.info('Planning time: {:.1f} s'.format(ptime))
  logger.info('Task execution time: {:.1f} s'.format(etime))
  # Visualize trajectories
  speed = 2
  logger.info('Visualizing the trajectory. Speed {0}X'.format(speed))
  # Configure the viewer
  env.SetDefaultViewer()
  while env.GetViewer() is None:
    time.sleep(0.1)
  viewer = env.GetViewer()
  T = br.euler.to_transform(*np.deg2rad([-160,0,-90]))
  T[:3,3] = [-0.125, 0, 1.65]
  viewer.SetCamera(T, 1.85)
  handles = visualize_trajectories(robot, trajs, speed)
  # Interactive console
  import IPython
  IPython.embed(banner1='')
  exit()
