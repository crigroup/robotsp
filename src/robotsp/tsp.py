#!/usr/bin/env python
import itertools
import numpy as np
import networkx as nx


def compute_tour_cost(graph, tour, weight='weight', is_cycle=True):
  cost = 0
  start = 0 if is_cycle else 1
  for idx in xrange(start, len(tour)):
    u = tour[idx-1]
    v = tour[idx]
    cost += graph.edge[u][v][weight]
  return cost

def nearest_neighbor(graph, restarts=10, weight='weight'):
  """
  Recursive Nearest-Neighbor algorithm to solve the traveling salesman problem.

  These are the steps of the algorithm for each restart:

  #. Start on a random node as current node;
  #. Find the shortest edge connecting the current node with an unvisited node;
  #. Mark the found node as current and visited;
  #. Go to step 2 until all nodes are visited

  Parameters
  ----------
  graph: NetworkX graph
  restarts: int, optional (default=10)
    Number of times the algorithm will try a different, randomly selected, node
    as starting point
  weight: string, optional (default='weight')
    Edge data key corresponding to the edge weight

  Returns
  -------
  best_tour: list
    The sequence for visiting all the nodes
  """
  restarts = min(restarts, graph.number_of_nodes())
  nodelist = graph.nodes()
  tried_nodes = []
  min_cost = float('inf')
  for _ in xrange(restarts):
    while True:
      current_node = np.random.choice(nodelist)
      if current_node not in tried_nodes:
        tried_nodes.append(current_node)
        break
    g = graph.copy()
    tour = [current_node]
    while g.number_of_nodes() > 1:
      min_weight = float('inf')
      for neighbor in g.neighbors_iter(current_node):
        w = g.edge[current_node][neighbor][weight]
        if w < min_weight:
          nn = neighbor
          min_weight = w
      g.remove_node(current_node)
      tour.append(nn)
      current_node = nn
    cost = compute_tour_cost(graph, tour)
    if cost < min_cost:
      min_cost = cost
      best_tour = list(tour)
  return best_tour

def rotate_tour(tour, start=0):
  """
  Rotate a tour so that it starts at the given ``start`` index. This is
  equivalent to rotate the input list to the left.

  Parameters
  ----------
  tour: list
    The input tour
  start: int, optional (default=0)
    New start index for the tour

  Returns
  -------
  rotated: list
    The rotated tour
  """
  idx = tour.index(start)
  if idx != 0:
    rotated = tour[idx:] + tour[:idx]
  else:
    rotated = tour
  return rotated

def scip_solver(graph, weight='weight'):
  try:
    from pyscipopt import Model, quicksum
  except ImportError:
    raise ImportError('SCIP Optimization Suit with Python support not found')
  nodes = graph.nodes()
  num_nodes = graph.number_of_nodes()
  c = nx.adjacency_matrix(graph, weight=weight)
  # Define the optimization problem
  model = Model('tsp')
  model.hideOutput() # silent/verbose mode
  # Create the variables
  x = {}
  for i in xrange(num_nodes):
    for j in xrange(i+1, num_nodes):
      x[i,j] = model.addVar(ub=1, name='x(%s,%s)'%(i,j))
  # Add the constraints
  for i in xrange(num_nodes):
    model.addCons(quicksum([x[j,i] for j in xrange(i)]) +
        quicksum([x[i,j] for j in xrange(i+1, num_nodes)]) == 2, 'Degree(%s)'%i)
  # Set minimization objective
  model.setObjective(quicksum(c[i,j]*x[i,j] for i in xrange(num_nodes)
                                  for j in xrange(i+1, num_nodes)), 'minimize')
  # Limit the number of edges in a connected component S to |S|-1
  def addcut(cut_edges):
    G = nx.Graph()
    G.add_edges_from(cut_edges)
    Components = list(nx.connected_components(G))
    if len(Components) == 1:
      return False
    model.freeTransform()
    for S in Components:
      model.addCons(quicksum(x[i,j] for i in S for j in S if j>i) <= len(S)-1)
    return True
  # Solve
  EPS = 1.e-6
  isMIP = False
  while True:
    model.optimize()
    edges = []
    for (i,j) in x:
      if model.getVal(x[i,j]) > EPS:
        edges.append( (i,j) )
    if addcut(edges) == False:
      if isMIP:     # integer variables, components connected: solution found
        break
      model.freeTransform()
      for (i,j) in x:     # all components connected, switch to integer model
        model.chgVarType(x[i,j], 'B')
        isMIP = True
  # Extract the tour from the edges
  G = nx.Graph()
  for e in edges:
    G.add_edge(nodes[e[0]], nodes[e[1]])
  tour_edges = nx.eulerian_circuit(G, source=graph.nodes_iter().next())
  tour = [e[0] for e in tour_edges]
  return tour

def two_opt(graph, weight='weight'):
  num_nodes = graph.number_of_nodes()
  tour = graph.nodes()
  # min_cost = compute_tour_cost(graph, tour)
  start_again = True
  while start_again:
    start_again = False
    for i in xrange(0, num_nodes-1):
      for k in xrange(i+1, num_nodes):
        # 2-opt swap
        a, b = tour[i-1], tour[i]
        c, d = tour[k], tour[(k+1)%num_nodes]
        if (a == c) or (b == d):
          continue
        ab_cd_dist = graph.edge[a][b][weight] + graph.edge[c][d][weight]
        ac_bd_dist = graph.edge[a][c][weight] + graph.edge[b][d][weight]
        if ab_cd_dist > ac_bd_dist:
          tour[i:k+1] = reversed(tour[i:k+1])
          start_again = True
        if start_again:
          break
      if start_again:
        break
  return tour
