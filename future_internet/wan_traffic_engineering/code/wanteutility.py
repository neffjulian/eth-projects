# The MIT License (MIT)
#
# Copyright (c) 2019 Simon Kassing (ETH)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import networkx as nx
import ortoolslpparser


def read_graph(graph_filename):
    """
    Read in graph.

    :param graph_filename: .graph file

    :return: Networkx DiGraph instance
    """
    graph = nx.DiGraph()
    with open(graph_filename, "r") as graph_file:
        for line in graph_file:
            spl = line.split(",")
            graph.add_edge(int(spl[0]), int(spl[1]), weight=float(spl[2]))
    return graph


def read_demands(demands_filename):
    """
    Read in demands.

    :param demands_filename: .demand file

    :return: List of (src, dst) pairs.
    """
    demands = []
    with open(demands_filename, "r") as demand_file:
        for line in demand_file:
            spl = line.split(",")
            demands.append((int(spl[0]), int(spl[1])))
    return demands


def read_all_paths(paths_filename):
    """
    Read all paths from .path file and return in a full list.

    :param paths_filename: .path file

    :return: List of all paths, i.e. [ (0, 3, 4, 9), (0, 8, 9), ... ]
    """
    all_paths = []
    with open(paths_filename, "r") as path_file:
        for line in path_file:
            int_split = tuple(list(map(int, line.split("-"))))
            all_paths.append(int_split)
    return all_paths


def get_paths_x_to_y(all_paths, graph):
    """
    Read paths and return in a 2D array "from x to y".

    :param all_paths: List of all paths
    :param graph: Networkx DiGraph instance

    :return: 2D array of a tuple of paths, i.e. from node 0 to 9 with 2 paths (0-3-4-9, 0-8-9):
            paths[0][9] = [ (0, 3, 4, 9), (0, 8, 9) ]
    """
    paths_x_to_y = []
    for i in range(graph.number_of_nodes()):
        paths_x_to_y.append([])
        for j in range(graph.number_of_nodes()):
            paths_x_to_y[i].append([])
    for path in all_paths:
        paths_x_to_y[path[0]][path[-1]].append(path)
    return paths_x_to_y


def get_all_flows(all_paths, demands):
    """
    Get all the flows (i.e. the paths for which there is a demand).

    :param all_paths: List of all paths
    :param demands:   Demands

    :return: List of all flows (i.e. if [(0, 1, 3), (3, 4, 5), (9, 0)] and demands = [(0, 3), (0, 9)] the resulting
             flow list is [(0, 1, 3), (9, 0)].
    """
    all_flows = []
    for path in all_paths:
        if (path[0], path[-1]) in demands:
            all_flows.append(path)
    return all_flows


def ortools_solve_lp_and_get_var_val_map(lp_filename):
    """
    Solve the linear program with GLOP and store the variable values in a mapping.

    :param lp_filename: Linear program filename appended to the Gurobi command array at the end before execution
    :return: Mapping (i.e. { "var0" : 8935.24525, "var1" : 3802.3, ... }
    """

    # Solve the linear program
    parse_result = ortoolslpparser.parse_lp_file(lp_filename)
    solver = parse_result["solver"]
    result = solver.Solve()

    if result == solver.OPTIMAL:
        var_val_map = {}
        for var_name in parse_result["var_names"]:
            var_val_map[var_name] = solver.LookupVariable(var_name).solution_value()
        return var_val_map

    else:

        print("Linear program was not solved.")
        error_msg = "UNKNOWN"
        if result == solver.OPTIMAL:
            error_msg = "OPTIMAL"
        elif result == solver.FEASIBLE:
            error_msg = "FEASIBLE"
        elif result == solver.INFEASIBLE:
            error_msg = "INFEASIBLE"
        elif result == solver.UNBOUNDED:
            error_msg = "UNBOUNDED"
        elif result == solver.ABNORMAL:
            error_msg = "ABNORMAL"
        elif result == solver.NOT_SOLVED:
            error_msg = "NOT SOLVED"
        print("Error result provided by OR-tools: %s (%d)"  % (error_msg, result))
        exit(1)
