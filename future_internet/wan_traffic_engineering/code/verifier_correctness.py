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

try:
    from . import wanteutility
except (ImportError, SystemError):
    import wanteutility

PRECISION = 0.001


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def is_valid_path(graph, value):

    str_split = value.split("-")
    for j in str_split:
        if not is_int(j):
            return False, "FORMAT ERROR: %s is an invalid path." % value

    int_split = list(map(int, str_split))

    if len(int_split) < 2:
        return False, "FORMAT ERROR: %s is an invalid path (too short)." % value

    for j in int_split:
        if int_split.count(j) > 1:
            return False, "FORMAT ERROR: %s is cyclic." % value

    for i in range(len(int_split) - 1):
        if int_split[i] < 0 or int_split[i] >= graph.number_of_nodes() \
                or int_split[i + 1] < 0 or int_split[i + 1] >= graph.number_of_nodes():
            return False, "FORMAT ERROR: %s is an invalid path (node index out of graph range)." % value
        if not graph.has_edge(int_split[i], int_split[i + 1]):
            return False, "FORMAT ERROR: %s is an invalid path (edge (%d, %d) does not exist)." % \
                   (value, int_split[i], int_split[i + 1])

    return True, "SUCCESS"


def to_path(value):
    return list(map(int, value.split("-")))


def is_valid_rate(value):
    if not is_float(value):
        return False, "FORMAT ERROR: rate %s is not a float." % value
    if float(value) <= -PRECISION:
        return False, "FORMAT ERROR: rate %f is not positive within precision." % float(value)
    return True, "SUCCESS"


def verify(graph_filename, demand_filename, path_filename, rate_filename, k_limit):

    # Read in input
    graph = wanteutility.read_graph(graph_filename)
    demands = wanteutility.read_demands(demand_filename)

    # Manually read in paths w/ rates
    paths_with_rate = []
    edge_to_total_rate = {}
    for edge in graph.edges:
        edge_to_total_rate[edge] = 0.0
    for i in range(graph.number_of_nodes()):
        paths_with_rate.append([])
        for j in range(graph.number_of_nodes()):
            paths_with_rate[i].append([])
    with open(path_filename, "r") as path_file, open(rate_filename, "r") as rate_file:
        for line_path, line_rate in zip(path_file, rate_file):
            line_path = line_path.strip()
            line_rate = line_rate.strip()
            rate_valid = is_valid_rate(line_rate)
            if not rate_valid[0]:
                return False, rate_valid[1]
            path_valid = is_valid_path(graph, line_path)
            if not path_valid[0]:
                return False, path_valid[1]
            rate = float(line_rate)
            path = to_path(line_path)
            for i in range(len(path) - 1):
                edge_to_total_rate[(path[i], path[i + 1])] += rate
            paths_with_rate[path[0]][path[len(path) - 1]].append((path, rate))
            if len(paths_with_rate[path[0]][path[len(path) - 1]]) > k_limit:
                return False, "K EXCEEDED: too many paths between %d and %d" % (path[0], path[len(path) - 1])
        if path_file.readline() != "" or rate_file.readline() != "":
            return False, "Different length path and rate file."

    # Check edge capacities
    for edge in graph.edges:
        if edge_to_total_rate[edge] > graph.get_edge_data(edge[0], edge[1])["weight"] + PRECISION:
            return False, "FORMAT ERROR: capacity on edge %d -> %d is exceeded (%f > %f)." \
                   % (edge[0], edge[1], edge_to_total_rate[edge], graph.get_edge_data(edge[0], edge[1])["weight"])

    # Finally check how optimal it is
    minimum_demand = 100000000.0
    for demand in demands:
        total = 0.0
        for (_, rate) in paths_with_rate[demand[0]][demand[1]]:
            total += rate
        minimum_demand = min(minimum_demand, total)

    # Print the final optimality
    return True, minimum_demand
