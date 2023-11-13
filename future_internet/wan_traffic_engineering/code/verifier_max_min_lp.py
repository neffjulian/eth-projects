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

import numpy as np

try:
    from . import wanteutility
except (ImportError, SystemError):
    import wanteutility

try:
    from . import verifier_correctness
except (ImportError, SystemError):
    import verifier_correctness

PRECISION = 0.001


def verify_max_min_lp(graph_filename, demand_filename, path_filename, rate_filename, ground_truth_rate_filename,
                      k_limit):

    # Perform verification
    verification = verifier_correctness.verify(graph_filename, demand_filename, path_filename, rate_filename, k_limit)
    if not verification[0]:
        return verification

    # Read in ground truth
    graph = wanteutility.read_graph(graph_filename)
    demand_to_flow_sum = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    with open(path_filename, "r") as path_file:
        with open(ground_truth_rate_filename, "r") as rate_file:
            line_path = path_file.readline()
            line_rate = rate_file.readline()
            while line_path != "" and line_rate != "":
                path = list(map(int, line_path.split("-")))
                rate = float(line_rate)
                demand_to_flow_sum[path[0]][path[len(path) - 1]] += rate
                line_path = path_file.readline()
                line_rate = rate_file.readline()
            if line_path != "" or line_rate != "":
                return False, "Different length path and rate file."

    # Determine from ground truth the objective value
    min_demand = 1000000.0
    demands = wanteutility.read_demands(demand_filename)
    for demand in demands:
        min_demand = min(min_demand, demand_to_flow_sum[demand[0]][demand[1]])

    # Check objective
    if abs(verification[1] - min_demand) > PRECISION:
        return False, "ERROR: non-optimal solution (got %f but expected %f)" % (min_demand, verification[1])

    # Else we have success
    return verification
