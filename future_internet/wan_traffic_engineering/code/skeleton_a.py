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

from typing import Any
import networkx as nx
from networkx.classes.digraph import DiGraph
from networkx.classes.reportviews import OutEdgeView
from networkx.generators import small
import numpy as np
from numpy.lib import savetxt
import pandas as pd

try:
    from . import wanteutility
except (ImportError, SystemError):
    import wanteutility

try:
    from . import assignment_parameters
except (ImportError, SystemError):
    import assignment_parameters


def get_edge_index_by_srcdest(src: int, dest: int, edge_list) -> int:
    edge_index = np.where(
        (edge_list[:, 0] == src) * (edge_list[:, 1] == dest))
    return edge_index


def compute_current_edge_capacity(edge_index: int, edge_list) -> int:
    capacity = edge_list[edge_index, 2]
    count = edge_list[edge_index, -1]
    return capacity/count


def match_link_with_flow(cur_flows: list, flow: list, u: int, v: int) -> list:
    for i in range(len(flow)-1):
        if (flow[i] == u) & (flow[i+1] == v):
            cur_flows.append(flow)
    return cur_flows


def match_flow_with_flow(cur_flows: list, u: list, v: list, index) -> list:
    if len(u) != len(v):
        return cur_flows
    else:
        for i in range(len(u)):
            if u[i] != v[i]:
                return cur_flows
        return cur_flows.append(index)


def get_specific_flows_for_desired_link(cur_flows: list, all_flows: list, link: Any) -> list:
    for flow in all_flows:
        flows = match_link_with_flow(cur_flows, flow, link[0], link[1])

    # Check if there is a flow at all
    if len(flows):
        return flows, 1
    else:
        return flows, 0


def get_paths_of_flows(paths: list, flows: list, all_paths: list) -> list:
    for flow in flows:
        for i in range(len(all_paths)):
            match_flow_with_flow(paths, flow, all_paths[i], i)
    return paths


def get_link_capacity(link: tuple, graph: DiGraph) -> int:
    return graph.edges[link[0], link[1]]['weight']


MAXIMUM_CAPACITY = float('inf')


def solve(in_graph_filename, in_demands_filename, in_paths_filename, out_rates_filename):

    # Read in input
    print("Read in input")
    graph = wanteutility.read_graph(in_graph_filename)
    demands = wanteutility.read_demands(in_demands_filename)
    all_paths = wanteutility.read_all_paths(in_paths_filename)
    paths_x_to_y = wanteutility.get_paths_x_to_y(all_paths, graph)
    all_flows = wanteutility.get_all_flows(all_paths, demands)

    path_updates = []

    # Count how many actual flows in all_flows are left
    while len(list(filter(None, all_flows))) > 0:
        print(
            f'Len of actual all_flows : {len(list(filter(None, all_flows)))}')
        # Look for the smallest flow
        minimal_bandwidth = MAXIMUM_CAPACITY
        # Iterate through each edge in the graph
        for link in graph.edges:
            # print(link, iteration)
            link_capacity = get_link_capacity(link, graph)
            # print(all_flows_w_demand_for_link)
            # Get all flows with a current demand on this link
            all_flows_w_demand_for_link, valid = get_specific_flows_for_desired_link(
                [], all_flows, link)
            # print(link_capacity)
            # print(all_flows_w_demand_for_link)
            # print(valid)
            # If there is no flow, we can save the time and stop this iteration now
            if not valid:
                # print(all_flows_w_demand_for_link)
                continue
            # Calculate the minimal bandwidth for this flow
            # print(f"Test Value is : {test_value}")
            test_value = link_capacity / len(all_flows_w_demand_for_link)
            print(f"Test Value has become : {test_value}")
            # print(
            #     f"The bandwidth check now results in {test_value > minimal_bandwidth} because min_bw is {minimal_bandwidth}")
            # If there is already asmaller link, we can leave stop now
            if test_value > minimal_bandwidth:
                #print(test_value, minimal_bandwidth)
                continue
            # We update the current smallest bandwidth & corresponding link
            minimal_bandwidth = test_value
            print(f"New minimal Bandwidth updated to: {minimal_bandwidth}")
            # Select link which will be updated
            (congested_link_u, congested_link_v) = link

        # Get all paths which need to be updated
        satisfied_flows, _ = get_specific_flows_for_desired_link(
            [], all_flows, (congested_link_u, congested_link_v))

        # Update Path rates
        # print(f"The path rates will be updated by {minimal_bandwidth}")
        for path in get_paths_of_flows([], satisfied_flows, all_paths):
            path_updates.append((path, minimal_bandwidth))

        # Reduce capacities by fair split on flows
        for flow in satisfied_flows:
            for step_index in range(len(flow) - 1):
                graph.edges[flow[step_index],
                            flow[step_index+1]]['weight'] -= minimal_bandwidth

        # Search for satisfied paths
        # print(f"Len of all_flows is {len(all_flows)}")
        i = 0
        to_delete = []
        for original_flow in all_flows:
            for flow in satisfied_flows:
                if flow == original_flow:
                    to_delete.append(i)
            i += 1

        # Prune satisfied flows
        # print(len(to_delete))
        for i in to_delete:
            all_flows[i] = []

        # all_flows = list(filter(None, all_flows))

        # Prune congested link out of graph
        # print(f"Removing edge from graph. Current size is: {graph.size()}")
        graph.remove_edge(congested_link_u, congested_link_v)
        print(f"Removed edge from graph. New size is: {graph.size()}")

    # Calculate the flow for each rate
    path_rates = np.zeros(len(all_paths))
    for (i, value) in path_updates:
        path_rates[i] += value

    # Finally, write the rates to the output file
    print("Finally, write the rates to the output file")
    with open(out_rates_filename, "w+") as rate_file:
        np.savetxt(rate_file, path_rates)


def main():
    for appendix in range(assignment_parameters.num_tests_a):
        solve(
            "../ground_truth/input/a/graph%s.graph" % appendix,
            "../ground_truth/input/a/demand%s.demand" % appendix,
            "../ground_truth/input/a/path%s.path" % appendix,
            "../myself/output/a/rate%s.rate" % appendix
        )


if __name__ == "__main__":
    main()
