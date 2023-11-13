# The MIT License (MIT)
#
# Copyright (c) 2019 Debopam Bhattacherjee
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

from collections import defaultdict
import networkx as nx

try:
    from . import util
except (ImportError, SystemError):
    import util

G = nx.Graph()


def generate_html(city1, city2):
    """
        Generate html for visualizing the constellation and city-city paths
    """
    writer = open(html_file, 'w')
    with open(top_file, 'r') as fi:
        writer.write(fi.read())
    for sat in sat_positions:
        G.add_node(sat)
    for city in city_positions:
        G.add_node(city)
    for city, sat in city_coverage:
        G.add_edge(city, sat, length=city_coverage[city, sat])
    for sat1, sat2 in selected_isls:
        G.add_edge(sat1, sat2, length=selected_isls[sat1, sat2])
    nodes = G.nodes()
    for node in nodes:
        if node < 10000:
            writer.write("createNode( "
                         + str(sat_positions[node]["lat_deg"])
                         + " ,  " + str(sat_positions[node]["long_deg"])
                         + " , 'grey', 1);\n")
        elif node == city1 or node == city2:
            writer.write("createNode( "
                         + str(city_positions[node]["lat_deg"])
                         + " ,  " + str(city_positions[node]["long_deg"])
                         + " , 'green', 4);\n")
    writer.flush()
    edges = G.edges()
    for edge in edges:
        if (edge[0] < 10000) and (edge[1] < 10000):
            writer.write("createEdge( "
                         + str(sat_positions[edge[0]]["lat_deg"])
                         + " , " + str(sat_positions[edge[0]]["long_deg"])
                         + " , " + str(sat_positions[edge[1]]["lat_deg"])
                         + " , " + str(sat_positions[edge[1]]["long_deg"])
                         + " , 'blue', 0.1);\n")
    writer.flush()
    path = nx.shortest_path(G, source=city1, target=city2, weight='length')
    for j in range(0, path.__len__() - 1):
        if path[j] > 10000:
            writer.write("createEdge( " + str(city_positions[path[j]]["lat_deg"]) + " , " + str(
                city_positions[path[j]]["long_deg"]) + " , " + str(sat_positions[path[j + 1]]["lat_deg"]) + " , " + str(
                sat_positions[path[j + 1]]["long_deg"]) + " , 'red', 0.8);\n")
        elif path[j + 1] > 10000:
            writer.write("createEdge( " + str(sat_positions[path[j]]["lat_deg"]) + " , " + str(
                sat_positions[path[j]]["long_deg"]) + " , " + str(city_positions[path[j + 1]]["lat_deg"]) + " , " + str(
                city_positions[path[j + 1]]["long_deg"]) + " , 'red', 0.8);\n")
        else:
            writer.write("createEdge( " + str(sat_positions[path[j]]["lat_deg"]) + " , " + str(
                sat_positions[path[j]]["long_deg"]) + " , " + str(sat_positions[path[j + 1]]["lat_deg"]) + " , " + str(
                sat_positions[path[j + 1]]["long_deg"]) + " , 'red', 0.8);\n")
    writer.flush()
    with open(bottom_file, 'r') as fb:
        writer.write(fb.read())
    writer.close()


sat_pos_file = "../input_data/sat_positions.txt"
city_pos_file = "../input_data/cities.txt"
city_coverage_file = "../input_data/city_coverage.txt"
city_pair_file = "../input_data/city_pairs.txt"
valid_isls_file = "../input_data/valid_isls.txt"
sat_links_file = "../output_data/sat_links.txt"

top_file = "static_html/top.html"
bottom_file = "static_html/bottom.html"
html_file = "viz.html"

sat_positions = util.read_sat_positions(sat_pos_file)
city_positions = util.read_city_positions(city_pos_file)
city_coverage = util.read_coverage(city_coverage_file)
valid_isls = util.read_valid_isls(valid_isls_file)

selected_isls = util.read_and_filter_selected_links(valid_isls, sat_links_file)
generate_html(10002, 10041)
