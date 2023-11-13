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

import networkx as nx
import traceback

try:
    from . import util
except (ImportError, SystemError):
    import util

MAX_ISLS_PER_SAT = 4
G = nx.Graph()


def add_isls_to_graph():
    """
    Add inter-satellite links and check if satellites have more than X links
    """
    counter = 0
    for sat in sat_positions:
        G.add_node(sat)
    for sat1, sat2 in selected_isls:
        G.add_edge(sat1, sat2, length=selected_isls[sat1, sat2])
    for sat in sat_positions:
        if G.degree(sat) > MAX_ISLS_PER_SAT:
            print("Satellite", sat, "has more than", MAX_ISLS_PER_SAT, "ISLs")
            counter += 1
    if counter > 0:
        exit()


def compute_metric():
    """
    Compute metric, which is a linear combination of weighted stretch and
    hop count, taking into consideration shortest paths from top 10 cities
    to all other cities.
    """
    for city in city_positions:
        G.add_node(city)
    for city, sat in city_coverage:
        G.add_edge(city, sat, length=city_coverage[city, sat])
    w_sum = 0
    w_stretch_sum = 0
    w_hopcount_sum = 0
    for i in range(10001, 10100):
        path = nx.single_source_dijkstra_path(G, i, weight='length')
        for j in range(i + 1, 10101):
            try:
                hops = len(path[j]) - 1
                path_len = 0.0
                for n in range(0, len(path[j]) - 1):
                    link_len = 0
                    if path[j][n] < 10000 and path[j][n + 1] < 10000:
                        link_len = selected_isls[path[j][n], path[j][n + 1]]
                    else:
                        if path[j][n] > 10000:
                            link_len = city_coverage[path[j][n], path[j][n + 1]]
                        else:
                            link_len = city_coverage[path[j][n + 1], path[j][n]]
                    path_len += link_len
                stretch = path_len / city_pairs[i, j]
                weight = city_positions[i]["gdp"] * city_positions[j]["gdp"] / 10000000
                w_sum += weight
                w_stretch_sum += stretch * weight
                w_hopcount_sum += hops * weight
            except Exception as e:
                print("Error:", e)
                traceback.print_exc()
                exit()
    avg_w_stretch = w_stretch_sum / w_sum
    avg_w_hopcount = w_hopcount_sum / w_sum
    w_metric = avg_w_stretch + avg_w_hopcount
    w_metric = round(w_metric, 2)
    print("Your score:", w_metric)


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
city_pairs = util.read_city_pairs(city_pair_file)
city_coverage = util.read_coverage(city_coverage_file)
valid_isls = util.read_valid_isls(valid_isls_file)

print("Checking if all selected ISLs are valid")
selected_isls = util.read_and_filter_selected_links(valid_isls, sat_links_file)
print("Checking if any satellite has more than the maximum number of ISLs")
add_isls_to_graph()
print("Computing metric")
compute_metric()
