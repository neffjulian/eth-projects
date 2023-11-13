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

from numpy import positive


try:
    from . import wanteutility
except (ImportError, SystemError):
    import wanteutility

try:
    from . import assignment_parameters
except (ImportError, SystemError):
    import assignment_parameters

def solve(in_graph_filename, in_demands_filename, in_paths_filename, out_rates_filename):

    # Read in input
    graph = wanteutility.read_graph(in_graph_filename)
    demands = wanteutility.read_demands(in_demands_filename)
    all_paths = wanteutility.read_all_paths(in_paths_filename)
    paths_x_to_y = wanteutility.get_paths_x_to_y(all_paths, graph)

    print(in_demands_filename)
    edges = graph.edges

    # Write the linear program
    with open("../myself/output/b/program.lp", "w+") as program_file:
        # Write target
        program_file.write('min: Z; \n')

        # Writing first constraints: 
        for (u,v) in demands:
            # For each demand the sum of its path rates must be largerthan the barrier
            c = ''
            for i, path in enumerate(all_paths):
                # Look for demands and sum it up
                if (u == path[0]) & (v == path[len(path) -1]):
                    c += 'r' + str(i) + ' + '
            program_file.write(c + 'Z > 0; \n')
        
        # Writing down the second constraint
        for (u,v) in edges:
            # Write down link capacity
            c = ''
            for i, path in enumerate(all_paths):
                for j in range(len(path) -1):
                    if (u == path[j]) & (v == path[j+1]):
                        c += 'r' + str(i) + ' + '
            # No link capacities are exceeded by the sum of the path rates on them
            if c:
                program_file.write(c + '0 < ' + str(graph[u][v]['weight']) + '; \n')

        # Writing down third constraint
        for i in range(len(all_paths)):
            # Path rates must be positive
            program_file.write('r' +str(i) + ' >= 0; \n')

        program_file.close()

    # Solve the linear program
    var_val_map = wanteutility.ortools_solve_lp_and_get_var_val_map(
        '../myself/output/b/program.lp'
    )

    # Retrieve the rates from the variable values

    # Finally, write the rates to the output file
    with open(out_rates_filename, "w+") as rate_file:
        for i in range(len(all_paths)):
            rate_file.write(str(var_val_map['r' + str(i)]) + '\n')


def main():
    for appendix in range(assignment_parameters.num_tests_b):
        solve(
            "../ground_truth/input/b/graph%s.graph" % appendix,
            "../ground_truth/input/b/demand%s.demand" % appendix,
            "../ground_truth/input/b/path%s.path" % appendix,
            "../myself/output/b/rate%s.rate" % appendix
        )


if __name__ == "__main__":
    main()
