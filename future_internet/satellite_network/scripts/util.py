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
import math

EARTH_RADIUS = 6371  # Kms


def read_sat_positions(sat_pos_file):
    """
        Read satellite positions from file
        :param sat_pos_file: Input file name
        :return: Data structure holding satellite position details
    """
    sat_positions = {}
    lines = [line.rstrip('\n') for line in open(sat_pos_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        sat_positions[int(val[0])] = {
            "lat_deg": float(val[3]),
            "lat_rad": math.radians(float(val[3])),
            "long_deg": float(val[4]),
            "long_rad": math.radians(float(val[4])),
            "alt_km": int(float(val[5]))
        }
    return sat_positions


def read_city_positions(city_pos_file):
    """
        Read city positions from file
        :param city_pos_file: Input file name
        :return: Data structure holding city coordinates
    """
    city_positions = {}
    lines = [line.rstrip('\n') for line in open(city_pos_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        city_positions[int(val[0])] = {
            "lat_deg": float(val[2]),
            "long_deg": float(val[3]),
            "gdp": float(val[4])
        }
    return city_positions


def read_coverage(coverage_file):
    """
        Read satellite coverage for cities
        :param coverage_file: Input file name
        :return: Data structure holding city-satellite coverage mapping
    """
    city_coverage = {}
    lines = [line.rstrip('\n') for line in open(coverage_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        city_coverage[int(val[0]), int(val[1])] = float(val[2])
    return city_coverage


def read_city_pairs(city_pair_file):
    """
    Read city pairs and corr. geodesic distances
    :param city_pair_file: Input file name
    :return: Dat ctructure holding city-pairs and corr.geodesic distances
    """
    city_pairs = {}
    lines = [line.rstrip('\n') for line in open(city_pair_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        city_pairs[int(val[0]), int(val[1])] = float(val[2])
        city_pairs[int(val[1]), int(val[0])] = float(val[2])
    return city_pairs


def read_valid_isls(valid_isl_file):
    """
    Read valid ISLs from file
    :param valid_isl_file: Input file name
    :return:  Data structure holding valid inter-satellite links
    """
    valid_isls = {}
    lines = [line.rstrip('\n') for line in open(valid_isl_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        valid_isls[int(val[0]), int(val[1])] = float(val[2])
    return valid_isls


def read_and_filter_selected_links(valid_isls, selected_links_file):
    """
        Read selected ISLs from file
        :param valid_isl_file: Input file name
        :return:  Data structure holding valid inter-satellite links
        """
    selected_isls = {}
    lines = [line.rstrip('\n') for line in open(selected_links_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        sat1 = int(val[0])
        sat2 = int(val[1])
        try:
            dist = valid_isls[sat1, sat2]
            selected_isls[int(val[0]), int(val[1])] = dist
            selected_isls[int(val[1]), int(val[0])] = dist
        except:
            print("Selected link", sat1, "-", sat2,  "is not valid")
            exit()
    return selected_isls

