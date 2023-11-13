import os
import csv


SAT_POSITIONS_PATH = "../input_data/sat_positions.txt"
ISL_PATH = "../output_data/sat_links.txt"

satellites_list = []
isl_list = []


def read_satellites_list():
    """
    Read the provided sat_positions.txt file into a usable structure
    """
    with open(SAT_POSITIONS_PATH, newline='') as csvfile:
        satellites = csv.reader(csvfile, delimiter=',')
        for sat in satellites:
            satellites_list.append(sat)


def build_naive_grid():
    """
    Build a naive satellite grid based on orbits and satellite positions
    """
    for satellite in satellites_list:
        satellite_ID, orbit_ID, satInOrbit_ID = int(
            satellite[0]), int(satellite[1]), int(satellite[2])
        # Top
        top_id = satellite_ID + \
            1 if satInOrbit_ID < 39 else (orbit_ID * 40)
        # Bottom
        bottom_id = satellite_ID - \
            1 if satInOrbit_ID > 0 else (((orbit_ID+1)*40 - 1))
        # Left
        left_id = satellite_ID - \
            40 if orbit_ID > 0 else (39*40 + satInOrbit_ID)
        # Right
        right_id = satellite_ID + 40 if orbit_ID < 39 else (satInOrbit_ID)

        if [top_id, satellite_ID] not in isl_list:
            isl_list.append([satellite_ID, top_id])
        if [bottom_id, satellite_ID] not in isl_list:
            isl_list.append([satellite_ID, bottom_id])
        if [left_id, satellite_ID] not in isl_list:
            isl_list.append([satellite_ID, left_id])
        if [right_id, satellite_ID] not in isl_list:
            isl_list.append([satellite_ID, right_id])


def write_isls():
    """
    Writes the list of ISLs to the designated output file
    """
    with open(ISL_PATH, mode="w") as isl_file:
        isl_writer = csv.writer(isl_file, delimiter=",")
        for isl in isl_list:
            isl_writer.writerow(isl)


if __name__ == "__main__":
    read_satellites_list()

    build_naive_grid()

    write_isls()
