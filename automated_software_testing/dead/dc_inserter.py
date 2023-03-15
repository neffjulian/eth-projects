#!/usr/bin/env python3

import os
import random

# variable ranges:
'''
uint8_range = {'min': 0, 'max': 255}
int8_range = {'min': -128, 'max': 127}
uint16_range = {'min': 0, 'max': 65535}
int16_range = {'min': -32768, 'max': 32767}
uint32_range = {'min': 0, 'max': 4294967295}
int32_range = {'min': -2147483648, 'max': 2147483647}
uint64_range = {'min': 0, 'max': 18446744073709551615}
int64_range = {'min': -9223372036854775808, 'max': 9223372036854775807}
ranges = [int8_range, uint8_range, int16_range, uint16_range, int32_range, uint32_range, int64_range, uint64_range]
'''
# Get range of variable values depending on possible types
max_vals = [127, 255, 32767, 65535, 2147483647, 4294967295, 9223372036854775807, 18446744073709551615]
min_vals = [0, -128, -32768, -2147483648, -9223372036854775808]

def get_range(min_val, max_val):
    r = {'min': 0, 'max': 127}
    r['max'] = min([x for x in max_vals if max_val <= x], default=0)
    r['min'] = max([x for x in min_vals if min_val >= x], default=127)
    return r
            
# get a list of all global variables and generate GDB commands to get values
def get_global(candidate_code):
    print("GET GLOBAL VARIABLE INFO")
    code = iter(candidate_code.splitlines())
    line_counter = 0
    global_definitions = False
    global_variables = []
    for code_line in code:
        # start global variable definitions
        if(code_line == "/* --- GLOBAL VARIABLES --- */"):
            global_definitions = True
        # end global variable definitions
        if(code_line == "/* --- FORWARD DECLARATIONS --- */"):
            break
        # check global definition
        if(global_definitions):
            tokens = code_line.split()
            if not (tokens == []) and not (tokens[2] == "GLOBAL"):
                # check that variable is not a pointer
                if(not (tokens[2][0] == "*" or tokens[2][0] != "g")):
                    # add to the variables list
                    global_variables.append(tokens[2])
    

    # print list of global variables
    gdb_global = ""
    for var in global_variables:
        gdb_global = gdb_global + "\nprint " + var
    return gdb_global, global_variables

# sets local variable declarations to avoid first usage in the if-statement
def set_locals(local_var_list, candidate_code):
    print("SET LOCAL VARIABLE DECLARATIONS")
    # prepare local variable declarations
    local_decl = ""
    for var in local_var_list:
        local_decl = local_decl + "int " + var + ";\n"
    # iterate through each code line
    code = iter(candidate_code.splitlines())
    new_candidate = ""
    for code_line in code:
        # start of global variable definitions
        if(code_line == "/* --- GLOBAL VARIABLES --- */"):
            # set local variable declarations
            new_candidate = new_candidate + "\n/* --- LOCAL VARIABLE DECLARATIONS --- */\n" + local_decl + code_line + "\n"
        else:
            # add normal code lines
            new_candidate = new_candidate + code_line + "\n"

    return(new_candidate)

# take code. transform to compile ready. keep track of DCEMarkers
# PRECONDITION: The markers are numbered continuously
def precompute(candidate_code):
    print("PRECOMPUTE FOR COMPILATION")
    os.system("rm tmp/gdb_log.txt tmp/candidate.c tmp/command_file.txt")
    counter = 0
    x = candidate_code.find("void DCEMarker{}_(void);".format(counter))
    gdb_commands = ""
    # get global variable info
    gdb_global, global_variables = get_global(candidate_code)
    
    # interate through all markers
    while(not x == -1):
        # complete DCEMarker to a explicit function definition
        candidate_code = candidate_code.replace("void DCEMarker{}_(void);".format(counter), "void DCEMarker{}_(void){{}}".format(counter))
        # add the marker as breakpoint
        gdb_commands = gdb_commands + "break DCEMarker{}_\n".format(counter)
        # next iteration
        counter += 1
        x = candidate_code.find("void DCEMarker{}_(void);".format(counter))
    
    # store candidate as .c file to compile
    new_candidate = open('tmp/candidate.c', 'w+')
    new_candidate.write(candidate_code)
    new_candidate.close()
    # complete gdb_commands to automatically run and log the program. \ninfo locals ommitted for only global variable use
    gdb_commands = gdb_commands + "set logging file gdb_log.txt\nset logging on\nr\nwhile 1\ns" + gdb_global + "\ncontinue\nend"
    # store gdb_commands as .txt file to run in a gdb session
    command_file = open('tmp/command_file.txt', 'w+')
    command_file.write(gdb_commands)
    command_file.close()
    return global_variables

# run gdb_script.sh
def run_gdb():
    print("COMPILE AND EXECUTE GDB AND GET LOCAL VARIABLE INFO")
    os.system("./gdb_script.sh")

# evaluate gdb log file
def eval_log(global_variables):
    print("START: EVALUTATE LOG FILE")
    global_count = len(global_variables)
    curr_break = ""
    var_vals = {}
    local_var_list = []
    
    with open("tmp/gdb_log.txt", "r") as log_file:
        # iterate through each line in the log file
        for line in log_file:
            stripped_line = line.strip()
            # split lines into list of tokens
            tokens = stripped_line.split()

            # catch breakpoint and update the current active marker
            if (len(tokens) > 0 and tokens[0] == "Breakpoint"):
                curr_break = tokens[2]
                if not (curr_break in var_vals):
                    var_vals[curr_break] = {}

            # catch variable values
            if(len(tokens) > 2 and tokens[1] == "="):
                var = tokens[0]
                val = tokens[2]
                # check if var is a global variable
                if var[0] == '$':
                    global_number = var[1:]
                    global_number = (int(global_number) - 1) % global_count
                    var = global_variables[global_number]
                # check if var is a local variable (starts with l_)
                elif var[0] == 'l':
                    local_var_list.append(var)
                
                # add new variable to dictionary
                if not (var in var_vals[curr_break]):
                    var_vals[curr_break][var] = []
                # bring value to int format
                try:
                    val = int(val, 0)
                    if not (val in var_vals[curr_break][var]):
                        var_vals[curr_break][var].append(val)
                except ValueError:
                    print("VALUE ERROR")                
    print("END: EVALUATED LOG FILE")
    return var_vals, local_var_list
    
# generates a unsat condition 
def unsatConditionGenerator(var_vals):
    print("GENERATE UNSATISFIABLE CONDITION WITH HELP OF VARIABLE VALUES")
    conditions = {}
    # iterate through markers; create a condition for a new marker
    for marker, variables in var_vals.items():
        conditions[marker] = ""
        # iterate through variables of the marker
        for var, vals in variables.items():
            # generate random value in specific range to fulfill type requirements
            if not (vals == []):
                min_val = min(vals)
                max_val = max(vals)
                r = get_range(min_val, max_val)
                new_var = random.randint(r['min'], r['max'])
                while(new_var in vals):
                    new_var = random.randint(r['min'], r['max'])
                # check if it is the first variable and create condition string
                if conditions[marker] == "":
                    conditions[marker] = str(new_var) + " == " + var
                else:
                    conditions[marker] = conditions[marker] + " || " + str(new_var) + " == " + var
    print("CONDITION GENERATED")
    # return generated conditions
    return conditions

# create a new marker with the next possible index. it fills up missing indices
def get_new_marker(code):
    counter = 0
    index = code.find("void DCEMarker{}_(void);".format(counter))
    old_index = 0
    while(not -1 == index):
        old_index = index
        counter += 1
        index = code.find("void DCEMarker{}_(void);".format(counter))
    new_marker = "void DCEMarker{}_(void);\n".format(counter)
    new_marker_call = "DCEMarker{}_();".format(counter)
    code = code[:old_index] + new_marker + code[old_index:]
    return code, new_marker_call
    
# put if-statements around markers
def instrument_code(conditions, code):
    print("INSTRUMENT CODE")
    # iterate through each marker with its unsat condition
    for marker, condition in conditions.items():
        # look for the maker call
        marker_call = marker + "();"
        # get new marker
        code, new_marker_call = get_new_marker(code)
        # create replacement code for the marker call (one line)
        replacement = "if (" + condition + "){" + marker_call + "} " + new_marker_call
        # replace the marker call with replacement putting it into dead code
        code = code.replace(marker_call, replacement)
    print("CODE INSTRUMENTED")
    return code
    
# enter program from generator.py
def entrance(candidate_code):
    # store program in tmp/
    program_txt = open('tmp/candidate.txt', 'w+')
    program_txt.write(candidate_code)
    program_txt.close()
    global_variables = precompute(candidate_code)
    run_gdb()
    var_vals, local_var_list = eval_log(global_variables)
    conditions = unsatConditionGenerator(var_vals)
    new_candidate = instrument_code(conditions, candidate_code)
    # new_candidate = set_locals(local_var_list, new_candidate)
    return new_candidate
    
# enter program standalone
def main():
    text_file = open("tmp/candidate.txt", "r")
    candidate_code = text_file.read()
    text_file.close()
    global_variables = precompute(candidate_code)
    run_gdb()
    var_vals, local_var_list = eval_log(global_variables)
    conditions = unsatConditionGenerator(var_vals)
    new_candidate = instrument_code(conditions, candidate_code)
    # new_candidate = set_locals(local_var_list, new_candidate)
    new_program_txt = open('tmp/candidate_new.txt', 'w+')
    new_program_txt.write(new_candidate)
    new_program_txt.close()
    
if __name__ == "__main__":
    main()


