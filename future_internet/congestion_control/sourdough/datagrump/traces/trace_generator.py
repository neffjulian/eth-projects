import random

def load_data(path):
    result = []
    with open(path) as f:
        for line in f:
            result.append(int(line[:-1]))
    return result

def find_index(data, value):
    for i in range(len(data)):
        if data[i]>value*1000:
            return i
    return -1


def gen_file(name):
    data = []

    for i in [1,2,3]:
        load_up = load_data('%d.up'%i)
        load_down = load_data('%d.down'%i)
        data.append((load_down, load_up))

    random.shuffle(data)

    current_time = 0
    final_up_trace = []
    final_down_trace = []
    for repeat in range(2):
        for i in range(3):
            span = random.randint(10,20)
            start = random.randint(0,120-span)
            start_index_down = find_index(data[i][0], start)
            end_index_down = find_index(data[i][0], start+span)
            start_index_up = find_index(data[i][1], start)
            end_index_up = find_index(data[i][1], start+span)

            down_part = list(map(lambda x: x-data[i][0][start_index_down]+current_time, data[i][0][start_index_down:end_index_down]))
            up_part = list(map(lambda x: x-data[i][1][start_index_up]+current_time, data[i][1][start_index_up:end_index_up]))

            final_down_trace += down_part
            final_up_trace += up_part
            current_time = max(final_down_trace[-1], final_up_trace[-1])

    with open('t%s.up'%name, 'w') as f:
        for x in final_up_trace:
            f.write('%d\n'%x)
    with open('t%s.down'%name, 'w') as f:
        for x in final_down_trace:
            f.write('%d\n'%x)


def main():
    for i in range(50):
        gen_file(str(i))


if __name__ == '__main__':
    main()