# Future Internet - Congestion Control Contest

## This assignment has been adopted from Stanford course [CS244](https://web.stanford.edu/class/cs344g/contest.html)

## Background

The goal of this project is to design a congestion control algorithm that operates well on cellular networks. It is easy to design a congestion control algorithm that is optimized only for packet latency or only for throughput. Being conservative and sending one packet only when an ACK signal is received gives near-perfect latency. On the other hand, sending as much data as possible will maximize the throughput, but in that scenario, the latency will significantly increase. The question is how to balance between these two dimensions.

In this exercise, we optimize the trade-off between throughput and latency in cellular networks where network conditions (i.e. available bandwidth) oscillate frequently.

For more information about the problem (and potential solutions) watch the following video [link](http://www.youtube.com/watch?v=FUgdF-2V8Cw):

[![Sprout: Stochastic Forecasts Achieve High Throughput and Low Delay over Cellular Networks](http://img.youtube.com/vi/FUgdF-2V8Cw/0.jpg)](http://www.youtube.com/watch?v=FUgdF-2V8Cw)

## Prepare for the contest

First, you should install libraries and tools you need to run the project

```bash
$ sudo apt-get install build-essential git debhelper autotools-dev \
 dh-autoreconf iptables protobuf-compiler libprotobuf-dev pkg-config \
 libssl-dev dnsmasq-base ssl-cert libxcb-present-dev libcairo2-dev \
 libpango1.0-dev iproute2 apache2-dev apache2-bin iptables dnsmasq-base \
 gnuplot iproute2 apache2-api-20120211 libwww-perl
```

If you do not have a Linux machine or you do not want to do pollute your system, you can us a VM with all dependencies preinstalled [VM link](https://polybox.ethz.ch/index.php/s/TRL7MuA7B59TfGm).

To setup the project, do the following:
```bash
$ git clone https://gitlab.inf.ethz.ch/COURSE-FI2021/grp<XXX>-fi2021.git
$ cd project1
$ cd mahimahi
$ ./autogen.sh && ./configure && make
$ sudo make install
$
$ cd ..
$ cd sourdough
$ ./autogen.sh && ./configure && make
$
$ sudo sysctl -w net.ipv4.ip_forward=1
```

Installation tips in case you need them:

- Mahimahi must be compiled with GCC version lower than 9. If you get error messages during the installation process, check this [link](https://www.mail-archive.com/debian-bugs-rc@lists.debian.org/msg555451.html).
    
- You can skip Mahimahi compilation by installing it directly:
```bash
$ sudo apt-get install mahimahi
```

To run an example code:
```bash
$ cd sourdough/datagrump
$ ./run-contest
```

The example code runs one client and one server, where the client sends a continuous stream of data to the server. The script varies the bandwidth between the two according a particular trace. The data is transmitted between the machines using a single link with an unlimited outgoing buffer. It is important to notice that in this scenario, there is no packet loss.

For your experiments, generate traces using:

```bash
sourdough/datagrump/traces/trace_generator.py
```

or use a default trace:

```bash
sourdough/datagrump/traces/t0.down
```

To change the trace you use, modify the _run-contest_ script.

## Submit your solution

After you commit changes to your solution, or server will automatically evaluate your code against the default trace and update the leaderboard. Note that the update is not instantaneous. It could take between 5 to 10 minutes before your new score is available.


# Contest

Your task is to optimize the trade-off between bandwidth and latency. Your score is calculated as:
```bash
log(average_throughput/(95_percentile_latency*0.001))
```

You see the score after executing the ./run-contest script.
To improve your score, you should focus on changing the congestion control controller placed in:
```bash
sourdough/datagrump/controller.cc
```

Every time you change the file, you must rerun the make command:
```bash
$ make
```

In general, if your implementation requires more advanced techniques, you are free to change other files in the _datagrump_ folder apart from the _controller_.

### Task A
Maximize your score by changing the fixed window size in the _controller_.

You should achieve the score higher than **1.90**.

Achieving score **1.80** will give you at least 4 points.

### Task B
Maximize your score by changing the window_size dynamically. Think about additive increase/multiplicative decrease [[link]](https://en.wikipedia.org/wiki/Additive_increase/multiplicative_decrease).

You should achieve score higher than **2.40**.

Achieving the score **2.40** will give you at least 8 points.

### Task C
Achieve the best possible score.

**Important:** after the deadline, your solution will be tested against a set of hidden traces created using the same trace generator that is available to you, but with a secret seed. The final leaderboard will be published after the deadline with scores of each team on our hidden test set and the grade fill be assigned according to that score.

Try not to overfit your solution to any concrete trace.


### Final notice
The maximum number of points is _12.5_.

In you algorithm, **do not hardcode the trace**. You can make decision based on the bandwidth you have observed, but not based on the bandwidth changes that will come in the future.

Before the deadline, write a short explanation (up to 5 sentences) about how your algorithm works and store it in:

```bash
project1/explain.txt
```

**You can find the leader board [here](http://bach020.ethz.ch/cc_contest.html)**
