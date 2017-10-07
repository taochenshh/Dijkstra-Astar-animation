#! /usr/bin/env python
'''
Path Planning using Dijkstra's Algorithm
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import Queue
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from heapq import heappush, heappop

# 0 free space, white
# 1 obstacle, black
# 2 start, green
# 3 end, magenta
# 4 active_set, blue
# 5 visited_set, red
# 6 path, cyan


def read_mapfile(filename):
    with open(filename, 'r') as f:
        row = int(f.readline().split()[1])
        column = int(f.readline().split()[1])
        start = f.readline().split()[1:]
        start = tuple([int(i) for i in start])
        end = f.readline().split()[1:]
        end = tuple([int(i) for i in end])
        mp = f.readline()
        mp = f.readlines()
        mapmatrix = []
        for line in mp:
            mpline = line.split()
            mpline = [int(i) for i in mpline]
            mapmatrix.append(mpline)
    mapmatrix = np.array(mapmatrix)
    assert mapmatrix.shape[0] == row and mapmatrix.shape[1] == column, 'Size mismatch'
    return mapmatrix, start, end

class Dijkstra:
    def __init__(self, map, start, end):
        self.found_path = False
        self.program_end = False
        self.map = map
        self.start = start
        self.end = end
        self.map_width = map.shape[1]
        self.map_height = map.shape[0]
        self.map[start[0], start[1]] = 2
        self.map[end[0], end[1]] = 3
        self.active_set = []
        self.visited_nodes = {}
        self.path = Queue.LifoQueue()
        self.visited_nodes[start] = {'parent': (-1, -1), 'cost': 0}
        heappush(self.active_set, (self.visited_nodes[start]['cost'], start))

        self.env_fig = plt.figure()
        self.ax = self.env_fig.add_subplot(111, aspect='equal')
        self.cmap = mpl.colors.ListedColormap(['w', 'k', 'g', 'm', 'b', 'r', 'c'])
        bounds = [i - 0.5 for i in range(8)]
        self.norm = mpl.colors.BoundaryNorm(bounds, self.cmap.N)
        self.im = self.ax.imshow(self.map, animated=True, cmap=self.cmap, norm=self.norm,
                                 extent=[0.0, map.shape[1], 0.0, map.shape[0]])


    def update_neighbors(self, current):
        neighbors = [(current[0] - 1, current[1]),
                     (current[0] + 1, current[1]),
                     (current[0], current[1] - 1),
                     (current[0], current[1] + 1)]
        for nb in neighbors:
            if 0 <= nb[0] < self.map_height and \
               0 <= nb[1] < self.map_width:
                if nb == self.end:
                    # Reach the end, then retrieve the path
                    print('Path Found')
                    self.found_path = True
                    self.program_end = True
                    while self.visited_nodes[current]['parent'] != (-1, -1):
                        self.path.put(current)
                        current = self.visited_nodes[current]['parent']
                    return
                if self.map[nb[0], nb[1]] != 1:
                    if nb not in self.visited_nodes or\
                       self.visited_nodes[nb]['cost'] > self.visited_nodes[current]['cost'] + 1:
                        self.visited_nodes[nb] = {'parent': current,
                                                  'cost': self.visited_nodes[current]['cost'] + 1}
                        heappush(self.active_set, (self.visited_nodes[nb]['cost'], nb))
                        self.map[nb[0], nb[1]] = 4

    def step(self, i):
        if not self.found_path:
            if self.active_set:
                cost, current = heappop(self.active_set)
                if current != self.start:
                    self.map[current] = 5
                self.update_neighbors(current)
            elif not self.program_end:
                self.program_end = True
                print('No Path Found')
        else:
            if not self.path.empty():
                p = self.path.get()
                self.map[p[0], p[1]] = 6
        self.im.set_array(self.map)
        return [self.im]

    def plt_show(self):
        anim = animation.FuncAnimation(self.env_fig,
                                       self.step,
                                       init_func=None,
                                       frames=10,
                                       interval=50,
                                       blit=False)
        self.ax.axis('equal')
        self.ax.axis('off')
        plt.title('Dijkstra\'s Algorithm')
        plt.show()


if __name__ == "__main__":
    map_file = 'map1.txt'
    mapmatrix, start, end = read_mapfile(map_file)
    dij = Dijkstra(mapmatrix, start, end)
    dij.plt_show()
