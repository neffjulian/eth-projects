First we used dijkstra to calculate the distances between all the nodes.
We excluded the cities from the graph and only included them when we calculated the specific path between two cities. (To be sure that we have no paths which include further cities).
During this we counted the occurence of each edge and weighted it by the length of the path
Then we took greadily each edge in descending order of the occurence
