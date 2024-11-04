import networkx as nx
import sys
sys.path.append("/root/autodl-tmp/shortest-path-graph/graph")
from graph.words_of_graph import WordsOfGraph
from graph.shortest_path_graph import ShortestPathGraph
from k_score.k_counter import KCounter

if __name__ == '__main__':
    s1 = 'e1 e2 period'
    s2 = 'e1 e2'
    s1 = "e1 was in e2 dated summer of 1938 along other dresses from this same period."
    s2 = "e1 was in e2 dated summer of 1938 along with several other dresses from this same period."
    G1 = WordsOfGraph(s1, 2).get_graph()
    G2 = WordsOfGraph(s2, 2).get_graph()
    C1 = ShortestPathGraph(G1, 2).get_graph()
    C2 = ShortestPathGraph(G2, 2).get_graph()
    score = KCounter(C1, C2).get_k_score()
    print(score)
