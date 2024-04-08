import networkx as nx

# Create a simple graph
G = nx.Graph()
G.add_edge(1, 2, weight=0.5)
G.add_edge(2, 3, features={'color': 'red'})

# Test conditions
if 'features' in G[1][2]:
    attribute_name = 'features'
elif 'weight' in G[1][2]:
    attribute_name = 'weight'
else:
    raise ValueError(f"Edge attribute 'features' or 'weight' not found in the graph. it must be one of {G[0][1]}")

print("Attribute name:", attribute_name)
