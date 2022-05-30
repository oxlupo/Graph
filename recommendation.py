import torch
import pandas as pd
from neo4j import GraphDatabase
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
url = 'bolt://44.200.159.63:7687'
user = 'neo4j'
password = 'shipment-blades-addressee'

driver = GraphDatabase.driver(url, auth=(user, password))

def fetch_data(query):
  with driver.session() as session:
    result = session.run(query)
    return pd.DataFrame([r.values() for r in result], columns=result.keys())


def load_node(cypher, index_col, encoders=None, **kwargs):
  # Execute the cypher query and retrieve data from Neo4j
  df = fetch_data(cypher)
  df.set_index(index_col, inplace=True)
  # Define node mapping
  mapping = {index: i for i, index in enumerate(df.index.unique())}
  # Define node features
  x = None
  if encoders is not None:
    xs = [encoder(df[col]) for col, encoder in encoders.items()]
    x = torch.cat(xs, dim=-1)

  return x, mapping