import torch
import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

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
  """Execute the cypher query and retrieve data from Neo4j"""
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


def load_edge(cypher, src_index_col, src_mapping, dst_index_col, dst_mapping,
              encoders=None, **kwargs):
  """Execute the cypher query and retrieve data from Neo4j"""

  df = fetch_data(cypher)

  # Define edge index
  src = [src_mapping[index] for index in df[src_index_col]]
  dst = [dst_mapping[index] for index in df[dst_index_col]]
  edge_index = torch.tensor([src, dst])

  # Define edge features
  edge_attr = None
  if encoders is not None:
    edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
    edge_attr = torch.cat(edge_attrs, dim=-1)

  return edge_index, edge_attr

class SequenceEncoder(object):
  # The 'SequenceEncoder' encodes raw column strings into embeddings.
  def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
    self.device = device
    self.model = SentenceTransformer(model_name, device=device)

  @torch.no_grad()
  def __call__(self, df):
    x = self.model.encode(df.values, show_progress_bar=True,
                          convert_to_tensor=True, device=self.device)
    return x.cpu()

