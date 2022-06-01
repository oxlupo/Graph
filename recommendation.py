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

class GenresEncoder(object):

  """The 'GenreEncoder' splits the raw column strings by 'sep' and converts
  # individual elements to categorical labels."""

  def __init__(self, sep='|'):
    self.sep = sep

  def __call__(self, df):
    genres = set(g for col in df.values for g in col.split(self.sep))
    mapping = {genre: i for i, genre in enumerate(genres)}

    x = torch.zeros(len(df), len(mapping))
    for i, col in enumerate(df.values):
      for genre in col.split(self.sep):
        x[i, mapping[genre]] = 1
    return x

class IdentityEncoder(object):
  """The 'IdentityEncoder' takes the raw column values and converts them to
  PyTorch tensors."""
  def __init__(self, dtype=None, is_list=False):
    self.dtype = dtype
    self.is_list = is_list

  def __call__(self, df):
    if self.is_list:
      return torch.stack([torch.tensor(el) for el in df.values])
    return torch.from_numpy(df.values).to(self.dtype)

movie_query = """
MATCH (m:Movie)-[:IN_GENRE]->(genre:Genre)
WITH m, collect(genre.name) AS genres_list
RETURN m.movieId AS movieId, m.title AS title, apoc.text.join(genres_list, '|') AS genres, m.fastrp AS fastrp
"""


movie_x, movie_mapping = load_node(
    movie_query,
    index_col='movieId', encoders={
        'title': SequenceEncoder(),
        'genres': GenresEncoder(),
        'fastrp': IdentityEncoder(is_list=True)
    })

