from neo4j import GraphDatabase
import pandas as pd


url = 'bolt://44.200.159.63:7687'
user = 'neo4j'
password = 'shipment-blades-addressee'

driver = GraphDatabase.driver(url, auth=(user, password))

def fetch_data(query):
  with driver.session() as session:
    result = session.run(query)
    return pd.DataFrame([r.values() for r in result], columns=result.keys())

