import json
import urllib.request
import pandas as pd
from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'yousef'))


def run_query(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())


def ie_pipeline(text, relation_threshold=0.9, entities_threshold=0.8):
    # Prepare the URL.
    data = urllib.parse.urlencode([
        ("text", text), ("relation_threshold", relation_threshold),
        ("entities_threshold", entities_threshold)])

    url = "http://localhost:7687?" + data
    req = urllib.request.Request(url, data=data.encode("utf8"), method="GET")
    with urllib.request.urlopen(req, timeout=150) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))
    # Output the annotations.
    return response


example_data = ie_pipeline("""
Elon Musk is a business magnate, industrial designer, and engineer.
He is the founder, CEO, CTO, and chief designer of SpaceX.
He is also early investor, CEO, and product architect of Tesla, Inc.
He is also the founder of The Boring Company and the co-founder of Neuralink. 
A centibillionaire, Musk became the richest person in the world in January 2021, with an estimated net worth of $185 billion at the time, surpassing Jeff Bezos.
Musk was born to a Canadian mother and South African father and raised in Pretoria, South Africa.
He briefly attended the University of Pretoria before moving to Canada aged 17 to attend Queen's University.
He transferred to the University of Pennsylvania two years later, where he received dual bachelor's degrees in economics and physics.
He moved to California in 1995 to attend Stanford University, but decided instead to pursue a business career.
He went on co-founding a web software company Zip2 with his brother Kimbal Musk.
  """)

print(example_data)

