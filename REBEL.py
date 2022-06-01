import spacy
import rebel.spacy_component

nlp = spacy.load("en_core_web_sm")

nlp.add_pipe("rebel", after="senter", config={
    'device': -1,  # Number of the GPU, -1 if want to use CPU
    'model_name': 'Babelscape/rebel-large'}  # Model used, will default to 'Babelscape/rebel-large' if not given
             )
input_sentence = "Gràcia is a district of the city of Barcelona, Spain."

doc = nlp(input_sentence)

for value, rel_dict in doc._.rel.items():
    print(f"{value}: {rel_dict}")