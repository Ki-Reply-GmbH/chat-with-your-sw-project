import sys
import io
import json
import os
import openai
from src.utils.git_handler import GitHandler
from src.utils.cache import DisabledCache, SimpleCache
from src.utils.directory_loader import DirectoryLoader
from src.utils.vector_db import VectorDB, MongoDBAtlasVectorDB
from src.agents.docstring_agent import DocstringAgent
from src.agents.chat_agent import ChatAgent
from src.agents.embedding_agent import EmbeddingAgent
from src.config import load_config
from src.models import LLModel
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
from openai.embeddings_utils import get_embedding, cosine_similarity

authors_query = """
# Credits

## Development Leads

- Audrey Roy Greenfeld ([@audreyfeldroy](https://github.com/audreyfeldroy))
- Daniel Roy Greenfeld ([@pydanny](https://github.com/pydanny))
- Raphael Pierzina ([@hackebrot](https://github.com/hackebrot))

## Core Committers

- Michael Joseph ([@michaeljoseph](https://github.com/michaeljoseph))
- Paul Moore ([@pfmoore](https://github.com/pfmoore))
- Andrey Shpak ([@insspb](https://github.com/insspb))
- Sorin Sbarnea ([@ssbarnea](https://github.com/ssbarnea))
- Fábio C. Barrionuevo da Luz ([@luzfcb](https://github.com/luzfcb))
- Simone Basso ([@simobasso](https://github.com/simobasso))
- Jens Klein ([@jensens](https://github.com/jensens))
- Érico Andrei ([@ericof](https://github.com/ericof))

## Contributors

- Steven Loria ([@sloria](https://github.com/sloria))
- Goran Peretin ([@gperetin](https://github.com/gperetin))
- Hamish Downer ([@foobacca](https://github.com/foobacca))
- Thomas Orozco ([@krallin](https://github.com/krallin))
- Jindrich Smitka ([@s-m-i-t-a](https://github.com/s-m-i-t-a))
- Benjamin Schwarze ([@benjixx](https://github.com/benjixx))
- Raphi ([@raphigaziano](https://github.com/raphigaziano))
- Thomas Chiroux ([@ThomasChiroux](https://github.com/ThomasChiroux))
- Sergi Almacellas Abellana ([@pokoli](https://github.com/pokoli))
- Alex Gaynor ([@alex](https://github.com/alex))
- Rolo ([@rolo](https://github.com/rolo))
- Pablo ([@oubiga](https://github.com/oubiga))
- Bruno Rocha ([@rochacbruno](https://github.com/rochacbruno))
- Alexander Artemenko ([@svetlyak40wt](https://github.com/svetlyak40wt))
- Mahmoud Abdelkader ([@mahmoudimus](https://github.com/mahmoudimus))
- Leonardo Borges Avelino ([@lborgav](https://github.com/lborgav))
- Chris Trotman ([@solarnz](https://github.com/solarnz))
- Rolf ([@relekang](https://github.com/relekang))
- Noah Kantrowitz ([@coderanger](https://github.com/coderanger))
- Vincent Bernat ([@vincentbernat](https://github.com/vincentbernat))
- Germán Moya ([@pbacterio](https://github.com/pbacterio))
- Ned Batchelder ([@nedbat](https://github.com/nedbat))
- Dave Dash ([@davedash](https://github.com/davedash))
- Johan Charpentier ([@cyberj](https://github.com/cyberj))
- Éric Araujo ([@merwok](https://github.com/merwok))
- saxix ([@saxix](https://github.com/saxix))
- Tzu-ping Chung ([@uranusjr](https://github.com/uranusjr))
- Caleb Hattingh ([@cjrh](https://github.com/cjrh))
- Flavio Curella ([@fcurella](https://github.com/fcurella))
- Adam Venturella ([@aventurella](https://github.com/aventurella))
- Monty Taylor ([@emonty](https://github.com/emonty))
- schacki ([@schacki](https://github.com/schacki))
- Ryan Olson ([@ryanolson](https://github.com/ryanolson))
- Trey Hunner ([@treyhunner](https://github.com/treyhunner))
- Russell Keith-Magee ([@freakboy3742](https://github.com/freakboy3742))
- Mishbah Razzaque ([@mishbahr](https://github.com/mishbahr))
- Robin Andeer ([@robinandeer](https://github.com/robinandeer))
- Rachel Sanders ([@trustrachel](https://github.com/trustrachel))
- Rémy Hubscher ([@Natim](https://github.com/Natim))
- Dino Petron3 ([@dinopetrone](https://github.com/dinopetrone))
- Peter Inglesby ([@inglesp](https://github.com/inglesp))
- Ramiro Batista da Luz ([@ramiroluz](https://github.com/ramiroluz))
- Omer Katz ([@thedrow](https://github.com/thedrow))
- lord63 ([@lord63](https://github.com/lord63))
- Randy Syring ([@rsyring](https://github.com/rsyring))
- Mark Jones ([@mark0978](https://github.com/mark0978))
- Marc Abramowitz ([@msabramo](https://github.com/msabramo))
- Lucian Ursu ([@LucianU](https://github.com/LucianU))
- Osvaldo Santana Neto ([@osantana](https://github.com/osantana))
- Matthias84 ([@Matthias84](https://github.com/Matthias84))
- Simeon Visser ([@svisser](https://github.com/svisser))
- Guruprasad ([@lgp171188](https://github.com/lgp171188))
- Charles-Axel Dein ([@charlax](https://github.com/charlax))
- Diego Garcia ([@drgarcia1986](https://github.com/drgarcia1986))
- maiksensi ([@maiksensi](https://github.com/maiksensi))
- Andrew Conti ([@agconti](https://github.com/agconti))
- Valentin Lab ([@vaab](https://github.com/vaab))
- Ilja Bauer ([@iljabauer](https://github.com/iljabauer))
- Elias Dorneles ([@eliasdorneles](https://github.com/eliasdorneles))
- Matias Saguir ([@mativs](https://github.com/mativs))
- Johannes ([@johtso](https://github.com/johtso))
- macrotim ([@macrotim](https://github.com/macrotim))
- Will McGinnis ([@wdm0006](https://github.com/wdm0006))
- Cédric Krier ([@cedk](https://github.com/cedk))
- Tim Osborn ([@ptim](https://github.com/ptim))
- Aaron Gallagher ([@habnabit](https://github.com/habnabit))
- mozillazg ([@mozillazg](https://github.com/mozillazg))
- Joachim Jablon ([@ewjoachim](https://github.com/ewjoachim))
- Andrew Ittner ([@tephyr](https://github.com/tephyr))
- Diane DeMers Chen ([@purplediane](https://github.com/purplediane))
- zzzirk ([@zzzirk](https://github.com/zzzirk))
- Carol Willing ([@willingc](https://github.com/willingc))
- phoebebauer ([@phoebebauer](https://github.com/phoebebauer))
- Adam Chainz ([@adamchainz](https://github.com/adamchainz))
- Sulé ([@suledev](https://github.com/suledev))
- Evan Palmer ([@palmerev](https://github.com/palmerev))
- Bruce Eckel ([@BruceEckel](https://github.com/BruceEckel))
- Robert Lyon ([@ivanlyon](https://github.com/ivanlyon))
- Terry Bates ([@terryjbates](https://github.com/terryjbates))
- Brett Cannon ([@brettcannon](https://github.com/brettcannon))
- Michael Warkentin ([@mwarkentin](https://github.com/mwarkentin))
- Bartłomiej Kurzeja ([@B3QL](https://github.com/B3QL))
- Thomas O'Donnell ([@andytom](https://github.com/andytom))
- Jeremy Carbaugh ([@jcarbaugh](https://github.com/jcarbaugh))
- Nathan Cheung ([@cheungnj](https://github.com/cheungnj))
- Abdó Roig-Maranges ([@aroig](https://github.com/aroig))
- Steve Piercy ([@stevepiercy](https://github.com/stevepiercy))
- Corey ([@coreysnyder04](https://github.com/coreysnyder04))
- Dmitry Evstratov ([@devstrat](https://github.com/devstrat))
- Eyal Levin ([@eyalev](https://github.com/eyalev))
- mathagician ([@mathagician](https://github.com/mathagician))
- Guillaume Gelin ([@ramnes](https://github.com/ramnes))
- @delirious-lettuce ([@delirious-lettuce](https://github.com/delirious-lettuce))
- Gasper Vozel ([@karantan](https://github.com/karantan))
- Joshua Carp ([@jmcarp](https://github.com/jmcarp))
- @meahow ([@meahow](https://github.com/meahow))
- Andrea Grandi ([@andreagrandi](https://github.com/andreagrandi))
- Issa Jubril ([@jubrilissa](https://github.com/jubrilissa))
- Nytiennzo Madooray ([@Nythiennzo](https://github.com/Nythiennzo))
- Erik Bachorski ([@dornheimer](https://github.com/dornheimer))
- cclauss ([@cclauss](https://github.com/cclauss))
- Andy Craze ([@accraze](https://github.com/accraze))
- Anthony Sottile ([@asottile](https://github.com/asottile))
- Jonathan Sick ([@jonathansick](https://github.com/jonathansick))
- Hugo ([@hugovk](https://github.com/hugovk))
- Min ho Kim ([@minho42](https://github.com/minho42))
- Ryan Ly ([@rly](https://github.com/rly))
- Akintola Rahmat ([@mihrab34](https://github.com/mihrab34))
- Jai Ram Rideout ([@jairideout](https://github.com/jairideout))
- Diego Carrasco Gubernatis ([@dacog](https://github.com/dacog))
- Wagner Negrão ([@wagnernegrao](https://github.com/wagnernegrao))
- Josh Barnes ([@jcb91](https://github.com/jcb91))
- Nikita Sobolev ([@sobolevn](https://github.com/sobolevn))
- Matt Stibbs ([@mattstibbs](https://github.com/mattstibbs))
- MinchinWeb ([@MinchinWeb](https://github.com/MinchinWeb))
- kishan ([@kishan](https://github.com/kishan3))
- tonytheleg ([@tonytheleg](https://github.com/tonytheleg))
- Roman Hartmann ([@RomHartmann](https://github.com/RomHartmann))
- DSEnvel ([@DSEnvel](https://github.com/DSEnvel))
- kishan ([@kishan](https://github.com/kishan3))
- Bruno Alla ([@browniebroke](https://github.com/browniebroke))
- nicain ([@nicain](https://github.com/nicain))
- Carsten Rösnick-Neugebauer ([@croesnick](https://github.com/croesnick))
- igorbasko01 ([@igorbasko01](https://github.com/igorbasko01))
- Dan Booth Dev ([@DanBoothDev](https://github.com/DanBoothDev))
- Pablo Panero ([@ppanero](https://github.com/ppanero))
- Chuan-Heng Hsiao ([@chhsiao1981](https://github.com/chhsiao1981))
- Mohammad Hossein Sekhavat ([@mhsekhavat](https://github.com/mhsekhavat))
- Amey Joshi ([@amey589](https://github.com/amey589))
- Paul Harrison ([@smoothml](https://github.com/smoothml))
- Fabio Todaro ([@SharpEdgeMarshall](https://github.com/SharpEdgeMarshall))
- Nicholas Bollweg ([@bollwyvl](https://github.com/bollwyvl))
- Jace Browning ([@jacebrowning](https://github.com/jacebrowning))
- Ionel Cristian Mărieș ([@ionelmc](https://github.com/ionelmc))
- Kishan Mehta ([@kishan3](https://github.com/kishan3))
- Wieland Hoffmann ([@mineo](https://github.com/mineo))
- Antony Lee ([@anntzer](https://github.com/anntzer))
- Aurélien Gâteau ([@agateau](https://github.com/agateau))
- Axel H. ([@noirbizarre](https://github.com/noirbizarre))
- Chris ([@chrisbrake](https://github.com/chrisbrake))
- Chris Streeter ([@streeter](https://github.com/streeter))
- Gábor Lipták ([@gliptak](https://github.com/gliptak))
- Javier Sánchez Portero ([@javiersanp](https://github.com/javiersanp))
- Nimrod Milo ([@milonimrod](https://github.com/milonimrod))
- Philipp Kats ([@Casyfill](https://github.com/Casyfill))
- Reinout van Rees ([@reinout](https://github.com/reinout))
- Rémy Greinhofer ([@rgreinho](https://github.com/rgreinho))
- Sebastian ([@sebix](https://github.com/sebix))
- Stuart Mumford ([@Cadair](https://github.com/Cadair))
- Tom Forbes ([@orf](https://github.com/orf))
- Xie Yanbo ([@xyb](https://github.com/xyb))
- Maxim Ivanov ([@ivanovmg](https://github.com/ivanovmg))

## Backers

We would like to thank the following people for supporting us in our efforts to maintain and improve Cookiecutter:

- Alex DeBrie
- Alexandre Y. Harano
- Bruno Alla
- Carol Willing
- Russell Keith-Magee

## Sprint Contributors

### PyCon 2016 Sprint

The following people made contributions to the cookiecutter project at the PyCon sprints in Portland, OR from June 2-5 2016.
Contributions include user testing, debugging, improving documentation, reviewing issues, writing tutorials, creating and updating project templates, and teaching each other.

- Adam Chainz ([@adamchainz](https://github.com/adamchainz))
- Andrew Ittner ([@tephyr](https://github.com/tephyr))
- Audrey Roy Greenfeld ([@audreyfeldroy](https://github.com/audreyfeldroy))
- Carol Willing ([@willingc](https://github.com/willingc))
- Christopher Clarke ([@chrisdev](https://github.com/chrisdev))
- Citlalli Murillo ([@citmusa](https://github.com/citmusa))
- Daniel Roy Greenfeld ([@pydanny](https://github.com/pydanny))
- Diane DeMers Chen ([@purplediane](https://github.com/purplediane))
- Elaine Wong ([@elainewong](https://github.com/elainewong))
- Elias Dorneles ([@eliasdorneles](https://github.com/eliasdorneles))
- Emily Cain ([@emcain](https://github.com/emcain))
- John Roa ([@jhonjairoroa87](https://github.com/jhonjairoroa87))
- Jonan Scheffler ([@1337807](https://github.com/1337807))
- Phoebe Bauer ([@phoebebauer](https://github.com/phoebebauer))
- Kartik Sundararajan ([@skarbot](https://github.com/skarbot))
- Katia Lira ([@katialira](https://github.com/katialira))
- Leonardo Jimenez ([@xpostudio4](https://github.com/xpostudio4))
- Lindsay Slazakowski ([@lslaz1](https://github.com/lslaz1))
- Meghan Heintz ([@dot2dotseurat](https://github.com/dot2dotseurat))
- Raphael Pierzina ([@hackebrot](https://github.com/hackebrot))
- Umair Ashraf ([@umrashrf](https://github.com/umrashrf))
- Valdir Stumm Junior ([@stummjr](https://github.com/stummjr))
- Vivian Guillen ([@viviangb](https://github.com/viviangb))
- Zaro ([@zaro0508](https://github.com/zaro0508))

"""

def json_serializer(obj):
    """Custom JSON serializer for unsupported types."""
    if isinstance(obj, ObjectId):
        return str(obj)  # Convert ObjectId to string
    # Add more custom serialization rules here if needed
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def main():
    # Allow prinint utf-8 characters in console
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")

    config = load_config()
    print("Model name: ")
    print(config.LLM_MODEL_NAME)
    cache = SimpleCache(tmp_path="./.tmp")

    dir_loader = DirectoryLoader(directory="./resources/cookiecutter")
    # Ermitteln des absoluten Pfads
    absolute_path = os.path.abspath(dir_loader.directory)
    print("Absoluter Pfad:", absolute_path)
    documents = dir_loader.load()

    py_file_paths = []
    print("Loaded files in total: ", len(documents))
    print("File paths:")
    for document in documents:
        print(document.metadata["source"])
        if document.metadata["source"].endswith(".py"):
            py_file_paths.append(document.metadata["source"])
    
    print("Python files:")
    print(py_file_paths)

    doc_agent = DocstringAgent(
        config.WORKING_DIR,
        py_file_paths,
        config.prompts,
        LLModel(config, cache)
    )
    doc_agent.make_docstrings()
    doc_agent.make_module_descriptions()
    
    with open("responses.json", "w", encoding="utf-8") as f:
        json.dump(doc_agent.responses, f, ensure_ascii=False, indent=4)

    with open("module_responses.json", "w", encoding="utf-8") as f:
        json.dump(doc_agent.module_responses, f, ensure_ascii=False, indent=4)

    
    """
    docstr_agent = DocstringAgent(
        config.WORKING_DIR,
        py_filepaths,
        config.prompts,
        LLModel(config, cache)
    )
    print("Creating documentation for python code...")
    docstr_agent.make_in_code_docs()
    print("Found these python files:") 
    keys = [key for key in docstr_agent.responses.keys()]
    print(keys)

    with open("responses.json", "w") as f:
        json.dump(docstr_agent.responses, f, indent=4)
    """

def main_embedding():
    module_responses = json.loads(open("module_responses.json").read())

    # Allow prinint utf-8 characters in console
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")

    config = load_config()
    print("Model name: ")
    print(config.LLM_MODEL_NAME)
    cache = SimpleCache(tmp_path="./.tmp")

    dir_loader = DirectoryLoader(directory="./resources/cookiecutter")
    # Ermitteln des absoluten Pfads
    absolute_path = os.path.abspath(dir_loader.directory)
    print("Absoluter Pfad:", absolute_path)
    documents = dir_loader.load()

    #print("Loaded files in total: ", len(documents))
    #print("File paths:")
    for document in documents:
        print(document.metadata["source"] + ":")
        if document.metadata["source"].endswith(".py"):
            #print(module_responses[document.metadata["source"]])
            #print("---------------------------------------------------------")
            document.page_content = module_responses[document.metadata["source"]] # Ersetze Python Code mit den textuellen Beschreibungen vom Code
    
    print(documents)

    # Make the embedding
    emb_agent = EmbeddingAgent(documents)
    emb_agent.make_embeddings()

    print("Embeddings:")
    print(emb_agent.document_embeddings)
    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(emb_agent.document_embeddings, f, ensure_ascii=False, indent=4)

def main_db():

    """ Create DB connection and insert the document embeddings """
    connection_string = "mongodb+srv://tkubera:UBWiWVbOrWHgxcAL@cluster0.jteqk2p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    vector_db = MongoDBAtlasVectorDB(connection_string)
    vector_db.connect()

    embeddings = json.loads(open("embeddings.json").read())

    # Daten in die Collection einfügen
    try:
        for key in embeddings.keys():
            print("Inserting document: ", key)
            vector_db.insert_document(embeddings[key])
        print("Embeddings erfolgreich in die Datenbank eingefügt.")
    except Exception as e:
        print(f"Fehler beim Einfügen der Embeddings: {e}")

    
def main_chat():
    connection_string = "mongodb+srv://tkubera:UBWiWVbOrWHgxcAL@cluster0.jteqk2p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    vector_db = MongoDBAtlasVectorDB(connection_string)
    vector_db.connect()
    # Embedding vom User Query berechnen.
    i = 0
    while True:
        user_input = input("Geben Sie einen Text ein (oder 'x' zum Beenden): ")
        if user_input.lower() == "x":
            break

        # Einzelnes Dokument für die Benutzereingabe erstellen
        user_document = {
            "document_id": "user_input_" + str(i),
            "document_name": "user_input_" + str(i),
            "text": user_input
        }
        emb_agent = EmbeddingAgent([user_document], mode="user_query")
        emb_agent.make_embeddings()

        # Hier können Sie das Embedding mit den anderen Embeddings vergleichen
        # Vergleichslogik hier einfügen
        print("Embedding vector for user input: ", emb_agent.document_embeddings["user_input_"+str(i)])
        similar_documents = vector_db.find_similar_vectors(emb_agent.document_embeddings["user_input_"+str(i)]["embeddings"], k=5)

        for j, document in enumerate(similar_documents):
            filename  = f"./.tmp/vectors/document_{i}.{j}.json"
            with open(filename, "w") as file:
                json.dump(document, file, default=json_serializer)

        i += 1

def authors_chat():
    connection_string = "mongodb+srv://tkubera:UBWiWVbOrWHgxcAL@cluster0.jteqk2p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    vector_db = MongoDBAtlasVectorDB(connection_string)
    vector_db.connect()
    user_document = {
        "document_id": "user_input_0",
        "document_name": "user_input_0",
        "text": authors_query
    }
    emb_agent = EmbeddingAgent([user_document], mode="user_query")
    emb_agent.make_embeddings()

    # Hier können Sie das Embedding mit den anderen Embeddings vergleichen
    # Vergleichslogik hier einfügen
    print("Embedding vector for user input: ", emb_agent.document_embeddings["user_input_0"])
    similar_documents = vector_db.find_similar_vectors(emb_agent.document_embeddings["user_input_0"]["embeddings"], k=5)

    for j, document in enumerate(similar_documents):
        filename  = f"./.tmp/vectors/document_authors.{j}.json"
        with open(filename, "w") as file:
            json.dump(document, file, default=json_serializer)

def embedding_all_incl():
    module_responses = json.loads(open("module_responses.json").read())

    # Allow prinint utf-8 characters in console
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")

    config = load_config()
    print("Model name: ")
    print(config.LLM_MODEL_NAME)
    cache = SimpleCache(tmp_path="./.tmp")

    dir_loader = DirectoryLoader(directory="./resources/cookiecutter")
    # Ermitteln des absoluten Pfads
    absolute_path = os.path.abspath(dir_loader.directory)
    print("Absoluter Pfad:", absolute_path)
    documents = dir_loader.load()

    #print("Loaded files in total: ", len(documents))
    #print("File paths:")
    for document in documents:
        print(document.metadata["source"] + ":")
        if document.metadata["source"].endswith(".py"):
            #print(module_responses[document.metadata["source"]])
            #print("---------------------------------------------------------")
            document.page_content = module_responses[document.metadata["source"]] # Ersetze Python Code mit den textuellen Beschreibungen vom Code
    print(documents)

    #TODO User Input o.ä.




    # Make the embedding
    emb_agent = EmbeddingAgent(documents)
    emb_agent.make_embeddings()

    print("Embeddings:")
    print(emb_agent.document_embeddings)
    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(emb_agent.document_embeddings, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    authors_chat()