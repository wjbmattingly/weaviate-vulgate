import pandas as pd
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc
from weaviate.collections.classes.filters import Filter
from tqdm import tqdm
import os
from dotenv import load_dotenv


load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME = "Vulgate"


df = pd.read_csv("data/clem_vulgate.csv")
model = SentenceTransformer('sentence-transformers/LaBSE')
embeddings = model.encode(df.latin.tolist())
df["embedding"] = list(embeddings)
df.to_parquet("data/clem_vulgate_vectors.parquet")



client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
)

if client.collections.exists(COLLECTION_NAME):  # In case we've created this collection before
    client.collections.delete(COLLECTION_NAME)  # THIS WILL DELETE ALL DATA IN THE COLLECTION

vulgate = client.collections.create(
    name=COLLECTION_NAME,
    properties=[
        wvc.config.Property(
            name="text",
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="book",
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="chapter",
            data_type=wvc.config.DataType.INT
        ),
        wvc.config.Property(
            name="verse",
            data_type=wvc.config.DataType.INT
        ),
    ]
)

# Prepare all the data rows first
data_rows = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing data"):
    data_rows.append({
        "properties": {
            "text": row['latin'],
            "book": row['book'],
            "chapter": int(row['chapter']),
            "verse": int(row['verse'])
        },
        "vector": row['embedding']
    })


# Now perform the batch insertion
with vulgate.batch.dynamic() as batch:
    for data_row in tqdm(data_rows, desc="Inserting data"):
        batch.add_object(
            properties=data_row['properties'],
            vector=data_row['vector']
        )