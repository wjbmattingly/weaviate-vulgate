# Vulgate Weaviate Loader

This project loads the Clementine Vulgate (Latin Bible) into a Weaviate vector database, using sentence embeddings for semantic search and retrieval.

## Features
- Reads the Clementine Vulgate from a CSV file (`data/clem_vulgate.csv`).
- Generates sentence embeddings using the LaBSE model from `sentence-transformers`.
- Stores the text, book, chapter, and verse, along with the embedding, in a Weaviate collection.
- Supports batch insertion for efficient data loading.

## Dataset

The Vulgate version used is the Clementine version available from [The Clementine Text Project](https://vulsearch.sourceforge.net/). This was cleaned and structured into a CSV file. I would like to thank Marjorie Burghart for drawing this dataset to my attention. This dataset is licensed by the project as GPLv2.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/wjbmattingly/weaviate-vulgate.git
   cd weaviate-vulgate
   ```

2. **Create a free Weaviate sandbox instance:**
   - Visit [Weaviate Cloud Console](https://console.weaviate.cloud/).
   - Sign up or log in, then click "Create Cluster" and select the free sandbox option.
   - Once your cluster is ready, copy the endpoint URL and API key for use in the next step.

3. **Prepare environment variables:**
   Create a `.env` file in the project root with the following content:
   ```env
   WEAVIATE_URL=your-weaviate-endpoint
   WEAVIATE_API_KEY=your-api-key
   ```

4. **Prepare data:**
   Ensure `data/clem_vulgate.csv` exists with columns: `latin`, `book`, `chapter`, `verse`.

## Usage

Run the main script to process the data and upload it to Weaviate:

```bash
python main.py
```

This will:
- Generate embeddings for each verse.
- Save the embeddings to `data/clem_vulgate_vectors.parquet`.
- Create (or recreate) the `Vulgate` collection in Weaviate.
- Batch insert all verses with their embeddings.


## Streamlit App

The `streamlit_app.py` file provides a web interface for semantic search in the Latin Vulgate using your Weaviate instance.

### Features
- Enter a search query in Latin or any language supported by LaBSE.
- Select specific books of the Bible to search within.
- Adjust similarity threshold and number of results.
- View the most similar verses, their references, and similarity scores.

### How to Run

1. Make sure you have set up your Weaviate instance and loaded the data as described above.
2. Add your Weaviate credentials to a `.streamlit/secrets.toml` file:
   ```toml
   WEAVIATE_URL = "your-weaviate-endpoint"
   WEAVIATE_API_KEY = "your-api-key"
   COLLECTION_NAME = "Vulgate"
   ```
3. Install Streamlit if you haven't already:
   ```bash
   pip install streamlit
   ```
4. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```
5. Open the provided local URL in your browser to use the app.
