from weaviate.classes.query import MetadataQuery
import streamlit as st
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.auth import Auth
from weaviate.collections.classes.filters import Filter
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+Pro:wght@400;600&display=swap');

/* CSS Variables for color palette */
:root {
  --primary-bg: #f4f1e9;
  --secondary-bg: #e6e2d3;
  --primary-text: #2c2c2c;
  --secondary-text: #4a4a4a;
  --accent: #8b7d6b;
  --highlight: #b7a99a;
}

/* Base styles */
body {
  font-family: 'Source Sans Pro', sans-serif;
  background-color: var(--primary-bg);
  color: var(--primary-text);
  line-height: 1.6;
  padding: 20px;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
  font-family: 'Playfair Display', serif;
  color: var(--secondary-text);
}

h1 {
  font-size: 2.5em;
  border-bottom: 2px solid var(--accent);
  padding-bottom: 10px;
}

/* Streamlit specific styles */
.stTextInput > div > div > input {
  background-color: var(--secondary-bg);
  border: 1px solid var(--accent);
  color: var(--primary-text);
}

.stButton > button {
  background-color: var(--accent);
  color: var(--primary-bg);
  border: none;
  padding: 10px 20px;
  font-family: 'Playfair Display', serif;
  font-weight: 700;
  transition: background-color 0.3s ease;
}

.stButton > button:hover {
  background-color: var(--highlight);
}

/* Multiselect styles */
.stMultiSelect > div > div > div {
  background-color: var(--secondary-bg);
}

.stMultiSelect > div > div > div:hover {
  border-color: var(--highlight);
}

.stMultiSelect > div[data-baseweb="select"] > div > div > div[role="option"] {
  background-color: var(--accent);
  color: var(--primary-bg);
}

/* Slider styles */
.stSlider > div > div > div > div {
  background-color: var(--accent);
}

/* Expander styles */
.streamlit-expanderHeader {
  background-color: var(--secondary-bg);
  border: 1px solid var(--accent);
  border-radius: 4px;
  font-family: 'Playfair Display', serif;
  color: var(--secondary-text);
}

.streamlit-expanderContent {
  background-color: var(--primary-bg);
  border: 1px solid var(--accent);
  border-top: none;
  border-radius: 0 0 4px 4px;
  padding: 10px;
}

/* Progress bar styles */
.stProgress > div > div > div > div {
  background-color: var(--accent);
}

/* Custom classes */
.bible-verse {
  font-style: italic;
  color: var(--secondary-text);
  margin: 10px 0;
  padding: 10px;
  background-color: var(--secondary-bg);
  border-left: 3px solid var(--accent);
}

.similarity-score {
  font-weight: 600;
  color: var(--accent);
}
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = SentenceTransformer('sentence-transformers/LaBSE')
    return model

def find_similar(query, model, threshold, limit=10, books=[]):
    results = []
    query_vector = model.encode([query])[0]
    if books:
        response = vulgate.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
            filters=Filter.by_property("book").contains_any(books)
        )
    else:
        response = vulgate.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )

    for o in response.objects:
        if o.metadata.distance < threshold:
            results.append({
                "book": o.properties["book"],
                "chapter": o.properties["chapter"],
                "verse": o.properties["verse"],
                "text": o.properties["text"],
                "distance": o.metadata.distance
            })
    return results

vulgate_books = {"Genesis": "Gn", "Exodus": "Ex", "Leviticus": "Lv", "Numbers": "Nm", "Deuteronomy": "Dt", "Joshua": "Jos", "Judges": "Jdc", "Ruth": "Rt", "1 Samuel": "1Rg", "2 Samuel": "2Rg", "1 Kings": "3Rg", "2 Kings": "4Rg", "1 Chronicles": "1Par", "2 Chronicles": "2Par", "Ezra": "Esr", "Nehemiah": "Neh", "Tobit": "Tob", "Judith": "Jdt", "Esther": "Est", "1 Maccabees": "1Mcc", "2 Maccabees": "2Mcc", "Job": "Job", "Psalms": "Ps", "Proverbs": "Pr", "Ecclesiastes": "Ecl", "Song of Solomon": "Ct", "Wisdom": "Sap", "Sirach": "Sir", "Isaiah": "Is", "Jeremiah": "Jr", "Lamentations": "Lam", "Baruch": "Bar", "Ezekiel": "Ez", "Daniel": "Dn", "Hosea": "Os", "Joel": "Joel", "Amos": "Am", "Obadiah": "Abd", "Jonah": "Jon", "Micah": "Mch", "Nahum": "Nah", "Habakkuk": "Hab", "Zephaniah": "Soph", "Haggai": "Agg", "Zechariah": "Zach", "Malachi": "Mal", "Matthew": "Mt", "Mark": "Mc", "Luke": "Lc", "John": "Jo", "Acts": "Act", "Romans": "Rom", "1 Corinthians": "1Cor", "2 Corinthians": "2Cor", "Galatians": "Gal", "Ephesians": "Eph", "Philippians": "Phlp", "Colossians": "Col", "1 Thessalonians": "1Thes", "2 Thessalonians": "2Thes", "1 Timothy": "1Tim", "2 Timothy": "2Tim", "Titus": "Tit", "Philemon": "Phlm", "Hebrews": "Hbr", "James": "Jac", "1 Peter": "1Ptr", "2 Peter": "2Ptr", "1 John": "1Jo", "2 John": "2Jo", "3 John": "3Jo", "Jude": "Jud", "Revelation": "Apc"}


st.title("Latin Vulgate Verse Similarity Search")

model = load_model()

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=st.secrets["WEAVIATE_URL"],
    auth_credentials=Auth.api_key(st.secrets["WEAVIATE_API_KEY"]),
)
vulgate = client.collections.get(st.secrets["COLLECTION_NAME"])

query = st.text_input("Enter your search query:")
books = st.multiselect("Select book(s)", vulgate_books.keys())
select_books = [vulgate_books[book] for book in books]
col1, col2 = st.columns(2)
threshold = col1.slider("Similarity threshold:", 0.0, 1.0, value=0.5, step=0.01)
limit = col2.slider("Number of results:", 1, 10, value=5, step=1)

if st.button("Search"):
    results = find_similar(query, model, threshold, limit, select_books)
    
    if results:
        st.subheader(f"Found {len(results)} similar verses:")
        for i, result in enumerate(results, 1):
            with st.expander(f"{i}. {result['book']} {result['chapter']}:{result['verse']} (Similarity: {1 - result['distance']:.2f})", expanded=True):
                st.markdown(f"<div class='bible-verse'>{result['text']}</div>", unsafe_allow_html=True)
                st.progress(1 - result['distance'])
    else:
        st.warning("No results found. Try adjusting the similarity threshold or search query.")