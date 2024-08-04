from weaviate.classes.query import MetadataQuery
import streamlit as st
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.auth import Auth
from weaviate.collections.classes.filters import Filter

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
                st.markdown(f"**{result['text']}**")
                st.progress(1 - result['distance'])
    else:
        st.warning("No results found. Try adjusting the similarity threshold or search query.")