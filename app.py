import gradio as gr
import weaviate
from weaviate.auth import Auth
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import MetadataQuery
from weaviate.collections.classes.filters import Filter
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import pandas as pd
import re
from functools import lru_cache

# Load environment variables
load_dotenv()

# Validate environment variables
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if not all([WEAVIATE_URL, WEAVIATE_API_KEY, COLLECTION_NAME]):
    raise ValueError(
        "Missing required environment variables. Please ensure the following are set:\n"
        "WEAVIATE_URL\n"
        "WEAVIATE_API_KEY\n"
        "COLLECTION_NAME"
    )

# Initialize the model
model = SentenceTransformer('sentence-transformers/LaBSE')

# Book mappings
VULGATE_BOOKS = {
    "Genesis": "Gn", "Exodus": "Ex", "Leviticus": "Lv", "Numbers": "Nm", 
    "Deuteronomy": "Dt", "Joshua": "Jos", "Judges": "Jdc", "Ruth": "Rt", 
    "1 Samuel": "1Rg", "2 Samuel": "2Rg", "1 Kings": "3Rg", "2 Kings": "4Rg", 
    "1 Chronicles": "1Par", "2 Chronicles": "2Par", "Ezra": "Esr", 
    "Nehemiah": "Neh", "Tobit": "Tob", "Judith": "Jdt", "Esther": "Est", 
    "1 Maccabees": "1Mcc", "2 Maccabees": "2Mcc", "Job": "Job", "Psalms": "Ps", 
    "Proverbs": "Pr", "Ecclesiastes": "Ecl", "Song of Solomon": "Ct", 
    "Wisdom": "Sap", "Sirach": "Sir", "Isaiah": "Is", "Jeremiah": "Jr", 
    "Lamentations": "Lam", "Baruch": "Bar", "Ezekiel": "Ez", "Daniel": "Dn", 
    "Hosea": "Os", "Joel": "Joel", "Amos": "Am", "Obadiah": "Abd", 
    "Jonah": "Jon", "Micah": "Mch", "Nahum": "Nah", "Habakkuk": "Hab", 
    "Zephaniah": "Soph", "Haggai": "Agg", "Zechariah": "Zach", 
    "Malachi": "Mal", "Matthew": "Mt", "Mark": "Mc", "Luke": "Lc", 
    "John": "Jo", "Acts": "Act", "Romans": "Rom", "1 Corinthians": "1Cor", 
    "2 Corinthians": "2Cor", "Galatians": "Gal", "Ephesians": "Eph", 
    "Philippians": "Phlp", "Colossians": "Col", "1 Thessalonians": "1Thes", 
    "2 Thessalonians": "2Thes", "1 Timothy": "1Tim", "2 Timothy": "2Tim", 
    "Titus": "Tit", "Philemon": "Phlm", "Hebrews": "Hbr", "James": "Jac", 
    "1 Peter": "1Ptr", "2 Peter": "2Ptr", "1 John": "1Jo", "2 John": "2Jo", 
    "3 John": "3Jo", "Jude": "Jud", "Revelation": "Apc"
}

@lru_cache(maxsize=1)
def load_vulgate_csv():
    df = pd.read_csv("data/clem_vulgate.csv")
    # Expect columns: book, chapter, verse, text
    return df

def highlight_matching_words(text: str, query: str) -> str:
    if not query.strip():
        return text
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    if not query_words:
        return text
    partial_pattern = re.compile(r'(' + '|'.join(re.escape(w) for w in query_words) + r')', re.IGNORECASE)
    tokens = re.findall(r'\w+|\W+', text)
    highlighted = []
    for token in tokens:
        token_lc = token.lower()
        if token_lc in query_words:
            highlighted.append(f'<span style="background:yellow">{token}</span>')
        elif token.strip() and token.isalpha() and any(w in token_lc and w != token_lc for w in query_words):
            def green_sub(m):
                return f'<span style="background:lightgreen">{m.group(0)}</span>'
            highlighted.append(partial_pattern.sub(green_sub, token))
        else:
            highlighted.append(token)
    return ''.join(highlighted)

def find_similar(query: str, books: List[str], limit: int = 50) -> List[Dict[str, Any]]:
    try:
        query_vector = model.encode([query])[0]
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        )
        try:
            vulgate = client.collections.get(COLLECTION_NAME)
            filter_condition = None
            if books:
                selected_books = [VULGATE_BOOKS[book] for book in books]
                filter_condition = Filter.by_property("book").contains_any(selected_books)
            response = vulgate.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
                filters=filter_condition
            )
            results = []
            for obj in response.objects:
                highlighted_text = highlight_matching_words(obj.properties["text"], query)
                results.append({
                    "Reference": f"{obj.properties['book']} {obj.properties['chapter']}:{obj.properties['verse']}",
                    "Book": obj.properties["book"],
                    "Chapter": obj.properties["chapter"],
                    "Verse": obj.properties["verse"],
                    "Text": highlighted_text,
                    "RawText": obj.properties["text"],
                    "Similarity": round(1 - obj.metadata.distance, 3)
                })
            return results
        finally:
            client.close()
    except Exception as e:
        return [{"Error": str(e)}]

def format_results_html(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "<div>No results found.</div>"
    if "Error" in results[0]:
        return f'<div style="color:red">Error: {results[0]["Error"]}</div>'
    html = [
        '<style>td,th{padding:8px;}th{background:#f4f1e9;}tr:nth-child(even){background:#f9f9f9;}tr:hover{background:#e6e2d3;}table{border-radius:8px;overflow:hidden;box-shadow:0 2px 8px #e6e2d3;}td{vertical-align:top;}</style>',
        '<table style="border-collapse:collapse;width:100%;font-size:1em;">',
        '<thead><tr>'
        '<th>Reference</th><th>Text</th><th>Similarity</th><th>Book</th><th>Chapter</th><th>Verse</th>'
        '</tr></thead><tbody>'
    ]
    for r in results:
        html.append(f'<tr>'
            f'<td>{r["Reference"]}</td>'
            f'<td>{r["Text"]}</td>'
            f'<td>{r["Similarity"]}</td>'
            f'<td>{r["Book"]}</td>'
            f'<td>{r["Chapter"]}</td>'
            f'<td>{r["Verse"]}</td>'
            f'</tr>')
    html.append('</tbody></table>')
    return ''.join(html)

def search(query: str, books: List[str], limit: int) -> str:
    if not query.strip():
        return "<div>Please enter a search query.</div>"
    results = find_similar(query, books, limit)
    return format_results_html(results)

with gr.Blocks(title="Latin Vulgate Verse Similarity Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Latin Vulgate Verse Similarity Search
    
    Search for similar verses in the Latin Vulgate Bible using semantic similarity.
    <br>Words matching your query will be <span style='background:yellow'>highlighted yellow</span> (exact) or <span style='background:lightgreen'>green</span> (partial).
    """)
    with gr.Row():
        query = gr.Textbox(
            label="Search Query",
            placeholder="Enter your search query...",
            lines=2,
            scale=3
        )
    with gr.Row():
        with gr.Column(scale=2):
            book_select = gr.Dropdown(
                choices=list(VULGATE_BOOKS.keys()),
                label="Select Books (Optional)",
                multiselect=True
            )
    with gr.Row():
        limit = gr.Slider(
            minimum=1,
            maximum=50,
            value=20,
            step=1,
            label="Number of results"
        )
    with gr.Row():
        search_btn = gr.Button("Search", variant="primary")
    output = gr.HTML(label="Results")


    search_btn.click(
        fn=search,
        inputs=[query, book_select, limit],
        outputs=output
    )
    query.submit(
        fn=search,
        inputs=[query, book_select, limit],
        outputs=output
    )
if __name__ == "__main__":
    demo.launch()
