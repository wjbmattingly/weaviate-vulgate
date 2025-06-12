import argparse
import os
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.auth import Auth
from weaviate.classes.query import MetadataQuery
from weaviate.collections.classes.filters import Filter
from dotenv import load_dotenv

# Book abbreviation mapping (from streamlit_app.py)
vulgate_books = {"Genesis": "Gn", "Exodus": "Ex", "Leviticus": "Lv", "Numbers": "Nm", "Deuteronomy": "Dt", "Joshua": "Jos", "Judges": "Jdc", "Ruth": "Rt", "1 Samuel": "1Rg", "2 Samuel": "2Rg", "1 Kings": "3Rg", "2 Kings": "4Rg", "1 Chronicles": "1Par", "2 Chronicles": "2Par", "Ezra": "Esr", "Nehemiah": "Neh", "Tobit": "Tob", "Judith": "Jdt", "Esther": "Est", "1 Maccabees": "1Mcc", "2 Maccabees": "2Mcc", "Job": "Job", "Psalms": "Ps", "Proverbs": "Pr", "Ecclesiastes": "Ecl", "Song of Solomon": "Ct", "Wisdom": "Sap", "Sirach": "Sir", "Isaiah": "Is", "Jeremiah": "Jr", "Lamentations": "Lam", "Baruch": "Bar", "Ezekiel": "Ez", "Daniel": "Dn", "Hosea": "Os", "Joel": "Joel", "Amos": "Am", "Obadiah": "Abd", "Jonah": "Jon", "Micah": "Mch", "Nahum": "Nah", "Habakkuk": "Hab", "Zephaniah": "Soph", "Haggai": "Agg", "Zechariah": "Zach", "Malachi": "Mal", "Matthew": "Mt", "Mark": "Mc", "Luke": "Lc", "John": "Jo", "Acts": "Act", "Romans": "Rom", "1 Corinthians": "1Cor", "2 Corinthians": "2Cor", "Galatians": "Gal", "Ephesians": "Eph", "Philippians": "Phlp", "Colossians": "Col", "1 Thessalonians": "1Thes", "2 Thessalonians": "2Thes", "1 Timothy": "1Tim", "2 Timothy": "2Tim", "Titus": "Tit", "Philemon": "Phlm", "Hebrews": "Hbr", "James": "Jac", "1 Peter": "1Ptr", "2 Peter": "2Ptr", "1 John": "1Jo", "2 John": "2Jo", "3 John": "3Jo", "Jude": "Jud", "Revelation": "Apc"}

# Reverse mapping for abbreviation lookup
abbr_to_book = {abbr: name for name, abbr in vulgate_books.items()}


def main():
    parser = argparse.ArgumentParser(description="Query the Vulgate Weaviate DB by semantic similarity.")
    parser.add_argument("query", type=str, help="Query text (required)")
    parser.add_argument("--book", type=str, help="Book abbreviation (e.g., 'Gn' for Genesis) or full name (e.g., 'Genesis')", default=None)
    parser.add_argument("--threshold", type=float, help="Similarity threshold (default: 0.4)", default=0.4)
    parser.add_argument("--limit", type=int, help="Number of results to return (default: 5)", default=5)
    args = parser.parse_args()

    load_dotenv()
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Vulgate")

    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        print("Error: WEAVIATE_URL and WEAVIATE_API_KEY must be set in your .env file.")
        exit(1)

    # Normalize book argument
    book_abbr = None
    if args.book:
        if args.book in abbr_to_book:
            book_abbr = args.book
        elif args.book in vulgate_books:
            book_abbr = vulgate_books[args.book]
        else:
            print(f"Unknown book: {args.book}. Use abbreviation (e.g., 'Gn') or full name (e.g., 'Genesis').")
            exit(1)

    model = SentenceTransformer('sentence-transformers/LaBSE')
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )
    vulgate = client.collections.get(COLLECTION_NAME)

    query_vector = model.encode([args.query])[0]
    if book_abbr:
        response = vulgate.query.near_vector(
            near_vector=query_vector,
            limit=args.limit,
            return_metadata=MetadataQuery(distance=True),
            filters=Filter.by_property("book").equal(book_abbr)
        )
    else:
        response = vulgate.query.near_vector(
            near_vector=query_vector,
            limit=args.limit,
            return_metadata=MetadataQuery(distance=True)
        )

    found = False
    for o in response.objects:
        if o.metadata.distance < args.threshold:
            found = True
            print(f"{o.properties['book']} {o.properties['chapter']}:{o.properties['verse']} | {o.properties['text']}")
            print(f"  Similarity: {1 - o.metadata.distance:.2f}\n")
    if not found:
        print("No results found. Try adjusting the similarity threshold or search query.")
    client.close()

if __name__ == "__main__":
    main()
