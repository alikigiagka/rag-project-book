
"""LLM response generation module utilizing Google Gemini API.

This module formats retrieved text chunks into grounded prompt context and sends
requests to the Gemini model to synthesize domain-accurate answers.
"""

import os
from typing import List, Optional, Tuple
from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.config import BOOK, GEMINI_LLM_MODEL_NAME

load_dotenv()


def generate_response(
    query: str, context_texts: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Generates a grounded response using Gemini LLM from context snippets.

    Args:
        query: User input query string.
        context_texts: List of relevant text passages retrieved from the database.

    Returns:
        A tuple containing:
            - Response text string if successful, otherwise None.
            - Error message string if an exception occurred, otherwise None.
    """
    context_block = "\n\n---\n\n".join(context_texts)

    prompt = f"""ΣΗΜΑΝΤΙΚΟ: Απάντησε ΑΠΟΚΛΕΙΣΤΙΚΑ ΣΤΑ ΕΛΛΗΝΙΚΑ. Όλη η απάντηση πρέπει να είναι στα ελληνικά.

Είσαι βοηθός μελέτης για το βιβλίο «{BOOK['name']}».
Το βιβλίο καλύπτει τεχνολογίες Big Data: Hadoop, HDFS, MapReduce, Spark, Kafka, HBase, NoSQL, κ.λπ.
Τα αποσπάσματα μπορεί να περιέχουν Ελληνικό κείμενο, Αγγλικούς τεχνικούς όρους, ή περιγραφές εικόνων/διαγραμμάτων.

Κανόνες:
1. Απάντησε ΜΟΝΟ βασισμένος στα παρακάτω αποσπάσματα.
2. Αν τα αποσπάσματα δεν περιέχουν επαρκή πληροφορία, πες: «Δεν βρίσκω αυτή την πληροφορία στο διαθέσιμο υλικό.»
3. ΜΗΝ προσθέτεις γνώσεις εκτός των αποσπασμάτων.
4. Αν κάποιο απόσπασμα φέρει ετικέτα [ΕΙΚΟΝΑ], αξιοποίησε την περιγραφή της.
5. Διατήρησε τους Αγγλικούς τεχνικούς όρους (π.χ. MapReduce, DataFrame, RDD) όπου εμφανίζονται.
6. Απάντησε ΣΤΑ ΕΛΛΗΝΙΚΑ με πλήρεις, σαφείς προτάσεις.
7. ΜΗΝ απαντάς σε αγγλικά ή σε άλλη γλώσσα.

Ερώτηση: {query}

Αποσπάσματα βιβλίου:
{context_block}

Απάντησε στα ελληνικά:
"""

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "Δεν βρέθηκε GEMINI_API_KEY στο .env"

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL_NAME,
            contents=prompt,
            # Temperature is set to 0.0 to enforce deterministic output and minimize hallucinations
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return response.text.strip(), None
    except Exception as e:
        return None, str(e)