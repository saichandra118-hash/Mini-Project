"""
Text Preprocessing Module for Fake News Detection System.

Provides shared text normalization, leak prevention, and input validation routines
used consistently across both training (train_model.py) and inference (app.py).
"""

import html
import re
import string


def clean_text(text: str) -> str:
    """
    Cleans and normalizes news text:
    1. Unescapes HTML entities (e.g. &amp; -> &).
    2. Converts text to lowercase.
    3. Strips news agency prefixes (e.g. 'WASHINGTON (Reuters) - ')
       to prevent trivial data leakage from the Kaggle dataset.
    4. Removes HTML tags.
    5. Removes URLs and web links.
    6. Removes editorial bracketed notes (e.g. [Photo/File]).
    7. Retains valuable informative signals including dates, numbers,
       and statistics, while removing distracting symbols/punctuation.
    8. Normalizes consecutive whitespace.

    Parameters:
        text (str): Raw headline or article body.

    Returns:
        str: Cleaned, normalized text ready for vectorization.
    """
    if not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities
    text = html.unescape(text)

    # 2. Convert to lowercase
    text = text.lower()

    # 3. Strip news-wire prefixes to prevent data leakage
    # Kaggle's True.csv overwhelmingly begins with "CITY (Reuters) - "
    text = re.sub(r"^.*?\(reuters\)\s*[-–—]\s*", " ", text)
    text = re.sub(r"^reuters\s*[-–—]\s*", " ", text)
    text = re.sub(r"\(reuters\)", " ", text)
    text = re.sub(r"^.*?\((?:ap|afp)\)\s*[-–—]\s*", " ", text)

    # 4. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 5. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 6. Remove bracketed editorial annotations
    text = re.sub(r"\[.*?\]", " ", text)

    # 7. Clean unwanted symbols and punctuation, but PRESERVE numbers, dates, and statistics
    # Replace punctuation and symbols with a space, keeping alphanumeric characters (letters and digits)
    text = re.sub(r"[^\w\s]|_", " ", text)

    # 8. Normalize consecutive whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_news_input(text: str, min_words: int = 3, min_chars: int = 15):
    """
    Validates whether the user input contains enough content for analysis.

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    if not text or not isinstance(text, str) or not text.strip():
        return False, "Please enter a news headline or article. Input cannot be empty."

    cleaned = clean_text(text)
    words = cleaned.split()

    if len(text.strip()) < min_chars:
        return (
            False,
            f"Input is too short ({len(text.strip())} characters). Please enter at least {min_chars} characters for a reliable prediction.",
        )

    if len(words) < min_words:
        return (
            False,
            f"Input contains too few words ({len(words)}). Please enter at least {min_words} words.",
        )

    return True, ""
