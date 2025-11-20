"""
Text preprocessing utilities.
Light cleaning: keep punctuation and repeated characters, remove URLs, normalize whitespace.
"""
import re
import string


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.sub('', text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace characters."""
    # Replace multiple spaces/tabs/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()


def preprocess_text(text: str, remove_urls_flag: bool = True) -> str:
    """
    Preprocess text with light cleaning.
    
    - Keep punctuation
    - Keep repeated characters (e.g., "sooo" remains "sooo")
    - Remove URLs
    - Normalize whitespace
    
    Args:
        text: Input text string
        remove_urls_flag: Whether to remove URLs
        
    Returns:
        Preprocessed text string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs if flag is set
    if remove_urls_flag:
        text = remove_urls(text)
    
    # Normalize whitespace
    text = normalize_whitespace(text)
    
    return text


def preprocess_batch(texts, remove_urls_flag: bool = True):
    """
    Preprocess a batch of texts.
    
    Args:
        texts: List or array of text strings
        remove_urls_flag: Whether to remove URLs
        
    Returns:
        List of preprocessed texts
    """
    return [preprocess_text(text, remove_urls_flag) for text in texts]


if __name__ == "__main__":
    # Test preprocessing
    test_texts = [
        "Check this out: http://example.com/test   multiple   spaces",
        "Normal text with punctuation!!!",
        "Repeated characters: sooo good!!!",
        "Mixed\n\nwhitespace\t\there"
    ]
    
    for text in test_texts:
        processed = preprocess_text(text)
        print(f"Original: {repr(text)}")
        print(f"Processed: {repr(processed)}\n")

