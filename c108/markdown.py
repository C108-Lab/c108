#
# C108 Markdown Tools
#

# Standard library -----------------------------------------------------------------------------------------------------
import re


# Methods --------------------------------------------------------------------------------------------------------------

def get_markdown_h1_heading(file_name: str) -> str:
    """Find the first line with # Heading in Markdown file."""
    with open(file_name, 'r') as f:
        for line in f:
            if line.startswith("# "):
                return line.strip("#").strip()


def get_markdown_h1_text(file_name: str) -> str:
    """Find the text between first two # Headings in Markdown file.
    Return all text after the first heading if second heading not found.
    Heading is a line starting with # or ## or ###"""
    with open(file_name, 'r') as f:
        text = f.read()

    # Look for headings with # or ## (supports H1 and H2)
    pattern = r"(?:^|\n)(##?\s+[^#\n]*\n)(.*?)(?=\n##?\s+|$)"
    match = re.search(pattern, text, flags=re.DOTALL)

    if match:
        # Return text between headings or everything after the first heading
        return match.group(2).strip()
    else:
        # No headings found, return empty string
        return ""
