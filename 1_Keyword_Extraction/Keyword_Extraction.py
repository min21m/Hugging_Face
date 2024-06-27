import nltk
nltk.download('stopwords')
nltk.download('punkt')
from rake_nltk import Rake
rake_nltk_var = Rake()

text = """Experienced Electrical Design Engineer, who is in a process of  with a robust 
five-year background in the Aerospace industry, specializing 
in hardware design, prototyping, and testing. Eager to leverage
my technical expertise in electrical engineering while.
I am seeking a dynamic position that supports growth in
these areas alongside core electrical engineering responsibilities."""

rake_nltk_var.extract_keywords_from_text(text)
keyword_extracted = rake_nltk_var.get_ranked_phrases()
print()
print(keyword_extracted)
print()

"""
Primarily interested in leveraging AI models for helping businesses incorporating 
legacy and current business and engineering data.
Eager to apply my 5 years of technical expertise in electrical engineering to
transition into machine learning.
"""