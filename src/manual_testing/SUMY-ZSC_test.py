from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
import json

with open("../../data/appeals.json", "r", encoding="utf-8") as f:
    data = json.load(f)

text = data["appeals"][0]["text"]

parser = PlaintextParser.from_string(text, Tokenizer("russian"))

summarizer = TextRankSummarizer()
summary = summarizer(parser.document, 2)

for sentence in summary:
    print(sentence)