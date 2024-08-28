import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GPT import GPT
model = GPT()

ans1 = model.extract_esg_sentence("Who is the Prime Minister of Malaysia", temperature=0.7, verbose=False)
ans2 = model.extract_esg_sentence("How old is he?", temperature=0.7, verbose=False)

print(ans1)
print(ans2)