import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
sentences = [
    "Andy loved to sleep on a bed of nails.",
"She was amazed by the large chunks of ice washing up on the beach.",
"Having no hair made him look even hairier.",
"Joyce enjoyed eating pancakes with ketchup.",
"The best key lime pie is still up for debate.",
"The green tea and avocado smoothie turned out exactly as would be expected.",
"He had decided to accept his fate of accepting his fate.",
"He is good at eating pickles and telling women about his emotional problems.",
"The tears of a clown make my lipstick run, but my shower cap is still intact.",
"She insisted that cleaning out your closet was the key to good driving.",
"When nobody is around, the trees gossip about the people who have walked under them.",
"The toddler’s endless tantrum caused the entire plane anxiety.",
"Greetings from the galaxy MACS0647-JD, or what we call home.",
"My Mum tries to be cool by saying that she likes all the same things that I do.",
"8 of 25 is the same as 25 of 8 and one of them is much easier to do in your head.",
"The memory we used to share is no longer coherent.",
"Toddlers feeding raccoons surprised even the seasoned park ranger.",
"When she didn’t like a guy who was trying to pick her up, she started using sign language.",
"The book is in front of the table.",
"People who insist on picking their teeth with their elbows are so annoying!",
]

print(utils.calculate_pairwise_cosine_similarity_str(sentences))