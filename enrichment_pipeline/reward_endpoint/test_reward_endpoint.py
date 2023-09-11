import requests
import random
import time
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
VERIFY_TOKEN = "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n"

URLS = {
    "http://172.218.204.83:2701": "vast1",
    "http://142.182.6.112:55469": "vast2",
    "http://184.67.78.114:42036": "vast3",
    "http://37.27.2.44:60113": "vast4",
    "http://90.84.239.86:40264": "vast5",
}

PROMPT = """
Sure! Here is a summary of the text in seven sentences:\n\nYoichiro Nambu, a Japanese-American physicist, passed away on July 5th at the age of 94. He shared the 2008 Nobel Prize in Physics for his work on spontaneous symmetry breaking. Nambu's research inspired others, including Peter Higgs and François Englert, to develop the theoretical mechanism for the Higgs boson. Higgs and Englert went on to share the 2013 Nobel Prize for their work. Nambu's work was described as ""an act of imagination"" that was ahead of its time by Frank Wilczek. Nambu was born in Tokyo and studied at the Imperial University before moving to the United States. He spent the majority of his career at the University of Chicago and became an American citizen in 1970.\nPrevious Question \nQuestion:Sure! Here's my question:\nGiven that Nambu's work on spontaneous symmetry breaking was so influential and groundbreaking, why do you think it took so long (nearly 50 years) for him to receive recognition in the form of the Nobel Prize?\nAnswer:This question is asking for a step-by-step answer. I will break down the steps as follows:\n\n1. Understand the speaker's message: First, I will read the passage and fully comprehend the speaker's message.\n2. Identify the main points: Once I have understood the message, I will identify the main points that the speaker is trying to convey.\n3. Analyze the information: I will analyze the information and consider how it relates to the question I am being asked.\n4. Provide a step-by-step answer: Based on the analysis, I will provide a step-by-step answer that follows the question asked.\n5. Explain my thoughts: Finally, I will explain my thoughts on the question and why I made the choices that I did.\n\nAsk one relevant and insightful question about the preceding context and previous questions\n
"""

CONVERSATIONAL_WORDS = [
    "hello", "goodbye", "please", "thanks", "sorry", "yes", "no", "maybe", "okay",
    "hi", "bye", "welcome", "morning", "evening", "night", "sure", "absolutely", "definitely",
    "probably", "hope", "wish", "agree", "disagree", "understand", "remember", "forget",
    "think", "feel", "believe", "wonder", "guess", "chat", "talk", "speak", "listen",
    "hear", "tell", "ask", "answer", "question", "reply", "respond", "suggest", "recommend",
    "advise", "mention", "remark", "note",
]

def pick_random_words(word_list, num_words=40):
    """Return a list of randomly picked words with replacement."""
    return [random.choice(word_list) for _ in range(num_words)]

# Send requests and measure time
def test_url(url, note, num_requests=50, num_words=40):
    timings = []
    failed = False
    example_output = None

    for i in range(num_requests):
        random_combination = pick_random_words(CONVERSATIONAL_WORDS, num_words)
        data = {
            "verify_token": "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n",
            "prompt": PROMPT,
            "completions": [random_combination]
        }

        try:
            start_time = time.time()
            response = requests.post(url, json=data)
            response.raise_for_status()  # Will raise an exception for HTTP error codes
            end_time = time.time()

            timings.append(end_time - start_time)

            # Save the first response as an example output
            if example_output is None:
                example_output = response.json()

        except requests.RequestException as e:
            print(f"Request to {url} ({note}) failed: {e}")
            failed = True
            break  # Exit the loop early if there's a failure

    return timings, failed, example_output

def collect_statistics(url, note, timings):
    total_time = sum(timings)
    avg_time = total_time / len(timings)
    min_time = min(timings)
    max_time = max(timings)

    return {
        "URL": url,
        "Note": note,
        "Total time": total_time,
        "Average time per request": avg_time,
        "Min time": min_time,
        "Max time": max_time
    }

def main():
    failed_urls = []
    all_statistics = []
    example_outputs = []

    # Loop through URLs and notes from the dictionary
    for url, note in URLS.items():
        print(f"Testing for URL: {url} ({note})")
        timings, failed, example_output = test_url(url, note)
        if not failed:
            stats = collect_statistics(url, note, timings)
            all_statistics.append(stats)
            example_outputs.append((url, note, example_output))
        else:
            # Store both the URL and its note when a URL fails
            failed_urls.append((url, note))

    print("\nStatistics for all URLs:")
    for stats in all_statistics:
        print(f"URL: {stats['URL']} ({stats['Note']})")
        print(f"  Total time: {stats['Total time']:.4f} seconds")
        print(f"  Average time per request: {stats['Average time per request']:.4f} seconds")
        print(f"  Min time: {stats['Min time']:.4f} seconds")
        print(f"  Max time: {stats['Max time']:.4f} seconds\n")

    print("\nExample Outputs:")
    for url, note, output in example_outputs:
        print(f"Example output for URL: {url} ({note}):")
        print(output)
        print("-" * 80)

    if failed_urls:
        print("\nThe following URLs failed:")
        for url, note in failed_urls:
            print(f"{url} ({note})")

if __name__ == "__main__":
    main()

# data = {
#     "verify_token": "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n",  # Your authentication token
#     "prompt": prompt,
#     "completions": [
#     '''
# Groundbreaking Work: The initial work by Nambu on spontaneous symmetry breaking was indeed groundbreaking. However, groundbreaking work is often ahead of its time, and it can take years, sometimes decades, for the full impact of such work to be understood and appreciated by the wider scientific community.
# Experimental Confirmation: Theoretical physics is full of brilliant ideas, but not all of them are correct. It often takes considerable time to gather the experimental evidence needed to confirm or refute theoretical predictions. In Nambu's case, his theories were eventually confirmed by the discovery of the Higgs boson in 2012, which required the construction of the Large Hadron Collider, a project of unprecedented scale in experimental physics.
# Nobel Selection Process: The process for awarding the Nobel Prize is notoriously selective and conservative. It often takes many years, sometimes decades, for a discovery to be deemed "Nobel worthy". This is in part because the committee tends to award those whose contributions have stood the test of time and have been thoroughly vetted by the scientific community.
# Given the above, it's not unusual that it took nearly 50 years for Nambu's work to be recognized with a Nobel Prize.
# As for a related and insightful question: Can you elaborate on the specific contributions Nambu made in the field of spontaneous symmetry breaking, and how these contributions influenced the work of other physicists like Higgs and Englert?
#  ''',
#  'This is a bad answer that does not relate to the prompt so it will receive lower scores for both reward models and a 0 for the relevance filter, making the total score 0.'
#     ]
# }
