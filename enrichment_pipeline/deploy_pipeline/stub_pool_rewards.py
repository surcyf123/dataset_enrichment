import queue
import threading

REWARD_ENDPOINTS = [
    "http://70.52.53.190:50305",
    "http://70.52.53.190:50336",
    "http://70.52.53.190:50365",
    "http://70.52.53.190:50334",
    "http://47.189.79.46:50159",
    "http://47.189.79.46:50108",
    "http://47.189.79.46:50193",
    "http://47.189.79.46:50060",
    "http://93.206.137.205:48183",
    "http://93.206.137.205:48076",
    "http://93.206.137.205:48182",
    "http://93.206.137.205:48038"
]

def create_queue_from_list(url_list):
    q = queue.Queue()
    for url in url_list:
        q.put(url)
    return q


reward_queue = create_queue_from_list(REWARD_ENDPOINTS)
   
        
try:
    scoring_url = reward_queue.get(block=True, timeout=10)  # wait 10 seconds
    
except queue.Empty:
    print("No scoring server available after waiting for 10 seconds.")

            
ttl = 9.4
            
try:
    openai_thread = threading.Thread(target=generate_and_score_openai)
    openai_thread.start()
    openai_thread.join(ttl)
            # Rest of the code remains as is...

finally:
    # This will ensure the scoring server URL is put back to the queue even if an exception occurs
    reward_queue.put(scoring_url)
    print("Finished generate_and_score function.")