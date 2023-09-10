from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from collections import defaultdict
import json

def get_gptq_models():
    # Set up Chrome options
    chrome_options = Options()
    
    # Initialize the browser
    browser = webdriver.Chrome(options=chrome_options)
    
    # URL of the HuggingFace profile
    url = "https://huggingface.co/TheBloke"
    browser.get(url)
    
    # Wait for the "Expand models" button to be clickable and click it
    expand_button = WebDriverWait(browser, 30).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Expand')]")))
    expand_button.click()
    
    # Wait for a moment to ensure all models are loaded
    WebDriverWait(browser, 30).until(EC.presence_of_element_located((By.TAG_NAME, 'a')))
    
    # Find all links
    model_elements = browser.find_elements(By.TAG_NAME, 'a')
    
    # Filter model names containing "GPTQ" or "gptq" and clean up the names
    gptq_models = [elem.text.split('\n')[0] for elem in model_elements if "GPTQ" in elem.text or "gptq" in elem.text]
    
    # Close the browser
    browser.quit()
    
    # Extract model parameter size
    def extract_size(model_name):
        match = re.search(r'(\d+)[Bb]', model_name)
        return int(match.group(1)) if match else "unknown"
    
    # Group models by their parameter size
    models_by_size = defaultdict(list)
    for model in gptq_models:
        size = extract_size(model)
        models_by_size[size].append(model)
    
    # Sort the dictionary by parameter size from largest to smallest, then "unknown"
    sorted_models_by_size = dict(sorted(models_by_size.items(), key=lambda x: (x[0] == "unknown", -x[0] if isinstance(x[0], int) else 0)))
    
    return sorted_models_by_size

# Get the models and write to a JSON file
models = get_gptq_models()
with open("thebloke_gptqs.json", "w") as outfile:
    json.dump(models, outfile, indent=4)

print("Data written to thebloke_gptqs.json")
