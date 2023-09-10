from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from collections import defaultdict
import json
import time  # Importing the required module

def get_gptq_models_from_huggingface():
    # Set up Chrome options
    chrome_options = Options()
    
    # Initialize the browser
    browser = webdriver.Chrome(options=chrome_options)
    
    # Start with the search URL
    url = "https://huggingface.co/models?search=gptq"
    browser.get(url)
    
    all_gptq_models = []

    while True:
        # Wait for the models to load on the current page
        WebDriverWait(browser, 60).until(EC.presence_of_element_located((By.XPATH, '//article[contains(@class, "overview-card-wrapper") and contains(@class, "group/repo")]//a')))
        
        # Extract model names from the current page
        model_elements = browser.find_elements(By.XPATH, '//article[contains(@class, "overview-card-wrapper") and contains(@class, "group/repo")]//a')
        current_page_models = []
        for elem in model_elements:
            try:
                if "GPTQ" in elem.text or "gptq" in elem.text:
                    current_page_models.append(elem.text.split('\n')[0])
            except:
                continue
        all_gptq_models.extend(current_page_models)

        # Check if there's a "Next" button to paginate
        try:
            next_button = WebDriverWait(browser, 10).until(EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Next')]")))
            next_button.click()
        except Exception as e:
            break

    # Close the browser
    browser.quit()
    
    # Extract model parameter size
    def extract_size(model_name):
        match = re.search(r'(\d+)[Bb]', model_name)
        return int(match.group(1)) if match else "unknown"
    
    # Group models by their parameter size
    models_by_size = defaultdict(list)
    for model in all_gptq_models:
        size = extract_size(model)
        models_by_size[size].append(model)
    
    # Sort the dictionary by parameter size from largest to smallest, then "unknown"
    sorted_models_by_size = dict(sorted(models_by_size.items(), key=lambda x: (x[0] == "unknown", -x[0] if isinstance(x[0], int) else 0)))
    
    return sorted_models_by_size

# Get the models and write to a JSON file
models = get_gptq_models_from_huggingface()
with open("all_gptqs_huggingface.json", "w") as outfile:
    json.dump(models, outfile, indent=4)

print("Data written to all_gptqs_huggingface.json")
