from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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
    
    return gptq_models

# Test the function
print(get_gptq_models())
