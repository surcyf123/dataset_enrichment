from selenium import webdriver
from selenium.webdriver.common.by import By

def get_gptq_models():
    # Path to your webdriver (e.g., chromedriver)
    driver_path = '/path/to/your/webdriver'
    
    # Initialize the browser
    browser = webdriver.Chrome(driver_path)
    
    # URL of the HuggingFace profile
    url = "https://huggingface.co/TheBloke"
    browser.get(url)
    
    # Wait for the models to load (you can adjust the time if needed)
    browser.implicitly_wait(10)
    
    # Find all model names
    model_elements = browser.find_elements(By.CSS_SELECTOR, 'a.hf-Link')  # Modify the selector if it's different
    
    # Filter model names containing "GPTQ" or "gptq"
    gptq_models = [elem.text for elem in model_elements if "GPTQ" in elem.text or "gptq" in elem.text]
    
    # Close the browser
    browser.quit()
    
    return gptq_models

# Test the function
print(get_gptq_models())
