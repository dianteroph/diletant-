import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time


CHROME_DRIVER_PATH = './chromedriver-mac-arm64/chromedriver'

def start_driver():
    service = Service(executable_path=CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(service=service)
    driver.maximize_window()
    return driver


url = 'https://diletant.media/'

driver = start_driver()
driver.get(url)

search_button = driver.find_element(By.CLASS_NAME, 'NavStyles__StyledRouterLink-sc-rnvz55-18')
search_button.click()
search_input = driver.find_element(By.XPATH, '//*[@id="__next"]/div/div[1]/div/input')
search_input.send_keys('Ленин')

time.sleep(1)
button = driver.find_element(By.XPATH, '//*[@id="__next"]/div/div[2]/div[2]')
i = 0
while i < 30:
    button.click()
    time.sleep(0.5)
    i += 1

card_elements = driver.find_elements(By.CLASS_NAME, 'Card_card__3eu6Y')
links = [card.find_element(By.CLASS_NAME, 'Card_card__title__1dG8R').
         get_attribute('href') for card in card_elements]

start_time = time.time()

parsed_text = []
for link in links:
    print(link)
    if 'articles' not in link:
        continue

    driver.get(link)
    if '404' in driver.title:
        continue

    views = driver.find_element(By.XPATH, '/html/body/div[2]/main/section[1]/div/div/header/div/span[1]').text
    abstract = driver.find_element(By.CLASS_NAME, 'publication-teaser__theme').text
    title = driver.find_element(By.CLASS_NAME, 'publication-teaser__title').text
    text = driver.find_element(By.TAG_NAME, 'article').text

    try:
        author = driver.find_element(By.XPATH, '//*[@id="comp_c5696094e2404a625775f2d6b4fcacdd"]/footer/div/div/span[1]').text
    except:
        author = 'No 1st element'

    try:
        publication_date = driver.find_element(By.XPATH, '//*[@id="comp_c5696094e2404a625775f2d6b4fcacdd"]/footer/div/div/span[2]').text
    except:
        publication_date = 'No 2nd element'

    parsed_text.append({
        'link': link,
        'views': views,
        'title': title,
        'abstact': abstract,
        'text': text,
        'author': author,
        'publication_date': publication_date
    })

    time.sleep(random.uniform(0,2))

df = pd.DataFrame(parsed_text)
print("--- %s seconds ---" % (time.time() - start_time))

df.to_excel('lenin_output.xlsx')
