from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import time
import random
import os

class AutoSenderSeleniumParser:
    def __init__(self, max_pages=150):
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.base_url = "https://autosender.ru/avto-iz-kitaya"
        self.cars = []
        self.max_pages = max_pages
        self.debug_dir = "debug"
        os.makedirs(self.debug_dir, exist_ok=True)

    def take_screenshot(self, name):
        path = os.path.join(self.debug_dir, f"{name}.png")
        self.driver.save_screenshot(path)
        print(f"Screenshot saved: {path}")

    def save_page_source(self, name):
        path = os.path.join(self.debug_dir, f"{name}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.driver.page_source)
        print(f"Page source saved: {path}")

    def load_all_pages(self):
        print("Loading initial page...")
        self.driver.get(self.base_url)
        self.take_screenshot("initial_page")
        time.sleep(3)

        try:
            cookie_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.cookie__btn")))
            cookie_btn.click()
            print("Cookies accepted")
            time.sleep(1)
        except Exception as e:
            print(f"Cookie notice not found: {str(e)}")

        page_num = 1
        while page_num <= self.max_pages:
            try:

                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.auctions__info-wrap")))
                
                self.parse_loaded_page()
                
                if page_num % 10 == 0:
                    self.save_to_file(f"autosender_part_{page_num}.json")
                
                next_button = None
                try:
                    next_buttons = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.btn.btn-primary")))
                    
                    for btn in next_buttons:
                        if "следующая" in btn.text.lower():
                            next_button = btn
                            break
                except Exception as e:
                    print(f"Error finding next button: {str(e)}")
                
                if not next_button:
                    print("No more pages found")
                    break
                
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                time.sleep(1)
                self.driver.execute_script("arguments[0].click();", next_button)
                print(f"Loading page {page_num + 1}/{self.max_pages}")
                
                WebDriverWait(self.driver, 15).until(
                    EC.staleness_of(next_button))
                time.sleep(2 + random.uniform(0.5, 1.5))
                
                page_num += 1
                
            except Exception as e:
                print(f"Error loading page {page_num}: {str(e)}")
                self.take_screenshot(f"error_page_{page_num}")
                break

    def parse_loaded_page(self):
        print("Parsing page content...")
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        cards = soup.select("div.auctions__info-wrap")
        
        if not cards:
            print("No cars found on this page!")
            self.take_screenshot("no_cars_found")
            return

        for card in cards:
            try:
                car = {
                    "Марка": None,
                    "Модель": None,
                    "Год выпуска": None,
                    "Цена в г.Владивосток": None,
                    "Цена на аукционе": None, 
                    "Двигатель": None,
                    "Мощность": None,
                    "Коробка передач": None,
                    "Пробег": None,
                    "Цвет": None,
                    "URL": None,
                    "Дата публикации": None,
                }

                title_tag = card.select_one("a.auctions__brand")
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    parts = title.split(" ", 1)
                    car["Марка"] = parts[0]
                    car["Модель"] = parts[1] if len(parts) > 1 else None

                price_tag = soup.select_one("div.auctions__cost-up")
                if price_tag:
                    car["Цена в г.Владивосток"] = price_tag.text.strip().replace("\n", "").replace(" ", "")

                auction_price_tag = card.select_one("div.auctions__cost-down")
                if auction_price_tag:
                    car["Цена на аукционе"] = auction_price_tag.get_text(strip=True).replace(" ", "")

                link_tag = card.select_one("a.auctions__brand")
                if link_tag and link_tag.has_attr('href'):
                    car["URL"] = "https://autosender.ru" + link_tag["href"] if not link_tag["href"].startswith("http") else link_tag["href"]

                specs = card.select("div.auctions__info")
                for spec in specs:
                    items = [item.strip() for item in spec.get_text().split('\n\n') if item.strip()]
                    
                    for item in items:
                        if ':' in item:
                            key, value = map(str.strip, item.split(':', 1))
                            if not value or value == '—':
                                continue
                                
                            if key == 'Год':
                                car['Год выпуска'] = value
                            elif key == 'Пробег':
                                car['Пробег'] = value
                            elif key == 'Объем':
                                car['Двигатель'] = f"{value} см³"
                            elif key == 'Мощность':
                                car['Мощность'] = value
                            elif key == 'КПП':
                                car['Коробка передач'] = value.replace(',', '').strip()
                            elif key == 'Цвет':
                                car['Цвет'] = value
                            elif key == 'Дата Добавления':
                                car['Дата публикации'] = value
                            elif key == 'Цена на аукционе':
                                car["Цена на аукционе"] = value

                self.cars.append(car)
                print(f"Added car: {car.get('Марка', '')} {car.get('Модель', '')}")
                
            except Exception as e:
                print(f"Error parsing car card: {str(e)}")

    def save_to_file(self, filename="autosender_all.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.cars, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.cars)} cars to {filename}")

    def run(self):
        try:
            start_time = time.time()
            self.load_all_pages()
            self.save_to_file()
            print(f"Total execution time: {(time.time() - start_time)/60:.2f} minutes")
        except Exception as e:
            print(f"Critical error: {str(e)}")
            self.take_screenshot("critical_error")
        finally:
            self.driver.quit()

if __name__ == "__main__":
    parser = AutoSenderSeleniumParser(max_pages=150)
    parser.run()