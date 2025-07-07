import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
import time
import random
import os

class DromAutoParser:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7'
        }
        self.base_url = "https://auto.drom.ru"
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.cars_data = []

    def parse_car_page(self, car_url):
        try:
            response = self.session.get(car_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            car_info = {
                "Марка": None,
                "Модель": None,
                "Год выпуска": None,
                "Цена": None,
                "Двигатель": None,
                "Мощность": None,
                "Коробка передач": None,
                "Привод": None,
                "Тип кузова": None,
                "Пробег": None,
                "Цвет": None,
                "Владельцы": None,
                "Руль": None,
                "Поколение": None,
                "Комплектация": None,
                "URL": car_url,
            }

            title = soup.find('h1', {'class': 'css-6tq1oz e18vbajn0'})
            if not title:
                title = soup.find('span', {'class': 'css-1kb7l9z e162wx9x0'})
            
            if title:
                title_text = title.text.split(',')[0].strip()
                parts = title_text.split()
                if len(parts) >= 2:
                    car_info["Марка"] = parts[1]
                    car_info["Модель"] = ' '.join(parts[2:])
                
                year_part = title.text.split(',')[1].split(' ')[1]
                
                car_info["Год выпуска"] = year_part

            price = soup.find('div', {'class': 'wb9m8q0'})
            
            if price:
                try:
                    car_info["Цена"] = price.text
                except ValueError:
                    pass

         
            specs_table = soup.find('table', class_='css-xalqz7 eo7fo180')
            if specs_table:
                rows = soup.find_all('tr')
                for row in rows:
                    th = row.find('th', class_='css-1dzcqnh eka0pcn1')
                    td = row.find('td', class_='css-1azz3as eka0pcn0')
                    if th and td:
                        label = th.text.strip()
                        value = td.text.strip()
                        
                        if 'Двигатель' in label:
                            car_info["Двигатель"] = value
                        elif 'Коробка передач' in label:
                            car_info["Коробка передач"] = value
                        elif 'Пробег' in label:
                            car_info["Пробег"] = value
                        elif 'Цвет' in label:
                            car_info["Цвет"] = value
                        elif 'Мощность' in label:
                            car_info["Мощность"] = value
                        elif 'Привод' in label:
                            car_info['Привод'] = value
                        elif 'Тип кузова' in label:
                            car_info['Тип кузова'] = value
                        elif 'Владельцы' in label:
                            car_info['Владельцы'] = value
                        elif 'Руль' in label:
                            car_info['Руль'] = value
                        elif 'Поколение' in label:
                            car_info['Поколение'] = value
                        elif 'Комплектация' in label:
                            car_info['Комплектация'] = value

            return car_info

        except Exception as e:
            print(f"Ошибка при парсинге страницы {car_url}: {str(e)}")
            return None

    def parse_listing_page(self, page_url):
        try:
            response = self.session.get(page_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            car_links = soup.find_all('a', {'class': '_1990l150'})
            for link in car_links:
                car_url = urljoin(self.base_url, link['href'])
                print(f"Парсинг автомобиля: {car_url}")
                
                car_data = self.parse_car_page(car_url)
                if car_data:
                    self.cars_data.append(car_data)
                
                time.sleep(random.uniform(1, 3))

            next_page = soup.find('a', {'data-ftid': 'component_pagination-item-next'})
            if not next_page:
                next_page = soup.find('a', {'class': 'css-4gbnjj e24vrp30'})
            
            if next_page:
                return urljoin(self.base_url, next_page['href'])
            return None

        except Exception as e:
            print(f"Ошибка при парсинге страницы {page_url}: {str(e)}")
            return None

    def parse(self, pages=10, output_file='drom_cars.json'):
        self.cars_data = []                        
        current_url = self.base_url

        for _ in range(pages):
            if not current_url:
                break
            print(f"Парсинг страницы: {current_url}")
            current_url = self.parse_listing_page(current_url)

        existing = []                              
        if os.path.isfile(output_file): 
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                print("Файл повреждён")

        seen_urls = {car['URL'] for car in existing} 
        new_unique = [c for c in self.cars_data if c['URL'] not in seen_urls]
        merged = existing + new_unique

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=4)

        print(f"Добавлено {len(new_unique)} новых авто "
              f"(всего в файле: {len(merged)}).")

        return new_unique                         
    

if __name__ == "__main__":
    parser = DromAutoParser()
    for i in range(3):
        print(f"\n=== Итерация {i+1} ===")
        parser.parse(pages=10, output_file='cars.json')
        time.sleep(random.uniform(2, 4))