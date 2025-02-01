from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Callable
import fitz
import os
import re
import uvicorn
from PIL import Image, ImageEnhance
from io import BytesIO
import logging
import uuid
import json

from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PDFExtractor")

# Инициализация приложения FastAPI
app = FastAPI()

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Предварительная обработка изображения перед OCR.
    Увеличение контрастности.
    """
    try:
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL.Image.Image object")
        
        grayscale = image.convert("L")  # Преобразование в оттенки серого
        enhanced = ImageEnhance.Contrast(grayscale).enhance(3.5)  # Усиление контрастности
        
        logger.info("Image preprocessing completed successfully")
        return enhanced
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise e

# Определение функций для обработки текста по шаблону
def extract_title_page(text: str) -> dict:
    """
    Извлечение данных из титульного
    """
    lines = text.splitlines()

    data = {
        "theme": "not found",
        "students": [],
        "department_number": 0,
        "department_name": "not found"
    }

    theme_found = False
    theme_str = ""

    department_found = False
    department_str = ""

    try:
        for idx, line in enumerate(lines):

            # Выделение темы
            if "тему" in line.lower():
                theme_str = line.split("«")[-1]
                theme_found = True
            elif theme_found and (theme_str[-1] != "»") and bool(line.strip()):
                theme_str = theme_str + " " + line.strip()
            elif theme_found and theme_str[-1] == "»":
                data["theme"] = theme_str.strip("»")
                theme_found = False

            # Выделение студентов
            elif "студенты" in line.lower():
                students_id = idx + 1
                while students_id < len(lines) \
                      and lines[students_id].strip() \
                      and "подпись" not in lines[students_id].lower():
                    
                    student = lines[students_id].strip()
                    data["students"].append(student)
                    students_id += 1

            # Выделение номера кафедры
            elif "кафедра" in line.lower() and not department_found:
                number_line = line.split()
                if number_line[-1].isdigit():
                    data["department_number"] = int(number_line[-1])

                # Проверка следующей строки на наличие названия кафедры
                if (idx + 1 < len(lines)) and ("«" in lines[idx + 1]):
                    department_str = lines[idx + 1].strip("«")
                    department_found = True
                    
            elif department_found \
                and (department_str != line.strip("«")) \
                and department_str[-1] != "»":

                department_str = department_str + " " + line.strip("«")
            elif department_found and department_str and department_str[-1] == "»":
                data["department_name"] = department_str.rstrip("»").strip()
                department_found = False

    except Exception as e:
        logger.error(f"Error while extracting title page: {e}")

    return data

def extract_act_date_one(text: str) -> dict:
    """
    Извлечение данных из акта
    """
    lines = text.splitlines()

    data = {
        "message": "not found",
        "item": 0,
        "date_dmy": "not found",
        "type": "Акт",
        "location": "not found"
    }

    # Словарь преобразования месяца в числовой формат
    months = {
        "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
        "мая": 5, "июня": 6, "июля": 7, "августа": 8,
        "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
    }

    try:
        for idx, line in enumerate(lines):

            # Выделение ...
            if idx == 0:
                data["message"] = line[2:]

            # Выделение пункта
            elif (mask_item := re.search(r"по (\d+) пункту", line.lower())):
                data["item"] = mask_item.group(1)

            # Выделение даты
            elif (mask_date := re.search(r"«(\d{1,2})» (\w+) (\d{4})", line)):
                day = mask_date.group(1)
                month_str = mask_date.group(2)
                year = mask_date.group(3)

                month = months.get(month_str.lower())

                data["date_dmy"] = f"{int(day):02d}.{month:02d}.{year}"

            # Выделение места проведения
            elif (mask_location := re.search(r"проводилась в (.+)", line)):
                data["location"] = mask_location.group(1).split(".")[0]           
        
    except Exception as e:
        logger.error(f"Error while extracting act_one: {e}")

    return data

def extract_act_date_two(text: str) -> dict:
    """
    Извлечение данных из акта 2
    """
    lines = text.splitlines()

    # Шаблоны для поиска данных
    rank_mask = re.compile(
        r"(лейтенант|старший лейтенант|капитан|майор|подполковник|полковник)\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ])\.([А-ЯЁ])\."
    )
    department_mask = re.compile(r"от\s+(.+?):")
    role_mask = re.compile(
        r"(Заместитель председателя экспертной комиссии|Председатель экспертной комиссии|Члены экспертной комиссии|Секретарь экспертной комиссии)"
    )

    # Словари для преобразования из родительного в именительный падеж
    rank_conversion = {
        "лейтенанта": "лейтенант",
        "старшего лейтенанта": "старший лейтенант",
        "капитана": "капитан",
        "майора": "майор",
        "подполковника": "подполковник",
        "полковника": "полковник",
    }

    role_conversion = {
        "Заместитель председателя экспертной комиссии": "Заместитель председателя ЭК",
        "Председатель экспертной комиссии": "Председатель ЭК",
        "Члены экспертной комиссии": "Член ЭК",
        "Секретарь экспертной комиссии": "Секретарь ЭК",
    }
    
    # Данные сотрудников
    data = []
    current_department = None

    try:
        # Первая проходка: предварительный сбор в родительном падеже
        logger.info("Начата первая проходка для сбора данных сотрудников.")
        for line in lines:
            # Проверяем, есть ли информация об отделе
            department_match = department_mask.search(line)
            if department_match:
                current_department = department_match.group(1).strip()
                logger.debug(f"Найден отдел: {current_department}")

            # Ищем сотрудников
            for match in rank_mask.finditer(line):
                rank_roditel = match.group(1)
                last_name = match.group(2)
                first_name = match.group(3)
                middle_name = match.group(4)

                # Преобразуем звание в именительный падеж
                rank = rank_conversion.get(rank_roditel, rank_roditel)

                logger.debug(f"Найден сотрудник: {rank} {last_name} {first_name}.{middle_name}.")

                data.append({
                    "role": "-",
                    "rank": rank,
                    "last_name_roditel": last_name,
                    "first_name": first_name,
                    "middle_name": middle_name,
                    "department": current_department or "-",
                })

        logger.info("Первая проходка завершена.")

        # Вторая проходка: сопоставление ролей и фамилий
        logger.info("Начата вторая проходка для сопоставления ролей и обновления фамилий.")
        for line in lines:
            for match in role_mask.finditer(line):
                logger.info("1")
                role_roditel = match.group(1)
                logger.info("1")
                name_match = match.group(2)

                logger.info("2")
                # Преобразуем звание в именительный падеж
                role = role_conversion.get(role_roditel, role_roditel)
                logger.info("2")
                for d in data:
                    # Ищем фамилию сотрудника (родительный падеж) в строке и обновляем на именительный
                    if re.search(rf"{d['last_name_roditel']}\s+", line):
                        # Извлекаем именительный падеж фамилии из строки
                        name_match = re.search(rf"([А-Я][а-я]+)\s+", line)
                        if name_match:
                            old_last_name = d.get("last_name", d["last_name_roditel"])
                            d["last_name"] = name_match.group(1)
                            d["role"] = role
                            logger.debug(f"Обновлена фамилия: {old_last_name} -> {d['last_name']}, роль: {role}")

        logger.info("Вторая проходка завершена.")

        # Убираем временное поле
        logger.info("Удаление временных полей и финализация данных.")
        for d in data:
            d.pop("last_name_roditel", None)

    except Exception as e:
        logger.error(f"Ошибка при обработке данных: {e}")


    return data

def extract_act_date_three(text: str) -> dict:
    """
    Извлечение данных из акта 3
    """
    lines = text.splitlines()

    # Переменные для хранения информации
    data = {}
    current_solution = None
    solution_number = None
    
    # Шаблон для поиска решения
    solution_pattern = re.compile(r"Акт № (\d+)")
    
    # Шаблон для поиска деталей решения (appellation, key, total)
    details_pattern = re.compile(r"«(.*?)»\s*(включить|отказать во включении)?")
    
    # Итерируем по строкам
    idx = 1
    id_sol = 0

    try:
        for line in lines:
            # Ищем номер решения
            match = solution_pattern.search(line)
            if match:
                id_sol += 1
                solution_number = match.group(1)  # Извлекаем номер решения
                current_solution = idx
                data[current_solution] = {
                    "appellation": "",
                    "key": "",
                    "total": "",
                    "organization": "-",
                    "flag": 0,
                    "number": f"{solution_number}.{id_sol}"
                }
                idx += 1
                logger.info(f"Solution number {solution_number}.{id_sol} found and added.")
                continue
            
            # Ищем ключевые фразы для решения
            if current_solution:
                match_details = details_pattern.search(line)
                if match_details:
                    # Обрабатываем фразу
                    if not data[current_solution]["appellation"]:
                        data[current_solution]["appellation"] = match_details.group(1)
                    
                    if match_details.group(2):
                        data[current_solution]["total"] = line
                    else:
                        data[current_solution]["total"] = f"{data[current_solution]['appellation']} {line}"

                    # Проверка на завершение решения
                    if "включить" in data[current_solution]["total"] or \
                    "отказать во включении" in data[current_solution]["total"]:
                        data[current_solution]["number"] = f"{solution_number}.{idx - 1}"
                        logger.info(f"Solution {solution_number}.{idx - 1} completed.")
                        current_solution = None

    except Exception as e:
        logger.error(f"Error while extracting act_three: {e}")


    return data

def extract_full_text(text: str) -> dict:
    """
    Обработка полного текста из PDF с применением .lower().
    """
    try:
        lowercased_text = text
        return {"full_text": lowercased_text}
    except Exception as e:
        logger.error(f"Ошибка при обработке полного текста: {e}")
        return {"error": "Ошибка при обработке полного текста"}

# Словари для обработки инструкций
template_handlers = {
    "title_page": extract_title_page,
    "full_text": extract_full_text,
    "act_one": extract_act_date_one,
    "act_two": extract_act_date_two,
    "act_three": extract_act_date_three,
}

template_descriptions = {
    "title_page": "Титульный лист",
    "full_text": "Полный текст",
    "act_one": "Акт 1",
    "act_two": "Акт 2",
    "act_three": "Акт 3",
}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Отображает HTML-страницу с динамическими инструкциями.
    """
    logger.info("GET / endpoint accessed")
    try:
        # Чтение шаблона HTML
        with open("index.html", "r", encoding="utf-8") as file:
            html_template = file.read()

        # Генерация опций для выпадающего списка
        options = "\n".join(
            f'<option value="{key}">{description}</option>'
            for key, description in template_descriptions.items()
        )

        # Замена {{ options }} на сгенерированные опции
        html_content = html_template.replace("{{ options }}", options)

        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error while rendering HTML: {e}")
        return HTMLResponse(content="<h1>Internal Server Error</h1>", status_code=500)


@app.post("/extract-text/")  
async def extract_text_from_pdf(
    pdf_file: UploadFile = File(...), 
    template: str = Form(...),
    use_page: str = Form(None)  
):
    """
    Извлекает данные из PDF-файла на основе выбранного шаблона, используя OCR,
    и возвращает либо HTML-страницу, либо JSON в зависимости от параметра use_page
    """
    logger.info(f"POST /extract-text/ endpoint accessed with template: {template} and use_page: {use_page}")

    try:
        # Генерация уникального имени для временного файла
        unique_id = uuid.uuid4().hex
        temp_pdf_path = f"temp_{unique_id}_{pdf_file.filename}"

        # Сохранение загруженного файла
        with open(temp_pdf_path, "wb") as f:
            f.write(await pdf_file.read())
        logger.info(f"Uploaded file saved as {temp_pdf_path}")

        try:
            # Открытие PDF-файла через PyMuPDF
            doc = fitz.open(temp_pdf_path)

            extracted_text = []
            combined_text = ""

            # Загрузка модели для OCR
            langs = ["ru"] 
            det_processor, det_model = load_det_processor(), load_det_model()
            rec_model, rec_processor = load_rec_model(), load_rec_processor()

            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]

                    # Конвертация страницы в изображение
                    pix = page.get_pixmap()
                    image = Image.open(BytesIO(pix.tobytes("ppm")))

                    # Предварительная обработка изображения
                    preprocessed_image = preprocess_image(image)

                    # Используем Surya OCR для OCR
                    predictions = run_ocr([preprocessed_image], [langs], det_model, det_processor, rec_model, rec_processor)
                    
                    # Получаем текст из предсказания
                    extracted_text = "\n".join([line.text for line in predictions[0].text_lines])
                    
                    # Объединение текста со всех страниц
                    combined_text = combined_text + "/n" + extracted_text

                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")

            # Удаление временного файла
            doc.close()
            os.remove(temp_pdf_path)
            logger.info(f"Temporary file {temp_pdf_path} removed")

            # Применение шаблона
            handler: Callable[[str], dict] = template_handlers.get(
                template, lambda text: {"error": "Template not found"}
            )

            try:
                processed_text = handler(combined_text)
                logger.info("Text extraction and template processing completed")

                # Проверяем флажок use_page
                if use_page == "yes":
                    # Чтение файла result.html
                    try:
                        with open("result.html", "r", encoding="utf-8") as file:
                            html_template = file.read()

                        # Замена {{ json_data }} на обработанный JSON
                        formatted_json = json.dumps(processed_text, ensure_ascii=False, indent=4)
                        html_content = html_template.replace("{{ json_data }}", formatted_json)

                        return HTMLResponse(content=html_content)

                    except Exception as e:
                        logger.error(f"Error reading result.html: {e}")
                        return JSONResponse(content={"error": "Error rendering result.html"}, status_code=500)
                else:
                    # Возвращаем JSON-ответ
                    return JSONResponse(content=processed_text)

            except Exception as e:
                logger.error(f"Error applying template: {e}")
                return JSONResponse(content={"error": "Error applying template"}, status_code=500)

        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
            return JSONResponse(content={"error": "Error during text extraction"}, status_code=500)

    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        return JSONResponse(content={"error": "Error saving uploaded file"}, status_code=500)


if __name__ == "__main__":
    logger.info("Starting FastAPI application")
    uvicorn.run(app="reader:app", host="127.0.0.1", port=8000, reload=True)
