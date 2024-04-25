import uuid
from typing import Annotated
from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from PIL import Image
import io
import asyncio

import mnist_model.mnist_model_eval as MNISTModel


class ProcessingRequest:
    """
    Класс, держащий информацию о запросе на обработку.
    Аттрибуты:
        imgs (list[bytes]): список изображений для обработки
        id (str): uuid, выдается при создании. По нему мы отличаем запросы друг от друга
        ready (bool): выполнен ли запрос
        failed (bool): провалился ли запрос
        results (list[int]): результаты выполнения для каждого изображения (В данном случае - цифра)
        pngs_paths (list[str]): список из путей к изображениям на сервере, каждое изображение соответствует
            результату. т.е. pngs_paths[0] это путь к изображению, а results[0] это распознанная цифра с
            этого изображения. Используется для страницы с результатами.
    """
    def __init__(self, imgs: list[bytes]):
        self.imgs = imgs
        self.id = str(uuid.uuid4())

        self.ready = False
        self.failed = False
        self.results: list[int] = []

        self.pngs_paths: list[str] = []

requests: list[ProcessingRequest] = []

def find_request(uuid: str) -> ProcessingRequest | None:
    for req in requests:
        if req.id == uuid:
            return req
    return None

async def process_files(uuid: str) -> None:
    """
    Обрабатывает все файлы с запроса, имеющего соотв. uuid.
    После обработки устанавливает ready запроса на True
    """
    req = find_request(uuid)
    if(req is None): return
    if(len(req.imgs) == 1 and req.imgs[0] == b''):
        req.failed = True
        return
        

    # load model and get model's opinion on each image
    MNISTModel.load_mnist_model()
    outputs: list[int] = []
    files_counter = 0
    for file in req.imgs:
        img = Image.open(io.BytesIO(file))
        output = MNISTModel.eval_mnist_on_file(img)
        outputs.append(output)

        # also save photos in temporary location for neat results page
        img.save(f'public/tempnumbers/{uuid}_{files_counter}.png')
        req.pngs_paths.append(f'/getnumberpng/{uuid}_{files_counter}.png')
        files_counter += 1
        img.close()
    
    await asyncio.sleep(2) # some useful work :)
        
    req.results = outputs
    req.ready = True
 
app = FastAPI()
 
@app.get("/")
async def main():
    return FileResponse("public/index.html")

@app.get("/processing/{uuid}")
async def processing_page(uuid: str):
    return FileResponse("public/processing.html")

@app.get("/results/{uuid}")
async def results_page(uuid):
    req = find_request(uuid)
    if req is None:
        return RedirectResponse("/", status_code=status.HTTP_404_NOT_FOUND)
    
    return FileResponse("public/results.html")

@app.get("/getresults/{uuid}")
async def on_get_results(uuid: str):
    """
    Когда юзер попадает на страницу с результатами она сразу посылает этот
    GET запрос и мы возвращаем результаты в виде массива где каждый результат
    это {data: pngs_paths[i], label: results[i]}.
    """
    req = find_request(uuid)
    if req is None:
        return JSONResponse({}, status.HTTP_404_NOT_FOUND)
    response = []
    for i in range(len(req.imgs)):
        response.append({"data": req.pngs_paths[i], "label":req.results[i]})
    return JSONResponse(response)

@app.get("/isprocessed/{uuid}")
async def is_processed(uuid: str):
    """
    Пользователь на странице ожидания (processing.html) каждую секунду
    отправляет этот GET запрос чтобы проверить состояние обработки своего
    запроса. Тут мы возвращаем ответ. (в виде json и хардкод строк за что
    меня нужно покарать но я не знаю как иначе (джсеры и похуже код пишут
    так что моя душа спокойна))
    """
    req = find_request(uuid)
    if req is None or req.failed == True:
        return JSONResponse({"message":"failed"})
    if req.ready == False:
        return JSONResponse({"message":"not ready"})
    return JSONResponse({"message":"ready"})

@app.post("/uploadfiles/")
async def on_upload_files(files: Annotated[list[bytes], File()]):
    """
    Пользователь нажал на "Отправить запрос". Создаем ProcessingRequest с его файлами
    и отправляем асинхронную задачу на обработку этого запроса. Асинхронная задача т.к. 
    запрос может долго обрабатываться и мы не хотим тупо вешать клиенту браузер и наш сервер.
    """
    req = ProcessingRequest(files)
    requests.append(req)
    asyncio.create_task(process_files(req.id))
    # Клиента отправляем на страницу ожидания
    return RedirectResponse(f"/processing/{req.id}", status.HTTP_302_FOUND)

@app.get("/getnumberpng/{png_name}")
async def get_number_png(png_name: str):
    """
    сто пудова дыра в безопасности лол
    """
    return FileResponse(f"public/tempnumbers/{png_name}")

@app.get("/failed_processing")
async def failed_processing_page():
    return FileResponse(f"public/failed_processing.html")

@app.get("/fail_gif")
async def fail_gif():
    return FileResponse(f"public/gifs/Ideya_kharosh_realizatsia_pizdets.gif")