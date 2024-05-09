const dropFileZone = document.querySelector(".upload-zone_dragover"); // устанавливает область обработки выбранного файла;
const uploadInput = document.querySelector(".form-upload__input"); // устанавливает область кнопки для загрузки файла без перетаскивания.

const uploadUrl = "/unicorns";

//Выполняется во время перемещения файла над областью обработки файла;
["dragover", "drop"].forEach(function (event) {
  document.addEventListener(event, function (evt) {
    //Когда при перетаскивании выбранный файл будет находиться в пределах активной страницы, браузер его откроет.
    //Чтобы файл был обработан в назначенной для этого области,
    //необходимо отменить стандартное поведение браузера для событий dragover и drop путём вызова метода preventDefault():
    evt.preventDefault();
    return false;
  });
});

//срабатывает, когда файл входит в область обработки файла
dropFileZone.addEventListener("dragenter", function () {
  dropFileZone.classList.add("_active");
});

//срабатывает, если файл покидает область обработки, но ещё не «сброшен»;
dropFileZone.addEventListener("dragleave", function () {
  dropFileZone.classList.remove("_active");
});

//выполняется в тот момент, когда пользователь отпустил кнопку мыши и выбранный файл был помещён («сброшен») в заданную область.
dropFileZone.addEventListener("drop", function () {
  dropFileZone.classList.remove("_active");
  const file = event.dataTransfer?.files[0];
  if (!file) {
    return;
  }

  if (file.type.startsWith("image/")) {
    uploadInput.files = event.dataTransfer.files;
    processingUploadFile();
  } else {
    setStatus("Можно загружать только изображения");
    return false;
  }
});

/*uploadInput.addEventListener("change", (event) => {
  const file = uploadInput.files?.[0];
  if (file && file.type.startsWith("image/")) {
    processingUploadFile();
  } else {
    setStatus("Можно загружать только изображения");
    return false;
  }
});

//принимает выбранный пользователем файл fileInstanceUpload и отправляет его на сервер:
function processingUploadFile(file) {
  if (file) {
    const dropZoneData = new FormData(); //Тут с использованием объекта FormData будут храниться данные для отправки на сервер;
    const xhr = new XMLHttpRequest(); // Чтобы отправить файл на сервер без перезагрузки страницы

    dropZoneData.append("file", file);

    xhr.open("POST", uploadUrl, true); // Метод open() выполняет POST-запрос к управляющему файлу, который хранится на сервере

    xhr.send(dropZoneData); // Выбранный пользователем файл передаётся на сервер.

    xhr.onload = function () {
      if (xhr.status == 200) {
        setStatus("Всё загружено");
      } else {
        setStatus("Oшибка загрузки");
      }
      HTMLElement.style.display = "none";
    };
  }
}*/

// Посылает GET запрос чтобы узнать четам по обработке
async function isProcessed(id) {
  const response = await fetch(`/isprocessed/${id}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (response.ok === true) {
    const res = await response.json();
    if (res.message == "failed") {
      window.location = "/failed_processing";
    } 
    else if (res.message == "ready") {
      window.location = `/results/${id}`;
    }
    else if (res.message == "not ready") {
      let processed_files = res.processed_files;
      let total_files = res.total_files;
      let percent = Math.round((1.0 * processed_files / total_files) * 100);

      document.getElementById("progress-text").textContent=`${percent}% (${processed_files} / ${total_files})`;
    }
  }
}

async function getResults(id) {
  // Этот GET запрос вернет пути к изображениям и оценку модели для каждого
  // из изображения для запроса 'id'.
  const response = await fetch(`/getresults/${id}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  // если запрос ОКЭЙ то для каждого изображения добавляем строку в таблицу(generateRow(r))
  if (response.ok === true) {
    const req = await response.json();
    const rows = document.querySelector("tbody");
    req.forEach((r) => rows.append(generateRow(r)));
  } else {
    console.log("response not ok");
  }
}

function generateRow(r) {
  // добавляем табличную строку
  const tr = document.createElement("tr");

  // пихаем в строку картинку и цифорку которую определила модель
  const r_img = document.createElement("td");
  let img = document.createElement("img");
  img.src = r.data;
  r_img.append(img);

  const r_name = document.createElement("td");
  r_name.append(r.label);
  tr.append(r_img);
  tr.append(r_name);

  return tr;
}