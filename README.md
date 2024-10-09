# Скачать репозиторий  
```
git clone https://github.com/AxuJIuyc/YOLO_HumanSegmentor.git
cd YOLO_HumanSegmentor
```

# Установить зависимости  
Note: Будут установлены torch и nvidia_cuda (> 1.5 GB)
```
pip install -r ./requirements.txt
```

# Запуск
```
python image_processor.py
```
Note: Прекратить воспроизведение можно по клавише `Q`

# Пример результатов
<table>
  <tr>
    <td><img src="data/IMG_8756.JPG" width="45%"/></td>
    <td><img src="results/0.png" width="45%"/></td>
  </tr>
  <tr>
    <td><img src="data/IMG_8648.jpg" width="45%"/></td>
    <td><img src="results/2.png" width="45%"/></td>
  </tr>
  <tr>
    <td><img src="data/IMG_8758.JPG" width="45%"/></td>
    <td><img src="results/3.png" width="45%"/></td>
  </tr>
  <tr>
    <td><img src="data/IMG_8815.JPG" width="45%"/></td>
    <td><img src="results/1.png" width="45%"/></td>
  </tr>
</table>
