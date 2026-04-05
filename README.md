# music_search_project

`video` 폴더의 영상에서 오디오를 추출한 뒤, [ACRCloud](https://www.acrcloud.com/)로 곡 정보를 조회합니다.

## 준비

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`acrcloud_credentials.example.py`를 `acrcloud_credentials.py`로 복사하고 HOST / ACCESS_KEY / ACCESS_SECRET을 입력하세요. (`acrcloud_credentials.py`는 Git에 포함되지 않습니다.)

`video`에 영상을 두고, 추출된 파일은 `audio`에 저장됩니다. (용량 때문에 `audio/`, `video/`는 기본적으로 Git에 올리지 않습니다.)

## 실행

```bash
python main.py
python acrcloud_recognize.py
```

결과: `results.csv`
