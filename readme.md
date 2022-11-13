[![Python 3.10.6](https://img.shields.io/badge/python-3.10.6-blue.svg)](https://www.python.org/downloads/release/python-3106/)
![coverage](https://img.shields.io/badge/test_coverage-73%25-yellowgreen)
### Usage
```shell
docker-compose up -d --build
```
### Documentation: 
http://0.0.0.0:9090/docs

### Development
Install dev environment:
```shell
pip install -r app/requirements-dev.txt
```
Set necessary env variables:
```shell
export APP_PROJECT_NAME='AI Software Engineering'
```
Run tests
```shell
pytest
```
Check tests coverage:
```shell
pytest --cov app
```

### Known bugs:
- invalid symlink remains at app folder after file recognition