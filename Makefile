.PHONY: build

init:
	pyenv install -sf 3.8.2
	pyenv local 3.8.2
	pip install --exists-action i -qqq pipenv

build:
	rm -fr dist/ build/
	pipenv --python 3.8.2 install --dev
	pipenv run python setup.py bdist_wheel

run:
	docker-compose up -d
	pipenv run alembic upgrade head

stop:
	docker-compose down
