FROM tensorflow/tensorflow:2.10.0

COPY requirements_prod.txt /requirements.txt
COPY .env /.env

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /fungai
# set the working directory to fungai instead of the os home file
WORKDIR /fungai
RUN rm requirements.txt && mv requirements_prod.txt requirements.txt
RUN pip install .

CMD uvicorn FungAI.api.api:app --host 0.0.0.0