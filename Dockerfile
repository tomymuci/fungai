FROM tensorflow/tensorflow:2.10.0

COPY requirements_prod.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install libgl1

COPY . /fungai
# set the working directory to taxifare instead of the os home file
WORKDIR /fungai
RUN pip install .

CMD uvicorn FungAI.api.api:app --host 0.0.0.0
