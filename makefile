reinstall_package:
	@pip uninstall -y FungAI || :
	@pip install -e .

run_preprocess:
	python -c 'from FungAI.interface.main import preprocessor; preprocessor();'

run_train:
	python -c 'from FungAI.interface.main import train; train();'

run_eval:
	python -c 'from FungAI.interface.main import evaluate; evaluate();'

run_api:
<<<<<<< HEAD
	uvicorn --port 1234  FungAI.api.api:app --reload
=======
	uvicorn --port 1234 FungAI.api.api:app --reload
>>>>>>> 4d38dc1fb02da4bfdd9ad7f15e75a1164ac7366c

run_predict:
	python -c 'from FungAI.interface.main import pred; pred();'

streamlit:
	-@streamlit run frontend/app.py
