reinstall_package:
	@pip uninstall -y FungAI || :
	@pip install -e .

run_preprocess:
	python -c 'from FungAI.interface.main import preprocessor; preprocessor();'

run_train:
	python -c 'from FungAI.interface.main import train; train();'

run_eval:
	python -c 'from FungAI.interface.main import evaluate; evaluate();'
