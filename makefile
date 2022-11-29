reinstall_package:
	@pip uninstall -y FungAI || :
	@pip install -e .

run_preprocess:
	python -c 'from FungAI.interface.main import preprocess; print(preprocess());'

run_model:
	python -c 'from FungAI.interface.main import train; train();'
