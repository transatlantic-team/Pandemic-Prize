
Xprize Codalab local testing


Usage:
python scoring_program/evaluate.py scoring_input scoring_output


DIRECTORIES
===========

scoring_program/
	evaluate.py


scoring_input/
	ref/ # this contains the input data
		OxCGRT_latest.csv
		historical_ip.csv
		readme.md
		training.csv
	res/ # this contains your submission
		predict.py
		train.py
		transatlantic/

scoring_output/ # where the results will end up


SAMPLE SUBMISSIONS READY TO GO
================================
trained_sample_submission.zip
untrained_same_submission.zip

Those can be submitted to:
Secret url: https://competitions.codalab.org/competitions/27794?secret_key=642ab409-1332-450b-a58a-7f6834dd5be6


EXPECTED BEHAVIOR
=================

After running:
python scoring_program/evaluate.py scoring_input scoring_output

the scoring_output directory will contain:
	detailed_results.html
	models.zip
	predictions30.csv
	scores.txt
Unfortunately the zip file does not contain the models, it seems corrupted, we need to check that

But a new directory models/
gets created in res/
containing the trained model

HOW TO CREATE A SUBMISSION WITH A TRAINED MODEL
===============================================

To create a submission containing the trained model, create a zip archive with only:
		predict.py
		transatlantic/
		models/
Do not include train.py; do not zip the top directory.



