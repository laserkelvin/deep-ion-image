
NAMES = baseline vae pixelvae aegan vaegan
#PROB = vae pixelvae vaegan

all: common probabilistic report

$(NAMES):
	papermill template.ipynb $@_results.ipynb -f model-parameters/$@.yml
	jupyter nbconvert --to PDF $@_results.ipynb

common: $(NAMES)

probabilistic: $(PROB)

$(PROB):
	papermill probabilistic_test.ipynb $@_uncertainty_tests.ipynb -f model-parameters/$@.yml
	jupyter nbconvert --to PDF $@_uncertainty_tests.ipynb

report:
	pandoc model_report.md -o model_report.pdf --template=eisvogel --pdf-engine=xelatex


