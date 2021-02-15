
NAMES = baseline vae pixelvae aegan vaegan

all: $(NAMES)

$(NAMES):
	papermill template.ipynb $@_results.ipynb -f model-parameters/$@.yml
	jupyter nbconvert --to PDF $@_results.ipynb

