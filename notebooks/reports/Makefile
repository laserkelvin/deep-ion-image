
baseline:
	papermill template.ipynb baseline_results.ipynb \
	  -f model-parameters/baseline.yml

vae:
	papermill template.ipynb vae_results.ipynb \
	  -f model-parameters/vae.yml

pixelvae:
	papermill template.ipynb pixelvae_results.ipynb \
	  -f model-parameters/pixelvae.yml

all:
	$(baseline) $(vae) $(pixelvae)
