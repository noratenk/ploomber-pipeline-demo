# ploomber-pipeline-demo

A demo project using ploomber to demonstate data science pipeline using Jupyter notebooks as building blocks.

#### At the first time run:
```sh
    conda env create --file environment.yml
```
This will create an anaconda environment called ploomber-demo.
```sh
    pip install -r requirements.txt
```
#### When your environment is ready:
```sh
    conda activate ploomber-demo
```
Run pipeline with:
```sh
    python pipeline/run.py
```

## Resources:
- [Ploomber example projects on GitHub](https://github.com/ploomber/projects)
- [Execute Jupyter notebook as a pipeline step, TDS](https://towardsdatascience.com/execute-jupyter-notebook-as-a-pipeline-step-4dba8c45aebf)
- [Ploomber: Maintainable and Collaborative Pipelines in Jupyter, Jupyter Blog](https://blog.jupyter.org/ploomber-maintainable-and-collaborative-pipelines-in-jupyter-acb3ad2101a7)
- [Data Science Project Folder Structure](https://dzone.com/articles/data-science-project-folder-structure)