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
Build pipeline with:
```sh
    ploomber build --entry-point pipeline/pipeline.yaml
```

