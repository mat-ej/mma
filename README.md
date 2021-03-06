# MMA project using ploomber MLops
Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

- pipeline-old.yaml - current project pipeline
- alt-pipeline.yaml - pipeline to analyze alternative MMA markets e.g.: KO, DEC, SUBMISSION
- pipeline.yaml - dev pipeline

## Ploomber notes
{{here}}: Absolute path to the parent folder of pipeline.yaml

{{cwd}}: Absolute path to the current working directory

{{root}}: Absolute path to project’s root folder. It is usually the same as {{here}}, except when the project is a package (i.e., it has setup.py file), in such a case, it points to the parent directory of the setup.py file.

{{user}}: Current username

{{now}}: Current timestamp in ISO 8601 format (Added in Ploomber 0.13.4)

A common use case for this is when passing paths to files to scripts/notebooks. For example, let’s say your script has to read a file from a specific location. Using {{here}} turns path into absolute so you can ready it when using Jupyter, even if the script is in a different location than your pipeline.yaml.

By default, paths in tasks[*].product are interpreted relative to the parent folder of pipeline.yaml. You can use {{cwd}} or {{root}} to override this behavior:


## NOTES:
    ploomber build --partially task
    
    /home/jovyan/shared-volume/conda/envs/mma
    conda env export --no-build --file environment.lock.yml

build until task C

## Setup development environment

```sh
# configure dev environment
ploomber install


# ...or use conda directly
conda env create --file environment.yml

# activate conda environment
conda activate mma
```

## Running the pipeline

```sh
ploomber build

# start an interactive session
ploomber interact
```

## Exporting to other systems

[soopervisor](https://soopervisor.readthedocs.io/) allows you to run ploomber projects in other environments (Kubernetes, AWS Batch, AWS Lambda and Airflow). Check out the docs to learn more.# mma
