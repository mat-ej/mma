# mma
Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

## NOTES:
    ploomber build --partially task
    
    /home/jovyan/shared-volume/conda/envs/mma

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

Default placeholders¶
There are a few default placeholders you can use in your pipeline.yaml, even if not defined in the env.yaml (or if you don’t have a env.yaml altogether)

{{here}}: Absolute path to the parent folder of pipeline.yaml

{{cwd}}: Absolute path to the current working directory

{{root}}: Absolute path to project’s root folder. It is usually the same as {{here}}, except when the project is a package (i.e., it has setup.py file), in such a case, it points to the parent directory of the setup.py file.

{{user}}: Current username

{{now}}: Current timestamp in ISO 8601 format (Added in Ploomber 0.13.4)

A common use case for this is when passing paths to files to scripts/notebooks. For example, let’s say your script has to read a file from a specific location. Using {{here}} turns path into absolute so you can ready it when using Jupyter, even if the script is in a different location than your pipeline.yaml.

By default, paths in tasks[*].product are interpreted relative to the parent folder of pipeline.yaml. You can use {{cwd}} or {{root}} to override this behavior:

## Running the pipeline

```sh
ploomber build

# start an interactive session
ploomber interact
```

## Exporting to other systems

[soopervisor](https://soopervisor.readthedocs.io/) allows you to run ploomber projects in other environments (Kubernetes, AWS Batch, AWS Lambda and Airflow). Check out the docs to learn more.# mma
