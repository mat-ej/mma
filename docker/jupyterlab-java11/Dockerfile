FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-scipy:v1.4

# Java installation
USER root
RUN apt-get update && apt-get install -y openjdk-11-jre-headless && apt-get clean


# pygraphviz
#TODO install environment.lock.yml, htop
USER ${NB_UID}