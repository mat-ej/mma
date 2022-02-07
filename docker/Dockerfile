# Let's use the image base-notebook to build our image on top of it
FROM jupyter/base-notebook

LABEL Matej Uhrin <uhrinmat@fel.cvut.cz>

# Let's change to root user to install java 8
USER root

# Install OpenJDK-11
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;
# Install kotlin kernel for jupyter using conda
#RUN conda install -y -c jetbrains kotlin-jupyter-kernel && echo "Kotlin Jupyter kernel installed via conda"

# Let's change to  "$NB_USER" command so the image runs as a non root user by default
USER $NB_UID

#Let's define this parameter to install jupyter lab instead of the default juyter notebook command so we don't have to use it when running the container with the option -e
ENV JUPYTER_ENABLE_LAB=yes