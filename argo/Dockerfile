FROM condaforge/mambaforge:4.10.1-0

COPY environment.lock.yml project/environment.lock.yml


RUN mamba env update --name base --file project/environment.lock.yml && conda clean --all --force-pkgs-dir --yes


COPY dist/* project/
WORKDIR /project/

# extract to get any config files at the root
RUN tar --strip-components=1 -zxvf *.tar.gz
RUN cp -r /project/ploomber/ /root/.ploomber/  || echo 'ploomber home does not exist'

