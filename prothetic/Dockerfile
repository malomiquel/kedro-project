ARG BASE_IMAGE=python:3.9-slim
FROM $BASE_IMAGE as runtime-environment

# Install necessary build dependencies
RUN apt-get update && \
    apt-get install -y gcc python3-dev pkg-config libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# install project requirements
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -U "pip>=21.2,<23.2"
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

# Copy the whole project except what is in .dockerignore
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

# Install Java and set JAVA_HOME
USER root
RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update && \
    apt-get install -y default-jdk && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

EXPOSE 8888

CMD ["python", "app.py"]
