FROM ubuntu:23.10

ARG CI
ENV CI ${CI}

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --assume-yes \
        build-essential \
        python3 \
        python3-dev \
        pipx \
        bash \
        ffmpeg \
        libsm6 \
        libxext6

COPY . /workspace

RUN /workspace/scripts/build.sh \
    && /workspace/scripts/test.sh

ENV CI=

ENTRYPOINT [ "/workspace/scripts/run.sh" ]
