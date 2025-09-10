# hash:sha256:517b2480acbdaf948b423e6c9209716d5e2767198c779763f6c4683410403ca4
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        jq=1.6-2.1ubuntu3.1 \
        ninja-build=1.10.1-1 \
        wget=1.21.2-2ubuntu1.1 \
        zip=3.0-12build2 \
    && rm -rf /var/lib/apt/lists/*

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
