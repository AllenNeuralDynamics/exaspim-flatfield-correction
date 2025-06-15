# hash:sha256:8bc506b306edd8125f91744a1d3da06471541d2cd291145d6747b284cc872280
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2 \
    && rm -rf /var/lib/apt/lists/*

COPY postInstall /
RUN chmod +x /postInstall
RUN /postInstall
