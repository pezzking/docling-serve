ARG BASE_IMAGE=quay.io/sclorg/python-312-c9s:c9s

ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.19

ARG UV_SYNC_EXTRA_ARGS=""

FROM ${BASE_IMAGE} AS docling-base

###################################################################################################
# OS Layer                                                                                        #
###################################################################################################

USER 0

RUN --mount=type=bind,source=os-packages.txt,target=/tmp/os-packages.txt \
    dnf -y install --best --nodocs --setopt=install_weak_deps=False dnf-plugins-core epel-release && \
    dnf config-manager --best --nodocs --setopt=install_weak_deps=False --save && \
    dnf config-manager --enable crb && \
    dnf -y update && \
    dnf install -y $(cat /tmp/os-packages.txt) && \
    dnf -y clean all && \
    rm -rf /var/cache/dnf

RUN /usr/bin/fix-permissions /opt/app-root/src/.cache

ENV TESSDATA_PREFIX=/usr/share/tesseract/tessdata/

FROM ${UV_IMAGE} AS uv_stage

###################################################################################################
# Docling layer                                                                                   #
###################################################################################################

FROM docling-base

USER 1001

WORKDIR /opt/app-root/src

ENV \
    OMP_NUM_THREADS=4 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/app-root \
    DOCLING_SERVE_ARTIFACTS_PATH=/opt/app-root/src/.cache/docling/models \
    LD_PRELOAD=/usr/lib64/libjemalloc.so.2 \
    MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000

ARG UV_SYNC_EXTRA_ARGS

RUN --mount=from=uv_stage,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/opt/app-root/src/.cache/uv,uid=1001 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    umask 002 && \
    echo "UV_SYNC_EXTRA_ARGS=${UV_SYNC_EXTRA_ARGS}" && \
    UV_SYNC_ARGS="--frozen --no-install-project --no-dev --extra ui --extra easyocr --extra rapidocr" && \
    echo "Running: uv sync ${UV_SYNC_ARGS} ${UV_SYNC_EXTRA_ARGS}" && \
    uv sync ${UV_SYNC_ARGS} ${UV_SYNC_EXTRA_ARGS}

# Models are mounted via PVC at runtime - skip downloading to reduce image size
# ARG MODELS_LIST="layout tableformer picture_classifier rapidocr easyocr"
# RUN echo "Downloading models..." && \
#     HF_HUB_DOWNLOAD_TIMEOUT="90" \
#     HF_HUB_ETAG_TIMEOUT="90" \
#     docling-tools models download -o "${DOCLING_SERVE_ARTIFACTS_PATH}" ${MODELS_LIST} && \
#     chown -R 1001:0 ${DOCLING_SERVE_ARTIFACTS_PATH} && \
#     chmod -R g=u ${DOCLING_SERVE_ARTIFACTS_PATH}

COPY --chown=1001:0 ./docling_serve ./docling_serve
COPY --chown=1001:0 ./custom-wheels ./custom-wheels

RUN --mount=from=uv_stage,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/opt/app-root/src/.cache/uv,uid=1001 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    umask 002 && \
    echo "Running final sync: uv sync --frozen --no-dev --extra ui --extra easyocr --extra rapidocr ${UV_SYNC_EXTRA_ARGS}" && \
    uv sync --frozen --no-dev --extra ui --extra easyocr --extra rapidocr ${UV_SYNC_EXTRA_ARGS}

# Install custom docling wheel (overrides the PyPI version)
# Use --no-deps to avoid re-resolving dependencies (which would pull GPU torch)
RUN --mount=from=uv_stage,source=/uv,target=/bin/uv \
    if ls ./custom-wheels/*.whl 1> /dev/null 2>&1; then \
        uv pip install --no-deps --force-reinstall ./custom-wheels/*.whl; \
    fi

EXPOSE 5001

CMD ["docling-serve", "run"]
