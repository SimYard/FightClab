FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive

COPY docker/apt-requirements.txt /apt-requirements.txt
# hadolint ignore=SC2046
RUN apt-get update && \
    apt-get install --no-install-recommends -y $(cat /apt-requirements.txt) && \
    rm -rf /var/lib/apt/lists/* /apt-requirements.txt

# The base image ships user "ubuntu" with uid 1000; replace it so files in the mounted repo keep the host owner.
ARG USER=dev
ARG UID=1000
RUN userdel -r ubuntu && \
    useradd -l -m -u $UID -s /bin/bash $USER && \
    echo "$USER ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER

USER $USER
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/home/$USER/venv \
    MUJOCO_GL=egl
WORKDIR /home/$USER/fightclab

# Dependencies only: the repo itself is mounted at runtime (see docker-compose.yml).
RUN --mount=type=cache,target=/home/dev/.cache/uv,uid=1000 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-install-project

CMD ["bash"]
