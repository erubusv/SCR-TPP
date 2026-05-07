FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl wget unzip vim tmux procps \
    && rm -rf /var/lib/apt/lists/*
RUN git config --global --add safe.directory /workspace

COPY . .

ENV PYTHONPATH=/workspace
ENV OMP_NUM_THREADS=12
ENV MKL_NUM_THREADS=12
ENV OPENBLAS_NUM_THREADS=12
ENV NUMEXPR_NUM_THREADS=12

CMD ["/bin/bash"]
