FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y git curl wget unzip vim
RUN git config --global --add safe.directory /workspace

COPY . .

ENV PYTHONPATH "${PYTHONPATH}:/src"
CMD ["/bin/bash"]