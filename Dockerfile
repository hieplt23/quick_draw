FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxcb1 \
    libxrender1 \
    libxi6 \
    libqt5x11extras5 \
    qtbase5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install opencv-python scikit-learn tensorboard

COPY painting_app.py  /workspace
COPY trained_models /workspace/trained_models
COPY images /workspace/images
COPY src /workspace/src

#WORKDIR /workspace
#
#CMD ["python", "train.py"]