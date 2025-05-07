FROM icr.io/quantum-public/qiskit-serverless/ray-node:0.21.1

# install all necessary dependencies for your custom image

# copy our function implementation in `/runner/runner.py` of the docker image
USER 0
RUN pip install -r chemistry/sqd_pcm/requirements.txt

WORKDIR /runner
COPY chemistry/sqd_pcm /runner
WORKDIR /


USER 1000