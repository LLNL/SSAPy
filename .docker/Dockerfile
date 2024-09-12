FROM quay.io/centos/centos:stream8

ARG BUILD_DATE
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="SSAPy Docker image" \
      org.label-schema.description="Provides an environment containing a working copy of SSAPy." \
      org.label-schema.vendor="LLNL"

# Copy over SSAPy
COPY ssapy/ /opt/ssapy/

# Install C/C++ compilers
RUN dnf -y update && \
    dnf groupinstall -y "Development Tools" && \
    dnf -y clean all && \
    gcc --version

# Install necessary cmake packages
RUN dnf install -y make cmake3 tree wget vi && \
    dnf -y clean all && \
    make --version && \
    cmake --version && \
    ls -l /

# Install the necessary python/pip packages
RUN dnf install -y wget python3.12 python3.12-devel && \
    dnf -y clean all && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python --version && \
    python -m pip --version && \
    ls -l /usr/bin/ && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --upgrade setuptools && \
    python -m pip install --no-cache-dir -r /opt/ssapy/requirements.txt && \
    ls -l /

# Install SSAPY
RUN ls -la / && \
    ls -la /opt && \
    ls -la /opt/ssapy && \
    tree -r /opt/ssapy && \
    cd /opt/ssapy && \
    python setup.py build && \
    python setup.py install && \
    chmod -R go+rwX ./ && \
    chmod -R go+rwX /usr/lib64/python3.12/site-packages/ && \
    chmod -R go+rwX /usr/local/lib64/python3.12/site-packages/
