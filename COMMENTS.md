> It wont work on a Linux container on Mac-M1 (arm64 chips). It'll ask for Linux-aarm64 and no PyTorch wheel is available.

When the devcontainer is spun off, it asks for a Debian with Python 3.10, for the system's architecture. This probably works because microsoft (who publishes the image downloaded) uploads multiplatform, multi architecture.For example, [the python images](https://hub.docker.com/r/microsoft/devcontainers-python) or [the linux base images](https://hub.docker.com/r/microsoft/devcontainers-base) to the Hub.

You can try [amd64 simulation](https://docs.docker.com/desktop/troubleshoot-and-support/troubleshoot/known-issues/). Then it will download Linux/X86_64. In both cases though, it won't use MPS, which is software available within MacOS-only.

Currently the best case 
See also a related pytorch [issue for MPS enabling docker image](https://github.com/pytorch/pytorch/issues/81224)