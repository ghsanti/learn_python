# Dev container

Configures dev containers. These initialize the development environment in **Codespaces** or **VS Code**; it includes dev dependencies (linter, formatter etc.)

- **GitHub Codespaces**. Create a codespace for the repo by clicking
  the "Code" button on the main page of the repo.

- **Visual Studio Code**. Open the root folder of the repo in VS Code. A
  notification will pop up to open it in a dev container with the
  configuration.

  After the install, you need to issue:

  ```bash
  source .venv/bin/activate
  ```

  And select the virtual environment's python interpreter. Using `Ctrl+Shift+P` VSCode command palette should let you choose it (the one in `.venv`, currently **3.10.15**).


  ## Installing software

  Your terminal is now in the container's OS. It's Ubuntu, you can add anything like:

```bash
sudo apt-get install git
```

Or one can directly modify the _devcontainer.json_'s features key. This way you
don't have to install things again when the container is rebuilt. Same is achieved
using a postcreate command. We have both ways within the file as example.
