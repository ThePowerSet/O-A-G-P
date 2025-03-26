# 🛠 Repository Setup Summary (Mac M1 + TensorFlow + GPflow)

## Overview

This markdown summarizes the steps taken to make the repository work on a Mac silicon, along with the issues encountered and how they were fixed.

---
- python --version : Python 3.8.10

## ✅ What Was Done
- Installed the required packages for Mac M1 GPU support:
  - `tensorflow-macos`
  - `tensorflow-metal`
  - `gpflow`
  - `tensorboard` (installed manually to remove warnings)

- Resolved version conflicts between TensorFlow, Keras, and other dependencies.

- Tests were run using `pytest` from the **root folder** and passed (only some deprecation warnings appeared, not critical).

---

## ⚠️ Issues and Fixes

- **TensorFlow was not using the GPU properly**, even though it recognized it. A warning like this appeared:
  > Created TensorFlow device (/device:GPU:0 with 0 MB memory)...

  This is a known limitation of Metal on macOS. No fix needed unless performance becomes an issue.

- **Gpflow is not correctely being imported**, remember to import first tensorflow and then gpflow

- **TensorBoard missing warnings**, despite being installed. These can be ignored after manual installation.

- **Tests failed when running from `/tests` folder**: likely due to the wrong Python version (e.g., 3.12 used instead of the virtual environment).

  **Fix**: Always run `pytest` from the project root (do not go in the test folder) and ensure the correct environment is active.

- **Environment issues**: Sometimes the terminal says the environment is active, but it's not.

  **Fix**: Kill the terminal, open a new one, and manually reactivate:
  ```bash
  source .venv/bin/activate

