# Writers Block

This repository contains a model which can evaluate semantic proximity of given text
to Hemingways' style.

Codebase consists of two parts: training and serving

## Launching serving part locally

```bash
# cloning repo
git clone git@github.com:kstolz/writers_block.git
cd writers_block

# installing prerequisites
pip install

# launching flask server locally
export FLASK_APP=main.py
flask run

# open your browser and point it to http://127.0.0.1:5000/
```