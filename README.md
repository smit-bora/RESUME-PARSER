This is Resume parser using bert, very initial stage. Only parsing resume part exists here. 

Clone the repo.

Prerequisites

- [Python 3.11+](https://www.python.org/)
- [Conda](https://docs.conda.io/en/latest/) (recommended) or `venv`
- [Node.js + npm](https://nodejs.org/) (for frontend)

Run:

1. Create and activate environment (only once):(just open cmd and run this first step in the base folder itself)(provided you have conda installed and added to the path as env variable)

  conda create -n resume-parser python=3.11
  
  conda activate resume-parser

2. then navigate to the backend folder and run pip install -r requirements.txt
3. python run.py
Backend will start on: http://127.0.0.1:8000

upload your own resume. It is not perfect and gets half the things correctly and the other half is incorrect.
