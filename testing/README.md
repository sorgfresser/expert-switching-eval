# Setup

# Requirements
Install the requirements for the test suite.

```bash
pip3 install -r requirements.txt
```

## Pythonpath
Set the pythonpath to include the current directory and the wandb directory. This is necessary to import the modules in the test suite.

```bash
export PYTHONPATH=".:../../wandb/:../../wandb/test_suite_sql_eval"
```

