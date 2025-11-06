# Welcome to the Team!

Glad to have you onboard for our predictive modeling project! During these early weeks while you're getting familiar, here are a few things I'd like you all to brush up on:

## 1. Linear Regression

For this project, I plan on starting with **Linear Regression** as it is a fundamental part of machine learning and also a lot easier to grasp and understand. 

**Resource:** [Codecademy - Simple Linear Regression Course](https://www.codecademy.com/learn/simple-linear-regression-course) (free!)

## 2. Git/GitHub

I can't stress this one enough. Understanding how to use Git is absolutely fundamental — so much that I actually want to take the time during our next standup to show you all the workflow we're looking for. 

**IF YOU'RE UNFAMILIAR WITH GIT:**
Please check out this resource, it's a great interactive tool that will teach you what "git" is and how crucial it is in our project - [Learn git branching](https://learngitbranching.js.org/?locale=en_US)


**General workflow:**
```
create feature branch → make edits → pull from main → open pull request → merge → repeat
```

*Don't forget to pull from main regularly to keep up to date*

**Important: Commit Message Standards**

Please follow this format for all commit messages:
```
[COMMIT TYPE]: [COMMIT MESSAGE]
```

**Commit types:**
- `feat` - New features
- `fix` - Bug fixes
- `doc` - Documentation changes
- `refactor` - Code refactoring
- `test` - Test additions or modifications

**Example:** `feat: add data preprocessing pipeline`

## 3. Python

Pretty obvious and not too hard. There's lots of great resources out there, but this one is a personal favorite of mine:

**Resource:** [futurecoder.io](https://futurecoder.io)

## 4. CI/CD and Code Quality Tools

We have automated code quality checks that run on every pull request. Here's what you need to know:

### What Gets Checked Automatically

Every time you open a PR, GitHub Actions will automatically run:
- **Ruff format** - Ensures code formatting consistency
- **Ruff lint** - Checks code quality and style
- **mypy** - Type checking for better code reliability
- **bandit** - Security scanning
- **pytest** - Runs all tests

### Setting Up Your Local Environment

Install the development tools:
```bash
pip install ruff mypy pytest bandit pre-commit
```

Install pre-commit hooks (highly recommended - runs checks before each commit):
```bash
pre-commit install
```

### Running Checks Locally

Before pushing your code, you can run these commands to catch issues early:

```bash
ruff format .          # Auto-format your code
ruff check --fix .     # Lint and auto-fix issues
mypy .                 # Check types
pytest                 # Run tests
```

Or run all pre-commit hooks manually:
```bash
pre-commit run --all-files
```

### Writing Tests

All tests go in the `tests/` directory. Example test structure:

```python
def test_data_preprocessing():
    """Test that preprocessing removes null values."""
    raw_data = load_sample_data()
    processed = preprocess(raw_data)
    assert not processed.isnull().any().any()
```

For ML projects, focus on testing:
- Data preprocessing functions
- Model input/output validation
- Feature engineering logic
- Edge cases and data quality

This is to be updated...

## 5. What's Next?

TBD
