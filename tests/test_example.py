"""Example test file to demonstrate test structure for ML projects."""

import pytest


def test_example():
    """Basic example test that always passes."""
    assert 1 + 1 == 2


def test_data_processing_example():
    """Example test for data processing functions.

    Replace this with actual tests for your preprocessing functions.
    """
    # Example: test that data cleaning removes nulls
    # sample_data = [1, 2, None, 4, 5]
    # cleaned_data = clean_data(sample_data)
    # assert None not in cleaned_data
    assert True  # Placeholder


def test_model_prediction_example():
    """Example test for model predictions.

    Replace this with actual tests for your model.
    """
    # Example: test that model predictions are in valid range
    # model = load_model()
    # prediction = model.predict([[1, 2, 3]])
    # assert 0 <= prediction <= 1
    assert True  # Placeholder


@pytest.fixture
def sample_dataset():
    """Example fixture providing sample data for tests.

    Fixtures are reusable test data/setup that can be shared across tests.
    """
    return {"features": [[1, 2], [3, 4], [5, 6]], "labels": [0, 1, 0]}


def test_with_fixture(sample_dataset):
    """Example test using a fixture."""
    assert len(sample_dataset["features"]) == len(sample_dataset["labels"])
