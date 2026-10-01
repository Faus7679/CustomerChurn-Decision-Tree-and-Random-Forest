import pandas as pd

from analyze_churn import make_preprocessor, make_models, prepare_data


def test_prepare_data_removes_identifier_and_converts_total_charges():
    frame = pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "tenure": [1, 2],
            "TotalCharges": ["10.5", " "],
            "Churn": ["Yes", "No"],
        }
    )
    features, target = prepare_data(frame)
    assert "customerID" not in features
    assert features["TotalCharges"].iloc[0] == 10.5
    assert pd.isna(features["TotalCharges"].iloc[1])
    assert target.tolist() == [1, 0]


def test_model_factory_has_all_comparable_models():
    features = pd.DataFrame({"tenure": [1, 2], "contract": ["Month-to-month", "Two year"]})
    models = make_models(make_preprocessor(features))
    assert set(models) == {
        "Logistic regression",
        "k-nearest neighbours",
        "Decision tree",
        "Random forest",
    }
    assert all("preprocess" in model.named_steps for model in models.values())
