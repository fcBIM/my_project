from my_project.model import MyAwesomeModel

def test_model():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
    assert len(train_dataset) == N_train, "Dataset did not have the correct number of samples"