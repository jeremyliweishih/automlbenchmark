from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.utils import call_script_in_same_dir, unsparsify


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    X_train, X_test = dataset.train.X, dataset.test.X
    y_train, y_test = dataset.train.y, dataset.test.y
    data = dict(
        train=dict(
            X=X_train,
            y=y_train
        ),
        test=dict(
            X=X_test,
            y=y_test
        )
    )

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config)

