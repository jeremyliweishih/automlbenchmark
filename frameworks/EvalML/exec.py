import logging
import math
import os
import shutil
import tempfile as tmp
import warnings

# os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'


from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, system_memory_mb, walk_apply, zip_path

log = logging.getLogger(__name__)

from evalml import AutoMLSearch, __version__
from evalml.problem_types import detect_problem_type
from frameworks.shared.callee import (call_run, output_subdir, result,
                                      save_metadata)
import pandas as pd

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** EvalML ****\n")
    save_metadata(config, version=__version__)

    is_classification = config.type == 'classification'
    
    X_train = dataset.train.X
    y_train = dataset.train.y.iloc[:, 0]
    

    problem_type = detect_problem_type(y_train)

    metrics_mapping = None
    if problem_type.value == "binary":
        metrics_mapping = dict(
            acc='accuracy binary',
            auc='auc',
            f1='f1',
            logloss='log loss binary',
            mae='mean absolute percentage error',
            mse='root mean squared error',
            msle='mean squared log error',
            r2='r2',
            rmse='root mean squared error',
        )
    else:
        metrics_mapping = dict(
            acc='accuracy multiclass',
            auc='auc',
            f1='f1',
            logloss='log loss multiclass',
            mae='mean absolute percentage error',
            mse='root mean squared error',
            msle='mean squared log error',
            r2='r2',
            rmse='root mean squared error',
        )

    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    n_jobs = config.framework_params.get('_n_jobs', config.cores)  # useful to disable multicore, regardless of the dataset config

    log.info('Running EvalML with a maximum time of %ss on %s cores, optimizing %s.',
             config.max_runtime_seconds, n_jobs, scoring_metric)

    automl = AutoMLSearch(
                          X_train=X_train,
                          y_train=y_train,
                          problem_type=problem_type, 
                          objective=scoring_metric,
                          max_time=config.max_runtime_seconds, 
                          random_seed=config.seed, 
                          **training_params)

    with Timer() as training:
        automl.search()

    log.info('Predicting on the test set.')
    X_test = dataset.test.X
    y_test = dataset.test.y

    best_pipeline = automl.best_pipeline
    best_pipeline.fit(X_train, y_train)
    
    with Timer() as predict:
        predictions = best_pipeline.predict(X_test)
    
    probabilities = best_pipeline.predict_proba(X_test) if is_classification else None

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=False,
                  models_count=len(automl.full_rankings),
                  training_duration=training.duration,
                  predict_duration=predict.duration)

if __name__ == '__main__':
    call_run(run)
