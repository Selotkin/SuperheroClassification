"""Peform hyperparemeters search"""

import argparse
import os

from helpers.utils import Params
from helpers.train_pipeline import tpipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model',  default='model_1',
                    help="Model in models directory")
parser.add_argument('--experiment', default='learning_rate',
                    help="Experiment in experiments directory wrt model")
parser.add_argument('--data_dir', default='data//64x64_SuperHeroes',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

if __name__ == "__main__":
    # Load the "reference" parameters from model_dir json file
    args = parser.parse_args()

    experiment_dir = 'experiments/{}/{}'.format(args.model, args.experiment)

    json_path = os.path.join(experiment_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Perform hypersearch over one parameter
    learning_rates = [3e-5, 3e-4, 3e-3]

    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate

        # Set job name
        args.job_name = "learning_rate_{}".format(learning_rate)

        # Create a new folder in parent_dir with unique_name "job_name"
        # If folder exists try to get params from there
        experiment_dir = os.path.join(experiment_dir, args.job_name)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
            # Write parameters in json file
            json_path = os.path.join(experiment_dir, 'params.json')
            params.save(json_path)

        # Start training pipeline
        tpipeline(args)
