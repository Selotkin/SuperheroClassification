"""Train the model"""

import argparse

from helpers.train_pipeline import tpipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model_1',
                    help="Model in models directory")
parser.add_argument('--experiment', default='1',
                    help="Experiment in experiments directory wrt model")
parser.add_argument('--job_name', default='',
                    help="Job name")
parser.add_argument('--data_dir', default='data\\64x64_SuperHeroes',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()

    # Start training pipeline
    tpipeline(args)
