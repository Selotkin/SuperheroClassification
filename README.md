# SuperHero classification with Tensorflow


## Data

You can download data. See instructions [here](https://cs230-stanford.github.io/project-starter-code.html).

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.

## Task

Given an image of a superhero recognize superhero name. The model was trained on 5 superheroes (Groot, Hulk, Ironman, Spider-Gwen and Spiderman)


## Download the dataset

The data set consists of 759 images related to 5 different classes. You can download the dataset [here] (https://drive.google.com/open?id=13qSvfXAcNJY7SYzrQhOzicuPO5kzmZVN).

This will download the SuperHeroes dataset (~86 MB) containing images of different superheroes (Groot, Hulk, Ironman, Spider-Gwen and Spiderman).
Here is the structure of the data:
```
SuperHeroes/
    {label}_{ID}.png
    ...
```

The images are named following `{label}_{ID}.png` where the label are `[Groot, Hulk, Ironman, Spider-Gwen and Spiderman]`.

Once the download is complete, move the dataset into `data/SuperHeroes`.
Run the script `build_dataset.py` which will resize the images to size `(64, 64)` and copy them 5 times. The new resized dataset will be located by default in `data/64x64_SuperHeroes`:
