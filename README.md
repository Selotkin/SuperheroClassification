# SuperHero classification with Tensorflow
![alt text](https://cdn.thedesigninspiration.com/wp-content/uploads/2009/04/civil-war-by-leinilyu.jpg)

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
