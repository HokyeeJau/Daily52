# Daily52

This project hosts the Daily52 dataset for the CIKM'26 resource paper ***[Daily52: An Air-exposed Smell Dataset for Real-world Odor Classification](https://dl.acm.org/doi/abs/10.1145/3799682.3840178)***.

# Usage

## Loading NumPy Arrays with Numeric Labels

```python
from dataloader import load_dataset 

data = load_dataset(backend="numpy", label_format = "index")
print(data['X'].shape, data['y'].shape, type(data['X']), type(data['y'][0]))
```
Output:
```bash 
>>> (530, 18, 535) (530,) <class 'numpy.ndarray'> <class 'numpy.int64'>
```

## Loading NumPy Arrays with String Labels

```python
data = load_dataset(backend="numpy", label_format = "name")
print(data['X'].shape, data['y'].shape, type(data['X']), type(data['y'][0]))
```
Output:
```bash 
>>> (530, 18, 535) (530,) <class 'numpy.ndarray'> <class 'numpy.str_'>
```

## 	Loading PyTorch Tensors with Numeric Labels

```python
data = load_dataset(backend="torch", label_format="index")
print(data['X'].shape, data['y'].shape, type(data['X']), type(data['y'][0]))
```
Output:
```bash 
>>> torch.Size([530, 18, 535]) torch.Size([530]) <class 'torch.Tensor'> <class 'torch.Tensor'>
```

## Loading PyTorch Tensors with String Labels

```python
data = load_dataset(backend="torch", label_format="name")
print(data['X'].shape, len(data['y']), type(data['X']), type(data['y'][0]))
```
Output:
```bash 
>>> torch.Size([530, 18, 535]) 530 <class 'torch.Tensor'> <class 'str'>
```


# References
If you find this dataset useful for your research, please cite the following paper in your publication:
```bibtex 
@inproceedings{Daily52,
author = {Zhou, Xueyi and Lu, Qi and Chae, Dong-Kyu},
title = {Daily52: An Air-exposed Smell Dataset for Real-world Odor Classification},
year = {2026},
isbn = {9798400725395},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3799682.3840178},
doi = {10.1145/3799682.3840178},
booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
pages = {9061–9066},
numpages = {6},
keywords = {olfactory computing, gas-sensing dataset, electronic nose},
location = {Rome, Italy},
series = {CIKM '26}
}
```


# Existing Gas-sensing Datasets

Comparison between existing gas-sensing datasets (for classification) and ours. The variables in the table: $N$, the
number of categories; $C$, the number of gas sensors; $M$, the number of samples; $A$, if exposed to the air. Note that,
the number of categories does not count the number of concentration levels.

| Category          | Source                                                                                                                                                                                                                                                                          | Year | $N$ | $C$ | $M$    | $A$ | Task                                            |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|-----|-----|--------|-----|-------------------------------------------------|
| Chemical          | [link](https://doi.org/10.24432/C5MK6M)                                                                                                                                                                                                                                         | 2013 | 6   | 16  | 13,910 | N   | Chemical Gas Classification                     |
| Chemical          | [link](https://doi.org/10.24432/C5JP5N)                                                                                                                                                                                                                                         | 2013 | 10  | 72  | 18,000 | N   | Chemical Gas Classification                     |
| Chemical          | [link](https://doi.org/10.24432/C5JS5P)                                                                                                                                                                                                                                         | 2014 | 30  | 8   | 180    | N   | Turbulent Gas Mixture Classification            |
| Chemical          | [link](https://data.csiro.au/collection/csiro:13032)                                                                                                                                                                                                                            | 2015 | 20  | 12  | 200    | N   | Chemical Classification                         |
| Chemical          | [link](https://doi.org/10.24432/C5MW3K)                                                                                                                                                                                                                                         | 2016 | 4   | 8   | 640    | N   | Chemical Classification                         |
| Chemical          | [link](https://dx.doi.org/10.21227/kgv5-sk55)                                                                                                                                                                                                                                   | 2022 | 14  | 41  | 210    | N   | Chemical Classification                         |
| Chemical          | [link](https://dx.doi.org/10.21227/75j4-nx17)                                                                                                                                                                                                                                   | 2023 | 10  | 15  | 240    | N   | Gas Classification                              |
| Chemical          | [link](https://dx.doi.org/10.21227/8e69-pe98)                                                                                                                                                                                                                                   | 2025 | 11  | 8   | 112    | N   | Chemical Classification                         |
| Safety & Security | [link](https://dx.doi.org/10.21227/exkq-c076)                                                                                                                                                                                                                                   | 2019 | 2   | 6   | 420    | N   | Mixed Explosive Detection                       |
| Safety & Security | [link](https://data.mendeley.com/datasets/wfn8z3pvhv/1)                                                                                                                                                                                                                         | 2020 | 2   | 8   | 75     | N   | Children Toy Contamination Detection            |
| Food/Beverages    | [link](https://data.mendeley.com/datasets/7spd6fpvyk/1)                                                                                                                                                                                                                         | 2009 | 3   | 8   | 30     | N   | Colombian Coffee Quality Monitoring             |
| Food/Beverages    | [link](https://data.mendeley.com/datasets/compare/mwmhh766fc)                                                                                                                                                                                                                   | 2018 | 4   | 11  | 48     | N   | Beef Cut Quality Monitoring                     |
| Food/Beverages    | [link](https://data.mendeley.com/datasets/vpc887d53s/1)                                                                                                                                                                                                                         | 2019 | 3   | 6   | 300    | N   | Wine Spoilage Detection                         |
| Food/Beverages    | [link](https://www.sciencedirect.com/science/article/pii/S221418042030101X)                                                                                                                                                                                                     | 2020 | 6   | 9   | 249    | N   | Meat Cut Classification                         |
| Food/Beverages    | [link](https://data.mendeley.com/datasets/5yhggs7zy7/1)                                                                                                                                                                                                                         | 2020 | 7   | 8   | 420    | N   | Pork Adulteration Detection                     |
| Food/Beverages    | [link](https://dx.doi.org/10.21227/7zch-5585)                                                                                                                                                                                                                                   | 2021 | 8   | 10  | 640    | N   | Chinese Liquor Classification                   |
| Food/Beverages    | [link](https://figshare.com/articles/dataset/The_E-nose_Dataset_of_Maofeng_green_tea_and_Maojian_green_tea_in_our_article_A_Machine_Learning_Method_for_the_Fine-Grained_Classification_of_Green_Tea_with_Geographical_Indication_Using_a_MOS-Based_Electronic_Nose/14375927/1) | 2021 | 12  | 10  | 120    | N   | Chinese Green Tea Classification                |
| Food/Beverages    | [link](https://figshare.com/articles/dataset/The_E-nose_Dataset_of_base_liquor_and_commercial_liquor_in_our_article_A_Machine_Learning_Method_for_the_Quality_Detection_of_Base_Liquor_and_Commercial_Liquor_Using_Multidimensional_Signals_from_an_Electronic_Nose/22351597)   | 2023 | 9   | 10  | 243    | N   | Chinese Liquor Quality Detection                |
| Food/Beverages    | [link](https://data.mendeley.com/datasets/v52nr5jky5/1)                                                                                                                                                                                                                         | 2023 | 2   | 10  | 100    | N   | Test Analysis of Cocoa Butter's Free Fatty Acid |
| Food/Beverages    | [link](https://doi.org/10.34820/FK2/NNAL9K)                                                                                                                                                                                                                                     | 2023 | 2   | 6   | 234    | N   | Gambung Green Tea Quality Identification        |
| Food/Beverages    | [link](https://dx.doi.org/10.21227/mddf-qn59)                                                                                                                                                                                                                                   | 2023 | 2   | 5   | 130    | N   | Seafood Quality Detection                       |
| Healthcare        | [link](https://data.mendeley.com/datasets/h5pcn99zw4/4 )                                                                                                                                                                                                                        | 2021 | 4   | 8   | 78     | N   | COPD Detection                                  |
| Healthcare        | [link](https://doi.org/10.5061/dryad.v3g17)                                                                                                                                                                                                                                     | 2021 | 5   | 6   | 101    | N   | Urinary Pathogens Detection                     |
| Healthcare        | [link](https://zenodo.org/records/14053366)                                                                                                                                                                                                                                     | 2025 | 2   | 3   | 900    | N   | Rapid Sepsis Diagnosis                          |
| Breeding          | [link](https://www.mdpi.com/1424-8220/24/5/1628)                                                                                                                                                                                                                                | 2024 | 3   | 8   | 150    | N   | Ammonia Emission Detection                      |
| Common            | [link](https://dx.doi.org/10.21227/nmmk-0t19)                                                                                                                                                                                                                                   | 2024 | 12  | 4   | 12     | N   | Odor Detection                                  |
| Common            | Ours                                                                                                                                                                                                                                                                            | 2026 | 52  | 16  | 520    | Y   | Daily Item Classification                       |

# Collected Items

| ID | Category  | Label                                      | Name                                  | Form         | Color              | Volume/Weight | Alcoholicity | Temperature (Celsius) | Container          |
|----|-----------|--------------------------------------------|---------------------------------------|--------------|--------------------|---------------|--------------|-----------------------|--------------------|
| 1  | Beer      | kelly                                      | Kelly                                 | Liquid       | Yellow Transparent | 150 ml        | 4.50\%       | -                     | Beer Glass         |
| 2  | Beer      | cass                                       | Cass                                  | Liquid       | Yellow Transparent | 150 ml        | 4.50\%       | -                     | Beer Glass         |
| 3  | Beverage  | apple\_juice                               | Apple Juice                           | Liquid       | Yellow Transparent | 150 ml        | -            | -                     | Beer Glass         |
| 4  | Beverage  | plum                                       | Plum                                  | Liquid       | Yellow Transparent | 150-200 ml    | -            | -                     | Beer Glass         |
| 5  | Beverage  | grapefruit\_sparkling                      | Grapefruit Sparkling                  | Liquid       | Transparent        | 150 ml        | -            | -                     | Beer Glass         |
| 6  | Beverage  | lime\_sparkling                            | Lime Sparkling                        | Liquid       | Transparent        | 150 ml        | -            | -                     | Beer Glass         |
| 7  | Beverage  | chilsung\_cider                            | Chilsung Cider                        | Liquid       | Transparent        | 150 ml        | -            | -                     | Beer Glass         |
| 8  | Beverage  | pear\_juice                                | 170 days pear                         | Liquid       | Yellow Transparent | 150 ml        | -            | -                     | Beer Glass         |
| 9  | Beverage  | lemonade                                   | Sotill Lemonade                       | Liquid       | Yellow Transparent | 150 ml        | -            | -                     | Beer Glass         |
| 10 | Commodity | garlin\_original                           | Garlin Original                       | Liquid       | Transparent        | 20-25 ml      | -            | -                     | Liquor Glass       |
| 11 | Commodity | garlin\_apple                              | Garlin Apple                          | Liquid       | Transparent        | 20-25 ml      | -            | -                     | Liquor Glass       |
| 12 | Commodity | garlin\_green\_grape                       | Garlin Green Grape                    | Liquid       | Transparent        | 20-25 ml      | -            | -                     | Liquor Glass       |
| 13 | Commodity | garlin\_strawberry                         | Garlin Strawberry                     | Liquid       | Transparent        | 20-25 ml      | -            | -                     | Liquor Glass       |
| 14 | Commodity | fruit\_veggie\_wash                        | Fruit Veggie Wash                     | Liquid       | Transparent        | 20-30 ml      | -            | -                     | One-off Sauce Dish |
| 15 | Commodity | bubble\_hand\_wash\_green\_herb            | Bubble Hand Wash Green Herb           | Liquid       | Transparent        | 10-15 ml      | -            | -                     | One-off Sauce Dish |
| 16 | Commodity | bubble\_hand\_wash\_lemon                  | Bubble Hand Wash Lemon                | Liquid       | Transparent        | 10-15 ml      | -            | -                     | One-off Sauce Dish |
| 17 | Liquor    | saero\_zero\_sugar                         | Saero zero sugar                      | Liquid       | Transparent        | 30 ml         | 16\%         | -                     | Liquor Glass       |
| 18 | Liquor    | good\_day\_clean                           | Good day clean                        | Liquid       | Transparent        | 30 ml         | 16\%         | -                     | Liquor Glass       |
| 19 | Liquor    | kong\_fu\_jia\_jiu                         | Kong Fu Jia Jiu                       | Liquid       | Transparent        | 20-25 ml      | 32\%         | -                     | Liquor Glass       |
| 20 | Liquor    | kaori\_hanayagu                            | KAORI-HANAYAGU                        | Liquid       | Yellow Transparent | 30 ml         | 15-16\%      | -                     | Liquor Glass       |
| 21 | Sauce     | carbo\_buldak\_sauce                       | Carbo Buldak Sauce                    | Liquid       | Red                | 10g           | -            | -                     | One-off Sauce Dish |
| 22 | Sauce     | hack\_buldak\_sauce                        | Hack Buldak Sauce                     | Liquid       | Red                | 10g           | -            | -                     | One-off Sauce Dish |
| 23 | Sauce     | ketchup                                    | Ketchup                               | Liquid       | Red                | 20-25g        | -            | -                     | One-off Sauce Dish |
| 24 | Sauce     | thai\_sriracha\_sauce                      | Thai Sriracha Sauce                   | Liquid       | Red                | 20-25g        | -            | -                     | One-off Sauce Dish |
| 25 | Sauce     | sweet\_sour\_gochujang                     | Sweet  Sour Gochujang                 | Liquid       | Red                | 20-25g        | -            | -                     | One-off Sauce Dish |
| 26 | Sauce     | pickled\_onion\_sauce                      | Pickled Onion Sauce                   | Liquid       | Red Transparent    | 10-15g        | -            | -                     | One-off Sauce Dish |
| 27 | Sauce     | vinegar\_with\_gochujang                   | Vinegar With Gochujang                | Liquid       | Red                | 20-25g        | -            | -                     | One-off Sauce Dish |
| 28 | Sauce     | distilled\_white\_vinegar                  | Distilled White Vinegar               | Liquid       | Transparent        | 20-25g        | -            | -                     | One-off Sauce Dish |
| 29 | Sauce     | teriyaki\_sauce                            | Teriyaki Sauce                        | Liquid       | Brown Transparent  | 15g           | -            | -                     | One-off Sauce Dish |
| 30 | Sauce     | apple\_cider\_vinegar                      | Apple Cider Vinegar                   | Liquid       | Yellow Transparent | 20-25g        | -            | -                     | One-off Sauce Dish |
| 31 | Sauce     | glaze\_with\_balsamic\_vinegar\_of\_modena | Glaze with Balsamic Vinegar of Modena | Liquid       | Black              | 15-20g        | -            | -                     | One-off Sauce Dish |
| 32 | Sauce     | cooking\_wine                              | Cooking Wine                          | Liquid       | Yellow Transparent | 15-20g        | -            | -                     | One-off Sauce Dish |
| 33 | Sauce     | arafuku\_sushi\_vinegar                    | Arafuku Sushi Vinegar                 | Liquid       | Yellow Transparent | 20\_25g       | -            | -                     | One-off Sauce Dish |
| 34 | Sauce     | soy\_sauce                                 | Soy Sauce                             | Liquid       | Black              | 10-15g        | -            | -                     | One-off Sauce Dish |
| 35 | Sauce     | korean\_oyster\_tuna\_sauce                | Korean Oyster Tuna Sauce              | Liquid       | Black              | 15-20g        | -            | -                     | One-off Sauce Dish |
| 36 | Sauce     | yondu\_strong                              | Yondu Strong                          | Liquid       | Yellow Transparent | 10-15g        | -            | -                     | One-off Sauce Dish |
| 37 | Sauce     | extra\_virgin\_olive\_oil                  | Extra Virgin Olive Oil                | Liquid       | Green Transparent  | 5-10g         | -            | -                     | One-off Sauce Dish |
| 38 | Sauce     | sesame\_oil                                | Sesame Oil                            | Liquid       | Yellow Transparent | 5-10g         | -            | -                     | One-off Sauce Dish |
| 39 | Seasoner  | cumin                                      | Cumin                                 | Solid Powder | Brown              | $<1$g         | -            | -                     | One-off Sauce Dish |
| 40 | Seasoner  | herb\_salt                                 | Herb Salt                             | Solid Powder | White              | 3g            | -            | -                     | One-off Sauce Dish |
| 41 | Seasoner  | black\_pepper                              | Black Pepper                          | Solid Powder | Black              | $<1$g         | -            | -                     | One-off Sauce Dish |
| 42 | Seasoner  | oregano                                    | Oregano                               | Solid Powder | Green              | $<1$g         | -            | -                     | One-off Sauce Dish |
| 43 | Seasoner  | garlic                                     | Garlic                                | Solid Powder | Yellow             | 3g            | -            | -                     | One-off Sauce Dish |
| 44 | Seasoner  | ginger                                     | Ginger                                | Solid Powder | Yellow             | 2g            | -            | -                     | One-off Sauce Dish |
| 45 | Tea       | earl\_grey\_tea                            | Earl Grey Tea                         | Liquid       | Yellow Transparent | 150-200 ml    | -            | 90-100                | Beer Glass         |
| 46 | Tea       | hibiscus\_apple\_blend\_tea                | Hibiscus Apple Blend Tea              | Liquid       | Red Transparent    | 200 ml        | -            | 90-100                | Beer Glass         |
| 47 | Tea       | rosehip\_hibiscus\_tea                     | Rosehip Hibiscus Tea                  | Liquid       | Red Transparent    | 150-200 ml    | -            | 90-100                | Beer Glass         |
| 48 | Tea       | muscat\_black\_tea                         | Muscat Black Tea                      | Liquid       | Yellow Transparent | 150-200 ml    | -            | 90-100                | Beer Glass         |
| 49 | Tea       | peach\_oolong                              | Peach Oolong                          | Liquid       | Yellow Transparent | 150-200 ml    | -            | 90-100                | Beer Glass         |
| 50 | Tea       | peacock\_organic\_green\_tea               | Peacock Organic Green Tea             | Liquid       | Green Transparent  | 150-200 ml    | -            | 90-100                | Beer Glass         |
| 51 | Tea       | schisandra\_berry                          | Schisandra Berry                      | Liquid       | Red Transparent    | 200-250 ml    | -            | 30-60                 | Beer Glass         |
| 52 | Water     | water                                      | Water                                 | Liquid       | Transparent        | 150 ml        | -            | -                     | Beer Glass         |
| 53 | Water     | water                                      | Hot Water                             | Liquid       | Transparent        | 150 ml        | -            | -                     | Beer Glass         |


# Copyright
```text
This project follows the CC BY-NC 4.0 license.
```
