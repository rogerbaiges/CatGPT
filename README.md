# CatGPT

![CatGPT Logo](logo/CatGPT.jpg)

## Project Objective

CatGPT is a Catalan natural language model created with the goal of providing a lightweight yet effective model that can continue sentences in Catalan. This project is designed to facilitate the generation of coherent and relevant text in Catalan.

## Model Structure

CatGPT is based on a structure similar to GPT-2, but with approximately 110 million parameters. This reduction in model size allows for faster training and inference while maintaining reasonable text generation quality.

## Training Datasets

The model has been trained using various datasets, including:

- **Oscar:** A massive multilingual corpus with Catalan data.
- **Catalan_Textual Dataset:** A specifically created dataset to cover a wide range of Catalan texts.

## Tokenizer

For this project, a specific tokenizer with 32,768 different tokens has been created. This tokenizer was generated using a 50 MB subset of the training data, ensuring adequate coverage of the Catalan vocabulary.

## Document Structure

1. **CatGPT_tokenizer.ipynb:** Contains the code to create and train the tokenizer for the CatGPT model.
2. **dataset.ipynb:** Describes the steps to prepare the dataset used to train the model.
3. **CatGPT_train.py:** Script to train the CatGPT model, including model configuration and hyperparameters.
4. **CatGPT_training.ipynb:** Details the model training process with metric visualization.
5. **main.ipynb:** Examples of using the trained model to generate text from phrases in Catalan.

## How to Use the Model

To use CatGPT, simply clone the repository and execute the cells in the `main.ipynb` file. This file provides examples of how to ask the model to generate text from a phrase and how to customize this generation according to your needs.

# CatGPT

![CatGPT Logo](logo/CatGPT.jpg)

## Objectiu del Projecte

CatGPT és un model de llenguatge natural en català, creat amb l'objectiu de proporcionar un model lleuger però efectiu que pugui continuar oracions en català. Aquest projecte està dissenyat per facilitar la generació de text coherent i rellevant en català.

## Estructura del Model

CatGPT està basat en una estructura similar a GPT-2, però amb aproximadament 110 milions de paràmetres. Aquesta reducció de la mida del model permet un entrenament i una inferència més ràpids, tot i mantenir una qualitat de generació de text raonable.

## Datasets d'Entrenament

El model ha estat entrenat utilitzant diversos datasets, incloent:

- **Oscar:** Un corpus multilingüe massiu amb dades en català.
- **Catalan_Textual Dataset:** Un conjunt de dades específicament creat per cobrir una àmplia gamma de textos en català.

## Tokenizer

Per a aquest projecte, s'ha creat un tokenizer específic amb 32,768 tokens diferents. Aquest tokenizer ha estat generat utilitzant un subset de 50 MB de les dades d'entrenament, assegurant una cobertura adequada del vocabulari català.

## Estructura dels Documents

1. **CatGPT_tokenizer.ipynb:** Conté el codi per crear i entrenar el tokenizer per al model CatGPT.
2. **dataset.ipynb:** Descriu els passos per preparar el dataset utilitzat per entrenar el model.
3. **CatGPT_train.py:** Script per entrenar el model CatGPT, incloent configuració del model i hiperparàmetres.
4. **CatGPT_training.ipynb:** Detalla el procés de formació del model amb visualització de mètriques.
5. **main.ipynb:** Exemples d'ús del model entrenat per generar text a partir de frases en català.

## Com Utilitzar el Model

Per utilitzar CatGPT, simplement cal clonar el repositori i executar les cèl·lules del fitxer `main.ipynb`. Aquest fitxer proporciona exemples de com demanar al model que generi text a partir d'una frase i com personalitzar aquesta generació segons les necessitats.
