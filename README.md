# Predicting droughts at short-term scale

![Project Logo](drought_logo.jpg)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains a series of scripts to predict drought at short term scale using Machine Learning models.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Use the provided .toml file to install the python environment

## Usage

The script contains pipelines to download MSG SEVIRI images using the eumdac library and the EUMETSAT Data Tailor. Further, there is a pipeline to apply a Whittaker-filter from the modape library.
It contains an implementation of a ConvLSTM and a Graph Neural Network in the WaveNet architecture. The code is currently under development

## Contributing

In progress

## Acknowledgments

This project was supported by the Italian National PhD Program in Artificial Intelligence “AI & agrifood and environment” pillar. The research activity was conducted at the “CNR-ISAC” with the partnership and support of the University of Naples Federico II.

## License

This project is licensed under the [MIT License](LICENSE).