# Mario GP-T

This repository contains the code for reproducing the experimental pipeline of two papers: "Large Language Model-based Test Case Generation for GP Agents" and "Policy Search through Genetic Programming and LLM-assisted Curriculum Learning" (see the full citations at the end of the readme).

The core idea of both papers is to optimize a controller for Super Mario Bros using Graph-based Genetic Programming (GGP).
The controller is optimized using curriculum learning on a set of levels generated with Mario-GPT, a fine tuned GPT model.

<img width="450" height="450" alt="super-mario-bros-image" src="https://github.com/user-attachments/assets/9aa3edc5-055f-48f1-8af0-8a4c8351dcea" />

## Repository content description
This repository contains various elements:
- `cgpax` is the Python package which contains all the artifacts related to graph optimization with GPP. It is taken and adapted from [cgpax](https://github.com/giorgia-nadizar/cgpax).
- `configs` contains all the config files in `yaml` format with the specifications of the various experiments (both optimization and validation).
- `curriculum` is the Python package dealing with the generation and organization of test cases (i.e., Super Mario Bros levels) into a curriculum.
- `levels` contains the Super Mario Bros levels used as test cases for optimization. All levels are specificied in a textual format; an image representation is provided in the subfolder `imgs`.
- `mario_gym` is the Python package containing the gym wrapper for the Super Mario Bros simulation environment, which is originally written in Java.
- `original_levels` contains a sample of 15 original levels from the Super Mario Bros game.

## Run instructions

### Python configuration

Create the conda environment and install the requirements.
```bash
conda create --name mario python=3.11
conda activate mario
pip install -r requirements.txt
```

### Running an experiment

1. Start the Java servers which will compute the simulation of Mario.
   The following command will start 100 servers on 100 consecutive ports starting from port 25000.
   These parameters can be passed to PlayPython when running it, with the initial port first and the number of servers
   second.

```bash
 java -cp "server/mario/*:server/libs/*" PlayPython
```

2. Start the Python evolution script with the appropriate config file.
   For the config file, it must be located in the `configs` folder, and its name should be specified without location.

```bash
python3 main.py config_file_name
```

3. Files will be saved to the `results` folder, within an internal folder named after the run.

4. Kill the Java server once done.
   While the Python process will close itself at the end of computation, the Java one will not.

## Notes

- The Java code for the server is available
  at [https://github.com/giorgia-nadizar/MarioAI](https://github.com/giorgia-nadizar/MarioAI)
- The code for generating the Mario levels is adapted
  from [https://github.com/shyamsn97/mario-gpt](https://github.com/shyamsn97/mario-gpt)

## Citation
If you use this code in your project please cite
```
@inproceedings{jorgensen2024large,
  title={Large Language Model-based Test Case Generation for GP Agents},
  author={Jorgensen, Steven and Nadizar, Giorgia and Pietropolli, Gloria and Manzoni, Luca and Medvet, Eric and O'Reilly, Una-May and Hemberg, Erik},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  pages={914--923},
  year={2024}
}
```
```
@article{jorgensen2025policy,
  title={Policy Search through Genetic Programming and LLM-assisted Curriculum Learning},
  author={Jorgensen, Steven and Nadizar, Giorgia and Pietropolli, Gloria and Manzoni, Luca and Medvet, Eric and O'Reilly, Una-May and Hemberg, Erik},
  journal={Under review},
  year={2025}
}
```
