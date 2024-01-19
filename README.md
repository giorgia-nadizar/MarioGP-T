# Mario GP-T

Graph GP for controlling Mario.
GPT for generating the levels.

## Run instructions

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