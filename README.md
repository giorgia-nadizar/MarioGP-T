# Mario GP-T

Graph GP for controlling Mario.
GPT for generating the levels.

## Run instructions

1. Start the Java servers which will compute the simulation of Mario.
   The following command will start 100 servers on 100 consecutive ports starting from port 25000.
   These parameters can be passed to PlayPython when running it, with the initial port first and the number of servers
   second.

```bash
 java -cp "server/target/*:server/libs/*" PlayPython
```

2. Start the Python evolution script with the appropriate config file.
   For the config file, it must be located in the `configs` folder, and its name should be specified without location.

```bash
python3 main.py config_file_name
```

