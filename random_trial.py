import os
import jax.numpy as jnp

path = "test_levels"

for filename in os.listdir(path):
    # if filename.endswith("txt"):
    #     tokens = filename.replace(".txt","").split("_")
    #     if tokens[-1] == "blocks":
    #         tokens[-1] = "blocks_0"
    #     else:
    #         tokens[-1] = str(int(tokens[-1]) + 1)
    #     new_filename = "R" + "_".join(tokens) + ".txt"
    #     os.rename(f"{path}/{filename}", f"{path}/{new_filename}")
    if filename.startswith("R"):
        new_filename = filename.replace("R", "")
        os.rename(f"{path}/{filename}", f"{path}/{new_filename}")
        # print(os.path.join(path, filename))
