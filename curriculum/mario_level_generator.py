import os

from mario_gpt import MarioLM
from py4j.java_gateway import JavaGateway

import torch

# code adapted from https://github.com/shyamsn97/mario-gpt


max_attempts = 10

quantifiers = ["no", "little", "some", "many"]
pipes = [f"{n} pipes" for n in quantifiers]
enemies = [f"{n} enemies" for n in quantifiers]
blocks = [f"{n} blocks" for n in quantifiers]

# elevations = ["", "low elevation", "high elevation"]


all_prompts = []
for pipe in pipes:
    for enemy in enemies:
        for block in blocks:
            # for elevation in elevations:
            prompt = f"{pipe}, {enemy}, {block}"
            # if len(elevation) > 0:
            #     prompt += f", {elevation}"
            all_prompts.append(prompt)

mario_lm = MarioLM()

gateway = JavaGateway()
level_scorer = gateway.entry_point

# use cuda to speed stuff up
device = torch.device("cuda")
mario_lm = mario_lm.to(device)

for i, prompt in enumerate(all_prompts):
    print(f"{i}/{len(all_prompts)} - {prompt}")
    file_name = f"levels/{prompt.replace(' ', '_').replace(',_', ',')}"
    if os.path.exists(f"{file_name}.txt"):
        print("already done")
        continue

    # TODO maybe try to decrease temp if it fails
    for attempt in range(max_attempts):
        # generate level of size 1400, pump temperature up to ~2.4 for more stochastic but playable levels
        generated_level = mario_lm.sample(
            prompts=[prompt],
            num_steps=1400,
            temperature=2.0 - 0.05 * attempt,
            use_tqdm=True
        )

        # level to string
        level_string = "\n".join(generated_level.current_levels)
        playability_result = level_scorer.score(level_string)
        if playability_result.getCompletionPercentage() < 1:
            print(f"retry {attempt + 1}")
            continue

        # save image
        generated_level.img.save(file_name + ".png")

        # save text level to file
        generated_level.save(file_name + ".txt")

        break
