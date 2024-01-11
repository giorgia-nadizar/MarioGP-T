import os

from mario_gpt import MarioLM

# code originally from https://github.com/shyamsn97/mario-gpt

quantifiers = ["no", "little", "some", "many"]
pipes = [f"{n} pipes" for n in quantifiers]
enemies = [f"{n} enemies" for n in quantifiers]
blocks = [f"{n} blocks" for n in quantifiers]
elevations = ["", "low elevation", "high elevation"]

all_prompts = []
for pipe in pipes:
    for enemy in enemies:
        for block in blocks:
            for elevation in elevations:
                prompt = f"{pipe}, {enemy}, {block}"
                if len(elevation) > 0:
                    prompt += f", {elevation}"
                all_prompts.append(prompt)

mario_lm = MarioLM()

# use cuda to speed stuff up
# import torch
# device = torch.device("cuda")
# mario_lm = mario_lm.to(device)

for i, prompt in enumerate(all_prompts):
    print(f"{i}/{len(all_prompts)} - {prompt}")
    file_name = f"levels/{prompt.replace(' ', '_')}"
    if os.path.exists(f"{file_name}.txt"):
        print("already done")
        continue

    # generate level of size 1400, pump temperature up to ~2.4 for more stochastic but playable levels
    generated_level = mario_lm.sample(
        prompts=[prompt],
        num_steps=1400,
        temperature=2.0,
        use_tqdm=True
    )

    # show string list
    print(generated_level.level)

    # save image
    generated_level.img.save(file_name + ".png")

    # save text level to file
    generated_level.save(file_name + ".txt")
    with open(file_name +"-lvl.txt", "w") as f:
        f.write("\n".join(generated_level.level))
    generated_level.play()

    # Continue generation
    # generated_level_continued = mario_lm.sample(
    #     seed=generated_level,
    #     prompts=prompts,
    #     num_steps=1400,
    #     temperature=2.0,
    #     use_tqdm=True
    # )
