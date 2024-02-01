import argparse
import glob
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def is_safetensor(path):
    files = glob.glob(f"{path}/*.safetensors")
    return len (files) > 0

def removeTensorKeysStartingWith(path, rm):
    checkpoint = safe_open(path, framework="pt")
    output = {}
    for k in checkpoint.keys():
        if not k.startswith(rm):
            output[k] = checkpoint.get_tensor(k)
    return output

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="Path to LLaVA v1.5 model")
args = ap.parse_args()

if (is_safetensor(args.model)):
    print("processing safetensor model")
    paths = sorted(glob.glob(f"{args.model}/*.safetensors"))

    # HACK: remove model.image_newline 
    rm_image_newline = removeTensorKeysStartingWith(paths[0], "model.image_newline")
    save_file(rm_image_newline, paths[0])

    # split out all the tensors that are part of the multimodal projector
    mmproj_checkpoints = []
    clip_checkpoints = []
    for path in paths:
        checkpoint = safe_open(path, framework="pt")
        mmproj = [k for k in checkpoint.keys() if k.startswith("model.mm_projector")]
        clip = [k for k in checkpoint.keys() if k.startswith("model.vision_tower")]
        if len(mmproj) > 0:
            mmproj_checkpoints.append({"path": path, "mmproj": mmproj})
        if len(clip) > 0:
            clip_checkpoints.append({"path": path, "clip": clip})

    # build projector and remove mmproj from each checkpoint containing it
    projector = {}
    for c in mmproj_checkpoints:
        checkpoint = safe_open(c["path"], framework="pt")
        projector.update({name: checkpoint.get_tensor(name).float() for name in c["mmproj"]})

        print("removing {} from {}".format(c["mmproj"], c["path"]))
        rm_mmproj = removeTensorKeysStartingWith(c["path"], "model.mm_projector")
        save_file(rm_mmproj, c["path"])
    
    # save the projector
    torch.save(projector, f"{args.model}/llava.projector")

    # build llava.clip and remove clip from each checkpoint containing it
    clip = {}
    for c in clip_checkpoints:
        checkpoint = safe_open(c["path"], framework="pt")
        clip.update({name.replace("vision_tower.vision_tower.", ""): checkpoint.get_tensor(name).float() for name in c["clip"]})

        print("removing {} from {}".format(len(c["clip"]), c["path"]))
        rm_clip = removeTensorKeysStartingWith(c["path"], "model.vision_tower")
        save_file(rm_clip, c["path"])

    torch.save(clip, f"{args.model}/llava.clip")

else:
    print("processing torch model")
    # find the model part that includes the the multimodal projector weights
    path = sorted(glob.glob(f"{args.model}/pytorch_model*.bin"))[-1]
    checkpoint = torch.load(path)

    # get a list of mm tensor names
    mm_tensors = [k for k, v in checkpoint.items() if k.startswith("model.mm_projector")]

    # store these tensors in a new dictionary and torch.save them
    projector = {name: checkpoint[name].float() for name in mm_tensors}
    torch.save(projector, f"{args.model}/llava.projector")

    # remove these tensors from the checkpoint and save it again
    for name in mm_tensors:
        del checkpoint[name]

    # BakLLaVA models contain CLIP tensors in it
    clip_tensors = [k for k, v in checkpoint.items() if k.startswith("model.vision_tower")]
    if len(clip_tensors) > 0:
        clip = {name.replace("vision_tower.vision_tower.", ""): checkpoint[name].float() for name in clip_tensors}
        torch.save(clip, f"{args.model}/llava.clip")

        # remove these tensors
        for name in clip_tensors:
            del checkpoint[name]



    torch.save(checkpoint, path)

# added tokens should be removed to be able to convert Mistral models
if os.path.exists(f"{args.model}/added_tokens.json"):
    with open(f"{args.model}/added_tokens.json", "w") as f:
        f.write("{}\n")

print("Done!")
print(f"Now you can convert {args.model} to a a regular LLaMA GGUF file.")
print(f"Also, use {args.model}/llava.projector to prepare a llava-encoder.gguf file.")
