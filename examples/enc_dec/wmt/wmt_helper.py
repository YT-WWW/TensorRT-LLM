'''
download WMT model from 
https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md
https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2
'''
import torch
from fairseq.models.transformer import TransformerModel

en_bin = torch.tensor([329, 14044, 682, 812, 2])
model = TransformerModel.from_pretrained("<your local path>/wmt14.en-fr.joined-dict.transformer", checkpoint_file="model.pt", bpe=None)
model.eval()
# initial run
model.generate(tokenized_sentences=en_bin, beam=1, prefix_allowed_tokens_fn=None, verbose=True)

class GetIntermediateData:
    def __init__(self, layer):
        self.layer_content = []
        self.hook = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, net_self, input, output):
        # print layer input
        # self.layer_content.append(input)
        # print layer output
        self.layer_content.append(output)

    def get(self):
        return self.layer_content
        del self.layer_content[:]

    def remove_hook(self):
        self.hook.remove()


# hook the output of selected layers
module = "encoder"
num_layer = 0

hook_wrapper = []
hook_wrapper.append(GetIntermediateData(model.models[0].encoder.embed_tokens))
hook_wrapper.append(GetIntermediateData(model.models[0].encoder.embed_positions))
hook_wrapper.append(GetIntermediateData(model.models[0].encoder.layers[num_layer].self_attn))
hook_wrapper.append(GetIntermediateData(model.models[0].encoder.layers[num_layer].self_attn_layer_norm))
hook_wrapper.append(GetIntermediateData(model.models[0].encoder.layers[num_layer].final_layer_norm))

layer_names = [
    "encoder.embed_tokens",
    "encoder.embed_positions",
    "encoder.layers[num_layer].self_attn",
    "encoder.layers[num_layer].self_attn_layer_norm",
    "encoder.layers[num_layer].final_layer_norm"
]
assert len(layer_names) == len(hook_wrapper)
# this is the call to the network 
model.generate(en_bin, beam=1, verbose=True)

# this is how to get intermediate representations
for i in range(len(layer_names)):
    print("----------------")
    print(layer_names[i])
    print(hook_wrapper[i].get())
    print("----------------")
    
