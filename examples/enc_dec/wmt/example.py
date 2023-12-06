import torch
'''
Hello World: [329, 14044, 682, 812, 2]
Good Morning! How are you doing? : [9938, 5384, 9328,  812, 3619,   53,  181, 3829, 1735,  171,    2]
'''
# Load an En-Fr Transformer model trained on WMT'14 data :
en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')

# Use the GPU (optional):
en2fr.cuda()
text = "Good Morning! How are you doing today?"
# Translate with beam search:
fr = en2fr.translate(text, beam=1)

# Manually tokenize:
en_toks = en2fr.tokenize(text)

# Manually apply BPE:
en_bpe = en2fr.apply_bpe(en_toks)

# Manually binarize:
en_bin = en2fr.binarize(en_bpe)
print(f"input_text : {text}, input_ids: {en_bin}")
# assert en_bin.tolist() == [329, 14044, 682, 812, 2]

# Generate five translations with top-k sampling:
fr_bin = en2fr.generate(en_bin, beam=1)

# Convert one of the samples to a string and detokenize
fr_sample = fr_bin[0]['tokens']
print(f"output_text: {fr}, output_ids={fr_bin[0]}")
# fr_bpe = en2fr.string(fr_sample)
# fr_toks = en2fr.remove_bpe(fr_bpe)
# fr = en2fr.detokenize(fr_toks)
# assert fr == en2fr.decode(fr_sample)
