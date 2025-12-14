# Scale it up

The tiny 40M param MDLM seems to work, so i intend to scale it up a bit to see if there are substantial performance jump

## Scaled MDLM

* 164M parameters, roughly GPT-2 size (40M --> 164M)
* Trained on TinyStories dataset of 766M tokens under this tokenizer (1M --> 766M)
* A Rust implementation of BPE tokenizer for training and encoding (fast and low mem footprint)
* Encode data as uint16 np array, using memory map lazy dataloader for minimum memory overhead
* Swapped in Flash Attention and gradient accumulation for better efficiency
* Use standard Train/Val this time
* A React app for inference denoising visualization

## Resources

* The tokenized and encoded data:
    - Training data: https://drive.google.com/file/d/1PKbSRIQECqhHPBsb5-71Cn3vbRlQtV82/view?usp=drive_link

    - Validation data: https://drive.google.com/file/d/1wGjONQ6HZvHpDIKhFxeOv--IqT-VaRGG/view?usp=drive_link

    - Tokenizer: https://drive.google.com/file/d/17Nd8zuZKwqysSUw3Yo3VJKX9w-D6ugDw/view?usp=sharing

* Trained model: https://drive.google.com/file/d/1U_uArGN4Q3LKfag4hw4DRUhl89pg-vEj/view?usp=drive_link