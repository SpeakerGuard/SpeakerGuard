
# News
- The paper releasing **SpeakerGuard** has been accepted by IEEE Transactions on Dependable and Secure Computing (TDSC), 2022.

# SpeakerGuard
<!-- This repository contains the code for SpeakerGuard, a Pytorch library for adversarial machine learning research on speaker recognition. -->
This repository contains the code for SpeakerGuard, a Pytorch library for security research on speaker recognition.

<!-- Paper: Anonymous Submission to a conference (Under Review Currently) -->
Paper: [SpeakerGuard Paper](https://arxiv.org/abs/2206.03393)

Website: [SpeakerGuard Website](https://speakerguard.github.io/)

Feel free to use SpeakerGuard for academic purpose 😄. For commercial purpose, please contact us 📫.

Cite our paper as follow:
```
@article{SpeakerGuard,
  author    = {Guangke Chen and
               Zhe Zhao and
               Fu Song and
               Sen Chen and
               Lingling Fan and
               Feng Wang and 
               Jiashui Wang},
  title     = {Towards Understanding and Mitigating Audio Adversarial Examples for Speaker Recognition},
  journal   = {IEEE Transactions on Dependable and Secure Computing},
  year      = {2022}
}
```

# 1. Usage
## 1.1 Requirements
pytorch=1.6.0, torchaudio=0.6.0, numpy=1.19.2, scipy=1.4.1, 
[libKMCUDA=6.2.3](https://github.com/src-d/kmcuda), kmeans-pytorch=0.3, torch-lfilter=0.0.3, 
pesq=0.0.2, pystoi=0.3.3, librosa=0.8.0, kaldi-io=0.9.4

If you don't have GPU, you can skip libKMCUDA.

If you want to use speech_compression methods in `defense/speech_compression.py`, you should also install `ffmpeg` and the required de/en-coders. See this [instructions](instructions_ffmpeg.md).

## 1.2 Dataset Preparation
We provide five datasets, namely, Spk10_enroll, Spk10_test, Spk10_imposter, Spk251_train and Spk_251_test. They cover all the recognition tasks (i.e., CSI-E, CSI-NE, SV and OSI). The code in `./dataset/Dataset.py` will download them automatically when they are used. You can also manually download them using the follwing links:

[Spk10_enroll.tar.gz, 18MB, MD5:0e90fb00b69989c0dde252a585cead85](https://drive.google.com/uc?id=1BBAo64JOahk0F3yBAovnRLZ1NvjwBy7y&export\=download)

[Spk10_test.tar.gz, 114MB, MD5:b0f8eb0db3d2eca567810151acf13f16](https://drive.google.com/uc?id=1WctqJtP5Es74-U7y3cFXqfHi7JkDz6g5&export\=download)

[Spk10_imposter.tar.gz, 212MB, MD5:42abd80e27b78983a13b74e44a67be65](https://drive.google.com/uc?id=1f1GULs0aj_Xrw8JRxe6zzvTN3r2nnOf6&export\=download)

[Spk251_train.tar.gz, 10GB, MD5:02bee7caf460072a6fc22e3666ac2187](https://drive.google.com/uc?id=1iGcMPiPMzcCLI7xKJLwH1L0Ff_95-tmB&export\=download)

[Spk251_test.tar.gz, 1GB, MD5:182dd6b17f8bcfed7a998e1597828ed6](https://drive.google.com/uc?id=1rsXzuEyi5Zqd1XAsr1_Op7mC7hqY0tsp&export\=download)

After downloading, untar them inside `./data` directory.

## 1.3 Model Preparation
### 1.3.1 Speaker Enroll (CSI-E/SV/OSI tasks)
<!-- - Download [iv_system, MD5:bfe90ec7782b54dc295e72b5bf789377](https://drive.google.com/uc?id=13yDZvM6a7W1Str2KEI7Vrm2xSdxWe7Vv&export\=download) and [xv_system, MD5:37cb3e7ca48c0da3ae72a35195aacf58](https://drive.google.com/uc?id=1HbpR6cUuPzDQLVvQTFUpIAflEa1eP-XF&export\=download), and untar them inside the reposity directory (i.e., `./`). Iv_system and xv_system contain the pre-trained ivector-PLDA and xvector-PLDA background models. -->
- Download [pre-trained-models.tar.gz, 340MB, MD5:b011ead1e6663d557afa9e037f30a866](https://drive.google.com/uc?id=1kxsSr7V_DbPloUPeqgsxUbuHsV6rBveH&export\=download) and untar it inside the reposity directory (i.e., `./`). It contains the pre-trained ivector-PLDA and xvector-PLDA background models.
- Run `python enroll.py iv_plda` and `python enroll.py xv_plda` 
to enroll the speakers in Spk10_enroll for ivector-PLDA and xvector-PLDA systems. 
Multiple speaker models for CSI-E and OSI tasks are stored as `speaker_model_iv_plda` and `speaker_model_xv_plda` inside `./model_file`. 
Single speaker models for SV task are  stored as `speaker_model_iv_plda_{ID}` and `speaker_model_xv_plda_{ID}` inside `./model_file`.
- Run `python set_threshold.py iv_plda` and 
`python set_threshold.py xv_plda` to set the threshold of SV/OSI tasks (also test the EER of SV/OSI tasks and the accuracy of CSI-E task).

### 1.3.2 Natural Training (CSI-NE task)
- Sole natural training: 
  ```
  python natural_train.py -num_epoches 30 -batch_size 128 -model_ckpt ./model_file/natural-audionet -log ./model_file/natural-audionet-log
  ```
- Natural training with QT (q=512)
  ```
  python natural_train.py -defense QT -defense_param 512 -defense_flag 0 -model_ckpt ./model_file/QT-512-natural-audionet -log ./model_file/QT-512-natural-audionet-log
  ```
  Note: `-defense_flag 0` means QT operates at the waveform level.

### 1.3.3 Adversarial Training (CSI-NE task)
- Sole FGSM adversarial training:
  ```
  python adver_train.py -attacker FGSM -epsilon 0.002 -model_ckpt ./model_file/fgsm-adver-audionet -log ./model_file/fgsm-adver-audionet-log -evaluate_adver
  ```
- Sole PGD adversarial training:
  ```
  python adver_train.py -attacker PGD -epsilon 0.002 -max_iter 10 -model_ckpt ./model_file/pgd-adver-audionet -log ./model_file/pgd-adver-audionet-log
  ```
- Combining adversarial training with input transformation AT (randomized, should use EOT during training)
    ```
  python adver_train.py -defense AT -defense_param 16 -defense_flag 0 -attacker PGD -epsilon 0.002 -max_iter 10 -EOT_size 10 -EOT_batch_size 5 -model_ckpt ./model_file/AT-16-pgd-adver-audionet -log ./model_file/AT-16-pgd-adver-audionet-log
  ```

## 1.4 Generate Adversarial Examples
- Example 1: FAKEBOB attack on naturally-trained audionet model with QT (q=512)
  ```
  python attackMain.py -task CSI -root ./data -name Spk251_test -des ./adver-audio/QT-512-audionet-fakebob audionet_csine -extractor ./model_file/QT-512-natural-audionet FAKEBOB -epsilon 0.002
  ```

- Example 2: PGD targeted attack on FeCo-defended xvector-plda model for OSI task. FeCo is randomized, using EOT
  ```
  python attackMain.py -threshold 18.72 -defense FeCo -defense_param "kmeans 0.2 L2" -defense_flag 1 -root ./data -name Spk10_imposter -des ./adver-audio/xv-pgd -task OSI -EOT_size 5 -EOT_batch_size 5 -targeted xv_plda -model_file ./model_file/xv_plda/speaker_model_xv_plda PGD -epsilon 0.002 -max_iter 5 -loss Margin
  ```

  Note: `-defense_flag 1` means we want FeCo to operate at the raw acoustic feature level. 
  Set `-defense_flag 2` or `-defense_flag 3` for delta or cmvn acoustic feature level. 

## 1.5 Evaluate Adversarial Examples
- Example 1: Testing for unadaptive attack
  ```
  python test_attack.py -defense QT -defense_param 512 -defense_flag 0 -root ./adver-audio -name QT-512-audionet-fakebob -root_ori ./data -name_ori Spk251_test audionet_csine -extractor ./model_file/QT-512-natural-audionet
  ```
- Example 2: Testing for adaptive attack
  ```
  python test_attack.py -threshold 18.72 -defense FeCo -defense_param "kmeans 0.2 L2" -defense_flag 1 -root ./adver-audio -name xv-pgd xv_plda -model_file ./model_file/xv_plda/speaker_model_xv_plda
  ```

In Example 1, the adversarial examples are generated on undefended audionet model, but tested on QT-defended audionet model, so it is **non-adaptive** attack.

In Example 2, the adversarial examples are generated on FeCo-defended xvector-plda model using EOT (to overcome the randomness of FeCo), and also tested on FeCo-defended xvector-plda model, so it is **adaptive** attack. 
In this example, the adaptive attack may be not strong enough. 
You can improve its attack capacity by setting a larger max_iter or larger EOT_size at the cost of increased attack overhead.

By default, targeted attack randomly selects the targeted label. If you want to control the targeted label, you can run `specify_target_label.py` and input the generated target label file to `attackMain.py` and `test_attack.py`.

`test_attack.py` can also be used to test the benign accuracy of systems. Just let `-root` and `-name` point to the benign dataset.

You can also try the combination of different transformation-based defenses, e.g., 
```
-defense QT AT FeCo -defense_param 512 16 "kmeans 0.5 L2" -defense_flag 0 0 1 -defense_order sequential
```
where `-defense_order` specifies the combination way (sequential or average).

# 2. Extension
If you would like to incorporate your attacks/defenses/models/datasets into our official repositor 
so that everyone can access them (also as a way to propaganda your works), feel free to make a pull resuest or contact us.

## MC (Model Component)
MC contains three state-of-the-art embedding-based speaker recognition models, i.e., ivector-PLDA, xvector-PLDA and AudioNet. Xvector-PLDA and AudioNet are based on neural networks while ivector-PLDA on statistic model (i.e Gaussian Mixture Model).

The flexibility and extensibility of SpeakerGuard make it easy to add new models. 
<!-- Just wrap the model as `torch.nn.Module` and implement `forward`, `score` and `make_decision` methods. -->
To add a new model, one can define a new subclass of the `torch.nn.Module` class and implement three methods: `forward`, `score`, and `make_decision` , then it can be evaluated using different attacks.

## DAC (Dataset Component)
We provide five datasets, namely, Spk10_enroll, Spk10_test, Spk10_imposter, Spk251_train and Spk_251_test. They cover all the recognition tasks (i.e., CSI-E, CSI-NE, SV and OSI). 

<!-- To add new datasets, one just need to define a class inheriting from `torch.utils.data.Dataset`, just like `dataset/Dataset.py`. -->
All our datasets are subclasses of the class `torch.utils.data.Dataset`. Hence, to add a new dataset, one just need to define a new subclass of `torch.utils.data.Dataset` and implement two methods: `__len__` and `__getitem__`, which defines the length and loading sequence of the dataset.

## AC (Attack Component)
SpeakerGuard currently incorporate four white-box attacks (FGSM, PGD, CW$_\infty$ and CW$_2$) and two black-box attacks (FAKEBOB and SirenAttack). 
<!-- To incorporate new attack algorithms, one just need to inhert from the class in `attack/Attack.py` and implement the abstract method `attack`. -->
To add a new attack, one can define a new subclass of the abstract class `Attack` and implement the `attack` method. This design ensures that the `attack` methods in different concrete `Attack` classes have the same method signature, i.e., unified API.

## DEC (Defense Component)
To secure SRSs from adversarial attack, SpeakerGuard provides 2 robust training methods (FGSM and PGD adversarial training) and 22 speech/speaker-dedicated input transformation methods, including our feature-level approach FEATURE COMPRESSION (FeCo). 
<!-- All input transformation methods are implemented as standalone python functions, making it easy to extend new methods. -->
Since all our defenses are standalone functions, adding a new defense is straightforward, one just needs to implement it as a python function accepting the input audios or features as one of its arguments.

## ADAC (Adaptive Attack Component)
All these adaptive attack techniques are implemented as standalone wrappers so that they can be easily plugged into attacks to mount adaptive attacks.
